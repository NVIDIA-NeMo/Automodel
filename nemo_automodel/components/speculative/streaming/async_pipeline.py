# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Background-thread prefetch pipeline for speculative-decoding draft training.

:class:`AsyncFeaturePipeline` wraps a :class:`FeatureProducer` and a
:class:`SampleRefQueue` and runs the target forward in a background
thread, pushing every produced :class:`SampleRef` onto the queue
through :meth:`SampleRefQueue.put_blocks_until_below` so the queue's
HWM/LWM hysteresis (PR 1's fix) governs the producer's pacing.

The trainer-side code is unchanged from PR 2:
``FeatureDataLoader`` iterates the queue; the trainer consumes
``Eagle3TargetBatch`` instances. The only difference is that the queue
is now filled by a background thread rather than by the trainer's
main thread, so target-side forward and draft-side backward overlap.

Distributed-training note (per the Automodel principle): the pipeline
is per-rank. Each rank owns its own :class:`FeatureProducer`,
:class:`SampleRefQueue`, and :class:`FeatureStore`; FSDP / CP / EP
happen inside the trainer's forward / backward. Cross-rank
coordination (DP resharding) is PR 4.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Iterator, Protocol, runtime_checkable

import torch

from nemo_automodel.components.speculative.streaming.producer import FeatureProducer
from nemo_automodel.components.speculative.streaming.queue import SampleRefQueue

logger = logging.getLogger(__name__)


@runtime_checkable
class PromptSource(Protocol):
    """Anything the background thread can pull a prompt batch from.

    Either a zero-arg callable returning
    ``(input_ids, attention_mask, loss_mask)`` (or ``None`` when the
    source is exhausted) or an :class:`Iterator` whose items are the
    same tuple. ``StopIteration`` from an iterator is normalized to
    ``None`` so the loop has a single exhausted signal.
    """

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None: ...


class AsyncFeaturePipeline:
    """Run a :class:`FeatureProducer` in a background thread, draining a prompt source.

    Args:
        producer: The :class:`FeatureProducer` (PR 2) to invoke on each
            prompt. It carries the wrapped target backend, the store,
            and the per-call metadata.
        queue: The :class:`SampleRefQueue` to push the resulting
            :class:`SampleRef` onto. The queue's HWM/LWM hysteresis
            (PR 1's fix) paces the producer against the consumer.
        prompt_source: A :class:`PromptSource` -- either a zero-arg
            callable or an :class:`Iterator`. The callable is invoked
            from the background thread; it must be thread-safe (e.g.
            a :class:`torch.utils.data.DataLoader` iterator is).
        poll_interval: Seconds between ``prompt_source`` invocations
            after exhaustion when ``stop_on_exhausted`` is ``False``.
            Defaults to 100ms -- cheap, lets the producer resume
            quickly if more data lands.
        stop_on_exhausted: When ``True`` (default), the background
            thread exits as soon as ``prompt_source`` returns ``None``
            and :meth:`close` drains. When ``False``, the thread keeps
            polling so a streaming dataset can refill.

    Lifecycle:
        Construct, then :meth:`start` (or use the context manager).
        The background thread is daemon; ``stop`` joins it within
        ``join_timeout`` seconds. Outstanding leases are the trainer's
        responsibility -- the pipeline does not drop them on close.
    """

    def __init__(
        self,
        producer: FeatureProducer,
        queue: SampleRefQueue,
        prompt_source: PromptSource | Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        *,
        poll_interval: float = 0.1,
        stop_on_exhausted: bool = True,
    ) -> None:
        self._producer = producer
        self._queue = queue
        # Normalize an iterator into a callable that pulls ``next()``
        # and converts ``StopIteration`` into ``None`` so the loop has
        # a single "exhausted" signal.
        if isinstance(prompt_source, Iterator):
            self._iterator: Iterator | None = prompt_source
            self._prompt_source: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None] = (
                self._pull_from_iterator
            )
        else:
            self._iterator = None
            self._prompt_source = prompt_source
        self._poll_interval = poll_interval
        self._stop_on_exhausted = stop_on_exhausted
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._joined_event = threading.Event()
        self._error: BaseException | None = None

    def _pull_from_iterator(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        assert self._iterator is not None
        try:
            value = next(self._iterator)
        except StopIteration:
            return None
        # Iterators that yield a single ``input_ids`` tensor are common
        # in tests; fall back to (input_ids, attention_mask, loss_mask)
        # only when the iterator yields the full 3-tuple.
        if isinstance(value, tuple) and len(value) == 3:
            return value
        if isinstance(value, torch.Tensor):
            attn = torch.ones_like(value, dtype=torch.long)
            loss = torch.ones_like(value, dtype=torch.long)
            return value, attn, loss
        raise TypeError(
            f"prompt iterator must yield (input_ids, attention_mask, loss_mask) or a single "
            f"Tensor; got {type(value).__name__}"
        )

    def start(self) -> None:
        """Spawn the background producer thread.

        Idempotent: a second call while the thread is alive is a
        no-op. The thread is named ``streaming-async-producer`` so
        test failures and runtime traces show which thread is hung.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._joined_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="streaming-async-producer",
            daemon=True,
        )
        self._thread.start()
        logger.debug("AsyncFeaturePipeline started")

    def stop(self, *, join_timeout: float | None = 10.0) -> None:
        """Signal the background thread to exit and join it.

        Outstanding leases are the trainer's; this method does not
        ack them. Idempotent. If the background thread raises, the
        exception is re-raised here so the trainer sees the failure.
        """
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout)
        if thread is not None and thread.is_alive():
            logger.warning("AsyncFeaturePipeline background thread did not exit within timeout")
        if self._error is not None:
            err = self._error
            self._error = None
            raise err

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._joined_event.set()

    def __enter__(self) -> "AsyncFeaturePipeline":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                prompt = self._prompt_source()
                if prompt is None:
                    if self._stop_on_exhausted:
                        logger.debug("AsyncFeaturePipeline prompt source exhausted; stopping")
                        break
                    # Streaming mode: wait for more data. ``stop_event``
                    # short-circuits the wait so shutdown is responsive.
                    if self._stop_event.wait(timeout=self._poll_interval):
                        break
                    continue
                input_ids, attention_mask, loss_mask = prompt
                ref = self._producer.produce(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    loss_mask=loss_mask,
                )
                # put_blocks_until_below honors the queue's HWM/LWM
                # hysteresis; a fast producer here naturally blocks
                # when the store is at high watermark.
                self._queue.put_blocks_until_below(
                    ref,
                    poll_interval=self._poll_interval,
                )
        except BaseException as e:  # noqa: BLE001 -- capture then re-raise from stop()
            logger.exception("AsyncFeaturePipeline background thread failed")
            self._error = e
        finally:
            self._queue.close()


__all__ = ["AsyncFeaturePipeline", "PromptSource"]

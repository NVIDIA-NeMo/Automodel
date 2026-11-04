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
HWM/LWM hysteresis governs the producer's pacing.

The trainer-side :class:`~nemo_automodel.components.speculative.streaming.loader.FeatureDataLoader`
iterates the queue; the trainer consumes ``Eagle3TargetBatch`` instances.
The queue is filled by a background thread rather than by the trainer's
main thread, so target-side forward and draft-side backward overlap.

Distributed-training note: the pipeline is per-rank. Each rank owns its own
:class:`FeatureProducer`, :class:`SampleRefQueue`, and :class:`FeatureStore`;
FSDP / CP / EP happen inside the trainer's forward / backward. Cross-rank
sample routing is handled above this layer.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Iterator, Protocol, TypeAlias, runtime_checkable

import torch

from nemo_automodel.components.speculative.streaming.producer import FeatureProducer
from nemo_automodel.components.speculative.streaming.queue import SampleRefQueue

logger = logging.getLogger(__name__)

_PromptBatch: TypeAlias = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
_PackedPromptBatch: TypeAlias = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


@runtime_checkable
class PromptSource(Protocol):
    """Zero-argument callable that supplies the next prompt batch."""

    def __call__(self) -> _PromptBatch | _PackedPromptBatch | None:
        """Return the next prompt batch or ``None`` when the source is exhausted.

        Returns:
            A three-tuple ``(input_ids, attention_mask, loss_mask)`` where
            each item is a ``torch.long`` Tensor of shape ``[batch, sequence]``;
            or a six-tuple that appends ``position_ids`` and ``doc_remaining``
            (each a ``torch.long`` Tensor of shape ``[batch, sequence]``) and
            ``seq_lens`` (a ``torch.long`` Tensor of shape ``[batch, max_docs]``
            containing packed document lengths); or ``None`` when exhausted.
        """
        ...


class AsyncFeaturePipeline:
    """Run a :class:`FeatureProducer` in a background thread, draining a prompt source.

    Args:
        producer: The :class:`FeatureProducer` to invoke on each prompt.
            It carries the wrapped target backend, the store, and the
            per-call metadata.
        queue: The :class:`SampleRefQueue` to push the resulting
            :class:`SampleRef` onto. The queue's HWM/LWM hysteresis
            paces the producer against the consumer.
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
        prompt_source: PromptSource | Iterator[_PromptBatch | _PackedPromptBatch | torch.Tensor],
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
            self._iterator: Iterator[_PromptBatch | _PackedPromptBatch | torch.Tensor] | None = prompt_source
            self._prompt_source: Callable[[], _PromptBatch | _PackedPromptBatch | None] = self._pull_from_iterator
        else:
            self._iterator = None
            self._prompt_source = prompt_source
        self._poll_interval = poll_interval
        self._stop_on_exhausted = stop_on_exhausted
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._error: BaseException | None = None

    def _pull_from_iterator(self) -> _PromptBatch | _PackedPromptBatch | None:
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
        if isinstance(value, tuple) and len(value) == 6:
            return value
        if isinstance(value, torch.Tensor):
            attn = torch.ones_like(value, dtype=torch.long)
            loss = torch.ones_like(value, dtype=torch.long)
            return value, attn, loss
        raise TypeError(
            f"prompt iterator must yield (input_ids, attention_mask, loss_mask) or a six-tuple "
            f"with packing metadata (position_ids, seq_lens, doc_remaining), or a single "
            f"Tensor; got {type(value).__name__} of length {len(value) if isinstance(value, tuple) else 'n/a'}"
        )

    def start(self) -> None:
        """Spawn the background producer thread.

        Idempotent while the thread is alive: a second call is a no-op. The
        thread is named ``streaming-async-producer`` so test failures and
        runtime traces show which thread is hung.

        The pipeline is single-use. When the producer thread exits -- prompt
        source exhausted, :meth:`stop`, or an error -- the bound queue is
        closed in ``_run``'s ``finally`` so the consumer drains and stops. A
        closed queue cannot reopen, so restarting is not supported; construct
        a fresh pipeline (and queue) to stream again. Restart is rejected here
        rather than failing later with an opaque "queue is closed" from a
        blocking put.
        """
        if self._thread is not None and self._thread.is_alive():
            return
        if self._queue.is_closed:
            raise RuntimeError(
                "AsyncFeaturePipeline is single-use: its queue is already closed. "
                "Construct a new pipeline and queue to stream again."
            )
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
                packing_kwargs = {}
                if isinstance(prompt, tuple) and len(prompt) == 6:
                    input_ids, attention_mask, loss_mask, position_ids, seq_lens, doc_remaining = prompt
                    packing_kwargs = {
                        "position_ids": position_ids,
                        "seq_lens": seq_lens,
                        "doc_remaining": doc_remaining,
                    }
                else:
                    input_ids, attention_mask, loss_mask = prompt
                ref = self._producer.produce(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    loss_mask=loss_mask,
                    **packing_kwargs,
                )
                # put_blocks_until_below honors the queue's HWM/LWM
                # hysteresis; a fast producer here naturally blocks
                # when the store is at high watermark.
                try:
                    self._queue.put_blocks_until_below(
                        ref,
                        poll_interval=self._poll_interval,
                        abort_when=self._stop_event.is_set,
                    )
                except RuntimeError as e:
                    # The ref was produced (its sample is resident in the store)
                    # but never entered the queue, so no consumer can release it.
                    # Discard it or it orphans -- an on-disk .safetensors file for
                    # SharedDirFeatureStore -- until this instance's close().
                    self._producer.discard(ref)
                    if self._stop_event.is_set() and (
                        "aborted during shutdown" in str(e) or "closed while put was waiting" in str(e)
                    ):
                        break
                    raise
        except BaseException as e:  # noqa: BLE001 -- capture then re-raise from stop()
            logger.exception("AsyncFeaturePipeline background thread failed")
            self._error = e
        finally:
            self._queue.close()


__all__ = ["AsyncFeaturePipeline", "PromptSource"]

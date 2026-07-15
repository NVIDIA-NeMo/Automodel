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

"""Streaming consumer for speculative-decoding draft training.

:class:`FeatureDataLoader` is a Python iterator over
:class:`Eagle3TargetBatch`. It pulls one :class:`SampleRef` lease at a
time from a :class:`SampleRefQueue`, materializes the tensors through a
:class:`FeatureStore`, hands the trainer a fresh
:class:`Eagle3TargetBatch`, and releases the *previous* lease on every
``__next__`` call -- so the trainer can hold one batch across one full
forward without it being freed mid-forward.

The loader is per-rank and the leases stay on-rank; the queue + store
live in the same Python process as the consumer. FSDP / CP / EP
parallelism lives inside the trainer's forward / backward and is
unaffected by the loader's lifecycle.
"""

from __future__ import annotations

import logging

import torch

from nemo_automodel.components.speculative.eagle.target import Eagle3TargetBatch
from nemo_automodel.components.speculative.streaming.eagle3 import validate_eagle3_ref
from nemo_automodel.components.speculative.streaming.queue import Lease, SampleRefQueue
from nemo_automodel.components.speculative.streaming.store import FeatureStore, StoreHandle

logger = logging.getLogger(__name__)


class FeatureDataLoader:
    """Iterator over :class:`Eagle3TargetBatch` materialized from a streaming queue.

    Args:
        queue: The metadata-only queue the producer puts :class:`SampleRef`
            onto. ``queue.close()`` at any time cuts the iterator short.
        store: The :class:`FeatureStore` each lease will be materialized
            through. Must match ``ref.store_uri`` for the leased refs.
        algorithm: :class:`FeatureAlgorithm` the loader runs the
            per-algorithm schema check for; defaults to EAGLE-3.

    Lifecycle:
        Each iterator pull yields an :class:`Eagle3TargetBatch` whose
        tensors come from a fresh ``store.get()``. The previous
        batch's lease is ack'd and its store handle released on the
        NEXT pull -- so the trainer can hold one batch across one
        forward pass without it being freed mid-forward. A consumer
        that wants eager reclamation (e.g. to free memory before
        pulling the next batch) calls :meth:`consume_now` after
        computing its loss. Iteration ends on :meth:`close` or when
        the queue drains.
    """

    def __init__(
        self,
        queue: SampleRefQueue,
        store: FeatureStore,
        *,
        algorithm=None,
    ) -> None:
        from nemo_automodel.components.speculative.streaming.refs import FeatureAlgorithm

        if algorithm is None:
            algorithm = FeatureAlgorithm.EAGLE3
        self._queue = queue
        self._store = store
        self._algorithm = algorithm
        self._pending_lease: Lease | None = None
        self._pending_handle: StoreHandle | None = None
        self._closed = False

    def __iter__(self):
        return self

    def __next__(self) -> Eagle3TargetBatch:
        self._release_pending()
        if self._closed:
            raise StopIteration
        lease = self._queue.acquire()
        if lease is None:
            raise StopIteration
        ref = lease.ref
        if ref.algorithm is not self._algorithm:
            self._queue.fail(lease)
            raise ValueError(
                f"FeatureDataLoader is bound to algorithm={self._algorithm!r} but "
                f"received ref with algorithm={ref.algorithm!r} for sample_id={ref.sample_id!r}"
            )
        if self._algorithm.value == "eagle3":
            try:
                validate_eagle3_ref(ref)
            except ValueError:
                self._queue.fail(lease)
                raise
        try:
            tensors, handle = self._store.get(ref)
        except Exception:
            logger.exception("store.get failed for sample_id=%s; failing lease", ref.sample_id)
            self._queue.fail(lease)
            raise
        self._pending_lease = lease
        self._pending_handle = handle
        return _materialize_batch(ref.algorithm, tensors)

    def consume_now(self) -> None:
        """Release the most recently yielded batch eagerly.

        Useful for trainer hooks that want to free memory before
        pulling the next batch (e.g. immediately after ``backward()``).
        Idempotent.
        """
        self._release_pending()

    def close(self) -> None:
        """Ack the most recent lease / release its store handle and shut the queue.

        After ``close`` the iterator raises :class:`StopIteration` on
        the next pull. Idempotent so a trainer can use ``with`` safely.
        """
        self._release_pending()
        self._queue.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _release_pending(self) -> None:
        if self._pending_handle is not None:
            try:
                self._store.release(self._pending_handle)
            except Exception:
                logger.exception(
                    "store.release failed for sample_id=%s; ignoring",
                    self._pending_handle.sample_id,
                )
            self._pending_handle = None
        if self._pending_lease is not None:
            try:
                self._queue.ack(self._pending_lease)
            except Exception:
                logger.exception(
                    "queue.ack failed for sample_id=%s; ignoring",
                    self._pending_lease.ref.sample_id,
                )
            self._pending_lease = None


def _materialize_batch(algorithm, tensors: dict[str, torch.Tensor]) -> Eagle3TargetBatch:
    """Build an :class:`Eagle3TargetBatch` from the store's tensors.

    Pulled into a module-level helper so the per-algorithm batch-building
    logic lives in one place and adding DFlash / DSpark only requires a
    new branch here.
    """
    if algorithm.value == "eagle3":
        logits = tensors.get("logits")
        if logits is not None:
            return Eagle3TargetBatch(
                aux_hidden_states=tensors["aux_hidden_states"],
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
                loss_mask=tensors["loss_mask"],
                logits=logits,
            )
        target_probs = tensors.get("target_probs")
        position_mask = tensors.get("position_mask")
        if target_probs is not None and position_mask is not None:
            return Eagle3TargetBatch(
                aux_hidden_states=tensors["aux_hidden_states"],
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
                loss_mask=tensors["loss_mask"],
                target_probs=target_probs,
                position_mask=position_mask,
            )
        raise ValueError(
            f"FeatureDataLoader (EAGLE-3) received a ref with no supervision encoding; feature keys={sorted(tensors)}"
        )
    raise ValueError(
        f"FeatureDataLoader cannot materialize batches for algorithm={algorithm!r}; only EAGLE-3 is implemented"
    )


__all__ = ["FeatureDataLoader"]

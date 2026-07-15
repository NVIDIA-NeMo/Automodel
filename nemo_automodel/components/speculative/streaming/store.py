# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Pluggable data-plane transport for the speculative-training stream.

The :class:`FeatureStore` ABC abstracts where the supervision tensors actually
live -- an in-process dict for build/test, a POSIX shared mount for
multi-node/colocated, or NCCL for GPU-to-GPU in later PRs. The
:class:`SampleRefQueue` reads the store's :meth:`FeatureStore.health` to
decide whether to back off.

The contract is deliberately small (5 methods + 1 property) so PR 2 and
later have an obvious surface to extend:

- :meth:`put` -- produce-side: stash tensors for a sample.
- :meth:`get`  -- consume-side: materialize them; returns a :class:`StoreHandle`
  the consumer must hand to :meth:`release` once it's done with them.
- :meth:`release` -- consume-once: free / drop the materials.
- :meth:`gc` -- sweep partially-released handles (a stale lease, a crashed
  consumer) so the store cannot leak.
- :meth:`health` -- ints only; the queue uses these for backpressure.
- :meth:`close` -- dispose store resources at shutdown.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

import torch

from nemo_automodel.components.speculative.streaming.refs import FeatureAlgorithm, SampleRef


@dataclass(frozen=True)
class StoreHandle:
    """Opaque token the consumer must return to :meth:`FeatureStore.release`.

    Holds the producing store, the sample id, and the originating :class:`SampleRef`
    so :meth:`release` can both free the cached tensors and decrement the
    resident-byte counter. Treated as opaque by callers -- ``release`` is the
    only sanctioned way to dispose of one.
    """

    store: "FeatureStore"
    sample_id: str
    ref: SampleRef


@dataclass(frozen=True)
class StoreHealth:
    """Integers-only snapshot of store residency for backpressure decisions.

    Attributes:
        resident_bytes: Bytes currently held in the store across un-acked
            samples. Compared against ``capacity_bytes`` for the high/low
            watermark hysteresis.
        capacity_bytes: Configured hard cap from
            :class:`~nemo_automodel.components.speculative.streaming.stores.local.LocalFeatureStore`
            (PR 3's shared-dir / PR 4's NCCL store report the analogous cap).
        sample_count: Number of un-acked samples currently held. Used for
            the second cap (sample count) the RFC §"Open questions" Q2 keeps
            alongside the byte backstop.
        high_watermark_hit: True iff ``resident_bytes >= high_watermark_bytes``
            on the last :meth:`health` call. The queue pauses a producer that
            sees this transition.
        low_watermark_hit: True iff ``resident_bytes <= low_watermark_bytes``
            on the last :meth:`health` call. The queue resumes a producer
            that has been paused and now sees this transition. The
            hysteresis band between the two is what prevents flapping.
    """

    resident_bytes: int
    capacity_bytes: int
    sample_count: int
    high_watermark_hit: bool
    low_watermark_hit: bool


class FeatureStore(ABC):
    """Abstract transport every feature-store backend implements.

    Implementations must be safe to call from one thread per operation;
    cross-thread concurrency is the queue's responsibility and outside this
    contract. The :class:`~nemo_automodel.components.speculative.streaming.queue.SampleRefQueue`
    uses :meth:`health` for backpressure; everything else is the
    produce/consume pair plus ``release`` / ``gc`` for lifecycle.
    """

    @abstractmethod
    def put(
        self,
        sample_id: str,
        tensors: Mapping[str, torch.Tensor],
        *,
        run_id: str,
        algorithm: FeatureAlgorithm,
        schema_version: int,
        target_model_version: str,
        draft_weight_version: str,
        num_tokens: int,
    ) -> SampleRef:
        """Stash ``tensors`` for ``sample_id`` and return a tensor-free :class:`SampleRef`.

        The producer-side metadata (``run_id``, ``algorithm``, ``schema_version``,
        ``target_model_version``, ``draft_weight_version``, ``num_tokens``) is
        the data the :class:`SampleRef` carries on the control plane, so the
        store must accept it here even though it does not inspect the values
        beyond building the ref. Consumers see the ref unchanged.

        Implementations must validate that the tensors match what they
        declare (``dtype``, ``shape`` per feature, ``numel * element_size``
        summed across features == ``ref.estimated_bytes``) and reject the
        put with a specific exception when they do not, before any partial
        write is observable.
        """

    @abstractmethod
    def get(
        self,
        ref: SampleRef,
        device: torch.device | str | None = None,
    ) -> tuple[dict[str, torch.Tensor], StoreHandle]:
        """Materialize ``ref``'s tensors on ``device`` and hand back a :class:`StoreHandle`.

        The returned tensors are detached copies (``clone`` for ``cuda``
        tensors, plain ``detach`` for CPU views) so prefetch cannot observe
        aliasing through :meth:`release`. Returns one tensor per key in
        :attr:`SampleRef.feature_keys`, in the same insertion order, so the
        consumer can wire them straight into the per-algorithm batch.
        """

    @abstractmethod
    def release(self, handle: StoreHandle) -> None:
        """Free the resources backing ``handle``; idempotent.

        After this returns, the sample is no longer present in the store and
        a second :meth:`get` on the same :class:`SampleRef` raises
        :class:`KeyError`. ``gc`` retries releases that a previous call
        rejected (e.g. transient I/O), so the queue can rely on
        ``gc + release`` being idempotent + retriable.
        """

    @abstractmethod
    def gc(self) -> int:
        """Sweep stale entries (e.g. failed releases from a crashed consumer).

        Returns the number of entries reclaimed. Called opportunistically by
        the :class:`SampleRefQueue` between leases and unconditionally by
        :meth:`close`.
        """

    @abstractmethod
    def health(self) -> StoreHealth:
        """Return a ints-only :class:`StoreHealth` snapshot for backpressure.

        Must not block on tensor I/O (no ``.cpu()``, no ``.to()``); in-memory
        counters are sufficient. Backends that do background I/O MUST serve
        :meth:`health` from cached counters, not from the in-flight I/O
        thread.
        """

    @abstractmethod
    def close(self) -> None:
        """Release every resource owned by the store (file handles, NCCL groups, ...).

        After ``close``, all subsequent calls raise :class:`RuntimeError`.
        Idempotent with respect to an already-closed store.
        """


__all__ = ["FeatureStore", "StoreHandle", "StoreHealth"]

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

"""In-process feature store for the speculative-training stream.

The local store keeps tensors in a Python dict under the
:class:`~nemo_automodel.components.speculative.streaming.refs.SampleRef.sample_id`,
with a resident-byte counter so the queue can drive backpressure from the
same store it puts into. It is the build-and-test surface for the entire
streaming pipeline: enough to wire up a colocated producer and consumer
without a network, without a shared mount, and without GPUDirect.

Residency policy (RFC §"Open questions" Q2 answer: bytes as the hard
backstop, sample count as a soft cap). When the next ``put`` would exceed
either cap, :meth:`put` raises :class:`MemoryError` -- the producer is then
expected to retry after the consumer drains below the low watermark.
"""

from __future__ import annotations

import logging
import threading
from typing import Mapping

import torch

from nemo_automodel.components.speculative.streaming.refs import FeatureAlgorithm, FeatureSpec, SampleRef
from nemo_automodel.components.speculative.streaming.store import FeatureStore, StoreHandle, StoreHealth

logger = logging.getLogger(__name__)


class LocalFeatureStore(FeatureStore):
    """In-process :class:`FeatureStore` implementation.

    Args:
        max_samples: Hard cap on simultaneously-stored samples. ``None`` means
            unbounded sample count (still bounded by ``max_bytes``).
        max_bytes: Hard cap on resident bytes. ``None`` means unbounded
            (still bounded by ``max_samples``). At least one of
            ``max_samples`` / ``max_bytes`` must be set, otherwise a
            misconfigured store silently behaves as unbounded.
        high_watermark_bytes: Threshold above which :attr:`StoreHealth.high_watermark_hit`
            is ``True``. The producer pauses here.
        low_watermark_bytes: Threshold below which :attr:`StoreHealth.low_watermark_hit`
            is ``True``. The producer resumes here. Must be
            strictly less than ``high_watermark_bytes``; a hysteresis band
            of zero flaps the producer on every step.

    Thread safety: every public method holds a single :class:`threading.Lock`,
    so concurrent puts and gets from the same Python process are safe. Async
    / cross-process safety is the queue's responsibility and is out of scope
    for PR 1.
    """

    def __init__(
        self,
        *,
        max_samples: int | None = 64,
        max_bytes: int | None = 256 * 1024 * 1024,
        high_watermark_bytes: int | None = 192 * 1024 * 1024,
        low_watermark_bytes: int | None = 64 * 1024 * 1024,
    ) -> None:
        if max_samples is None and max_bytes is None:
            raise ValueError("LocalFeatureStore requires at least one of max_samples / max_bytes to be set")
        if max_samples is not None and max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        if high_watermark_bytes is not None and low_watermark_bytes is not None:
            if low_watermark_bytes >= high_watermark_bytes:
                raise ValueError(
                    f"low_watermark_bytes must be strictly less than high_watermark_bytes; "
                    f"got low={low_watermark_bytes} high={high_watermark_bytes}"
                )
        self._max_samples = max_samples
        self._max_bytes = max_bytes
        self._high_watermark = high_watermark_bytes if high_watermark_bytes is not None else max_bytes
        self._low_watermark = low_watermark_bytes if low_watermark_bytes is not None else 0
        # The lock that guards every public method.
        self._lock = threading.Lock()
        self._storage: dict[str, dict[str, torch.Tensor]] = {}
        self._handle_refs: dict[str, int] = {}  # sample_id -> outstanding get() handle count
        self._resident_bytes = 0
        self._closed = False

    # --- helpers ------------------------------------------------------------

    @staticmethod
    def _tensor_bytes(tensor: torch.Tensor) -> int:
        # numel * element_size counts the logical bytes (incl. padding for
        # CUDA storage); matches what the consumer allocates, so put's
        # estimated_bytes == get's actual storage.
        return int(tensor.numel()) * tensor.element_size()

    def _make_ref(
        self,
        sample_id: str,
        tensors: Mapping[str, torch.Tensor],
        run_id: str,
        schema_version: int,
        target_model_version: str,
        draft_weight_version: str,
        algorithm: FeatureAlgorithm,
        num_tokens: int,
    ) -> SampleRef:
        feature_specs: dict[str, FeatureSpec] = {}
        feature_keys: dict[str, str] = {}
        estimated_bytes = 0
        for name, tensor in tensors.items():
            feature_keys[name] = f"{sample_id}/{name}"
            feature_specs[name] = FeatureSpec(shape=tuple(tensor.shape), dtype=tensor.dtype)
            estimated_bytes += self._tensor_bytes(tensor)
        return SampleRef(
            sample_id=sample_id,
            run_id=run_id,
            store_uri=self.store_uri,
            feature_keys=feature_keys,
            feature_specs=feature_specs,
            algorithm=algorithm,
            schema_version=schema_version,
            num_tokens=num_tokens,
            estimated_bytes=estimated_bytes,
            target_model_version=target_model_version,
            draft_weight_version=draft_weight_version,
        )

    @property
    def store_uri(self) -> str:
        # Stable URI the queue's lease / ack protocols can match on. PR 3's
        # SharedDirFeatureStore will return a different scheme ("file://"),
        # so the queue can refuse to lease a ref whose URI does not match
        # its bound store.
        return f"mem://local-{id(self):x}"

    # --- public API ---------------------------------------------------------

    def put(
        self,
        sample_id: str,
        tensors: Mapping[str, torch.Tensor],
        *,
        run_id: str,
        algorithm: FeatureAlgorithm = FeatureAlgorithm.EAGLE3,
        schema_version: int = 1,
        target_model_version: str = "0",
        draft_weight_version: str = "0",
        num_tokens: int = 0,
    ) -> SampleRef:
        """Store ``tensors`` under ``sample_id`` and return a tensor-free :class:`SampleRef`.

        Raises:
            MemoryError: if the put would exceed ``max_samples`` or
                ``max_bytes``. The producer is expected to retry after the
                store drains below the low watermark (see
                :meth:`health`).
            RuntimeError: if the store has been closed.
        """
        if not sample_id:
            raise ValueError("sample_id must be a non-empty str")
        if not isinstance(sample_id, str):
            raise ValueError(f"sample_id must be str, got {type(sample_id).__name__}")
        if not tensors:
            raise ValueError("tensors must be non-empty so a SampleRef has at least one feature")
        bytes_in = sum(self._tensor_bytes(t) for t in tensors.values())
        ref = self._make_ref(
            sample_id=sample_id,
            tensors=tensors,
            run_id=run_id,
            schema_version=schema_version,
            target_model_version=target_model_version,
            draft_weight_version=draft_weight_version,
            algorithm=algorithm,
            num_tokens=num_tokens,
        )
        with self._lock:
            if self._closed:
                raise RuntimeError("LocalFeatureStore is closed; no further puts accepted")
            if sample_id in self._storage:
                raise ValueError(f"sample_id already present in store: {sample_id}")
            if self._max_samples is not None and len(self._storage) >= self._max_samples:
                raise MemoryError(
                    f"LocalFeatureStore at sample-count cap ({len(self._storage)}/{self._max_samples}); "
                    f"refusing put for sample_id={sample_id}"
                )
            if self._max_bytes is not None and self._resident_bytes + bytes_in > self._max_bytes:
                raise MemoryError(
                    f"LocalFeatureStore at byte cap ({self._resident_bytes + bytes_in} > "
                    f"{self._max_bytes}); refusing put for sample_id={sample_id}"
                )
            # Stash a detached, contiguous copy so a follow-up caller mutating
            # the source tensor cannot disturb what we just stored, and so the
            # bytes counted in resident_bytes match what we hand out later.
            self._storage[sample_id] = {name: t.detach().clone().contiguous() for name, t in tensors.items()}
            self._handle_refs[sample_id] = 0
            self._resident_bytes += bytes_in
            logger.debug(
                "LocalFeatureStore put sample_id=%s features=%d bytes=%d resident=%d",
                sample_id,
                len(tensors),
                bytes_in,
                self._resident_bytes,
            )
            return ref

    def get(
        self,
        ref: SampleRef,
        device: torch.device | str | None = None,
    ) -> tuple[dict[str, torch.Tensor], StoreHandle]:
        if ref.store_uri != self.store_uri:
            raise KeyError(
                f"SampleRef.store_uri {ref.store_uri!r} does not match this store's URI {self.store_uri!r}; "
                f"consumer is bound to a different store"
            )
        target_device = torch.device(device) if device is not None else None
        with self._lock:
            if self._closed:
                raise RuntimeError("LocalFeatureStore is closed; no further gets accepted")
            tensors = self._storage.get(ref.sample_id)
            if tensors is None:
                raise KeyError(
                    f"sample_id {ref.sample_id!r} is not present in this LocalFeatureStore (released or never put)"
                )
            out: dict[str, torch.Tensor] = {}
            for name in ref.feature_names():
                tensor = tensors[name]
                # Validate the producer's claim against what's actually
                # stored; mismatch means the producer's feature_specs drifted
                # from the tensors it put, which is a programming error, not
                # a transient failure.
                spec = ref.feature_specs[name]
                if tuple(tensor.shape) != spec.shape or tensor.dtype != spec.dtype:
                    raise RuntimeError(
                        f"stored tensor for {name!r} shape/dtype mismatch with SampleRef spec: "
                        f"stored=(shape={tuple(tensor.shape)}, dtype={tensor.dtype}) "
                        f"ref=(shape={spec.shape}, dtype={spec.dtype})"
                    )
                if target_device is not None and tensor.device != target_device:
                    out[name] = tensor.to(target_device)
                else:
                    # clone() so the consumer holds an independent copy and
                    # cannot disturb what release()'s gc sweep might
                    # subsequently do.
                    out[name] = tensor.clone()
            self._handle_refs[ref.sample_id] += 1
            handle = StoreHandle(store=self, sample_id=ref.sample_id, ref=ref)
            logger.debug(
                "LocalFeatureStore get sample_id=%s device=%s handles=%d",
                ref.sample_id,
                target_device,
                self._handle_refs[ref.sample_id],
            )
            return out, handle

    def release(self, handle: StoreHandle) -> None:
        if handle.store is not self:
            raise ValueError(
                f"StoreHandle was issued by a different store (id {id(handle.store):x}), cannot release it here"
            )
        with self._lock:
            count = self._handle_refs.get(handle.sample_id, 0)
            if count <= 0:
                logger.debug("LocalFeatureStore release sample_id=%s is a no-op (already released)", handle.sample_id)
                return
            count -= 1
            # Refcounts: drop the storage only when the last outstanding
            # handle is released, so concurrent readers of the same sample
            # each get their own tensors and see them vanish at the same
            # point.
            if count == 0:
                tensors = self._storage.pop(handle.sample_id, None)
                if tensors is not None:
                    bytes_in = sum(self._tensor_bytes(t) for t in tensors.values())
                    self._resident_bytes -= bytes_in
                self._handle_refs.pop(handle.sample_id, None)
            else:
                self._handle_refs[handle.sample_id] = count
            logger.debug(
                "LocalFeatureStore release sample_id=%s remaining_handles=%d resident=%d",
                handle.sample_id,
                count,
                self._resident_bytes,
            )

    def gc(self) -> int:
        # Local store has no transient I/O; gc is a no-op for now, but the
        # method exists so the queue can call it unconditionally and PR 4's
        # NCCL store can override with real cleanup of failed-release handles.
        with self._lock:
            return 0

    def health(self) -> StoreHealth:
        with self._lock:
            capacity = self._max_bytes if self._max_bytes is not None else 0
            return StoreHealth(
                resident_bytes=self._resident_bytes,
                capacity_bytes=capacity,
                sample_count=len(self._storage),
                high_watermark_hit=(self._high_watermark is not None and self._resident_bytes >= self._high_watermark),
                low_watermark_hit=(self._low_watermark is not None and self._resident_bytes <= self._low_watermark),
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._storage.clear()
            self._handle_refs.clear()
            self._resident_bytes = 0
            logger.debug("LocalFeatureStore closed")


__all__ = ["LocalFeatureStore"]

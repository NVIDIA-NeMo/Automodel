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

"""File-backed :class:`FeatureStore` for cross-process / cross-node streaming.

:class:`SharedDirFeatureStore` writes one ``<sample_id>.safetensors`` file
per produced sample into a shared directory. Reads return cloned
tensors the consumer can mutate; ``release`` deletes the file once
the last outstanding handle is dropped. Writes are atomic
(tmp file + ``os.replace``) so a concurrent reader never observes a
partial write.

The store is the rendezvous for a multi-process run:

- Each process owns its own :class:`SharedDirFeatureStore` instance
  pointing at the same shared directory. Producers write files; consumers
  in other processes materialize them by ``sample_id`` without needing a
  process-local ownership record.
- The shared directory is the cross-process rendezvous; cross-rank
  routing of which rank reads which ``sample_id`` is left to distributed
  resharding layers above this store.
- Distributed parallelism (FSDP / CP / EP) lives in the trainer's
  forward / backward; this store is rank-local state plus a shared
  filesystem.

Backpressure is identical to :class:`LocalFeatureStore`: sample and
byte caps with the high / low-watermark hysteresis the queue reads.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from typing import Mapping

import torch

from nemo_automodel.components.speculative.streaming.refs import FeatureAlgorithm, FeatureSpec, SampleRef
from nemo_automodel.components.speculative.streaming.store import FeatureStore, StoreHandle, StoreHealth
from nemo_automodel.shared.import_utils import safe_import

logger = logging.getLogger(__name__)

# safetensors is used elsewhere in Automodel (checkpointing, model init,
# bagel, deepseek_v4 state_dict_adapter) and is in the project's
# dependency tree, but guard it through ``safe_import`` so a host
# without the optional dep fails clean rather than at first put.
_safetensors_torch_ok, _safetensors_torch = safe_import("safetensors.torch", alt=None)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * tensor.element_size()


class SharedDirFeatureStore(FeatureStore):
    """:class:`FeatureStore` backed by one ``<sample_id>.safetensors`` per sample.

    Args:
        directory: Filesystem path used as the rendezvous. Created if it
            does not exist. Concurrent producers and consumers in
            separate processes / ranks coordinate via unique
            ``sample_id`` values -- collision is the caller's problem.
        max_samples, max_bytes, high_watermark_bytes, low_watermark_bytes:
            Same residency contract as :class:`LocalFeatureStore`; the
            queue's HWM/LWM hysteresis reads them off
            :meth:`health`.

    Thread safety: every public method holds a single
    :class:`threading.Lock`. Cross-process / cross-node is supported
    as long as distinct processes use distinct ``sample_id`` values;
    the lock does not extend across processes.
    """

    def __init__(
        self,
        directory: str,
        *,
        max_samples: int | None = 64,
        max_bytes: int | None = 256 * 1024 * 1024,
        high_watermark_bytes: int | None = 192 * 1024 * 1024,
        low_watermark_bytes: int | None = 64 * 1024 * 1024,
    ) -> None:
        if not _safetensors_torch_ok:
            raise RuntimeError(
                "SharedDirFeatureStore requires the optional `safetensors` dependency; "
                "install with `uv pip install safetensors`"
            )
        if not directory:
            raise ValueError("SharedDirFeatureStore.directory must be a non-empty path")
        if max_samples is None and max_bytes is None:
            raise ValueError("SharedDirFeatureStore requires at least one of max_samples / max_bytes to be set")
        if max_samples is not None and max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError(f"max_bytes must be positive, got {max_bytes}")
        if (
            high_watermark_bytes is not None
            and low_watermark_bytes is not None
            and low_watermark_bytes >= high_watermark_bytes
        ):
            raise ValueError(
                f"low_watermark_bytes must be strictly less than high_watermark_bytes; "
                f"got low={low_watermark_bytes} high={high_watermark_bytes}"
            )
        self._directory = directory
        self._max_samples = max_samples
        self._max_bytes = max_bytes
        self._high_watermark = high_watermark_bytes if high_watermark_bytes is not None else max_bytes
        self._low_watermark = low_watermark_bytes if low_watermark_bytes is not None else 0
        os.makedirs(directory, exist_ok=True)
        self._lock = threading.Lock()
        # Files this store instance put, keyed by sample_id. Tracks
        # ownership so ``close`` only unlinks files we wrote.
        self._owned_files: dict[str, int] = {}
        # Track per-file handle refcount so concurrent get() calls on
        # the same sample each get their own tensors and the file
        # deletes only after the last release -- mirrors
        # ``LocalFeatureStore``'s consume-once semantics.
        self._handle_refs: dict[str, int] = {}
        self._closed = False

    @property
    def store_uri(self) -> str:
        return f"file://{os.path.abspath(self._directory)}"

    def _path_for(self, sample_id: str) -> str:
        # Defensive: refuse sample_ids that would escape the directory.
        if "/" in sample_id or ".." in sample_id or "\x00" in sample_id:
            raise ValueError(
                f"sample_id {sample_id!r} contains characters that would escape the store directory; refusing to write"
            )
        return os.path.join(self._directory, f"{sample_id}.safetensors")

    def _directory_residency(self) -> tuple[int, int]:
        """Return ``(sample_count, resident_bytes)`` from on-disk sample files."""
        sample_count = 0
        resident_bytes = 0
        try:
            names = os.listdir(self._directory)
        except OSError:
            return 0, 0
        for name in names:
            if not name.endswith(".safetensors") or name.startswith(".tmp."):
                continue
            path = os.path.join(self._directory, name)
            try:
                if os.path.isfile(path):
                    resident_bytes += os.path.getsize(path)
                    sample_count += 1
            except OSError:
                continue
        return sample_count, resident_bytes

    def _atomic_write(self, path: str, tensors: Mapping[str, torch.Tensor]) -> int:
        # Atomic write: serialize to a tmp file in the same directory,
        # then os.replace. A concurrent reader either sees the old file
        # (tmp file not yet replaced) or the full new file (after
        # replace); never a partial write. Mirrors how
        # ``nemo_automodel/components/checkpoint/checkpointing.py``
        # guards its ``save_file`` calls.
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=".tmp.", suffix=".safetensors", dir=self._directory)
        os.close(tmp_fd)
        try:
            detached = {name: t.detach().clone().contiguous().cpu() for name, t in tensors.items()}
            _safetensors_torch.save_file(detached, tmp_path)
            os.replace(tmp_path, path)
            return sum(_tensor_bytes(t) for t in detached.values())
        except Exception:
            # Best-effort cleanup on failure so a half-written tmp file
            # does not accumulate.
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise

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
        if not sample_id:
            raise ValueError("sample_id must be a non-empty str")
        if not isinstance(sample_id, str):
            raise ValueError(f"sample_id must be str, got {type(sample_id).__name__}")
        if not tensors:
            raise ValueError("tensors must be non-empty so a SampleRef has at least one feature")
        path = self._path_for(sample_id)
        with self._lock:
            if self._closed:
                raise RuntimeError("SharedDirFeatureStore is closed; no further puts accepted")
            if sample_id in self._owned_files or os.path.isfile(path):
                raise ValueError(f"sample_id already present in store: {sample_id}")
            sample_count, resident_bytes = self._directory_residency()
            if self._max_samples is not None and sample_count >= self._max_samples:
                raise MemoryError(
                    f"SharedDirFeatureStore at sample-count cap ({sample_count}/{self._max_samples}); "
                    f"refusing put for sample_id={sample_id}"
                )
            bytes_in = sum(_tensor_bytes(t) for t in tensors.values())
            if self._max_bytes is not None and resident_bytes + bytes_in > self._max_bytes:
                raise MemoryError(
                    f"SharedDirFeatureStore at byte cap ({resident_bytes + bytes_in} > "
                    f"{self._max_bytes}); refusing put for sample_id={sample_id}"
                )
            bytes_written = self._atomic_write(path, tensors)
            self._owned_files[sample_id] = bytes_written
            self._handle_refs[sample_id] = 0
            logger.debug(
                "SharedDirFeatureStore put sample_id=%s bytes=%d resident=%d",
                sample_id,
                bytes_written,
                resident_bytes + bytes_written,
            )

        feature_specs: dict[str, FeatureSpec] = {
            name: FeatureSpec(shape=tuple(t.shape), dtype=t.dtype) for name, t in tensors.items()
        }
        feature_keys = {name: f"{sample_id}/{name}" for name in tensors}
        return SampleRef(
            sample_id=sample_id,
            run_id=run_id,
            store_uri=self.store_uri,
            feature_keys=feature_keys,
            feature_specs=feature_specs,
            algorithm=algorithm,
            schema_version=schema_version,
            num_tokens=num_tokens,
            estimated_bytes=bytes_written,
            target_model_version=target_model_version,
            draft_weight_version=draft_weight_version,
        )

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
        path = self._path_for(ref.sample_id)
        with self._lock:
            if self._closed:
                raise RuntimeError("SharedDirFeatureStore is closed; no further gets accepted")
        if not os.path.isfile(path):
            raise KeyError(
                f"sample_id {ref.sample_id!r} is not present in this SharedDirFeatureStore (released or never put)"
            )
        # Read outside the lock: the file is owned by this store and
        # is not mutated after the atomic put-write, so concurrent
        # reads of the same file are safe. Holding the lock across the
        # disk read would serialize independent get() calls on the same
        # file and waste the OS page cache.
        loaded = _safetensors_torch.load_file(path)
        out: dict[str, torch.Tensor] = {}
        for name in ref.feature_names():
            tensor = loaded[name]
            spec = ref.feature_specs[name]
            if tuple(tensor.shape) != spec.shape or tensor.dtype != spec.dtype:
                raise RuntimeError(
                    f"loaded tensor for {name!r} shape/dtype mismatch with SampleRef spec: "
                    f"loaded=(shape={tuple(tensor.shape)}, dtype={tensor.dtype}) "
                    f"ref=(shape={spec.shape}, dtype={spec.dtype})"
                )
            if target_device is not None and tensor.device != target_device:
                tensor = tensor.to(target_device)
            else:
                tensor = tensor.clone()
            out[name] = tensor
        with self._lock:
            self._handle_refs[ref.sample_id] = self._handle_refs.get(ref.sample_id, 0) + 1
            handle = StoreHandle(store=self, sample_id=ref.sample_id, ref=ref)
            logger.debug(
                "SharedDirFeatureStore get sample_id=%s device=%s handles=%d",
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
                logger.debug(
                    "SharedDirFeatureStore release sample_id=%s is a no-op (already released)",
                    handle.sample_id,
                )
                return
            count -= 1
            if count == 0:
                path = self._path_for(handle.sample_id)
                self._owned_files.pop(handle.sample_id, None)
                self._handle_refs.pop(handle.sample_id, None)
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.exception("SharedDirFeatureStore unlink failed for %s; ignoring", path)
            else:
                self._handle_refs[handle.sample_id] = count
            _, resident_bytes = self._directory_residency()
            logger.debug(
                "SharedDirFeatureStore release sample_id=%s remaining_handles=%d resident=%d",
                handle.sample_id,
                count,
                resident_bytes,
            )

    def gc(self) -> int:
        # Release already unlinks files when the last handle drops; a
        # controller could override this to reclaim files from crashed producers.
        return 0

    def health(self) -> StoreHealth:
        sample_count, resident_bytes = self._directory_residency()
        with self._lock:
            if self._closed:
                sample_count = 0
                resident_bytes = 0
            capacity = self._max_bytes if self._max_bytes is not None else 0
            return StoreHealth(
                resident_bytes=resident_bytes,
                capacity_bytes=capacity,
                sample_count=sample_count,
                high_watermark_hit=(self._high_watermark is not None and resident_bytes >= self._high_watermark),
                low_watermark_hit=(self._low_watermark is not None and resident_bytes <= self._low_watermark),
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            outstanding = sum(count for count in self._handle_refs.values() if count > 0)
            if outstanding:
                raise RuntimeError(
                    f"SharedDirFeatureStore.close() called with {outstanding} outstanding handle(s); "
                    f"release all StoreHandles before closing"
                )
            self._closed = True
            # Delete every file this store instance still owns. A
            # crashed process leaks its files into the directory;
            # ``gc()`` on a healthy process (which we are) cleans
            # them up on shutdown.
            for sample_id in list(self._owned_files):
                path = self._path_for(sample_id)
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
                except OSError:
                    logger.exception("SharedDirFeatureStore close unlink failed for %s; ignoring", path)
            self._owned_files.clear()
            self._handle_refs.clear()


__all__ = ["SharedDirFeatureStore"]

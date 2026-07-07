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

"""Shared on-disk helpers for speculative offline cache readers."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler

from nemo_automodel.shared.import_utils import safe_import_from

MANIFEST_NAME = "manifest.json"
SHARD_RE = re.compile(r"^shard-(\d{6})\.safetensors$")


def load_safetensors(cache_name: str):
    """Return ``(save_file, safe_open)`` or raise a clear dependency error."""
    has_save, save_file = safe_import_from("safetensors.torch", "save_file")
    has_open, safe_open = safe_import_from("safetensors", "safe_open")
    if not (has_save and has_open):
        raise ImportError(
            f"The {cache_name} offline cache requires the 'safetensors' package. "
            "Install it with `uv sync --locked` and re-run."
        )
    return save_file, safe_open


def atomic_write(path: str, write_fn: Callable[[str], None]) -> str:
    """Write through a sibling temporary file, then atomically replace the target path."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    write_fn(tmp_path)
    os.replace(tmp_path, path)
    return path


def shard_path(cache_dir: str, shard_index: int) -> str:
    """Return the path of shard ``shard_index`` inside ``cache_dir``."""
    return os.path.join(cache_dir, f"shard-{shard_index:06d}.safetensors")


def manifest_path(cache_dir: str) -> str:
    """Return the manifest path inside ``cache_dir``."""
    return os.path.join(cache_dir, MANIFEST_NAME)


def existing_shard_indices(cache_dir: str) -> set[int]:
    """Return the shard indices already present in ``cache_dir``."""
    indices: set[int] = set()
    if not os.path.isdir(cache_dir):
        return indices
    for name in os.listdir(cache_dir):
        match = SHARD_RE.match(name)
        if match is not None:
            indices.add(int(match.group(1)))
    return indices


def write_manifest(cache_dir: str, manifest: dict[str, Any], format_version: int) -> str:
    """Persist an offline-cache manifest atomically."""

    def _write(tmp_path: str) -> None:
        with open(tmp_path, "w") as f:
            json.dump({"format_version": format_version, **manifest}, f, indent=2, sort_keys=True)

    return atomic_write(manifest_path(cache_dir), _write)


def read_manifest(
    cache_dir: str,
    *,
    cache_name: str,
    format_version: int,
    producer_name: str | None = None,
) -> dict[str, Any]:
    """Load and validate an offline-cache manifest."""
    path = manifest_path(cache_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{cache_name} cache manifest not found at {path}. Was the cache fully written?")
    with open(path) as f:
        manifest = json.load(f)
    version = manifest.get("format_version")
    if version != format_version:
        producer = producer_name or f"precompute_{cache_name.lower().replace('-', '')}"
        raise ValueError(
            f"{cache_name} cache at {cache_dir} has format_version={version}, expected {format_version}. "
            f"Regenerate the cache with the current {producer}."
        )
    return manifest


def validate_contiguous_shards(cache_dir: str, cache_name: str, num_samples: int, shard_size: int) -> list[int]:
    """Return shard indices, requiring exactly ``0..N-1`` for the manifest size."""
    indices = sorted(existing_shard_indices(cache_dir))
    expected = (num_samples + shard_size - 1) // shard_size
    expected_indices = list(range(expected))
    if indices != expected_indices:
        raise ValueError(
            f"{cache_name} cache at {cache_dir} declares {num_samples} samples "
            f"({expected} shards) but found shard indices {indices}."
        )
    return indices


def write_tensor_shard(
    cache_dir: str,
    shard_index: int,
    samples: dict[str, torch.Tensor],
    keys: tuple[str, ...],
    cache_name: str,
) -> str:
    """Write one cache shard containing exactly the requested tensor keys."""
    missing = [key for key in keys if key not in samples]
    if missing:
        raise ValueError(f"write_shard is missing required {cache_name} cache fields: {missing}")
    save_file, _ = load_safetensors(cache_name)
    tensors = {key: samples[key].contiguous() for key in keys}
    return atomic_write(shard_path(cache_dir, shard_index), lambda tmp: save_file(tensors, tmp))


class CachedTensorDataset(Dataset):
    """Lazily read fixed-key safetensors shards by sample index."""

    def __init__(
        self,
        *,
        cache_dir: str,
        cache_name: str,
        format_version: int,
        disk_keys: tuple[str, ...],
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_name = cache_name
        self.manifest = read_manifest(cache_dir, cache_name=cache_name, format_version=format_version)
        self.shard_size = int(self.manifest["shard_size"])
        self.num_samples = int(self.manifest["num_samples"])
        self._disk_keys = disk_keys
        self._shard_indices = validate_contiguous_shards(cache_dir, cache_name, self.num_samples, self.shard_size)
        self._open_handles: dict[int, Any] = {}

    def __len__(self) -> int:
        return self.num_samples

    def _handle(self, shard_index: int):
        handle = self._open_handles.get(shard_index)
        if handle is None:
            _, safe_open = load_safetensors(self.cache_name)
            handle = safe_open(shard_path(self.cache_dir, shard_index), framework="pt")
            self._open_handles[shard_index] = handle
        return handle

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += self.num_samples
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        shard_index = index // self.shard_size
        offset = index % self.shard_size
        handle = self._handle(shard_index)
        return {key: handle.get_slice(key)[offset] for key in self._disk_keys}


def collate_cached(features: list[dict[str, torch.Tensor]], keys: tuple[str, ...]) -> dict[str, torch.Tensor]:
    """Stack per-sample cache dicts into a batch."""
    return {key: torch.stack([feature[key] for feature in features], dim=0) for key in keys}


def build_cached_dataloader(
    *,
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    collate_fn: Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]],
    num_workers: int = 0,
    distributed: bool = False,
) -> DataLoader:
    """Build a dataloader over a precomputed offline cache dataset."""
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle and sampler is None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
        drop_last=False,
    )


def resume_start_sample(existing_shards: set[int], shard_size: int) -> int:
    """Return the sample index where a deterministic sequential resume should start."""
    expected = set(range(len(existing_shards)))
    if existing_shards != expected:
        raise ValueError(
            f"--resume expects existing shards to be a contiguous prefix {sorted(expected)}, "
            f"but found {sorted(existing_shards)}. Delete stale shards or restart from a clean directory."
        )
    return len(existing_shards) * shard_size


def dataloader_from_sample(dataloader: DataLoader, start_sample: int) -> DataLoader:
    """Return an equivalent sequential dataloader beginning at ``start_sample``."""
    if start_sample <= 0:
        return dataloader
    dataset = dataloader.dataset
    batch_size = getattr(dataloader, "batch_size", None) or 1
    collate_fn = getattr(dataloader, "collate_fn", None)
    if start_sample >= len(dataset):
        return DataLoader(
            Subset(dataset, []),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=False,
        )
    worker_kwargs: dict[str, Any] = {}
    num_workers = int(getattr(dataloader, "num_workers", 0))
    if num_workers > 0:
        worker_kwargs["persistent_workers"] = getattr(dataloader, "persistent_workers", False)
        multiprocessing_context = getattr(dataloader, "multiprocessing_context", None)
        if multiprocessing_context is not None:
            worker_kwargs["multiprocessing_context"] = multiprocessing_context
    return DataLoader(
        Subset(dataset, range(start_sample, len(dataset))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(getattr(dataloader, "pin_memory", False)),
        collate_fn=collate_fn,
        drop_last=False,
        **worker_kwargs,
    )


def write_cache_shards(
    *,
    dataloader: DataLoader,
    output_dir: str,
    shard_size: int,
    start_shard_index: int,
    compute_batch: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
    write_shard_fn: Callable[[str, int, dict[str, torch.Tensor]], str],
    logger: logging.Logger,
) -> None:
    """Run a precompute loop and write sequential cache shards."""
    shard_index = start_shard_index
    chunks: list[dict[str, torch.Tensor]] = []
    buffered = 0

    def _flush(max_samples: int | None = None) -> None:
        nonlocal shard_index, chunks, buffered
        if buffered == 0:
            return
        samples_to_write = buffered if max_samples is None else min(buffered, max_samples)
        merged = {k: torch.cat([c[k] for c in chunks], dim=0) for k in chunks[0]}
        shard = {k: v[:samples_to_write] for k, v in merged.items()}
        path = write_shard_fn(output_dir, shard_index, shard)
        logger.info("Wrote %s (%d samples)", path, shard["input_ids"].shape[0])
        remainder = {k: v[samples_to_write:] for k, v in merged.items()}
        remaining = buffered - samples_to_write
        chunks = [remainder] if remaining else []
        buffered = remaining
        shard_index += 1

    for batch in dataloader:
        chunks.append(compute_batch(batch))
        buffered += batch["input_ids"].shape[0]
        while buffered >= shard_size:
            _flush(shard_size)

    _flush()

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

"""On-disk format + reader for the EAGLE-3 offline target-output cache.

This is the SpecForge "offline" training data path: the frozen target model's
per-token supervision (auxiliary hidden states + the draft-vocab target
distribution) is precomputed once and stored on disk, so draft training reads it
back instead of re-running the (large, frozen) target every step.

It is **extremely disk-intensive** -- on the order of tens of MB per sample for
an 8B target (``aux_hidden_states`` is ``3 * target_hidden_size`` wide), i.e.
multiple TB for a large dataset -- and is largely superseded by online training,
where the target forward is cheap relative to the cache I/O. It is kept for
completeness / reproducibility of the SpecForge offline recipe; prefer the online
path unless you are re-training repeatedly on a fixed, bounded dataset.

This module owns the format (so the producer in
``components/speculative/precompute_eagle3.py`` and the training-time reader
agree on one schema):

* ``<cache_dir>/manifest.json`` -- run config + the ``selected_token_ids`` used
  to build the draft vocabulary (the recipe reuses these instead of rescanning).
* ``<cache_dir>/shard-000000.safetensors`` -- one shard holds a contiguous block
  of samples, each field stacked along dim 0:
  ``input_ids[n,S]``, ``attention_mask[n,S]``, ``loss_mask[n,S]`` (int64),
  ``aux_hidden_states[n,S,3H]``, ``target_probs[n,S,draft_vocab]`` (float),
  ``position_mask[n,S,1]`` (bool).

Each ``CachedEagle3Dataset`` item is exactly the keyword arguments
``Eagle3TrainerModule.forward`` consumes on its precomputed-distribution path.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from nemo_automodel.shared.import_utils import safe_import_from

_MANIFEST_NAME = "manifest.json"
_EMBEDDINGS_NAME = "target_embeddings.safetensors"
_SHARD_RE = re.compile(r"^shard-(\d{6})\.safetensors$")
_FORMAT_VERSION = 1

# Fields stored per sample. The float fields are cast to the cache dtype; the id /
# mask fields stay int64 / bool. This key set is exactly what the trainer's
# precomputed-distribution path consumes.
_FLOAT_KEYS = ("aux_hidden_states", "target_probs")
_INT_KEYS = ("input_ids", "attention_mask", "loss_mask")
_BOOL_KEYS = ("position_mask",)
CACHE_KEYS = _FLOAT_KEYS + _INT_KEYS + _BOOL_KEYS

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _load_safetensors():
    """Return ``(save_file, safe_open)`` or raise a clear error if safetensors is missing."""
    has_save, save_file = safe_import_from("safetensors.torch", "save_file")
    has_open, safe_open = safe_import_from("safetensors", "safe_open")
    if not (has_save and has_open):
        raise ImportError(
            "The EAGLE-3 offline cache requires the 'safetensors' package. "
            "Install it with `uv pip install safetensors` and re-run."
        )
    return save_file, safe_open


def _atomic_write(path: str, write_fn: Callable[[str], None]) -> str:
    """Run ``write_fn`` against a sibling ``.tmp`` path, then ``os.replace`` it into place.

    A crash mid-write never leaves a half-written file a later run would load.
    """
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
    return os.path.join(cache_dir, _MANIFEST_NAME)


def existing_shard_indices(cache_dir: str) -> set[int]:
    """Return the set of shard indices already present in ``cache_dir``."""
    indices: set[int] = set()
    if not os.path.isdir(cache_dir):
        return indices
    for name in os.listdir(cache_dir):
        match = _SHARD_RE.match(name)
        if match is not None:
            indices.add(int(match.group(1)))
    return indices


def write_manifest(cache_dir: str, manifest: dict[str, Any]) -> str:
    """Persist the cache manifest atomically (``.tmp`` + ``os.replace``)."""

    def _write(tmp_path: str) -> None:
        with open(tmp_path, "w") as f:
            json.dump({"format_version": _FORMAT_VERSION, **manifest}, f, indent=2, sort_keys=True)

    return _atomic_write(manifest_path(cache_dir), _write)


def read_manifest(cache_dir: str) -> dict[str, Any]:
    """Load the cache manifest, raising if it is missing or the wrong format version."""
    path = manifest_path(cache_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"EAGLE-3 cache manifest not found at {path}. Was the cache fully written?")
    with open(path) as f:
        manifest = json.load(f)
    version = manifest.get("format_version")
    if version != _FORMAT_VERSION:
        raise ValueError(
            f"EAGLE-3 cache at {cache_dir} has format_version={version}, expected {_FORMAT_VERSION}. "
            "Regenerate the cache with the current precompute_eagle3."
        )
    return manifest


def write_target_embeddings(cache_dir: str, weight: torch.Tensor) -> str:
    """Persist the target input-embedding table the draft initializes from.

    The offline training path never loads the target model, but the draft's
    ``embed_tokens`` must still be seeded from the target's embeddings (EAGLE-3
    concatenates token embeddings with the carried hidden state), so the
    producer stores them once alongside the cache.
    """
    save_file, _ = _load_safetensors()
    tensors = {"weight": weight.detach().to(torch.float32).cpu().contiguous()}
    return _atomic_write(os.path.join(cache_dir, _EMBEDDINGS_NAME), lambda tmp: save_file(tensors, tmp))


def read_target_embeddings(cache_dir: str) -> torch.Tensor:
    """Load the target input-embedding table written by ``write_target_embeddings``."""
    _, safe_open = _load_safetensors()
    path = os.path.join(cache_dir, _EMBEDDINGS_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. The EAGLE-3 offline cache must include the target embeddings "
            "so the draft can initialize from them; regenerate the cache with the current precompute_eagle3."
        )
    with safe_open(path, framework="pt") as handle:
        return handle.get_tensor("weight")


def write_shard(cache_dir: str, shard_index: int, samples: dict[str, torch.Tensor]) -> str:
    """Write one shard atomically. ``samples`` maps each ``CACHE_KEYS`` field to a stacked tensor."""
    save_file, _ = _load_safetensors()
    missing = [k for k in CACHE_KEYS if k not in samples]
    if missing:
        raise ValueError(f"write_shard is missing required cache fields: {missing}")
    tensors = {k: samples[k].contiguous() for k in CACHE_KEYS}
    return _atomic_write(shard_path(cache_dir, shard_index), lambda tmp: save_file(tensors, tmp))


class CachedEagle3Dataset(Dataset):
    """Reads the EAGLE-3 offline cache; each item is one sample's trainer inputs.

    Shards are opened lazily with ``safetensors.safe_open`` (memory-mapped) and
    sliced per sample, so the full cache is never loaded into memory at once.
    Handles are reopened per worker after a DataLoader fork.
    """

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.manifest = read_manifest(cache_dir)
        self.shard_size = int(self.manifest["shard_size"])
        self.num_samples = int(self.manifest["num_samples"])
        indices = sorted(existing_shard_indices(cache_dir))
        expected = (self.num_samples + self.shard_size - 1) // self.shard_size
        if len(indices) != expected:
            raise ValueError(
                f"EAGLE-3 cache at {cache_dir} declares {self.num_samples} samples "
                f"({expected} shards) but found {len(indices)} shard files."
            )
        self._shard_indices = indices
        self._open_handles: dict[int, Any] = {}

    def __len__(self) -> int:
        return self.num_samples

    def _handle(self, shard_index: int):
        handle = self._open_handles.get(shard_index)
        if handle is None:
            _, safe_open = _load_safetensors()
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
        return {key: handle.get_slice(key)[offset] for key in CACHE_KEYS}


def _collate_cached(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack per-sample cache dicts into a batch."""
    return {key: torch.stack([feature[key] for feature in features], dim=0) for key in CACHE_KEYS}


def build_cached_eagle3_dataloader(
    *,
    cache_dir: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    distributed: bool = False,
) -> DataLoader:
    """Build a dataloader over a precomputed EAGLE-3 cache directory."""
    dataset = CachedEagle3Dataset(cache_dir)
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle and sampler is None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_cached,
        drop_last=False,
    )

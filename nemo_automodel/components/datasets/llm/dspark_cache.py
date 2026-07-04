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

"""On-disk format and reader for DSpark offline target supervision.

The DSpark online recipe runs a frozen target model every step to capture the
intermediate target features used by the draft and the final hidden state used
for the TV / confidence losses. This module owns the disk format that lets a
precompute job write those tensors once and lets training stream them later via
``recipe_args.cached_target_path`` without loading the target model.
"""

from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from nemo_automodel.shared.import_utils import safe_import_from

_MANIFEST_NAME = "manifest.json"
_TARGET_WEIGHTS_NAME = "target_weights.safetensors"
_SHARD_RE = re.compile(r"^shard-(\d{6})\.safetensors$")
_FORMAT_VERSION = 1

CACHE_KEYS = (
    "input_ids",
    "attention_mask",
    "loss_mask",
    "target_hidden_states",
    "target_last_hidden_states",
)

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _load_safetensors():
    """Return ``(save_file, safe_open)`` or raise a clear error if safetensors is missing."""
    has_save, save_file = safe_import_from("safetensors.torch", "save_file")
    has_open, safe_open = safe_import_from("safetensors", "safe_open")
    if not (has_save and has_open):
        raise ImportError(
            "The DSpark offline cache requires the 'safetensors' package. "
            "Install it with `uv sync --locked` and re-run."
        )
    return save_file, safe_open


def _atomic_write(path: str, write_fn: Callable[[str], None]) -> str:
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
    return os.path.join(cache_dir, _MANIFEST_NAME)


def existing_shard_indices(cache_dir: str) -> set[int]:
    """Return the shard indices already present in ``cache_dir``."""
    indices: set[int] = set()
    if not os.path.isdir(cache_dir):
        return indices
    for name in os.listdir(cache_dir):
        match = _SHARD_RE.match(name)
        if match is not None:
            indices.add(int(match.group(1)))
    return indices


def write_manifest(cache_dir: str, manifest: dict[str, Any]) -> str:
    """Persist the DSpark cache manifest."""

    def _write(tmp_path: str) -> None:
        with open(tmp_path, "w") as f:
            json.dump({"format_version": _FORMAT_VERSION, **manifest}, f, indent=2, sort_keys=True)

    return _atomic_write(manifest_path(cache_dir), _write)


def read_manifest(cache_dir: str) -> dict[str, Any]:
    """Load and validate the DSpark cache manifest."""
    path = manifest_path(cache_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"DSpark cache manifest not found at {path}. Was the cache fully written?")
    with open(path) as f:
        manifest = json.load(f)
    version = manifest.get("format_version")
    if version != _FORMAT_VERSION:
        raise ValueError(
            f"DSpark cache at {cache_dir} has format_version={version}, expected {_FORMAT_VERSION}. "
            "Regenerate the cache with the current precompute_dspark."
        )
    return manifest


def _target_weights_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, _TARGET_WEIGHTS_NAME)


def write_target_weights(cache_dir: str, embed_tokens: torch.nn.Module, lm_head: torch.nn.Module) -> str:
    """Persist target embedding and lm_head weights for target-free draft initialization."""
    embed_weight = getattr(embed_tokens, "weight", None)
    head_weight = getattr(lm_head, "weight", None)
    if embed_weight is None:
        raise ValueError("Target input embeddings do not expose a .weight tensor.")
    if head_weight is None:
        raise ValueError("Target output embeddings / lm_head do not expose a .weight tensor.")
    save_file, _ = _load_safetensors()
    tensors = {
        "embed_tokens.weight": embed_weight.detach().to(torch.float32).cpu().contiguous(),
        "lm_head.weight": head_weight.detach().to(torch.float32).cpu().contiguous(),
    }
    return _atomic_write(_target_weights_path(cache_dir), lambda tmp: save_file(tensors, tmp))


def read_target_weight_modules(cache_dir: str) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Load target weights and return module-like objects exposing ``.weight``."""
    _, safe_open = _load_safetensors()
    path = _target_weights_path(cache_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. The DSpark offline cache must include target embed_tokens and lm_head "
            "weights so training can skip loading the target model."
        )
    with safe_open(path, framework="pt") as handle:
        embed_weight = handle.get_tensor("embed_tokens.weight")
        head_weight = handle.get_tensor("lm_head.weight")
    return SimpleNamespace(weight=embed_weight), SimpleNamespace(weight=head_weight)


def write_shard(cache_dir: str, shard_index: int, samples: dict[str, torch.Tensor]) -> str:
    """Write one DSpark cache shard."""
    missing = [key for key in CACHE_KEYS if key not in samples]
    if missing:
        raise ValueError(f"write_shard is missing required DSpark cache fields: {missing}")
    save_file, _ = _load_safetensors()
    tensors = {key: samples[key].contiguous() for key in CACHE_KEYS}
    return _atomic_write(shard_path(cache_dir, shard_index), lambda tmp: save_file(tensors, tmp))


class CachedDSparkDataset(Dataset):
    """Read DSpark offline cache shards lazily."""

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.manifest = read_manifest(cache_dir)
        self.shard_size = int(self.manifest["shard_size"])
        self.num_samples = int(self.manifest["num_samples"])
        indices = sorted(existing_shard_indices(cache_dir))
        expected = (self.num_samples + self.shard_size - 1) // self.shard_size
        expected_indices = list(range(expected))
        if indices != expected_indices:
            raise ValueError(
                f"DSpark cache at {cache_dir} declares {self.num_samples} samples "
                f"({expected} shards) but found shard indices {indices}."
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


def build_cached_dspark_dataloader(
    *,
    cache_dir: str,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    distributed: bool = False,
) -> DataLoader:
    """Build a dataloader over a precomputed DSpark cache directory."""
    dataset = CachedDSparkDataset(cache_dir)
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


__all__ = [
    "CACHE_KEYS",
    "DTYPE_MAP",
    "CachedDSparkDataset",
    "build_cached_dspark_dataloader",
    "existing_shard_indices",
    "manifest_path",
    "read_manifest",
    "read_target_weight_modules",
    "shard_path",
    "write_manifest",
    "write_shard",
    "write_target_weights",
]

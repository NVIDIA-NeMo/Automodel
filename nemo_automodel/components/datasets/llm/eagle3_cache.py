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
  ``aux_hidden_states[n,S,3H]`` (float), ``position_mask[n,S,1]`` (bool), and the
  draft-vocab target distribution. The distribution is stored in full as
  ``target_probs[n,S,draft_vocab]`` (float) by default, or -- when the manifest's
  ``target_probs_topk`` enables it -- as the top-k factorization
  ``target_probs_values[n,S,k]`` (float) + ``target_probs_indices[n,S,k]`` (int32),
  which the reader scatters back to full width. Top-k bounds the dominant field's
  footprint at the cost of the dropped tail mass (a disk/fidelity trade-off, off by
  default; see ``precompute_eagle3``).

Each ``CachedEagle3Dataset`` item is exactly the keyword arguments
``Eagle3TrainerModule.forward`` consumes on its precomputed-distribution path
(``target_probs`` always yielded at full width).
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch.utils.data import DataLoader

from nemo_automodel.components.datasets.llm.offline_cache import (
    CachedTensorDataset,
    atomic_write,
    build_cached_dataloader,
    collate_cached,
    load_safetensors,
    shard_path,
)
from nemo_automodel.components.datasets.llm.offline_cache import (
    existing_shard_indices as _existing_shard_indices,
)
from nemo_automodel.components.datasets.llm.offline_cache import (
    manifest_path as _manifest_path,
)
from nemo_automodel.components.datasets.llm.offline_cache import (
    read_manifest as _read_manifest,
)
from nemo_automodel.components.datasets.llm.offline_cache import (
    write_manifest as _write_manifest,
)

_EMBEDDINGS_NAME = "target_embeddings.safetensors"
_FORMAT_VERSION = 2
_CACHE_NAME = "EAGLE-3"

# Fields the trainer's precomputed-distribution path consumes (what each
# ``CachedEagle3Dataset`` item yields). ``target_probs`` is the full draft-vocab
# distribution; under top-k compression the reader reconstructs it from the
# stored factorization, otherwise it is read back as-is.
_COMMON_KEYS = ("input_ids", "attention_mask", "loss_mask", "aux_hidden_states", "position_mask")
_LOSSLESS_PROBS_KEYS = ("target_probs",)
CACHE_KEYS = _COMMON_KEYS + _LOSSLESS_PROBS_KEYS

# A shard stores the draft-vocab ``target_probs`` either in full (lossless, the
# default) or, to bound the dominant field's footprint, as the top-k
# ``(values, indices)`` factorization the reader scatters back to full width.
# Which layout is in use is recorded by ``target_probs_topk`` in the manifest.
_TARGET_PROBS_VALUES = "target_probs_values"
_TARGET_PROBS_INDICES = "target_probs_indices"
_TOPK_PROBS_KEYS = (_TARGET_PROBS_VALUES, _TARGET_PROBS_INDICES)

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def is_compressed(target_probs_topk: int | None, draft_vocab_size: int) -> bool:
    """True when ``target_probs_topk`` actually compresses (a positive k below the vocab width)."""
    return target_probs_topk is not None and 0 < int(target_probs_topk) < draft_vocab_size


def compress_target_probs(target_probs: torch.Tensor, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor a draft-vocab distribution into its top-``topk`` ``(values, indices)``.

    ``target_probs`` is ``[..., draft_vocab]``. The kept values are renormalized to
    sum to 1 so the reconstructed distribution stays a proper distribution for soft
    cross-entropy. The dropped tail mass is a disk/fidelity trade-off that grows with
    how diffuse the target is (it is *not* negligible for a real LLM: top-64 of a
    Qwen3-8B next-token distribution keeps only ~0.92 of the mass), so callers choose
    ``topk`` deliberately; see ``precompute_eagle3``. ``topk`` is clamped to the vocab
    width.
    """
    k = min(topk, target_probs.shape[-1])
    values, indices = torch.topk(target_probs, k, dim=-1)
    values = values / values.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(target_probs.dtype).tiny)
    return values, indices.to(torch.int32)


def reconstruct_target_probs(values: torch.Tensor, indices: torch.Tensor, draft_vocab_size: int) -> torch.Tensor:
    """Scatter stored top-k ``(values, indices)`` back into a full ``[..., draft_vocab]`` distribution."""
    full = torch.zeros((*values.shape[:-1], draft_vocab_size), dtype=values.dtype, device=values.device)
    return full.scatter_(-1, indices.to(torch.long), values)


def write_manifest(cache_dir: str, manifest: dict[str, Any]) -> str:
    """Persist the cache manifest atomically (``.tmp`` + ``os.replace``)."""
    return _write_manifest(cache_dir, manifest, _FORMAT_VERSION)


def read_manifest(cache_dir: str) -> dict[str, Any]:
    """Load the cache manifest, raising if it is missing or the wrong format version."""
    return _read_manifest(
        cache_dir,
        cache_name=_CACHE_NAME,
        format_version=_FORMAT_VERSION,
        producer_name="precompute_eagle3",
    )


def existing_shard_indices(cache_dir: str) -> set[int]:
    """Return EAGLE-3 shard indices already present in ``cache_dir``."""
    return _existing_shard_indices(cache_dir)


def manifest_path(cache_dir: str) -> str:
    """Return the EAGLE-3 manifest path inside ``cache_dir``."""
    return _manifest_path(cache_dir)


def write_target_embeddings(cache_dir: str, weight: torch.Tensor) -> str:
    """Persist the target input-embedding table the draft initializes from.

    The offline training path never loads the target model, but the draft's
    ``embed_tokens`` must still be seeded from the target's embeddings (EAGLE-3
    concatenates token embeddings with the carried hidden state), so the
    producer stores them once alongside the cache.
    """
    save_file, _ = load_safetensors(_CACHE_NAME)
    tensors = {"weight": weight.detach().to(torch.float32).cpu().contiguous()}
    return atomic_write(os.path.join(cache_dir, _EMBEDDINGS_NAME), lambda tmp: save_file(tensors, tmp))


def read_target_embeddings(cache_dir: str) -> torch.Tensor:
    """Load the target input-embedding table written by ``write_target_embeddings``."""
    _, safe_open = load_safetensors(_CACHE_NAME)
    path = os.path.join(cache_dir, _EMBEDDINGS_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. The EAGLE-3 offline cache must include the target embeddings "
            "so the draft can initialize from them; regenerate the cache with the current precompute_eagle3."
        )
    with safe_open(path, framework="pt") as handle:
        return handle.get_tensor("weight")


def write_shard(cache_dir: str, shard_index: int, samples: dict[str, torch.Tensor]) -> str:
    """Write one shard atomically.

    ``samples`` carries the ``_COMMON_KEYS`` plus exactly one ``target_probs``
    representation: the full ``target_probs`` (lossless) or the top-k pair
    ``target_probs_values`` + ``target_probs_indices``.
    """
    save_file, _ = load_safetensors(_CACHE_NAME)
    has_full = "target_probs" in samples
    has_topk = all(k in samples for k in _TOPK_PROBS_KEYS)
    if int(has_full) + int(has_topk) != 1:
        raise ValueError(
            "write_shard needs exactly one target_probs representation: the full 'target_probs', "
            "or the top-k 'target_probs_values' + 'target_probs_indices'."
        )
    keys = _COMMON_KEYS + (_LOSSLESS_PROBS_KEYS if has_full else _TOPK_PROBS_KEYS)
    missing = [k for k in keys if k not in samples]
    if missing:
        raise ValueError(f"write_shard is missing required cache fields: {missing}")
    tensors = {k: samples[k].contiguous() for k in keys}
    return atomic_write(shard_path(cache_dir, shard_index), lambda tmp: save_file(tensors, tmp))


class CachedEagle3Dataset(CachedTensorDataset):
    """Reads the EAGLE-3 offline cache; each item is one sample's trainer inputs.

    Shards are opened lazily with ``safetensors.safe_open`` (memory-mapped) and
    sliced per sample, so the full cache is never loaded into memory at once.
    Handles are reopened per worker after a DataLoader fork.
    """

    def __init__(self, cache_dir: str):
        manifest = read_manifest(cache_dir)
        # A cache stores target_probs either in full or as its top-k factorization;
        # the reader yields the full distribution either way.
        self.draft_vocab_size = int(manifest["draft_vocab_size"])
        self._compressed = is_compressed(manifest.get("target_probs_topk"), self.draft_vocab_size)
        disk_keys = _COMMON_KEYS + (_TOPK_PROBS_KEYS if self._compressed else _LOSSLESS_PROBS_KEYS)
        super().__init__(
            cache_dir=cache_dir,
            cache_name=_CACHE_NAME,
            format_version=_FORMAT_VERSION,
            disk_keys=disk_keys,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += self.num_samples
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        shard_index = index // self.shard_size
        offset = index % self.shard_size
        handle = self._handle(shard_index)
        sample = {key: handle.get_slice(key)[offset] for key in self._disk_keys}
        if self._compressed:
            sample["target_probs"] = reconstruct_target_probs(
                sample.pop(_TARGET_PROBS_VALUES),
                sample.pop(_TARGET_PROBS_INDICES),
                self.draft_vocab_size,
            )
        return sample


def _collate_cached(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack per-sample cache dicts into a batch."""
    return collate_cached(features, CACHE_KEYS)


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
    return build_cached_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_cached,
        num_workers=num_workers,
        distributed=distributed,
    )

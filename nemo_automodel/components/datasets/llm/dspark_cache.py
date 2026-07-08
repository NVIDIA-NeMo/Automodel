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

import hashlib
import os
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader

from nemo_automodel.components.datasets.llm.offline_cache import (
    CachedTensorDataset,
    atomic_write,
    build_cached_dataloader,
    collate_cached,
    existing_shard_indices,
    load_safetensors,
    manifest_path,
    shard_path,
    write_tensor_shard,
)
from nemo_automodel.components.datasets.llm.offline_cache import (
    read_manifest as _read_manifest,
)
from nemo_automodel.components.datasets.llm.offline_cache import (
    write_manifest as _write_manifest,
)

_TARGET_WEIGHTS_NAME = "target_weights.safetensors"
_FORMAT_VERSION = 1
_CACHE_NAME = "DSpark"

CACHE_KEYS = (
    "input_ids",
    "loss_mask",
    "target_hidden_states",
    "target_last_hidden_states",
)

DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def compute_batch_cache(target_batch, cache_dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Convert one captured DSpark target batch into the on-disk cache tensors.

    Shared by the single-process and distributed precompute entry points so both
    producers emit byte-identical cache fields (``CACHE_KEYS``).
    """
    return {
        "input_ids": target_batch.input_ids.to(torch.long).cpu(),
        "loss_mask": target_batch.loss_mask.to(torch.long).cpu(),
        "target_hidden_states": target_batch.target_hidden_states.to(cache_dtype).cpu(),
        "target_last_hidden_states": target_batch.target_last_hidden_states.to(cache_dtype).cpu(),
    }


def tokenizer_chat_template_sha256(tokenizer) -> str:
    """Return a stable identity for the tokenizer's effective (post-override) chat template."""
    template = getattr(tokenizer, "chat_template", None) or ""
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def build_cache_manifest(
    *,
    target_model: str,
    target_model_type: str,
    target_text_config,
    seq_length: int,
    dtype: str,
    num_samples: int,
    shard_size: int,
    target_layer_ids: list[int],
    train_data_path: str,
    train_split: str | None,
    shuffle_seed: int,
    mask_reasoning_content: bool,
    chat_template_sha256: str,
) -> dict[str, Any]:
    """Assemble the DSpark cache manifest.

    Single source of the manifest schema: both precompute entry points call this, and
    ``train_dspark`` validates cached training against these fields, so adding or
    renaming a field here is the one place the contract changes.

    Beyond the tensor-shaping target/cache settings, the manifest records the
    *input identity* of the run (dataset path/split, shuffle seed, masking
    settings, effective chat template): a rerun into an existing directory with
    a different input therefore fails the manifest-compatibility check instead
    of silently interleaving old and new supervision shard by shard.
    """
    hidden_size = int(target_text_config.hidden_size)
    return {
        "target_model": target_model,
        "target_model_type": target_model_type,
        "target_vocab_size": int(target_text_config.vocab_size),
        "hidden_size": hidden_size,
        "num_hidden_layers": int(target_text_config.num_hidden_layers),
        "seq_length": seq_length,
        "dtype": dtype,
        "num_samples": num_samples,
        "shard_size": shard_size,
        "target_hidden_dim": hidden_size * len(target_layer_ids),
        "target_last_hidden_dim": hidden_size,
        "target_layer_ids": list(target_layer_ids),
        "train_data_path": str(train_data_path),
        "train_split": train_split,
        "shuffle_seed": int(shuffle_seed),
        "mask_reasoning_content": bool(mask_reasoning_content),
        "chat_template_sha256": chat_template_sha256,
    }


_IDENTITY_EXEMPT_FIELDS = ("format_version", "complete")


def manifest_mismatch_fields(recorded: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    """Return the manifest keys whose values differ, ignoring bookkeeping fields.

    ``format_version`` and the ``complete`` marker describe the on-disk state,
    not the run configuration, so they never count as a mismatch.
    """
    return sorted(
        k
        for k in recorded.keys() | manifest.keys()
        if k not in _IDENTITY_EXEMPT_FIELDS and recorded.get(k) != manifest.get(k)
    )


def write_manifest(cache_dir: str, manifest: dict) -> str:
    """Persist the DSpark cache manifest."""
    return _write_manifest(cache_dir, manifest, _FORMAT_VERSION)


def ensure_manifest_complete(manifest: dict[str, Any], cache_dir: str) -> None:
    """Reject a cache whose producer did not finish writing.

    The precompute entry points write the manifest with ``complete: false``
    before the first shard and flip it to ``true`` only after every shard has
    been written, so an interrupted (or still-running) precompute cannot be
    consumed as a valid cache. Manifests written before the marker existed
    have no ``complete`` field and are accepted.
    """
    if manifest.get("complete", True) is not True:
        raise ValueError(
            f"DSpark cache at {cache_dir} is marked incomplete (its precompute was interrupted or is "
            "still running). Re-run the precompute with the same configuration to finish it, or delete "
            "the directory and regenerate."
        )


def read_manifest(cache_dir: str, allow_incomplete: bool = False) -> dict[str, Any]:
    """Load and validate the DSpark cache manifest.

    ``allow_incomplete`` is for the precompute producers themselves (compat
    checks against a partially written directory); consumers keep the default
    so an interrupted precompute is never read as a valid cache.
    """
    manifest = _read_manifest(
        cache_dir,
        cache_name=_CACHE_NAME,
        format_version=_FORMAT_VERSION,
        producer_name="precompute_dspark",
    )
    if not allow_incomplete:
        ensure_manifest_complete(manifest, cache_dir)
    return manifest


def _target_weights_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, _TARGET_WEIGHTS_NAME)


def write_target_weights(
    cache_dir: str, embed_tokens: torch.nn.Module, lm_head: torch.nn.Module, dtype: torch.dtype
) -> str:
    """Persist target embedding and lm_head weights for target-free draft initialization."""
    embed_weight = getattr(embed_tokens, "weight", None)
    head_weight = getattr(lm_head, "weight", None)
    if embed_weight is None:
        raise ValueError("Target input embeddings do not expose a .weight tensor.")
    if head_weight is None:
        raise ValueError("Target output embeddings / lm_head do not expose a .weight tensor.")
    save_file, _ = load_safetensors(_CACHE_NAME)
    tensors = {
        "embed_tokens.weight": embed_weight.detach().to(dtype).cpu().contiguous().clone(),
        "lm_head.weight": head_weight.detach().to(dtype).cpu().contiguous().clone(),
    }
    return atomic_write(_target_weights_path(cache_dir), lambda tmp: save_file(tensors, tmp))


def read_target_weight_modules(cache_dir: str) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Load target weights and return module-like objects exposing ``.weight``."""
    _, safe_open = load_safetensors(_CACHE_NAME)
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
    return write_tensor_shard(cache_dir, shard_index, samples, CACHE_KEYS, _CACHE_NAME)


class CachedDSparkDataset(CachedTensorDataset):
    """Read DSpark offline cache shards lazily."""

    def __init__(self, cache_dir: str):
        super().__init__(
            cache_dir=cache_dir,
            cache_name=_CACHE_NAME,
            format_version=_FORMAT_VERSION,
            disk_keys=CACHE_KEYS,
        )
        ensure_manifest_complete(self.manifest, cache_dir)


def _collate_cached(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack per-sample cache dicts into a batch."""
    return collate_cached(features, CACHE_KEYS)


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
    return build_cached_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=_collate_cached,
        num_workers=num_workers,
        distributed=distributed,
    )


__all__ = [
    "CACHE_KEYS",
    "DTYPE_MAP",
    "CachedDSparkDataset",
    "build_cache_manifest",
    "build_cached_dspark_dataloader",
    "compute_batch_cache",
    "ensure_manifest_complete",
    "existing_shard_indices",
    "manifest_mismatch_fields",
    "manifest_path",
    "read_manifest",
    "read_target_weight_modules",
    "shard_path",
    "tokenizer_chat_template_sha256",
    "write_manifest",
    "write_shard",
    "write_target_weights",
]

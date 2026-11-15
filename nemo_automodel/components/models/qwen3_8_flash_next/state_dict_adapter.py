# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""State-dict conversion for Qwen3.8-Flash-Next and its owner-sharded Engram table.

Dense, grouped-MoE, shared-expert, and GatedDeltaNet parameters reuse the
Qwen3.5-MoE checkpoint layouts.  The PLE table is special: the checkpoint
stores ``split_ngram_parts`` physical shard tensors while the native module
registers one contiguous rank-local row range.  ``to_hf`` exposes the local
range as narrow views of the native parameter, so the checkpoint reader writes
directly into final model storage and the 51.2B-parameter global table is
never materialized.  MTP checkpoint keys are ignored in both directions
because the SFT target does not construct MTP.

A checkpoint whose config sets ``ple_embedding_dtype: float8_e4m3fn`` stores
the table as FP8 shards plus one global BF16 ``weight_scale``.  That table is a
frozen host table outside the model's state dict: :meth:`from_hf` checks the
checkpoint format and loads the table and its scale directly from the
checkpoint directory, and the shards never pass through the DCP read.
"""

from __future__ import annotations

import json
import logging
import re
import struct
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5_moe.state_dict_adapter import Qwen3_5MoeStateDictAdapter
from nemo_automodel.components.models.qwen3_8_flash_next.config import Qwen3_8_FlashNextTextConfig
from nemo_automodel.components.models.qwen3_8_flash_next.engram import (
    QWEN3_8_FLASH_NEXT_FP8_PLE_DTYPE,
    Qwen3_8_FlashNextHostNGramTable,
    Qwen3_8_FlashNextOwnerShardedEmbedding,
)
from nemo_automodel.components.moe.layers import MoEConfig

logger = logging.getLogger(__name__)

# Safetensors header dtype name of a ``torch.float8_e4m3fn`` tensor.
_SAFETENSORS_FP8_E4M3 = "F8_E4M3"


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    """Return the JSON header of one safetensors file without reading tensor data."""
    with open(path, "rb") as checkpoint_file:
        (header_size,) = struct.unpack("<Q", checkpoint_file.read(8))
        return json.loads(checkpoint_file.read(header_size))


def _local_safetensors_weight_map(checkpoint_dir: Path) -> dict[str, str] | None:
    """Map each tensor key of a local safetensors checkpoint to its file name.

    Args:
        checkpoint_dir: Directory with ``model.safetensors.index.json`` or with
            ``*.safetensors`` files.

    Returns:
        Key to file-name mapping, or ``None`` when the directory holds no
        safetensors checkpoint.
    """
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.is_file():
        return dict(json.loads(index_path.read_text())["weight_map"])
    weight_map: dict[str, str] = {}
    for shard_path in sorted(checkpoint_dir.glob("*.safetensors")):
        for key in _read_safetensors_header(shard_path):
            if key != "__metadata__":
                weight_map[key] = shard_path.name
    return weight_map or None


def _local_checkpoint_dir(name_or_path: str | None) -> Path | None:
    """Return the local folder of a named checkpoint: the path itself or its Hugging Face cache snapshot.

    Args:
        name_or_path: The model's ``name_or_path``; empty for a model built
            from a config.

    Returns:
        The folder, or ``None`` for a model without a named checkpoint.

    Raises:
        RuntimeError: If a named checkpoint has no local copy.
    """
    if not name_or_path:
        return None
    path = Path(name_or_path)
    if path.is_dir():
        return path
    try:
        return Path(snapshot_download(str(name_or_path), local_files_only=True))
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Checkpoint {name_or_path!r} has no local copy, so the Qwen3.8-Flash-Next PLE table format cannot be "
            "checked"
        ) from error


def _safetensors_layouts(
    checkpoint_dir: Path, weight_map: dict[str, str], keys: Iterable[str]
) -> dict[str, tuple[str, list[int]]]:
    """Return the stored safetensors dtype name (for example ``BF16``) and shape of each key."""
    keys_by_file: dict[str, list[str]] = {}
    for key in keys:
        keys_by_file.setdefault(weight_map[key], []).append(key)
    layouts: dict[str, tuple[str, list[int]]] = {}
    for file_name, file_keys in keys_by_file.items():
        header = _read_safetensors_header(checkpoint_dir / file_name)
        for key in file_keys:
            layouts[key] = (header[key]["dtype"], list(header[key]["shape"]))
    return layouts


class Qwen3_8_FlashNextStateDictAdapter(Qwen3_5MoeStateDictAdapter):
    """Convert Qwen3.8-Flash-Next checkpoints without gathering the global PLE table.

    Args:
        config: Text architecture configuration.
        moe_config: Native grouped-MoE layout.
        backend: Kernel backend configuration.
        engram_table: The model's PLE table: the trainable owner-sharded BF16
            embedding, or the frozen FP8 host table when
            ``config.ple_embedding_dtype`` is ``"float8_e4m3fn"``.
        dtype: Native parameter dtype.
        pretrained_model_name_or_path: Checkpoint directory. The FP8 table is
            read from it directly, so FP8 checkpoints must be local.
    """

    _supports_low_memory_dcp_load = True

    def __init__(
        self,
        config: Qwen3_8_FlashNextTextConfig,
        moe_config: MoEConfig,
        backend: BackendConfig,
        engram_table: Qwen3_8_FlashNextOwnerShardedEmbedding | Qwen3_8_FlashNextHostNGramTable,
        dtype: torch.dtype = torch.bfloat16,
        pretrained_model_name_or_path: str | None = None,
    ) -> None:
        super().__init__(
            config=config,
            moe_config=moe_config,
            backend=backend,
            dtype=dtype,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            mtp_expert_hf_layout="grouped",
            text_only=False,
        )
        self.engram_table = engram_table
        self.split_ngram_parts = int(config.split_ngram_parts)
        ple_layer_idx = int(config.ple_layer_ids[0]) - 1
        self._table_native_key = f"model.language_model.layers.{ple_layer_idx}.ple.ple_embedding.ngram_embedding.weight"
        self._table_hf_prefix = self._table_native_key.removesuffix(".weight")
        self._table_hf_key_pattern = re.compile(rf"{re.escape(self._table_hf_prefix)}\.shard_(\d+)\.weight")
        self._table_shard_keys = [
            f"{self._table_hf_prefix}.shard_{shard_idx}.weight" for shard_idx in range(self.split_ngram_parts)
        ]
        self._table_scale_key = f"{self._table_hf_prefix}.weight_scale"
        self._view_loaded_native_keys: set[str] = set()

        self._fp8_table = config.ple_embedding_dtype == QWEN3_8_FLASH_NEXT_FP8_PLE_DTYPE
        if self._fp8_table != isinstance(engram_table, Qwen3_8_FlashNextHostNGramTable):
            raise TypeError(
                f"ple_embedding_dtype={config.ple_embedding_dtype!r} does not match the PLE table type "
                f"{type(engram_table).__name__}"
            )
        if self._fp8_table:
            # The frozen host table is outside the state dict: no row views.
            return

        self._rows_per_checkpoint_shard = engram_table.num_embeddings // self.split_ngram_parts
        start, end = int(engram_table.global_row_start), int(engram_table.global_row_end)
        if start % self._rows_per_checkpoint_shard or end % self._rows_per_checkpoint_shard:
            raise ValueError(
                "The owner-sharded Engram row range must align to complete physical checkpoint shards; "
                f"got [{start}, {end}) with {self._rows_per_checkpoint_shard} rows per shard."
            )
        self._first_local_checkpoint_shard = start // self._rows_per_checkpoint_shard
        self._end_local_checkpoint_shard = end // self._rows_per_checkpoint_shard

    @property
    def view_loaded_native_keys(self) -> set[str]:
        """Return native parameters already populated through checkpoint views."""
        return set(self._view_loaded_native_keys)

    def get_hf_state_dict_keys(self, state_dict: dict[str, Any]) -> list[str]:
        """Return the rank-independent global HF key set without gathering PLE.

        Consolidated checkpoint planning requires the same global key list on
        every rank, so the one local PLE weight is replaced by all
        ``split_ngram_parts`` physical shard names.  No data is touched.
        """
        keys: list[str] = []
        for fqn, tensor in state_dict.items():
            if fqn.startswith("mtp.") or re.match(r".*_extra_state.*", fqn):
                continue
            if fqn == self._table_native_key:
                keys.extend(
                    f"{self._table_hf_prefix}.shard_{shard_idx}.weight" for shard_idx in range(self.split_ngram_parts)
                )
                continue
            keys.extend(
                key
                for key, _ in self.convert_single_tensor_to_hf(
                    fqn,
                    tensor,
                    exclude_key_regex=r".*_extra_state.*",
                    quantization=False,
                )
            )
        return keys

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs: Any) -> list[tuple[str, Any]]:
        """Convert one native tensor, specializing the PLE and MTP entries."""
        if fqn.startswith("mtp."):
            return []
        if fqn != self._table_native_key:
            return super().convert_single_tensor_to_hf(fqn, tensor, **kwargs)

        # Split the local table rows into narrow views, one per complete
        # physical checkpoint shard owned by this rank.
        local_tensor = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        exclude_key_regex = kwargs.get("exclude_key_regex")
        views: list[tuple[str, torch.Tensor]] = []
        for shard_idx in range(self._first_local_checkpoint_shard, self._end_local_checkpoint_shard):
            key = f"{self._table_hf_prefix}.shard_{shard_idx}.weight"
            if exclude_key_regex and re.match(exclude_key_regex, key):
                continue
            local_row_start = shard_idx * self._rows_per_checkpoint_shard - int(self.engram_table.global_row_start)
            views.append((key, local_tensor.narrow(0, local_row_start, self._rows_per_checkpoint_shard)))
        return views

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Check the PLE table format, then drop PLE and MTP checkpoint keys.

        BF16 table: the PLE entries returned by :meth:`to_hf` alias the native
        parameter, so by the time DCP hands them back the table is already
        populated. ``view_loaded_native_keys`` records the native key as loaded.

        FP8 table: this call loads the frozen host table and its scale from the
        checkpoint directory (a no-op when it is already loaded from it).

        Supplied PLE tensors are checked before they are dropped: a scale is an
        error with a BF16 config and must equal the loaded scale with an FP8
        config; a supplied shard must be FP8 with an FP8 config and must not be
        FP8 with a BF16 config. DCP destinations (views of the BF16 table) carry
        the model's dtype, not the checkpoint's; for them only the checkpoint
        directory establishes the stored format.

        Args:
            hf_state_dict: Checkpoint-named tensors read from the checkpoint.
            device_mesh: Expert-parallel mesh for the plain-tensor init path.
            **kwargs: Passed to the Qwen3.5-MoE conversion.

        Returns:
            Native-named state dict without PLE table, scale, or MTP entries.
        """
        format_checked = self._check_and_load_checkpoint_table()
        self._check_supplied_table_tensors(hf_state_dict, format_checked=format_checked)
        self._view_loaded_native_keys.clear()
        filtered_state_dict: dict[str, Any] = {}
        for key, value in hf_state_dict.items():
            if key.startswith("mtp.") or key == self._table_scale_key:
                continue
            if self._table_hf_key_pattern.fullmatch(key):
                if not self._fp8_table:
                    self._view_loaded_native_keys.add(self._table_native_key)
                continue
            filtered_state_dict[key] = value
        return super().from_hf(filtered_state_dict, device_mesh=device_mesh, **kwargs)

    def _check_supplied_table_tensors(self, hf_state_dict: dict[str, Any], *, format_checked: bool) -> None:
        """Validate PLE table and scale tensors supplied to :meth:`from_hf` before they are dropped.

        Args:
            hf_state_dict: Checkpoint-named tensors. A scale has shape ``[1]``;
                a shard has shape ``[rows_per_shard, embedding_dim]``.
            format_checked: Whether the checkpoint directory already
                established the stored format.

        Raises:
            RuntimeError: On a supplied scale with a BF16 config, a scale that
                differs from the loaded one, or a shard whose dtype contradicts
                the config.
        """
        local_table = None
        if not self._fp8_table:
            weight = self.engram_table.weight
            local_table = weight.to_local() if isinstance(weight, DTensor) else weight
        unchecked_views = []
        for key, value in hf_state_dict.items():
            if key == self._table_scale_key:
                if not self._fp8_table:
                    raise RuntimeError(
                        f"A {key} was supplied, but the config does not set "
                        f"ple_embedding_dtype={QWEN3_8_FLASH_NEXT_FP8_PLE_DTYPE!r}: a scale with a BF16 PLE table"
                    )
                if not (
                    isinstance(value, torch.Tensor)
                    and value.dtype == torch.bfloat16
                    and tuple(value.shape) == (1,)
                    and torch.equal(value.cpu(), self.engram_table.scale)
                ):
                    raise RuntimeError(f"The supplied {key} differs from the scale loaded with the FP8 PLE table")
            elif self._table_hf_key_pattern.fullmatch(key):
                if self._fp8_table:
                    if value.dtype != torch.float8_e4m3fn:
                        raise RuntimeError(
                            f"The config sets ple_embedding_dtype={QWEN3_8_FLASH_NEXT_FP8_PLE_DTYPE!r}, but the "
                            f"supplied {key} is {value.dtype}"
                        )
                elif value.dtype == torch.float8_e4m3fn:
                    raise RuntimeError(
                        f"The supplied {key} is float8_e4m3fn, but the config does not set "
                        f"ple_embedding_dtype={QWEN3_8_FLASH_NEXT_FP8_PLE_DTYPE!r}"
                    )
                elif value.untyped_storage().data_ptr() == local_table.untyped_storage().data_ptr():
                    unchecked_views.append(key)
        if unchecked_views and not format_checked:
            # The checkpoint reader casts into these BF16 destinations, so their stored dtype is not visible here.
            if self.pretrained_model_name_or_path:
                raise RuntimeError(
                    f"Cannot check the stored format of {len(unchecked_views)} PLE table shards: checkpoint "
                    f"{self.pretrained_model_name_or_path!r} has no local safetensors files"
                )
            # A model built from a config has no base checkpoint; it loads tables that it saved itself.
            logger.warning(
                "Loaded %d Qwen3.8-Flash-Next PLE table shards into a model built from a config (no base "
                "checkpoint) without a check of their stored dtype and of a weight_scale",
                len(unchecked_views),
            )

    def _check_and_load_checkpoint_table(self) -> bool:
        """Check the checkpoint's PLE table format against the config; load an FP8 table.

        The config's ``ple_embedding_dtype`` names the format. The checkpoint
        must agree: equal FP8 shards with a ``weight_scale``, or BF16 shards
        without one. Any other combination is an error, never a silent fallback.

        Returns:
            Whether the checkpoint's safetensors files established the format.
            ``False`` for a BF16 table of a model built from a config, or of a
            checkpoint without safetensors files; :meth:`from_hf` then checks
            the supplied tensors.

        Raises:
            RuntimeError: On a format mismatch, for a named checkpoint without
                a local copy, or for an FP8 table without a local safetensors
                checkpoint.
        """
        checkpoint_dir = _local_checkpoint_dir(self.pretrained_model_name_or_path)
        weight_map = None if checkpoint_dir is None else _local_safetensors_weight_map(checkpoint_dir)
        if weight_map is None:
            if self._fp8_table:
                raise RuntimeError(
                    "The FP8 Qwen3.8-Flash-Next PLE table is read from a local safetensors checkpoint; got "
                    f"pretrained_model_name_or_path={self.pretrained_model_name_or_path!r}"
                )
            return False
        present_shard_keys = [key for key in self._table_shard_keys if key in weight_map]
        shard_layouts = _safetensors_layouts(checkpoint_dir, weight_map, present_shard_keys)
        shard_dtypes = {dtype for dtype, _ in shard_layouts.values()}
        has_scale = self._table_scale_key in weight_map

        if not self._fp8_table:
            if has_scale:
                raise RuntimeError(
                    f"Checkpoint {checkpoint_dir} has {self._table_scale_key}, but its config does not set "
                    f"ple_embedding_dtype={QWEN3_8_FLASH_NEXT_FP8_PLE_DTYPE!r}: a scale with a BF16 PLE table"
                )
            if _SAFETENSORS_FP8_E4M3 in shard_dtypes:
                raise RuntimeError(
                    f"Checkpoint {checkpoint_dir} stores the PLE table shards as {_SAFETENSORS_FP8_E4M3}, but its "
                    f"config does not set ple_embedding_dtype={QWEN3_8_FLASH_NEXT_FP8_PLE_DTYPE!r}"
                )
            return True

        if len(present_shard_keys) != len(self._table_shard_keys):
            raise RuntimeError(
                f"Checkpoint {checkpoint_dir} has {len(present_shard_keys)} of the "
                f"{len(self._table_shard_keys)} PLE table shards {self._table_hf_prefix}.shard_<k>.weight"
            )
        if shard_dtypes != {_SAFETENSORS_FP8_E4M3}:
            raise RuntimeError(
                f"Config sets ple_embedding_dtype={QWEN3_8_FLASH_NEXT_FP8_PLE_DTYPE!r}, but checkpoint "
                f"{checkpoint_dir} stores the PLE table shards as {sorted(shard_dtypes)}"
            )
        # Shard k holds rows [k * rows_per_shard, (k + 1) * rows_per_shard), so a shard of another
        # length would shift every later shard: require equal shards.
        num_embeddings, embedding_dim = self.engram_table.num_embeddings, self.engram_table.embedding_dim
        if num_embeddings % self.split_ngram_parts:
            raise RuntimeError(
                f"The PLE table's {num_embeddings} rows do not split into {self.split_ngram_parts} equal shards"
            )
        expected_shape = [num_embeddings // self.split_ngram_parts, embedding_dim]
        wrong_shapes = {key: shape for key, (_, shape) in shard_layouts.items() if shape != expected_shape}
        if wrong_shapes:
            raise RuntimeError(
                f"Checkpoint {checkpoint_dir} has PLE table shards whose shape is not {expected_shape}: "
                f"{dict(list(wrong_shapes.items())[:4])}"
            )
        if not has_scale:
            raise RuntimeError(
                f"Checkpoint {checkpoint_dir} has an FP8 PLE table but no {self._table_scale_key}: "
                "the table values would be unscaled"
            )
        self.engram_table.load(
            checkpoint_dir,
            weight_map=weight_map,
            shard_keys=self._table_shard_keys,
            scale_key=self._table_scale_key,
        )
        return True

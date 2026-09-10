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

"""State dict adapter for DeepSeek V4.1.

The released ``deepseek-ai/DeepSeek-V4.1-Flash`` safetensors follow the
reference inference module tree.  On-disk layout (from the shard headers):

* FP8 E4M3 projections with ``float8_e8m0fnu`` scales over **32x32** blocks
  (``attn.{wq_a,wq_b,wkv,wo_a,wo_b}``, ``attn.indexer.wq_b``,
  ``ffn.shared_experts.w{1,2,3}``, ``engram.wkv``).  DeepSeek V4 used 128x128
  fp32 scales, so the block size is inferred from the scale shape here.
* FP4 E2M1 routed experts packed two per ``int8`` with per-row / 32-column
  ``e8m0`` scales (same layout as V4-Flash, reused from the V4 adapter).
* Engram tables: FP8 E4M3 ``[rows, 256]`` with per-row / 32-column ``e8m0``
  scales ``[rows, 8]``.
* BF16 / FP32 for everything else (norms, gate, hyper-connection mixers,
  compressor, indexer keys, attention sink, embeddings, head).

Key mapping (HF -> internal):
  embed.weight                           -> model.embed_tokens.weight
  norm.weight                            -> model.norm.weight
  head.weight                            -> lm_head.weight
  layers.{i}.attn_norm.weight            -> model.layers.{i}.input_layernorm.weight
  layers.{i}.ffn_norm.weight             -> model.layers.{i}.post_attention_layernorm.weight
  layers.{i}.attn.attn_sink              -> model.layers.{i}.self_attn.sinks_param.weight
  layers.{i}.attn.*                      -> model.layers.{i}.self_attn.*   (compressor.*, indexer.* keep their names)
  layers.{i}.ffn.gate.bias               -> model.layers.{i}.mlp.gate.e_score_correction_bias
  layers.{i}.ffn.gate.weight             -> model.layers.{i}.mlp.gate.weight
  layers.{i}.ffn.shared_experts.w1/w3/w2 -> model.layers.{i}.mlp.shared_experts.gate_proj/up_proj/down_proj
  layers.{i}.ffn.experts.{j}.w1/w3/w2    -> stacked into model.layers.{i}.mlp.experts.gate_and_up_projs / down_projs
  layers.{i}.hc_attn_{fn,base,scale}     -> model.layers.{i}.attn_hc.{fn,base,scale}
  layers.{i}.hc_ffn_{fn,base,scale}      -> model.layers.{i}.ffn_hc.{fn,base,scale}
  layers.{i}.engram.*                    -> model.layers.{i}.engram.*

Dropped on load: ``mtp.*`` (DSpark draft), ``vision.*`` / ``aligner.*`` /
``image_*`` (vision tower) and ``ffn.gate.bias_vl`` (image-token routing
bias); ``engram.*`` when the config disables Engram.
"""

from __future__ import annotations

import math
import re
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import dequantize_from_fp8
from nemo_automodel.components.models.deepseek_v4.state_dict_adapter import (
    DeepSeekV4StateDictAdapter,
    _ExpertQuantLayout,
)
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_utils import is_dtensor

FP8_BLOCK_SIZE = 32
ENGRAM_SCALE_BLOCK = 32

_HF_TO_INTERNAL_RENAMES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^embed\.(.+)$"), r"model.embed_tokens.\1"),
    (re.compile(r"^norm\.(.+)$"), r"model.norm.\1"),
    (re.compile(r"^head\.(.+)$"), r"lm_head.\1"),
    (re.compile(r"^layers\.(\d+)\.attn_norm\.(.+)$"), r"model.layers.\1.input_layernorm.\2"),
    (re.compile(r"^layers\.(\d+)\.ffn_norm\.(.+)$"), r"model.layers.\1.post_attention_layernorm.\2"),
    (re.compile(r"^layers\.(\d+)\.attn\.attn_sink$"), r"model.layers.\1.self_attn.sinks_param.weight"),
    (re.compile(r"^layers\.(\d+)\.attn\.(.+)$"), r"model.layers.\1.self_attn.\2"),
    (re.compile(r"^layers\.(\d+)\.ffn\.gate\.bias$"), r"model.layers.\1.mlp.gate.e_score_correction_bias"),
    (re.compile(r"^layers\.(\d+)\.ffn\.gate\.(.+)$"), r"model.layers.\1.mlp.gate.\2"),
    (re.compile(r"^layers\.(\d+)\.ffn\.shared_experts\.w1\.(.+)$"), r"model.layers.\1.mlp.shared_experts.gate_proj.\2"),
    (re.compile(r"^layers\.(\d+)\.ffn\.shared_experts\.w3\.(.+)$"), r"model.layers.\1.mlp.shared_experts.up_proj.\2"),
    (re.compile(r"^layers\.(\d+)\.ffn\.shared_experts\.w2\.(.+)$"), r"model.layers.\1.mlp.shared_experts.down_proj.\2"),
    (re.compile(r"^layers\.(\d+)\.hc_attn_(base|fn|scale)$"), r"model.layers.\1.attn_hc.\2"),
    (re.compile(r"^layers\.(\d+)\.hc_ffn_(base|fn|scale)$"), r"model.layers.\1.ffn_hc.\2"),
    (re.compile(r"^layers\.(\d+)\.engram\.(.+)$"), r"model.layers.\1.engram.\2"),
]

_INTERNAL_TO_HF_RENAMES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^model\.embed_tokens\.(.+)$"), r"embed.\1"),
    (re.compile(r"^model\.norm\.(.+)$"), r"norm.\1"),
    (re.compile(r"^lm_head\.(.+)$"), r"head.\1"),
    (re.compile(r"^model\.layers\.(\d+)\.input_layernorm\.(.+)$"), r"layers.\1.attn_norm.\2"),
    (re.compile(r"^model\.layers\.(\d+)\.post_attention_layernorm\.(.+)$"), r"layers.\1.ffn_norm.\2"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.sinks_param\.weight$"), r"layers.\1.attn.attn_sink"),
    (re.compile(r"^model\.layers\.(\d+)\.self_attn\.(.+)$"), r"layers.\1.attn.\2"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.e_score_correction_bias$"), r"layers.\1.ffn.gate.bias"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.(.+)$"), r"layers.\1.ffn.gate.\2"),
    (
        re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.gate_proj\.(.+)$"),
        r"layers.\1.ffn.shared_experts.w1.\2",
    ),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.up_proj\.(.+)$"), r"layers.\1.ffn.shared_experts.w3.\2"),
    (
        re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.down_proj\.(.+)$"),
        r"layers.\1.ffn.shared_experts.w2.\2",
    ),
    (re.compile(r"^model\.layers\.(\d+)\.attn_hc\.(fn|base|scale)$"), r"layers.\1.hc_attn_\2"),
    (re.compile(r"^model\.layers\.(\d+)\.ffn_hc\.(fn|base|scale)$"), r"layers.\1.hc_ffn_\2"),
    (re.compile(r"^model\.layers\.(\d+)\.engram\.(.+)$"), r"layers.\1.engram.\2"),
]

# HF keys stored as FP8 E4M3 with 32x32 e8m0 block scales.
_FP8_ON_DISK_PATTERNS = [
    re.compile(r"^layers\.\d+\.attn\.(wq_a|wq_b|wkv|wo_a|wo_b)\.weight$"),
    re.compile(r"^layers\.\d+\.attn\.indexer\.wq_b\.weight$"),
    re.compile(r"^layers\.\d+\.ffn\.shared_experts\.w[123]\.weight$"),
    re.compile(r"^layers\.\d+\.engram\.wkv\.weight$"),
]
_ENGRAM_EMBED_PATTERN = re.compile(r"^layers\.(\d+)\.engram\.embed\.weight$")
_ENGRAM_PATTERN = re.compile(r"^layers\.\d+\.engram\.")
_GATE_BIAS_VL_PATTERN = re.compile(r"^layers\.\d+\.ffn\.gate\.bias_vl$")
_DROPPED_PREFIXES = ("mtp.", "vision.", "aligner.")
_DROPPED_KEYS = {"image_start", "image_end", "image_newline", "image_pad"}


def _rename_hf_key(key: str) -> str:
    for pattern, replacement in _HF_TO_INTERNAL_RENAMES:
        new_key, n = pattern.subn(replacement, key)
        if n:
            return new_key
    return key


def _internal_key_to_hf(key: str) -> str:
    for pattern, replacement in _INTERNAL_TO_HF_RENAMES:
        new_key, n = pattern.subn(replacement, key)
        if n:
            return new_key
    return key


def _scale_to_float(scale: torch.Tensor) -> torch.Tensor:
    """Decode an ``e8m0`` (or any float) scale tensor to fp32.

    ``e8m0`` stores ``2 ** (bits - 127)`` with ``bits == 0`` meaning zero; decode
    explicitly so the adapter does not depend on PyTorch's e8m0 cast support.
    """
    if scale.dtype == torch.float8_e8m0fnu:
        bits = scale.contiguous().view(torch.uint8).int()
        return torch.where(bits == 0, torch.zeros_like(bits, dtype=torch.float32), torch.pow(2.0, (bits - 127).float()))
    return scale.to(torch.float32)


def infer_fp8_block_size(weight_shape: tuple[int, ...], scale_shape: tuple[int, ...]) -> int:
    """Return the square block size that maps ``weight_shape`` onto ``scale_shape``."""
    rows, cols = weight_shape[-2], weight_shape[-1]
    block_rows, block_cols = scale_shape[-2], scale_shape[-1]
    for block_size in (FP8_BLOCK_SIZE, 128):
        if math.ceil(rows / block_size) == block_rows and math.ceil(cols / block_size) == block_cols:
            return block_size
    raise ValueError(f"Cannot infer an FP8 block size for weight {tuple(weight_shape)} and scale {tuple(scale_shape)}")


def dequantize_fp8_blocks(
    weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, name: str = ""
) -> torch.Tensor:
    """Dequantize a 2D FP8 weight with square block scales of any block size."""
    block_size = infer_fp8_block_size(tuple(weight.shape), tuple(scale.shape))
    scale_f32 = _scale_to_float(scale.to_local() if is_dtensor(scale) else scale)
    if is_dtensor(weight) or is_dtensor(scale):
        # Let the shared V3 helper handle DTensor slicing of the scale grid.
        return dequantize_from_fp8(weight, scale_f32, dtype=dtype, BLOCK_SIZE=block_size, name=name)
    rows, cols = weight.shape
    pad_rows, pad_cols = (-rows) % block_size, (-cols) % block_size
    w = weight.float()
    if pad_rows or pad_cols:
        w = torch.nn.functional.pad(w, (0, pad_cols, 0, pad_rows))
    block_rows, block_cols = w.shape[0] // block_size, w.shape[1] // block_size
    w.view(block_rows, block_size, block_cols, block_size).mul_(
        scale_f32.to(w.device).view(block_rows, 1, block_cols, 1)
    )
    return w[:rows, :cols].to(dtype)


def dequantize_engram_table(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize per-row / 32-column Engram scales on each row owner.

    Args:
        weight: FP8 tensor of shape [rows, channels], optionally a DTensor
            with placement Shard(0) and uneven local shape [local_rows, channels].
        scale: Scale tensor of shape [rows, channels / 32], with the same row
            ownership as weight. Channels must be divisible by 32.
        dtype: Output floating-point dtype.

    Returns:
        Independent tensor of shape [rows, channels], preserving the weight's
        global shape, device and row ownership, including empty local shards.
    """
    weight_local = weight.to_local() if is_dtensor(weight) else weight
    scale_local = scale.to_local() if is_dtensor(scale) else scale
    rows, dim = weight_local.shape
    if dim % ENGRAM_SCALE_BLOCK or scale_local.shape != (rows, dim // ENGRAM_SCALE_BLOCK):
        raise ValueError(
            f"Engram table {tuple(weight_local.shape)} does not match scale {tuple(scale_local.shape)} "
            f"(expected per-row / {ENGRAM_SCALE_BLOCK}-column scales)"
        )
    scale_f32 = _scale_to_float(scale_local).to(weight_local.device)
    # Tables have hundreds of millions of rows: dequantize in row chunks so the fp32
    # temporaries stay bounded instead of materializing the whole table twice.
    out = torch.empty(rows, dim, dtype=dtype, device=weight_local.device)
    chunk = max(1, (1 << 28) // dim)
    for start in range(0, rows, chunk):
        end = min(start + chunk, rows)
        values = weight_local[start:end].float().view(-1, dim // ENGRAM_SCALE_BLOCK, ENGRAM_SCALE_BLOCK)
        values.mul_(scale_f32[start:end].unsqueeze(-1))
        out[start:end] = values.view(-1, dim).to(dtype)
    if is_dtensor(weight):
        return DTensor.from_local(out, weight.device_mesh, weight.placements, shape=weight.shape, stride=(dim, 1))
    return out


class DeepSeekV41StateDictAdapter(DeepSeekV4StateDictAdapter):
    """State dict adapter for DeepSeek V4.1 (see module docstring for the layout)."""

    def __init__(
        self,
        config: DeepseekV41Config,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(config, moe_config, backend, dtype=dtype)
        self.engram_enabled = bool(config.engram_enabled) and bool(config.engram_layer_ids)
        self._engram_rows = dict(zip(config.engram_layer_ids, config.engram_num_embeddings))

    # ------------------------------------------------------------------
    # from_hf
    # ------------------------------------------------------------------

    def _keep_hf_key(self, key: str) -> bool:
        if key.startswith(_DROPPED_PREFIXES) or key in _DROPPED_KEYS:
            return False
        if _GATE_BIAS_VL_PATTERN.match(key):
            return False
        if not self.engram_enabled and _ENGRAM_PATTERN.match(key):
            return False
        return True

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert the released HF checkpoint to the internal format.

        Steps: drop out-of-scope tensors, dequantize FP8 / FP4 weights, stack the
        routed experts, restore Engram owner padding, rename.

        Args:
            hf_state_dict: Released-name tensors in the layouts documented in
                this module. Engram tables have logical shape [rows, channels]
                and optionally placement Shard(0) on a one-dimensional owner
                mesh, with uneven local shape [local_rows, channels].
            device_mesh: Expert aggregation mesh.
            **kwargs: Additional checkpoint protocol arguments.

        Returns:
            Internal-name tensors. Engram DTensors have global shape
            [ceil(rows / owners) * owners, channels] with equal local row counts
            and zero padding. Other layouts follow the base adapter. Floating
            tensors can alias input storage when no conversion is needed.
        """
        filtered = {key: value for key, value in hf_state_dict.items() if self._keep_hf_key(key)}
        filtered = self._dequantize(filtered)
        filtered = self._aggregate_experts(filtered, device_mesh)
        for key, value in filtered.items():
            match = _ENGRAM_EMBED_PATTERN.match(key)
            if match:
                filtered[key] = self._restore_engram_padding(value, int(match.group(1)))
        return {_rename_hf_key(key): value for key, value in filtered.items()}

    def _engram_checkpoint_tensor(self, tensor: torch.Tensor, layer_id: int) -> torch.Tensor:
        """Expose logical checkpoint rows without gathering owner storage.

        Args:
            tensor: Table of global shape [padded_rows, channels], either a
                complete local tensor or a DTensor with placement Shard(0) on
                a one-dimensional owner mesh. Each owner stores equal rows.
            layer_id: Decoder layer identifying the logical checkpoint row count.

        Returns:
            Aliasing view of shape [rows, channels]. A DTensor preserves its
            mesh and row placement; its final local shards may be short or empty.
        """
        rows = self._engram_rows[layer_id]
        if not is_dtensor(tensor):
            if tensor.shape[0] < rows:
                raise ValueError("Owner-local Engram storage must be represented as a global row-sharded DTensor")
            return tensor[:rows]
        if tensor.device_mesh.ndim != 1 or tensor.placements != (Shard(0),):
            raise ValueError("Engram checkpoint tables require Shard(0) on a one-dimensional owner mesh")
        local = tensor.to_local()
        start = tensor.device_mesh.get_local_rank() * math.ceil(tensor.shape[0] / tensor.device_mesh.size())
        valid_rows = max(0, min(local.shape[0], rows - start))
        channels = tensor.shape[1]
        return DTensor.from_local(
            local[:valid_rows],
            tensor.device_mesh,
            tensor.placements,
            shape=torch.Size((rows, channels)),
            stride=(channels, 1),
        )

    def _restore_engram_padding(self, tensor: torch.Tensor, layer_id: int) -> torch.Tensor:
        """Restore equal owner storage after reading logical checkpoint rows.

        Args:
            tensor: Table of global shape [rows, channels], optionally a DTensor
                with placement Shard(0) on a one-dimensional owner mesh and
                uneven local shape [local_rows, channels].
            layer_id: Decoder layer identifying the logical checkpoint row count.

        Returns:
            Tensor unchanged for a local table. Distributed tables have global
            shape [ceil(rows / owners) * owners, channels] and equal local row
            counts. Unpadded shards alias input storage; added rows are zero.
        """
        if not is_dtensor(tensor):
            return tensor
        if tensor.device_mesh.ndim != 1 or tensor.placements != (Shard(0),):
            raise ValueError("Engram checkpoint tables require Shard(0) on a one-dimensional owner mesh")
        owners = tensor.device_mesh.size()
        local_rows = math.ceil(self._engram_rows[layer_id] / owners)
        local = tensor.to_local()
        if local.shape[0] < local_rows:
            local = torch.nn.functional.pad(local, (0, 0, 0, local_rows - local.shape[0]))
        channels = tensor.shape[1]
        return DTensor.from_local(
            local,
            tensor.device_mesh,
            tensor.placements,
            shape=torch.Size((local_rows * owners, channels)),
            stride=(channels, 1),
        )

    def _dequantize(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Dequantize every ``<base>.weight`` that has a ``<base>.scale`` companion."""
        for key in list(state_dict.keys()):
            if not key.endswith(".weight"):
                continue
            scale_key = key[: -len(".weight")] + ".scale"
            if scale_key not in state_dict:
                continue
            weight = state_dict[key]
            scale = state_dict.pop(scale_key)
            if (
                self._is_expert_weight_key(key)
                and self._expert_quant_layout_from_tensors(weight, scale) is _ExpertQuantLayout.FP4
            ):
                state_dict[key] = self._dequantize_expert_fp4(weight, scale, self.dtype)
            elif _ENGRAM_EMBED_PATTERN.match(key):
                state_dict[key] = dequantize_engram_table(weight, scale, self.dtype)
            else:
                state_dict[key] = dequantize_fp8_blocks(weight, scale, self.dtype, name=key)
        return state_dict

    # ------------------------------------------------------------------
    # to_hf
    # ------------------------------------------------------------------

    def _internal_key_to_hf(self, key: str) -> str:
        return _internal_key_to_hf(key)

    @staticmethod
    def _is_fp8_on_disk(hf_key: str) -> bool:
        return any(pattern.match(hf_key) for pattern in _FP8_ON_DISK_PATTERNS)

    @staticmethod
    def _fp8_block_scale_placeholder(value: Any) -> torch.Tensor:
        rows, cols = value.shape[-2], value.shape[-1]
        shape = (math.ceil(rows / FP8_BLOCK_SIZE), math.ceil(cols / FP8_BLOCK_SIZE))
        device = value.to_local().device if is_dtensor(value) else value.device
        return torch.empty(shape, dtype=torch.float8_e8m0fnu, device=device)

    @staticmethod
    def _engram_placeholders(value: Any) -> tuple[Any, Any]:
        """Allocate released-layout placeholders with explicit uneven shapes.

        Args:
            value: Table of global shape [rows, channels], optionally a DTensor
                with placement Shard(0) and local shape [local_rows, channels].

        Returns:
            Independent FP8 weights [rows, channels] and E8M0 scales
            [rows, channels / 32], preserving the input's row ownership.
        """
        local = value.to_local() if is_dtensor(value) else value
        rows, dim = local.shape
        packed = torch.empty(rows, dim, dtype=torch.float8_e4m3fn, device=local.device)
        scale = torch.empty(rows, dim // ENGRAM_SCALE_BLOCK, dtype=torch.float8_e8m0fnu, device=local.device)
        if is_dtensor(value):
            return (
                DTensor.from_local(packed, value.device_mesh, value.placements, shape=value.shape, stride=(dim, 1)),
                DTensor.from_local(
                    scale,
                    value.device_mesh,
                    value.placements,
                    shape=torch.Size((value.shape[0], dim // ENGRAM_SCALE_BLOCK)),
                    stride=(dim // ENGRAM_SCALE_BLOCK, 1),
                ),
            )
        return packed, scale

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert one internal tensor to HF keys, optionally emitting on-disk quantized placeholders.

        With ``quantization=True`` the placeholders mirror the released layout so
        DCP can validate shapes / dtypes before the adapter dequantizes on load.

        Args:
            fqn: Internal parameter name.
            tensor: Parameter in its model layout. Engram tables have global
                shape [padded_rows, channels], optionally placement Shard(0) on
                a one-dimensional owner mesh with equal local row counts.
            **kwargs: Checkpoint protocol options, including quantization and
                exclude_key_regex.

        Returns:
            Released-name tensor pairs in the module's documented layouts.
            Engram weights expose only [rows, channels] and scales expose
            [rows, channels / 32], retaining uneven row ownership. Floating
            views alias input storage; quantized placeholders are independent.
        """
        quantization = kwargs.get("quantization", False)
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        result = self._split_merged_expert(fqn, tensor)
        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]
        result = [(_internal_key_to_hf(k), v) for k, v in result]
        for index, (key, value) in enumerate(result):
            match = _ENGRAM_EMBED_PATTERN.match(key)
            if match:
                result[index] = (key, self._engram_checkpoint_tensor(value, int(match.group(1))))
        if not quantization:
            return result

        quantized: list[tuple[str, Any]] = []
        for key, value in result:
            if not key.endswith(".weight"):
                quantized.append((key, value))
                continue
            base = key[: -len(".weight")]
            if self._is_expert_weight_key(key):
                if self._checkpoint_expert_quant_layout() is _ExpertQuantLayout.FP4:
                    packed, scale = self._build_fp4_expert_placeholders(value)
                else:
                    packed = self._fp8_cast(value)
                    scale = self._fp8_block_scale_placeholder(value)
                quantized.append((key, packed))
                quantized.append((base + ".scale", scale))
            elif _ENGRAM_EMBED_PATTERN.match(key):
                packed, scale = self._engram_placeholders(value)
                quantized.append((key, packed))
                quantized.append((base + ".scale", scale))
            elif self._is_fp8_on_disk(key):
                quantized.append((key, self._fp8_cast(value)))
                quantized.append((base + ".scale", self._fp8_block_scale_placeholder(value)))
            else:
                quantized.append((key, value))
        return quantized

    @classmethod
    def _fp8_cast(cls, value: Any) -> Any:
        if is_dtensor(value):
            local = cls._empty_or_cast_fp8(value.to_local())
            return DTensor.from_local(local, value.device_mesh, value.placements)
        return cls._empty_or_cast_fp8(value)

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
  layers.{i}.ffn.gate.bias_vl             -> model.layers.{i}.mlp.gate.bias_vl
  vision.* / aligner.*                   -> model.vision.* / model.aligner.*
  image_{start,end,newline}              -> model.image_{start,end,newline}

Dropped on load/export: ``mtp.*`` (DSpark draft), the unconstructed
``image_pad`` tensor, vision tower/delimiters when vision is disabled, and
``engram.*`` when the config disables Engram. The visual router bias remains
part of every text gate even when no vision tower is constructed.
"""

from __future__ import annotations

import json
import math
import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Shard

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import dequantize_from_fp8
from nemo_automodel.components.models.deepseek_v4.state_dict_adapter import (
    DeepSeekV4StateDictAdapter,
    _ExpertQuantLayout,
)
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin
from nemo_automodel.components.moe.state_dict_utils import is_dtensor, should_load_expert_for_rank

FP8_BLOCK_SIZE = 32
ENGRAM_SCALE_BLOCK = 32

_HF_TO_INTERNAL_RENAMES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^vision\.(.+)$"), r"model.vision.\1"),
    (re.compile(r"^aligner\.(.+)$"), r"model.aligner.\1"),
    (re.compile(r"^(image_start|image_end|image_newline)$"), r"model.\1"),
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
    (re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w1\.(.+)$"), r"model.layers.\1.mlp.experts.\2.gate_proj.\3"),
    (re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w3\.(.+)$"), r"model.layers.\1.mlp.experts.\2.up_proj.\3"),
    (re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w2\.(.+)$"), r"model.layers.\1.mlp.experts.\2.down_proj.\3"),
    (re.compile(r"^layers\.(\d+)\.hc_attn_(base|fn|scale)$"), r"model.layers.\1.attn_hc.\2"),
    (re.compile(r"^layers\.(\d+)\.hc_ffn_(base|fn|scale)$"), r"model.layers.\1.ffn_hc.\2"),
    (re.compile(r"^layers\.(\d+)\.engram\.(.+)$"), r"model.layers.\1.engram.\2"),
]

_INTERNAL_TO_HF_RENAMES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^model\.vision\.(.+)$"), r"vision.\1"),
    (re.compile(r"^model\.aligner\.(.+)$"), r"aligner.\1"),
    (re.compile(r"^model\.(image_start|image_end|image_newline|image_pad)$"), r"\1"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.(.+)$"), r"layers.\1.ffn.experts.\2.w1.\3"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.(.+)$"), r"layers.\1.ffn.experts.\2.w3.\3"),
    (re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.(.+)$"), r"layers.\1.ffn.experts.\2.w2.\3"),
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
_VISION_PREFIXES = ("vision.", "aligner.")
_VISION_DELIMITERS = {"image_start", "image_end", "image_newline"}


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


@dataclass(frozen=True)
class CheckpointLoadAudit:
    """Local streaming-load coverage and byte counts, excluding model storage."""

    loaded_keys: tuple[str, ...]
    source_bytes: int
    loaded_bytes: int
    max_chunk_source_bytes: int
    max_chunk_output_bytes: int


def _local_offsets(tensor: DTensor) -> tuple[int, ...]:
    """Locate a contiguous DTensor shard without gathering its values.

    Args:
        tensor: DTensor of arbitrary global shape, with Shard or Replicate
            placements. Repeated sharding on the same axis is supported.

    Returns:
        Global offsets of this rank's local shard along each tensor dimension.
    """
    offsets = [0] * tensor.ndim
    shape = list(tensor.shape)
    for mesh_dim, placement in enumerate(tensor.placements):
        if isinstance(placement, Partial):
            raise ValueError("Checkpoint conversion requires resolved Shard or Replicate placements, not Partial")
        if isinstance(placement, Shard):
            axis = placement.dim % tensor.ndim
            size, offset = Shard.local_shard_size_and_offset(
                shape[axis], tensor.device_mesh.size(mesh_dim), tensor.device_mesh.get_local_rank(mesh_dim)
            )
            shape[axis] = size
            offsets[axis] += offset
    return tuple(offsets)


def dequantize_checkpoint_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rowwise: bool = False,
) -> torch.Tensor:
    """Decode released FP8 or packed FP4 weights with bounded FP32 temporaries.

    Args:
        weight: FP8 tensor of shape [rows, columns], or packed INT8 tensor of
            shape [rows, columns / 2]. Packed E2M1 stores the even column in
            the low nibble and the odd column in the high nibble. A DTensor
            preserves its global shape and Shard/Replicate placements.
        scale: Tensor of shape [ceil(rows / 32), ceil(columns / 32)] for dense
            FP8, or [rows, ceil(columns / 32)] for FP4 and Engram FP8. A plain
            scale may cover the global matrix or exactly this rank's blocks;
            a DTensor scale must cover the weight shard at matching offsets.
        dtype: Dequantized floating-point storage dtype.
        rowwise: Use per-row scales for FP8 Engram tables. FP4 always uses them.

    Returns:
        Independent tensor of shape [rows, columns] in ``dtype``; DTensor
        inputs retain their mesh and placements. No input is modified, and
        FP32/expanded-scale temporaries cover at most 4M weight elements.
    """
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("Checkpoint weights and scales must both be two-dimensional")
    packed = weight.dtype == torch.int8
    if not packed and weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"Expected packed INT8 FP4 or E4M3 FP8 weights, got {weight.dtype}")
    multiplier = 2 if packed else 1
    row_block = 1 if packed or rowwise else 32
    global_shape = (weight.shape[0], weight.shape[1] * multiplier)
    local_weight = weight.to_local() if isinstance(weight, DTensor) else weight
    offsets = _local_offsets(weight) if isinstance(weight, DTensor) else (0, 0)
    offsets = (offsets[0], offsets[1] * multiplier)
    rows, columns = local_weight.shape[0], local_weight.shape[1] * multiplier
    starts = (offsets[0] // row_block, offsets[1] // 32)
    ends = ((offsets[0] + rows + row_block - 1) // row_block, (offsets[1] + columns + 31) // 32)
    expected_local = tuple(end - start for start, end in zip(starts, ends))
    global_scale_shape = ((global_shape[0] + row_block - 1) // row_block, (global_shape[1] + 31) // 32)
    if isinstance(scale, DTensor):
        local_scale = scale.to_local()
        if _local_offsets(scale) != starts or tuple(local_scale.shape) != expected_local:
            raise ValueError("Scale DTensor placement does not cover the corresponding weight shard")
    elif tuple(scale.shape) == global_scale_shape:
        local_scale = scale[starts[0] : ends[0], starts[1] : ends[1]]
    elif tuple(scale.shape) == expected_local:
        local_scale = scale
    else:
        raise ValueError(
            f"Scale shape {tuple(scale.shape)} does not match global {global_scale_shape} "
            f"or local block coverage {expected_local}"
        )
    if local_scale.device != local_weight.device:
        raise ValueError("Checkpoint weight and scale shards must reside on the same device")
    output = torch.empty((rows, columns), dtype=dtype, device=local_weight.device)
    if not local_weight.is_meta and rows and columns:
        column_ids = (torch.arange(columns, device=local_weight.device) + offsets[1]) // 32 - starts[1]
        row_step = max(1, (4 * 1024 * 1024) // columns)
        table = (
            torch.tensor(
                [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
                dtype=torch.float32,
                device=local_weight.device,
            )
            if packed
            else None
        )
        for begin in range(0, rows, row_step):
            end = min(rows, begin + row_step)
            if packed:
                raw = local_weight[begin:end].contiguous().view(torch.uint8)
                decoded = torch.empty((end - begin, columns), dtype=torch.float32, device=raw.device)
                decoded[:, 0::2] = table[(raw & 15).long()]
                decoded[:, 1::2] = table[(raw >> 4).long()]
            else:
                decoded = local_weight[begin:end].float()
            scale_begin = (offsets[0] + begin) // row_block - starts[0]
            scale_end = (offsets[0] + end + row_block - 1) // row_block - starts[0]
            row_ids = (
                (torch.arange(begin, end, device=local_weight.device) + offsets[0]) // row_block
                - starts[0]
                - scale_begin
            )
            scales = local_scale[scale_begin:scale_end].float()
            output[begin:end].copy_(decoded * scales[row_ids[:, None], column_ids])
    if isinstance(weight, DTensor):
        return DTensor.from_local(
            output,
            weight.device_mesh,
            weight.placements,
            shape=torch.Size(global_shape),
            stride=(global_shape[1], 1),
        )
    return output


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
    """Decode square-block FP8, with bounded scratch for the released V4.1 layout.

    Args:
        weight: FP8 tensor of shape [rows, columns], optionally a DTensor with
            expert or inner matrix sharding as supported by the shared adapter.
        scale: Tensor of shape [ceil(rows / block), ceil(columns / block)],
            where block is 32 for released V4.1 or 128 for legacy V4 weights.
            DTensor scales must cover the corresponding weight shard.
        dtype: Dequantized floating-point dtype.
        name: Checkpoint tensor name used by the legacy V3 decoder.

    Returns:
        Independent tensor of shape [rows, columns], preserving the input
        DTensor global shape, mesh and placements. The released 32x32 path
        bounds FP32 intermediates through ``dequantize_checkpoint_weight``.
    """
    block_size = infer_fp8_block_size(tuple(weight.shape), tuple(scale.shape))
    if block_size == FP8_BLOCK_SIZE:
        return dequantize_checkpoint_weight(weight, scale, dtype=dtype)
    # Use the same dtype conversion as the released 32x32 decoder: E8M0
    # byte 0 is 2**-127, and byte 255 represents NaN.
    scale_f32 = (scale.to_local() if is_dtensor(scale) else scale).float()
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
    if weight.shape[1] % ENGRAM_SCALE_BLOCK:
        raise ValueError(f"Engram channels must be divisible by {ENGRAM_SCALE_BLOCK}, got {weight.shape[1]}")
    return dequantize_checkpoint_weight(weight, scale, dtype=dtype, rowwise=True)


class DeepSeekV41StateDictAdapter(MoESplitExpertsStateDictMixin, DeepSeekV4StateDictAdapter):
    """Convert released V4.1 layouts and stream directly into prepared model storage.

    Floating DCP initialization uses shared MoE views and skips rebuilding
    experts already written into model storage. Export and quantized layouts
    retain V4 conversion. Explicit streaming initialization copies one bounded
    quantized chunk at a time, without gathering experts or Engram owner rows.
    """

    # Quantized DCP loads still allocate converted tensors. The explicit
    # streaming API below owns its bounded direct copies independently.
    _supports_low_memory_dcp_load = False

    def __init__(
        self,
        config: DeepseekV41Config,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(config, moe_config, backend, dtype=dtype)
        self._uses_model_prefix = True
        self.engram_enabled = bool(config.engram_enabled) and bool(config.engram_layer_ids)
        self._engram_rows = dict(zip(config.engram_layer_ids, config.engram_num_embeddings))

    def get_hf_state_dict_keys(self, state_dict: dict[str, Any]) -> list[str]:
        """Return global checkpoint names without inspecting owner-local values.

        Args:
            state_dict: Native model mapping, including pre-distribution local
                Engram parameters and meta tensors of arbitrary shapes.

        Returns:
            Rank-independent released names. Grouped expert keys expand over
            every expert, while each Engram owner reports the same table key.
        """
        keys = []
        for fqn in state_dict:
            if "_extra_state" in fqn or not self._keep_hf_key(_internal_key_to_hf(fqn)):
                continue
            expert = re.fullmatch(r"model\.layers\.(\d+)\.mlp\.experts\.(gate_and_up_projs|down_projs)", fqn)
            if expert:
                projections = (1, 3) if expert[2] == "gate_and_up_projs" else (2,)
                keys.extend(
                    f"layers.{expert[1]}.ffn.experts.{index}.w{projection}.weight"
                    for index in range(self.moe_config.n_routed_experts)
                    for projection in projections
                )
            else:
                keys.append(_internal_key_to_hf(fqn))
        return keys

    @torch.no_grad()
    def load_from_checkpoint(
        self,
        model: torch.nn.Module,
        checkpoint_path: str | Path,
        device_mesh: DeviceMesh | None = None,
    ) -> CheckpointLoadAudit:
        """Stream released safetensors into materialized, rank-owned parameters.

        Args:
            model: Prepared native model. Grouped expert tensors must alias
                model storage, and Engram owner tables must be Shard(0)
                DTensors when distributed. No meta parameters are accepted.
            checkpoint_path: Local snapshot containing the safetensors index
                and shards, or a directory with one ``model.safetensors``.
            device_mesh: Expert mesh accepted for checkpoint API consistency;
                ownership is read from the already distributed parameters.

        Returns:
            Actual local source keys and transferred/storage byte counts.
            Per-chunk counts exclude model storage and decoder scratch, which
            is bounded separately by ``dequantize_checkpoint_weight``. No
            checkpoint tensor or Engram table is gathered across ranks.

        Raises:
            ValueError: A tensor is missing, has incompatible dimensions, or
                its backend exposes temporary rather than model-owned experts.
        """
        del device_mesh
        if not self._expert_checkpoint_tensors_use_model_storage:
            raise ValueError("Streaming checkpoint loading requires experts with model-owned grouped storage")
        root = Path(checkpoint_path)
        index_path = root / "model.safetensors.index.json"
        if index_path.is_file():
            weight_map = json.loads(index_path.read_text())["weight_map"]
        else:
            with safe_open(root / "model.safetensors", framework="pt", device="cpu") as reader:
                weight_map = {key: "model.safetensors" for key in reader.keys()}
        loaded_keys: set[str] = set()
        source_bytes = loaded_bytes = max_source = max_output = 0
        strict_fp32 = getattr(model, "_keep_in_fp32_modules_strict", ()) or ()
        # Shared expert splitting records DCP views. This method performs the
        # copies itself, so those records must not leak into a later DCP load.
        previous_inplace = set(getattr(self, "_inplace_loaded_native_keys", None) or ())
        try:
            with ExitStack() as stack:
                readers = {}
                for fqn, tensor in model.state_dict().items():
                    if not isinstance(tensor, torch.Tensor):
                        continue
                    if tensor.is_meta:
                        raise ValueError(f"Streaming checkpoint destination {fqn} has not been materialized")
                    if tensor.is_floating_point() and any(keyword in fqn for keyword in strict_fp32):
                        local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
                        if tensor.dtype != torch.float32 or local.dtype != torch.float32:
                            raise ValueError(
                                f"Strict FP32 checkpoint destination {fqn} has dtype {tensor.dtype} "
                                f"and local storage dtype {local.dtype}; prepare FP32 storage before loading"
                            )
                    engram = re.fullmatch(r"model\.layers\.(\d+)\.engram\.embed\.weight", fqn)
                    if engram and isinstance(tensor, DTensor):
                        row_start = _local_offsets(tensor)[0]
                        valid_rows = max(0, self._engram_rows[int(engram[1])] - row_start)
                        tensor.to_local()[valid_rows:].zero_()
                    destinations = self.convert_single_tensor_to_hf(
                        fqn,
                        tensor,
                        for_checkpoint_load=True,
                        preserve_dtensor_load_views=True,
                        quantization=False,
                        exclude_key_regex=r".*_extra_state.*",
                    )
                    for key, destination in destinations:
                        local = destination.to_local() if isinstance(destination, DTensor) else destination
                        if f".{self._expert_path_segment}." in fqn and local.numel():
                            native_local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
                            if (
                                local.device != native_local.device
                                or local.untyped_storage().data_ptr() != native_local.untyped_storage().data_ptr()
                            ):
                                raise ValueError(
                                    f"Streaming expert destination {key} does not alias model storage {fqn}"
                                )
                        if key not in weight_map:
                            raise ValueError(f"Checkpoint is missing model tensor {key}")
                        shard_name = weight_map[key]
                        if shard_name not in readers:
                            readers[shard_name] = stack.enter_context(
                                safe_open(root / shard_name, framework="pt", device="cpu")
                            )
                        source = readers[shard_name].get_slice(key)
                        scale_key = key.removesuffix(".weight") + ".scale"
                        scale_source = None
                        if key.endswith(".weight") and scale_key in weight_map:
                            shard_name = weight_map[scale_key]
                            if shard_name not in readers:
                                readers[shard_name] = stack.enter_context(
                                    safe_open(root / shard_name, framework="pt", device="cpu")
                                )
                            scale_source = readers[shard_name].get_slice(scale_key)
                        packed = source.get_dtype() == "I8" and scale_source is not None
                        source_shape = list(source.get_shape())
                        if packed:
                            source_shape[1] *= 2
                        if tuple(source_shape) != tuple(destination.shape):
                            raise ValueError(
                                f"Checkpoint {key} has decoded shape {tuple(source_shape)}, "
                                f"expected {tuple(destination.shape)}"
                            )
                        offsets = _local_offsets(destination) if isinstance(destination, DTensor) else (0,) * local.ndim
                        if scale_source is not None and local.ndim != 2:
                            raise ValueError(f"Quantized checkpoint tensor {key} must be two-dimensional")
                        row_step = max(1, (4 * 1024 * 1024) // max(1, math.prod(local.shape[1:])))
                        row_step = max(32, row_step // 32 * 32) if scale_source is not None else row_step
                        rows = local.shape[0] if local.ndim else 1
                        for begin in range(0, rows, row_step):
                            end = min(rows, begin + row_step)
                            if scale_source is None:
                                slices = tuple(slice(start, start + size) for start, size in zip(offsets, local.shape))
                                if local.ndim:
                                    slices = (slice(offsets[0] + begin, offsets[0] + end), *slices[1:])
                                chunk = source[slices]
                                if chunk.dtype in (torch.int8, torch.float8_e4m3fn):
                                    raise ValueError(f"Quantized checkpoint tensor {key} is missing its scale")
                                target = local[begin:end] if local.ndim else local
                                target.copy_(chunk)
                                chunk_source = chunk.numel() * chunk.element_size()
                                chunk_output = target.numel() * target.element_size()
                            else:
                                rowwise = packed or ".engram.embed." in key
                                row_block = 1 if rowwise else 32
                                r0 = (offsets[0] + begin) // row_block * row_block
                                r1 = min(source_shape[0], (offsets[0] + end + row_block - 1) // row_block * row_block)
                                c0 = offsets[1] // 32 * 32
                                c1 = min(source_shape[1], (offsets[1] + local.shape[1] + 31) // 32 * 32)
                                divisor = 2 if packed else 1
                                raw = source[r0:r1, c0 // divisor : c1 // divisor]
                                scales = scale_source[
                                    r0 // row_block : (r1 + row_block - 1) // row_block, c0 // 32 : (c1 + 31) // 32
                                ]
                                chunk_source = raw.numel() * raw.element_size() + scales.numel() * scales.element_size()
                                decoded = dequantize_checkpoint_weight(
                                    raw.to(local.device), scales.to(local.device), dtype=local.dtype, rowwise=rowwise
                                )
                                local[begin:end].copy_(
                                    decoded[
                                        offsets[0] + begin - r0 : offsets[0] + end - r0,
                                        offsets[1] - c0 : offsets[1] - c0 + local.shape[1],
                                    ]
                                )
                                chunk_output = decoded.numel() * decoded.element_size()
                                del raw, scales, decoded
                            source_bytes += chunk_source
                            max_source = max(max_source, chunk_source)
                            max_output = max(max_output, chunk_output)
                        loaded_bytes += local.numel() * local.element_size()
                        loaded_keys.add(key)
                        if scale_source is not None:
                            loaded_keys.add(scale_key)
        finally:
            self._inplace_loaded_native_keys = previous_inplace
        return CheckpointLoadAudit(tuple(sorted(loaded_keys)), source_bytes, loaded_bytes, max_source, max_output)

    # ------------------------------------------------------------------
    # from_hf
    # ------------------------------------------------------------------

    def _keep_hf_key(self, key: str) -> bool:
        """Select released tensors for the configured backbone and optional towers."""
        if key.startswith(("mtp.", "model.mtp.")) or key == "image_pad":
            return False
        if self.config.vision_config.num_hidden_layers == 0 and (
            key.startswith(_VISION_PREFIXES) or key in _VISION_DELIMITERS
        ):
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

        Steps: discard unconstructed layers, non-local experts and disabled
        towers before dequantization, restore Engram owner padding, rename,
        and merge experts not already loaded through views into model storage.

        Args:
            hf_state_dict: Consumed mapping of released-name tensors. Per-expert
                projections have shape [output, input], with FP4 input columns
                packed two per byte before dequantization. Engram tables have
                logical shape [rows, channels]
                and optionally placement Shard(0) on a one-dimensional owner
                mesh, with uneven local shape [local_rows, channels]. Other
                tensors retain the layouts documented in this module.
            device_mesh: Optional expert mesh selecting local expert IDs and
                retaining any inner-axis expert FSDP sharding.
            **kwargs: Additional checkpoint protocol arguments.

        Returns:
            Internal-name tensors. Engram DTensors have global shape
            [ceil(rows / owners) * owners, channels] with equal local row counts
            and zero padding. Grouped experts have global shapes
            [experts, hidden, 2 * intermediate] and [experts, intermediate, hidden],
            with rank-local shards following the expert mesh. Experts already
            loaded through model-storage views are omitted and recorded in
            view_loaded_native_keys. Other tensors retain their layouts and
            can alias input storage when no conversion is needed.

        Raises:
            ValueError: A retained quantized weight has no scale, a scale has
                no weight, or multiple released keys map to one native key.
            RuntimeError: A retained expert layer lacks a required local projection.
        """
        for key in list(hf_state_dict):
            layer = re.match(r"layers\.(\d+)\.", key)
            expert = re.match(r"layers\.\d+\.ffn\.experts\.(\d+)\.", key)
            if (
                not self._keep_hf_key(key)
                or (layer and int(layer[1]) >= self.config.num_hidden_layers)
                or (
                    expert
                    and not should_load_expert_for_rank(int(expert[1]), device_mesh, self.config.n_routed_experts)
                )
            ):
                hf_state_dict.pop(key)
        self._dequantize(hf_state_dict)
        converted = {}
        for key in list(hf_state_dict):
            value = hf_state_dict.pop(key)
            if key.endswith(".scale"):
                raise ValueError(f"Checkpoint scale {key} has no matching weight")
            target = _rename_hf_key(key)
            if target in converted:
                raise ValueError(f"Multiple checkpoint tensors map to {target}")
            match = _ENGRAM_EMBED_PATTERN.match(key)
            converted[target] = self._restore_engram_padding(value, int(match.group(1))) if match else value
        return self._from_hf_w_merged_experts(converted, device_mesh)

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
        """Dequantize paired weights and require scales for retained packed tensors.

        Args:
            state_dict: Mutated released-name mapping. Dense FP8 matrices have
                shape [rows, columns] with scales [ceil(rows / 32), ceil(columns / 32)].
                FP4 experts have shape [rows, columns / 2] with scales [rows, columns / 32].
                FP8 Engram tables have shape [rows, channels] with scales [rows, channels / 32].
                DTensors retain their global shape and mesh placements, including
                uneven row owners and inner-axis expert shards. Other tensors
                retain their registered shapes.

        Returns:
            The same mapping with consumed scale entries and dequantized weights
            in self.dtype. Decoded matrices restore the unpacked input dimension
            and preserve their global layout and rank-local ownership. Unchanged
            tensors alias input storage; decoded tensors have independent storage.

        Raises:
            ValueError: A packed INT8 or FP8 E4M3 weight has no companion scale.
        """
        for key in list(state_dict.keys()):
            if not key.endswith(".weight"):
                continue
            weight = state_dict[key]
            scale_key = key[: -len(".weight")] + ".scale"
            if scale_key not in state_dict:
                if weight.dtype in (torch.float8_e4m3fn, torch.int8):
                    raise ValueError(f"Quantized weight {key} is missing its scale tensor {scale_key}")
                continue
            scale = state_dict.pop(scale_key)
            if (
                self._is_expert_weight_key(key)
                and self._expert_quant_layout_from_tensors(weight, scale) is _ExpertQuantLayout.FP4
            ):
                state_dict[key] = dequantize_checkpoint_weight(weight, scale, dtype=self.dtype)
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
            **kwargs: Checkpoint protocol options, including quantization,
                exclude_key_regex, for_checkpoint_load and the explicit
                preserve_dtensor_load_views direct-loader contract.

        Returns:
            Released-name tensor pairs in the module's documented layouts.
            Engram weights expose only [rows, channels] and scales expose
            [rows, channels / 32], retaining uneven row ownership. Floating
            views alias input storage; quantized placeholders are independent.
        """
        quantization = kwargs.get("quantization", False)
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        result = self._split_merged_expert(
            fqn,
            tensor,
            for_checkpoint_load=kwargs.get("for_checkpoint_load", False),
            preserve_dtensor_load_views=kwargs.get("preserve_dtensor_load_views", False),
            quantization=quantization,
        )
        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]
        result = [(_internal_key_to_hf(k), v) for k, v in result if self._keep_hf_key(_internal_key_to_hf(k))]
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

    def _split_merged_expert(
        self,
        fqn: str,
        tensor: torch.Tensor,
        *,
        for_checkpoint_load: bool = False,
        preserve_dtensor_load_views: bool = False,
        quantization: bool = False,
    ) -> list[tuple[str, torch.Tensor]]:
        """Split released projections, optionally preserving live load views.

        Args:
            fqn: Native fully qualified name.
            tensor: Grouped gate/up [experts, hidden, 2 * intermediate] or
                down [experts, intermediate, hidden] tensor; other registered
                parameter/buffer layouts pass through unchanged. DTensors may
                shard the expert axis and an inner matrix axis on separate
                mesh dimensions. Partial placements are unsupported.
            for_checkpoint_load: The returned tensors will be overwritten by
                checkpoint initialization outside an active autograd graph.
            preserve_dtensor_load_views: Use the shared expert splitter to
                retain non-contiguous local storage aliases for direct loading.
                Ordinary DCP retains contiguous conversion for inner-axis
                DTensor shards and rebuilds those experts after reading.
            quantization: Split matrices will be replaced by independent
                quantized load targets, so they must not be recorded as views
                that load directly into model storage.

        Returns:
            Released expert names and [output, input] projection tensors, or
            the unchanged native name/tensor for non-experts. Unquantized load
            views alias model storage for ordinary local expert matrices and,
            when explicitly enabled, inner-axis DTensor shards. Global shape,
            mesh, placements and dtype are preserved. Export retains the V4
            projection conversion.
        """
        if preserve_dtensor_load_views and not for_checkpoint_load:
            raise ValueError("Preserving DTensor load views requires for_checkpoint_load=True")
        if not for_checkpoint_load:
            return super()._split_merged_expert(fqn, tensor)
        result = self._convert_single_merged_expert_to_hf_split_experts(
            fqn,
            tensor,
            for_checkpoint_load=for_checkpoint_load,
            preserve_dtensor_load_views=preserve_dtensor_load_views,
            quantization=quantization,
        )
        return [(fqn, tensor)] if result is None else [(_internal_key_to_hf(key), value) for key, value in result]

    def forced_hf_dtype_mapping(self, state_dict: dict[str, Any]) -> dict[str, str]:
        """Preserve full-precision parameters when checkpoint export casts weights.

        Args:
            state_dict: Native parameter/buffer tensors with arbitrary registered
                shapes and layouts. Values are inspected only for their dtype.

        Returns:
            Released checkpoint keys that must remain float32, including mHC,
            router parameters, attention sinks and the full-precision head.
        """
        return {
            _internal_key_to_hf(key): "float32"
            for key, value in state_dict.items()
            if isinstance(value, torch.Tensor)
            and value.dtype == torch.float32
            and self._keep_hf_key(_internal_key_to_hf(key))
        }

    @classmethod
    def _fp8_cast(cls, value: Any) -> Any:
        """Create FP8 storage while retaining the global layout of uneven shards."""
        if is_dtensor(value):
            local = cls._empty_or_cast_fp8(value.to_local())
            return DTensor.from_local(
                local, value.device_mesh, value.placements, shape=value.shape, stride=value.stride()
            )
        return cls._empty_or_cast_fp8(value)

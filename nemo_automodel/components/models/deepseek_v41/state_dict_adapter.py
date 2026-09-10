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
  ``ffn.shared_experts.w{1,2,3}``, ``engram.wkv``). Initialization uses this
  fixed 32-column layout rather than DeepSeek V4's 128x128 blocks.
* FP4 E2M1 routed experts packed two per ``int8`` with per-row / 32-column
  ``e8m0`` scales.
* Engram tables: FP8 E4M3 ``[rows, 256]`` with per-row / 32-column ``e8m0``
  scales ``[rows, 8]``.
* BF16 / FP32 for everything else (norms, gate, hyper-connection mixers,
  compressor, indexer keys, attention sink, embeddings, head).

Key mapping (HF -> internal):
  embed.weight                           -> model.embed_tokens.weight
  norm.weight                            -> model.norm.weight
  head.weight                            -> lm_head.weight
  layers.{i}.attn_norm.weight            -> model.layers.{i}.attn_norm.weight
  layers.{i}.ffn_norm.weight             -> model.layers.{i}.ffn_norm.weight
  layers.{i}.attn.attn_sink              -> model.layers.{i}.attn.sinks_param.weight
  layers.{i}.attn.*                      -> model.layers.{i}.attn.*   (compressor.*, indexer.* keep their names)
  layers.{i}.ffn.gate.bias               -> model.layers.{i}.ffn.gate.e_score_correction_bias
  layers.{i}.ffn.gate.weight             -> model.layers.{i}.ffn.gate.weight
  layers.{i}.ffn.shared_experts.w1/w3/w2 -> model.layers.{i}.ffn.shared_experts.gate_proj/up_proj/down_proj
  layers.{i}.ffn.experts.{j}.w1/w3/w2    -> stacked into model.layers.{i}.ffn.experts.gate_and_up_projs / down_projs
  layers.{i}.hc_attn_{fn,base,scale}     -> model.layers.{i}.attn_hc.{fn,base,scale}
  layers.{i}.hc_ffn_{fn,base,scale}      -> model.layers.{i}.ffn_hc.{fn,base,scale}
  layers.{i}.engram.*                    -> model.layers.{i}.engram.*
  layers.{i}.ffn.gate.bias_vl             -> model.layers.{i}.ffn.gate.bias_vl
  vision.* / aligner.*                   -> model.vision.* / model.aligner.*
  image_{start,end,newline}              -> model.image_{start,end,newline}

The ``mtp.*`` DSpark draft is excluded. Eager loading also excludes layers
beyond the configured backbone and experts assigned to other ranks before
dequantization. Other checkpoint keys are not filtered by optional-tower
configuration.
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

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin
from nemo_automodel.components.moe.state_dict_utils import is_dtensor, should_load_expert_for_rank

_HF_TO_INTERNAL_RENAMES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^vision\.(.+)$"), r"model.vision.\1"),
    (re.compile(r"^aligner\.(.+)$"), r"model.aligner.\1"),
    (re.compile(r"^(image_start|image_end|image_newline)$"), r"model.\1"),
    (re.compile(r"^embed\.(.+)$"), r"model.embed_tokens.\1"),
    (re.compile(r"^norm\.(.+)$"), r"model.norm.\1"),
    (re.compile(r"^head\.(.+)$"), r"lm_head.\1"),
    (re.compile(r"^layers\.(\d+)\.attn_norm\.(.+)$"), r"model.layers.\1.attn_norm.\2"),
    (re.compile(r"^layers\.(\d+)\.ffn_norm\.(.+)$"), r"model.layers.\1.ffn_norm.\2"),
    (re.compile(r"^layers\.(\d+)\.attn\.attn_sink$"), r"model.layers.\1.attn.sinks_param.weight"),
    (re.compile(r"^layers\.(\d+)\.attn\.(.+)$"), r"model.layers.\1.attn.\2"),
    (re.compile(r"^layers\.(\d+)\.ffn\.gate\.bias$"), r"model.layers.\1.ffn.gate.e_score_correction_bias"),
    (re.compile(r"^layers\.(\d+)\.ffn\.gate\.(.+)$"), r"model.layers.\1.ffn.gate.\2"),
    (re.compile(r"^layers\.(\d+)\.ffn\.shared_experts\.w1\.(.+)$"), r"model.layers.\1.ffn.shared_experts.gate_proj.\2"),
    (re.compile(r"^layers\.(\d+)\.ffn\.shared_experts\.w3\.(.+)$"), r"model.layers.\1.ffn.shared_experts.up_proj.\2"),
    (re.compile(r"^layers\.(\d+)\.ffn\.shared_experts\.w2\.(.+)$"), r"model.layers.\1.ffn.shared_experts.down_proj.\2"),
    (re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w1\.(.+)$"), r"model.layers.\1.ffn.experts.\2.gate_proj.\3"),
    (re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w3\.(.+)$"), r"model.layers.\1.ffn.experts.\2.up_proj.\3"),
    (re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w2\.(.+)$"), r"model.layers.\1.ffn.experts.\2.down_proj.\3"),
    (re.compile(r"^layers\.(\d+)\.hc_attn_(base|fn|scale)$"), r"model.layers.\1.attn_hc.\2"),
    (re.compile(r"^layers\.(\d+)\.hc_ffn_(base|fn|scale)$"), r"model.layers.\1.ffn_hc.\2"),
    (re.compile(r"^layers\.(\d+)\.engram\.(.+)$"), r"model.layers.\1.engram.\2"),
]

_INTERNAL_TO_HF_RENAMES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^model\.vision\.(.+)$"), r"vision.\1"),
    (re.compile(r"^model\.aligner\.(.+)$"), r"aligner.\1"),
    (re.compile(r"^model\.(image_start|image_end|image_newline|image_pad)$"), r"\1"),
    (re.compile(r"^model\.layers\.(\d+)\.ffn\.experts\.(\d+)\.gate_proj\.(.+)$"), r"layers.\1.ffn.experts.\2.w1.\3"),
    (re.compile(r"^model\.layers\.(\d+)\.ffn\.experts\.(\d+)\.up_proj\.(.+)$"), r"layers.\1.ffn.experts.\2.w3.\3"),
    (re.compile(r"^model\.layers\.(\d+)\.ffn\.experts\.(\d+)\.down_proj\.(.+)$"), r"layers.\1.ffn.experts.\2.w2.\3"),
    (re.compile(r"^model\.embed_tokens\.(.+)$"), r"embed.\1"),
    (re.compile(r"^model\.norm\.(.+)$"), r"norm.\1"),
    (re.compile(r"^lm_head\.(.+)$"), r"head.\1"),
    (re.compile(r"^model\.layers\.(\d+)\.attn_norm\.(.+)$"), r"layers.\1.attn_norm.\2"),
    (re.compile(r"^model\.layers\.(\d+)\.ffn_norm\.(.+)$"), r"layers.\1.ffn_norm.\2"),
    (re.compile(r"^model\.layers\.(\d+)\.attn\.sinks_param\.weight$"), r"layers.\1.attn.attn_sink"),
    (re.compile(r"^model\.layers\.(\d+)\.attn\.(.+)$"), r"layers.\1.attn.\2"),
    (re.compile(r"^model\.layers\.(\d+)\.ffn\.gate\.e_score_correction_bias$"), r"layers.\1.ffn.gate.bias"),
    (re.compile(r"^model\.layers\.(\d+)\.ffn\.gate\.(.+)$"), r"layers.\1.ffn.gate.\2"),
    (
        re.compile(r"^model\.layers\.(\d+)\.ffn\.shared_experts\.gate_proj\.(.+)$"),
        r"layers.\1.ffn.shared_experts.w1.\2",
    ),
    (re.compile(r"^model\.layers\.(\d+)\.ffn\.shared_experts\.up_proj\.(.+)$"), r"layers.\1.ffn.shared_experts.w3.\2"),
    (
        re.compile(r"^model\.layers\.(\d+)\.ffn\.shared_experts\.down_proj\.(.+)$"),
        r"layers.\1.ffn.shared_experts.w2.\2",
    ),
    (re.compile(r"^model\.layers\.(\d+)\.attn_hc\.(fn|base|scale)$"), r"layers.\1.hc_attn_\2"),
    (re.compile(r"^model\.layers\.(\d+)\.ffn_hc\.(fn|base|scale)$"), r"layers.\1.hc_ffn_\2"),
    (re.compile(r"^model\.layers\.(\d+)\.engram\.(.+)$"), r"layers.\1.engram.\2"),
]

_ENGRAM_EMBED_PATTERN = re.compile(r"^layers\.(\d+)\.engram\.embed\.weight$")


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


class DeepseekV41StateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Convert released V4.1 layouts and stream directly into prepared model storage.

    Floating DCP initialization uses shared MoE views and skips rebuilding
    experts already written into model storage. Quantized load targets use the
    released V4.1 layout; floating export uses the shared expert splitter.
    Explicit streaming initialization copies one bounded quantized chunk at a
    time, without gathering experts or Engram owner rows.
    """

    # Quantized DCP loads still allocate converted tensors. The explicit
    # streaming API below owns its bounded direct copies independently.
    _supports_low_memory_dcp_load = False

    def __init__(
        self,
        config: DeepseekV41Config,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True
        self._engram_rows = dict(zip(config.text_config.engram_layer_ids, config.text_config.engram_num_embeddings))

    @property
    def _expert_path_segment(self) -> str:
        return "ffn.experts"

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
            if fqn.startswith("mtp.") or "_extra_state" in fqn:
                continue
            expert = re.fullmatch(r"model\.layers\.(\d+)\.ffn\.experts\.(gate_and_up_projs|down_projs)", fqn)
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

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert the released HF checkpoint to the internal format.

        Steps: discard DSpark draft tensors, unconstructed layers and non-local
        experts before dequantization, restore Engram owner padding, rename,
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
                key.startswith("mtp.")
                or (layer and int(layer[1]) >= self.config.text_config.num_hidden_layers)
                or (
                    expert
                    and not should_load_expert_for_rank(
                        int(expert[1]), device_mesh, self.config.text_config.n_routed_experts
                    )
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
            state_dict[key] = dequantize_checkpoint_weight(
                weight, scale, dtype=self.dtype, rowwise=_ENGRAM_EMBED_PATTERN.match(key) is not None
            )
        return state_dict

    # ------------------------------------------------------------------
    # to_hf
    # ------------------------------------------------------------------

    @staticmethod
    def _quantized_load_targets(key: str, value: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Allocate rank-local destinations matching the original quantized dump.

        Args:
            key: Released checkpoint matrix name.
            value: Dequantized matrix [rows, columns], possibly a DTensor. Row
                scales retain row sharding; FP4 column shards must start and
                end on 32-column block boundaries.

        Returns:
            Packed INT8 [rows, columns / 2] or FP8 [rows, columns] weight and
            E8M0 scales. Dense FP8 scales cover the small global 32x32 grid;
            expert/Engram scales [rows, columns / 32] are owner-local DTensors.
            Buffers are uninitialized and must only be passed to DCP loading.
        """
        expert = re.fullmatch(r"layers\.\d+\.ffn\.experts\.\d+\.w[123]\.weight", key) is not None
        rowwise = expert or ".engram.embed." in key
        local = value.to_local() if isinstance(value, DTensor) else value
        if local.ndim != 2:
            raise ValueError(f"Quantized checkpoint matrix {key} must be two-dimensional")
        offsets = _local_offsets(value) if isinstance(value, DTensor) else (0, 0)
        if rowwise and (value.shape[1] % 32 or local.shape[1] % 32 or offsets[1] % 32):
            raise ValueError(f"Rowwise checkpoint matrix {key} requires 32-column-aligned shards")
        divisor = 2 if expert else 1
        local_weight = torch.empty(
            (local.shape[0], local.shape[1] // divisor),
            dtype=torch.int8 if expert else torch.float8_e4m3fn,
            device=local.device,
        )
        shape = (value.shape[0], value.shape[1] // divisor)
        if isinstance(value, DTensor):
            weight = DTensor.from_local(
                local_weight,
                value.device_mesh,
                value.placements,
                shape=torch.Size(shape),
                stride=(shape[1], 1),
            )
        else:
            weight = local_weight
        if rowwise:
            local_scale = torch.empty(
                (local.shape[0], local.shape[1] // 32), dtype=torch.float8_e8m0fnu, device=local.device
            )
            if isinstance(value, DTensor):
                shape = (value.shape[0], value.shape[1] // 32)
                scale = DTensor.from_local(
                    local_scale,
                    value.device_mesh,
                    value.placements,
                    shape=torch.Size(shape),
                    stride=(shape[1], 1),
                )
            else:
                scale = local_scale
        else:
            scale = torch.empty(
                ((value.shape[0] + 31) // 32, (value.shape[1] + 31) // 32),
                dtype=torch.float8_e8m0fnu,
                device=local.device,
            )
        return [(key, weight), (key.removesuffix(".weight") + ".scale", scale)]

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        quantization: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Export native tensors under the released checkpoint's key names.

        Args:
            state_dict: Native tensor mapping, including grouped expert tensors
                [experts, hidden, 2 * intermediate] and [experts, intermediate,
                hidden]. Other values retain their registered shapes. DTensors
                may shard the expert axis and an inner matrix axis; Engram
                tables have global shape [padded_rows, channels] with Shard(0)
                on a one-dimensional owner mesh.
            exclude_key_regex: Optional regular expression for excluded HF keys.
            quantization: Whether initialization needs packed checkpoint targets.
            **kwargs: Compatibility options from checkpointing, including
                for_checkpoint_load for destinations overwritten by DCP and
                preserve_dtensor_load_views for direct streaming initialization.

        Returns:
            Released mapping with split expert matrices [output, input]. DTensor
            expert placement conversion follows the shared MoE adapter contract.
            Engram weights expose logical rows only. Floating load views may
            alias model storage; quantized targets are independent, uninitialized
            tensors and require for_checkpoint_load=True.
        """
        output = {}
        for key, value in state_dict.items():
            output.update(
                self.convert_single_tensor_to_hf(
                    key, value, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
                )
            )
        return output

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs: Any) -> list[tuple[str, Any]]:
        """Convert one internal tensor to HF keys, optionally emitting on-disk quantized placeholders.

        With ``quantization=True`` the placeholders mirror the released layout so
        DCP can validate shapes / dtypes before the adapter dequantizes on load.
        These uninitialized targets require ``for_checkpoint_load=True``;
        trained weights must be exported without quantization.

        Args:
            fqn: Internal parameter name.
            tensor: Parameter in its model layout. Grouped experts have shape
                [experts, hidden, 2 * intermediate] or [experts, intermediate,
                hidden], with optional DTensor sharding of the expert and inner
                matrix axes. Engram tables have global shape [padded_rows,
                channels], optionally placement Shard(0) on a one-dimensional
                owner mesh with equal local row counts. Other tensors retain
                their arbitrary registered shapes.
            **kwargs: Checkpoint protocol options, including quantization,
                exclude_key_regex, for_checkpoint_load and the explicit
                preserve_dtensor_load_views direct-loader contract.

        Returns:
            Released-name tensor pairs, with split experts in [output, input]
            layout and DTensor placements following the shared MoE contract.
            Engram weights expose only [rows, channels] and scales expose
            [rows, channels / 32], retaining uneven row ownership. Floating load
            views may alias input storage; quantized placeholders are independent.
        """
        quantization = kwargs.get("quantization", False)
        if quantization and not kwargs.get("for_checkpoint_load", False):
            raise ValueError(
                "Quantization targets are for checkpoint loading only; export trained weights without quantization"
            )
        if fqn.startswith("mtp."):
            return []
        expert = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        result = [(fqn, tensor)] if expert is None else expert
        exclude = kwargs.get("exclude_key_regex")
        converted = []
        for key, value in result:
            key = _internal_key_to_hf(key)
            if exclude and re.match(exclude, key):
                continue
            match = _ENGRAM_EMBED_PATTERN.match(key)
            if match:
                value = self._engram_checkpoint_tensor(value, int(match.group(1)))
            quantized = re.fullmatch(
                r"layers\.\d+\.(?:attn\.(?:wq_a|wq_b|wkv|wo_a|wo_b|indexer\.wq_b)"
                r"|ffn\.(?:shared_experts|experts\.\d+)\.w[123]|engram\.(?:embed|wkv))\.weight",
                key,
            )
            if quantization and quantized:
                converted.extend(self._quantized_load_targets(key, value))
            else:
                converted.append((key, value))
        return converted

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
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32
        }

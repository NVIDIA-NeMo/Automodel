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

The backbone adapter excludes the ``mtp.*`` DSpark draft. The draft adapter
loads that namespace separately so DSpark training can start from the released
weights without attaching the draft objective to the frozen backbone.
"""

from __future__ import annotations

import logging
import math
import re
from functools import partial
from typing import Any, Iterator, Mapping

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Shard

from nemo_automodel.components.checkpoint.state_dict_adapter import CheckpointLoadPart, StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config, DeepseekV41TextConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin
from nemo_automodel.components.moe.state_dict_utils import (
    get_expert_range_for_rank_from_mesh,
    is_dtensor,
    should_load_expert_for_rank,
)

_ENGRAM_EMBED_PATTERN = re.compile(r"^layers\.(\d+)\.engram\.embed\.weight$")
_DSPARK_EXPERT_PATTERN = re.compile(r"^mtp\.(\d+)\.ffn\.experts\.(gate_and_up_projs|down_projs)$")


def _native_key(key: str) -> str:
    if key.startswith("embed."):
        return "model.embed_tokens." + key.removeprefix("embed.")
    if key.startswith("head."):
        return "lm_head." + key.removeprefix("head.")
    if key.endswith(".attn.attn_sink"):
        key = key.removesuffix(".attn_sink") + ".sinks_param.weight"
    match = re.fullmatch(r"layers\.(\d+)\.hc_(attn|ffn)_(fn|base|scale)", key)
    if match:
        return f"model.layers.{match[1]}.{match[2]}_hc.{match[3]}"
    key = re.sub(r"(\.ffn\.(?:experts\.\d+|shared_experts))\.w1\.", r"\1.gate_proj.", key)
    key = re.sub(r"(\.ffn\.(?:experts\.\d+|shared_experts))\.w3\.", r"\1.up_proj.", key)
    key = re.sub(r"(\.ffn\.(?:experts\.\d+|shared_experts))\.w2\.", r"\1.down_proj.", key)
    if key.endswith(".ffn.gate.bias"):
        key = key.removesuffix(".bias") + ".e_score_correction_bias"
    if key.startswith(("layers.", "norm.", "vision.", "aligner.")) or key in (
        "image_start",
        "image_end",
        "image_newline",
    ):
        return "model." + key
    return key


def _released_key(key: str) -> str:
    if key.startswith("model.embed_tokens."):
        return "embed." + key.removeprefix("model.embed_tokens.")
    if key.startswith("lm_head."):
        return "head." + key.removeprefix("lm_head.")
    key = key.removeprefix("model.")
    if key.endswith(".attn.sinks_param.weight"):
        key = key.removesuffix(".sinks_param.weight") + ".attn_sink"
    match = re.fullmatch(r"layers\.(\d+)\.(attn|ffn)_hc\.(fn|base|scale)", key)
    if match:
        return f"layers.{match[1]}.hc_{match[2]}_{match[3]}"
    key = re.sub(r"(\.ffn\.(?:experts\.\d+|shared_experts))\.gate_proj\.", r"\1.w1.", key)
    key = re.sub(r"(\.ffn\.(?:experts\.\d+|shared_experts))\.up_proj\.", r"\1.w3.", key)
    key = re.sub(r"(\.ffn\.(?:experts\.\d+|shared_experts))\.down_proj\.", r"\1.w2.", key)
    if key.endswith(".ffn.gate.e_score_correction_bias"):
        key = key.removesuffix(".e_score_correction_bias") + ".bias"
    return key


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


_FUSED_EXPERT_KEY = re.compile(r"^model\.layers\.(\d+)\.ffn\.experts\.(gate_and_up_projs|down_projs)$")


def _split_weight_scale(required: frozenset[str]) -> tuple[str, str]:
    """Split a quantized checkpoint key pair into (weight key, scale key)."""
    scale_key = next(key for key in required if key.endswith(".scale"))
    weight_key = next(key for key in required if key.endswith(".weight"))
    return weight_key, scale_key


def _finish_part_loads(
    adapter_dtype: torch.dtype,
    generic: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool], ...],
    engrams: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int], ...],
    experts: tuple[tuple[torch.Tensor, int, str, torch.Tensor, torch.Tensor, int], ...],
) -> None:
    """Install one load part's FP8 tensors into final model storage.

    Args:
        adapter_dtype: Decoded storage dtype (``self.dtype`` of the owning adapter).
        generic: ``(target, fp8_temp, scale_temp, rowwise)`` per quantized tensor. Targets are
            final model tensors (plain or DTensor) of shape [rows, columns]; the decoded
            tensor of the same global shape is copied in.
        engrams: ``(target, fp8_temp, scale_temp, restore_fn)`` per Engram table. The temps
            are true-row-count owner-sharded views; the decoded tensor restores equal owner
            padding via ``restore_fn`` (the adapter's ``_restore_engram_padding`` partial)
            before copying into the padded target storage.
        experts: ``(fused_local, expert_index, projection, fp8_temp, scale_temp, inter_dim,
            row_offset, row_count)`` per per-expert projection. ``fused_local`` is the
            rank-local grouped storage; the decoded full [rows, columns] checkpoint tensor is
            transposed exactly as ``_from_hf_w_merged_experts`` does, then this rank's
            inner-axis shard rows ``[row_offset : row_offset + row_count]`` are written into
            the expert's slice (unsharded targets pass the full range).
    """
    for target, fp8, scale, rowwise in generic:
        target.copy_(dequantize_checkpoint_weight(fp8, scale, dtype=adapter_dtype, rowwise=rowwise))
    for target, fp8, scale, restore_fn in engrams:
        decoded = dequantize_checkpoint_weight(fp8, scale, dtype=adapter_dtype, rowwise=True)
        target.copy_(restore_fn(decoded))
    for (
        fused_local,
        expert_index,
        projection,
        fp8,
        scale,
        inter_dim,
        row_offset,
        row_count,
    ) in experts:
        decoded = dequantize_checkpoint_weight(fp8, scale, dtype=adapter_dtype)
        shard = decoded.transpose(0, 1)[row_offset : row_offset + row_count]
        if projection == "w1":
            fused_local[expert_index, :, :inter_dim] = shard
        elif projection == "w3":
            fused_local[expert_index, :, inter_dim:] = shard
        else:
            fused_local[expert_index] = shard


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
        Independent tensor of shape [rows, columns] in ``dtype`` on the operands'
        device; DTensor inputs retain their mesh and placements. No input is
        modified, and FP32/expanded-scale temporaries cover at most 4M weight
        elements.
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
    """Convert released V4.1 checkpoint layouts for DCP loading and export.

    Floating DCP initialization uses shared MoE views and skips rebuilding
    experts already written into model storage. Quantized load targets use the
    released V4.1 layout; floating export uses the shared expert splitter.
    """

    # Quantized DCP loads still allocate converted tensors.
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
                keys.append(_released_key(fqn))
        return keys

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
            target = _native_key(key)
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
            state_dict[key] = self._dequantize_one(weight, scale, key)
        return state_dict

    def _dequantize_one(self, weight: torch.Tensor, scale: torch.Tensor, key: str) -> torch.Tensor:
        """Dequantize one weight/scale pair.

        Args:
            weight: Quantized weight; plain tensor or DTensor of FP8/INT8 payload.
            scale: Companion block-wise scale tensor on the same device.
            key: State-dict key, used to select the rowwise Engram scale layout.

        Returns:
            Decoded tensor in ``self.dtype`` on the weight's device.
        """
        return dequantize_checkpoint_weight(weight, scale, dtype=self.dtype, rowwise=".engram.embed." in key)

    def iter_checkpoint_load_parts(
        self,
        model_state_dict: dict[str, torch.Tensor],
        device_mesh: "DeviceMesh" | None = None,
        checkpoint_metadata: "Mapping[str, Any] | None" = None,
    ) -> Iterator[CheckpointLoadPart] | None:
        """Load the released V4.1 FP8 checkpoint in bounded per-layer parts.

        Every quantized checkpoint tensor gets a temporary FP8 destination (plus its
        block-scale destination) while non-quantized tensors use their final model
        storage directly. ``finish`` decodes one part and writes it straight into the
        parameter shells: per-expert ``w1``/``w3``/``w2`` tensors are decoded, transposed,
        and written into their slices of the grouped expert storage, reproducing
        ``_from_hf_w_merged_experts``'s layout. Only one layer's temporaries co-reside
        with the parameter shells, so peak load memory is the model plus one layer
        instead of the model plus a full decoded copy (the small-world OOM).

        Engram tables keep their checkpoint-side shapes (the model pads rows up to a
        multiple of the owner count) and install through an offset-aware local copy.

        Requires the checkpoint's DCP metadata as ground truth; returns ``None``
        (falling back to the full-state path) for DSpark-shaped state dicts, non-tensor
        entries, dtype mismatches, grouped expert state-dict tensors that do not alias
        model storage (TE / MoK experts), or any model tensor the checkpoint cannot
        serve.

        Args:
            model_state_dict: Native names mapped to final parameter and persistent-buffer
                tensors, with their model-specific shapes and distributed placements.
            device_mesh: Optional mesh whose expert-parallel range selects this rank's
                experts.
            checkpoint_metadata: Checkpoint names mapped to DCP storage metadata.

        Returns:
            One load part per decoder layer plus one for shared tensors, or ``None``.

        Raises:
            ValueError: The checkpoint holds an expert layer's projections only partially,
                or a quantized weight without its scale companion. Corrupt checkpoints
                reject here instead of loading silently wrong values; the full-state
                path raises equivalent errors for the same checkpoints.
        """
        if checkpoint_metadata is None:
            logging.info("%s: part loading declined (no checkpoint metadata)", type(self).__name__)
            return None
        if any(key.startswith("mtp.") for key in model_state_dict):
            logging.info("%s: part loading declined (DSpark-shaped state dict)", type(self).__name__)
            return None
        non_tensor_keys = [key for key, value in model_state_dict.items() if not isinstance(value, torch.Tensor)]
        if non_tensor_keys:
            logging.info(
                "%s: part loading declined (%d non-tensor entries, examples=%s)",
                type(self).__name__,
                len(non_tensor_keys),
                non_tensor_keys[:5],
            )
            return None
        n_experts = self.moe_config.n_routed_experts if self.moe_config is not None else 0
        expert_range: tuple[int, int]
        if n_experts and device_mesh is not None:
            expert_range = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
        elif n_experts:
            expert_range = (0, n_experts)
        else:
            expert_range = (0, 0)
        absent_keys: list[str] = []
        for model_key, target in model_state_dict.items():
            classified = self._classify_model_key(model_key, expert_range, n_experts, checkpoint_metadata)
            if classified is None:
                # Keys absent from the checkpoint (LoRA parameters injected before the base
                # load, a tied lm_head the checkpoint omits, ...) keep their initialization,
                # matching the full-state path's missing-key semantics.
                absent_keys.append(model_key)
                continue
            kind, required = classified
            if kind == "fused" and not self._expert_checkpoint_tensors_use_model_storage:
                # GroupedExpertsTE and the MoK gate/up storage expose grouped experts through
                # stacked copies of the real parameters, so part destinations would write into
                # the copies while the model keeps its initialization. The full-state path
                # rebuilds and installs those tensors after the read; serve it there.
                logging.info(
                    "%s: part loading declined (grouped expert state-dict tensors do not alias model storage)",
                    type(self).__name__,
                )
                return None
            if kind == "quantized":
                if target.ndim != 2 or target.dtype != self.dtype:
                    logging.info(
                        "%s: part loading declined (%s has %dD %s targets, expected 2D %s)",
                        type(self).__name__,
                        model_key,
                        target.ndim,
                        target.dtype,
                        self.dtype,
                    )
                    return None
                weight_key, _ = _split_weight_scale(required)
                size = getattr(checkpoint_metadata[weight_key], "size", None)
                if not isinstance(target, DTensor) and tuple(size or ()) != tuple(target.shape):
                    # Plain rank-local destinations (FSDP2 state dicts) need the full-state
                    # path's mesh-planned reads; only DTensor or exact-shape targets are served.
                    logging.info(
                        "%s: part loading declined (%s is a plain rank-local destination)",
                        type(self).__name__,
                        model_key,
                    )
                    return None
            if (
                kind == "fused"
                and isinstance(target, DTensor)
                and not all(
                    placement.is_replicate() or (isinstance(placement, Shard) and placement.dim in (0, 1))
                    for placement in target.placements
                )
            ):
                # Grouped experts shard on dim 0 (EP) and dim 1 (the ep_shard FSDP axis,
                # see _moe_shard_placement). Anything else needs the full-state path.
                logging.info(
                    "%s: part loading declined (%s shards outside the expert/inner axes)",
                    type(self).__name__,
                    model_key,
                )
                return None
        if absent_keys:
            logging.info(
                "%s: %d model keys absent from the checkpoint keep their initialization (examples=%s)",
                type(self).__name__,
                len(absent_keys),
                absent_keys[:5],
            )
        return self._iter_checkpoint_load_parts(model_state_dict, expert_range, n_experts, checkpoint_metadata)

    def _classify_model_key(
        self,
        model_key: str,
        expert_range: tuple[int, int],
        n_experts: int,
        checkpoint_metadata: "Mapping[str, Any]",
    ) -> tuple[str, frozenset[str]] | None:
        """Classify one model key against the checkpoint and list its required checkpoint keys.

        Args:
            model_key: Native model state-dict name.
            expert_range: Inclusive-exclusive global expert IDs owned by this rank.
            n_experts: Total routed experts per MoE layer, used for a rank-independent
                completeness check over the classified layer.
            checkpoint_metadata: Checkpoint names mapped to DCP storage metadata.

        Returns:
            ``("fused"|"engram"|"quantized"|"direct", required checkpoint keys)``, or
            ``None`` when the checkpoint legitimately lacks this key, whose model tensor
            then keeps its initialization like on the full-state path.

        Raises:
            ValueError: The layer's expert projections are only partially present in the
                checkpoint, or a quantized checkpoint weight lacks its scale companion.
                Both are corrupt checkpoints and must reject exactly like the full-state
                conversion instead of silently keeping initialization or installing raw
                payload values.
        """
        fused = _FUSED_EXPERT_KEY.match(model_key)
        if fused is not None:
            required: set[str] = set()
            layer_num = fused.group(1)
            any_expert_key = False
            complete = True
            for global_expert in range(n_experts):
                for projection in ("w1", "w3", "w2"):
                    base = f"layers.{layer_num}.ffn.experts.{global_expert}.{projection}"
                    weight_key, scale_key = base + ".weight", base + ".scale"
                    has_weight = weight_key in checkpoint_metadata
                    has_scale = scale_key in checkpoint_metadata
                    any_expert_key = any_expert_key or has_weight or has_scale
                    if not (has_weight and has_scale):
                        complete = False
                    if expert_range[0] <= global_expert < expert_range[1]:
                        required.update((weight_key, scale_key))
            if not any_expert_key:
                return None
            if not complete:
                raise ValueError(
                    f"Checkpoint expert projections for layer {layer_num} are incomplete; expected a weight and a "
                    f"scale for w1/w3/w2 of all {n_experts} routed experts"
                )
            return ("fused", frozenset(required))
        released = _released_key(model_key)
        if released not in checkpoint_metadata:
            return None
        scale_key = released[: -len(".weight")] + ".scale" if released.endswith(".weight") else None
        if scale_key is not None and scale_key not in checkpoint_metadata:
            properties = getattr(checkpoint_metadata[released], "properties", None)
            if getattr(properties, "dtype", None) in (torch.float8_e4m3fn, torch.int8):
                raise ValueError(f"Quantized checkpoint weight {released} is missing its scale tensor {scale_key}")
        if scale_key is not None and scale_key in checkpoint_metadata:
            if ".engram.embed." in model_key:
                return ("engram", frozenset((released, scale_key)))
            return ("quantized", frozenset((released, scale_key)))
        return ("direct", frozenset((released,)))

    def _iter_checkpoint_load_parts(
        self,
        model_state_dict: dict[str, torch.Tensor],
        expert_range: tuple[int, int],
        n_experts: int,
        checkpoint_metadata: "Mapping[str, Any]",
    ) -> Iterator[CheckpointLoadPart]:
        """Build per-layer load parts after :meth:`iter_checkpoint_load_parts` validated the inputs.

        Args:
            model_state_dict: Native names mapped to final tensors with their placements.
            expert_range: Inclusive-exclusive global expert IDs owned by this rank.
            n_experts: Total routed experts per MoE layer.
            checkpoint_metadata: Checkpoint names mapped to DCP storage metadata; sizes
                allocate the temporary destinations at checkpoint-side shapes.

        Yields:
            One load part per decoder layer plus one for shared tensors.
        """
        start_expert, end_expert = expert_range

        groups: dict[str, list[str]] = {}
        for model_key in model_state_dict:
            layer_match = re.match(r"^model\.layers\.(\d+)\.", model_key)
            group = f"layer-{layer_match.group(1)}" if layer_match is not None else "shared"
            groups.setdefault(group, []).append(model_key)

        def _checkpoint_size(key: str) -> tuple[int, int]:
            size = getattr(checkpoint_metadata[key], "size", None)
            if size is None or len(size) != 2:
                raise ValueError(f"Checkpoint metadata for {key} lacks a two-dimensional size")
            return int(size[0]), int(size[1])

        def _checkpoint_dtype(key: str) -> torch.dtype:
            properties = getattr(checkpoint_metadata[key], "properties", None)
            dtype = getattr(properties, "dtype", None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"Checkpoint metadata for {key} lacks a tensor dtype")
            return dtype

        def _checkpoint_temp(key: str, device: torch.device) -> torch.Tensor:
            """Allocate a temporary destination at the checkpoint tensor's exact shape and dtype."""
            return torch.empty(_checkpoint_size(key), dtype=_checkpoint_dtype(key), device=device)

        for group_keys in groups.values():
            checkpoint_tensors: dict[str, torch.Tensor] = {}
            temporary_keys: set[str] = set()
            generic: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]] = []
            engrams: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]] = []
            experts: list[tuple[torch.Tensor, int, str, torch.Tensor, torch.Tensor, int, int, int]] = []
            for model_key in group_keys:
                target = model_state_dict[model_key]
                classified = self._classify_model_key(model_key, expert_range, n_experts, checkpoint_metadata)
                if classified is None:
                    # Absent from the checkpoint: no destination, covered keep-at-init.
                    continue
                kind, required = classified
                if kind == "fused":
                    fused_local = target.to_local() if isinstance(target, DTensor) else target
                    # Row offsets of this rank's inner-axis shard (dim 1): gate/up slice the
                    # hidden rows of the transposed checkpoint tensor, down slices the
                    # intermediate rows. Unsharded targets take the full range.
                    offsets = _local_offsets(target) if isinstance(target, DTensor) else (0, 0, 0)
                    layer_match = _FUSED_EXPERT_KEY.match(model_key)
                    layer_num, which = layer_match.group(1), layer_match.group(2)
                    row_offset, row_count = int(offsets[1]), int(fused_local.shape[-2])
                    inter_dim = (
                        int(fused_local.shape[-1]) // 2 if which == "gate_and_up_projs" else int(fused_local.shape[-1])
                    )
                    projections = ("w1", "w3") if which == "gate_and_up_projs" else ("w2",)
                    for global_expert in range(start_expert, end_expert):
                        for projection in projections:
                            base = f"layers.{layer_num}.ffn.experts.{global_expert}.{projection}"
                            weight_key, scale_key = base + ".weight", base + ".scale"
                            # Released experts are FP4-packed INT8 with per-row scales; the
                            # temp mirrors the checkpoint tensor exactly (the full per-expert
                            # tensor - the shard pair reads it twice, a few MB each) instead
                            # of guessing the quantization layout.
                            fp8 = _checkpoint_temp(weight_key, fused_local.device)
                            scale = _checkpoint_temp(scale_key, fused_local.device)
                            checkpoint_tensors[weight_key] = fp8
                            checkpoint_tensors[scale_key] = scale
                            temporary_keys.update((weight_key, scale_key))
                            experts.append(
                                (
                                    fused_local,
                                    global_expert - start_expert,
                                    projection,
                                    fp8,
                                    scale,
                                    inter_dim,
                                    row_offset,
                                    row_count,
                                )
                            )
                elif kind == "engram":
                    weight_key, scale_key = _split_weight_scale(required)
                    rows, _ = _checkpoint_size(weight_key)
                    _, scale_columns = _checkpoint_size(scale_key)
                    layer_id = int(re.search(r"layers\.(\d+)\.engram\.embed\.", model_key).group(1))
                    # The Engram table is a ~100GB monolith: the destination must be the
                    # true-row-count owner-sharded view of the model's padded storage
                    # (_engram_checkpoint_tensor), so each rank reads only its own rows.
                    checkpoint_view = self._engram_checkpoint_tensor(target, layer_id)
                    fp8 = checkpoint_view.to(_checkpoint_dtype(weight_key))
                    if isinstance(fp8, DTensor):
                        local_rows = int(fp8.to_local().shape[0])
                        scale = DTensor.from_local(
                            torch.empty(
                                (local_rows, scale_columns),
                                dtype=_checkpoint_dtype(scale_key),
                                device=target.device,
                            ),
                            fp8.device_mesh,
                            fp8.placements,
                            shape=torch.Size((rows, scale_columns)),
                            stride=(scale_columns, 1),
                        )
                    else:
                        scale = torch.empty(
                            (rows, scale_columns), dtype=_checkpoint_dtype(scale_key), device=target.device
                        )
                    checkpoint_tensors[weight_key] = fp8
                    checkpoint_tensors[scale_key] = scale
                    temporary_keys.update((weight_key, scale_key))
                    engrams.append((target, fp8, scale, partial(self._restore_engram_padding, layer_id=layer_id)))
                elif kind == "quantized":
                    weight_key, scale_key = _split_weight_scale(required)
                    fp8 = torch.empty_like(target, dtype=torch.float8_e4m3fn)
                    scale = _checkpoint_temp(scale_key, target.device)
                    checkpoint_tensors[weight_key] = fp8
                    checkpoint_tensors[scale_key] = scale
                    temporary_keys.update((weight_key, scale_key))
                    generic.append((target, fp8, scale, False))
                else:
                    checkpoint_tensors[next(iter(required))] = target
            yield CheckpointLoadPart(
                checkpoint_tensors=checkpoint_tensors,
                model_keys=frozenset(group_keys),
                temporary_checkpoint_keys=frozenset(temporary_keys),
                finish=partial(
                    _finish_part_loads,
                    self.dtype,
                    tuple(generic),
                    tuple(engrams),
                    tuple(experts),
                ),
            )

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
        expert = re.fullmatch(r"(?:layers|mtp)\.\d+\.ffn\.experts\.\d+\.w[123]\.weight", key) is not None
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
                for_checkpoint_load for destinations overwritten by DCP.

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
                exclude_key_regex and for_checkpoint_load.

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
            key = _released_key(key)
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
            _released_key(key): "float32"
            for key, value in state_dict.items()
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32
        }


def _dspark_native_key(key: str) -> str:
    """Map one released DSpark checkpoint name to the native draft namespace."""
    key = re.sub(r"^mtp\.(\d+)\.", r"layers.\1.", key)
    key = _native_key(key)
    if key.startswith("model.layers."):
        return "mtp." + key.removeprefix("model.layers.")
    return key.removeprefix("model.")


def _dspark_released_key(key: str) -> str:
    """Map one native DSpark parameter name to the released checkpoint namespace."""
    if key.startswith("mtp."):
        key = "model.layers." + key.removeprefix("mtp.")
    elif key.startswith("embed_tokens."):
        key = "model." + key
    key = _released_key(key)
    if key.startswith("layers."):
        return "mtp." + key.removeprefix("layers.")
    return key


class DeepseekV41DSparkStateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Convert the released ``mtp.*`` DSpark weights for training and export."""

    _supports_low_memory_dcp_load = False

    def __init__(
        self,
        config: DeepseekV41TextConfig,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = False

    @property
    def _expert_path_segment(self) -> str:
        return "ffn.experts"

    @property
    def view_loaded_native_keys(self) -> set[str]:
        """Return native DSpark keys populated through checkpoint views."""
        return {re.sub(r"^layers\.(\d+)\.", r"mtp.\1.", key) for key in super().view_loaded_native_keys}

    def get_hf_state_dict_keys(self, state_dict: dict[str, Any]) -> list[str]:
        """Return released names for the draft's native state-dict keys."""
        keys: list[str] = []
        for fqn in state_dict:
            expert = _DSPARK_EXPERT_PATTERN.fullmatch(fqn)
            if expert:
                projections = (1, 3) if expert[2] == "gate_and_up_projs" else (2,)
                keys.extend(
                    f"mtp.{expert[1]}.ffn.experts.{index}.w{projection}.weight"
                    for index in range(self.moe_config.n_routed_experts)
                    for projection in projections
                )
            elif "_extra_state" not in fqn:
                keys.append(_dspark_released_key(fqn))
        return keys

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convert released draft tensors to native grouped storage.

        Args:
            hf_state_dict: Consumed mapping containing ``embed.weight``,
                ``head.weight``, and ``mtp.*`` tensors. Split expert projections
                have shape [output, input]; packed FP4 projections store two input
                values per INT8 element. Other tensors keep their registered shapes.
            device_mesh: Optional expert mesh selecting rank-local expert IDs.
            **kwargs: Additional checkpoint protocol arguments.

        Returns:
            Native mapping with grouped experts shaped [experts, hidden,
            2 * expert_hidden] and [experts, expert_hidden, hidden].
        """
        for key in list(hf_state_dict):
            expert = re.match(r"mtp\.\d+\.ffn\.experts\.(\d+)\.", key)
            not_draft = key not in ("embed.weight", "head.weight") and not key.startswith("mtp.")
            nonlocal_expert = expert is not None and not should_load_expert_for_rank(
                int(expert[1]), device_mesh, self.moe_config.n_routed_experts
            )
            if not_draft or nonlocal_expert:
                hf_state_dict.pop(key)
        self._dequantize(hf_state_dict)

        converted: dict[str, Any] = {}
        expert_state: dict[str, Any] = {}
        for key in list(hf_state_dict):
            value = hf_state_dict.pop(key)
            if key.endswith(".scale"):
                raise ValueError(f"Checkpoint scale {key} has no matching weight")
            native_key = _dspark_native_key(key)
            if ".ffn.experts." in native_key:
                pseudo_key = re.sub(r"^mtp\.(\d+)\.", r"layers.\1.", native_key)
                expert_state[pseudo_key] = value
            else:
                if native_key in converted:
                    raise ValueError(f"Multiple checkpoint tensors map to {native_key}")
                converted[native_key] = value

        merged = self._from_hf_w_merged_experts(expert_state, device_mesh)
        converted.update({re.sub(r"^layers\.(\d+)\.", r"mtp.\1.", key): value for key, value in merged.items()})
        return converted

    def _dequantize(self, state_dict: dict[str, Any]) -> None:
        """Replace paired released draft weights and scales with BF16 tensors."""
        for key in list(state_dict):
            if not key.endswith(".weight"):
                continue
            weight = state_dict[key]
            scale_key = key.removesuffix(".weight") + ".scale"
            if scale_key not in state_dict:
                if weight.dtype in (torch.float8_e4m3fn, torch.int8):
                    raise ValueError(f"Quantized weight {key} is missing its scale tensor {scale_key}")
                continue
            state_dict[key] = dequantize_checkpoint_weight(
                weight,
                state_dict.pop(scale_key),
                dtype=self.dtype,
            )

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        quantization: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Export native draft tensors under released checkpoint names."""
        output: dict[str, Any] = {}
        for key, value in state_dict.items():
            output.update(
                self.convert_single_tensor_to_hf(
                    key,
                    value,
                    exclude_key_regex=exclude_key_regex,
                    quantization=quantization,
                    **kwargs,
                )
            )
        return output

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs: Any) -> list[tuple[str, Any]]:
        """Convert one native draft tensor to released checkpoint storage."""
        quantization = kwargs.get("quantization", False)
        if quantization and not kwargs.get("for_checkpoint_load", False):
            raise ValueError(
                "Quantization targets are for checkpoint loading only; export trained weights without quantization"
            )

        expert = _DSPARK_EXPERT_PATTERN.fullmatch(fqn)
        if expert:
            pseudo_fqn = re.sub(r"^mtp\.(\d+)\.", r"layers.\1.", fqn)
            result = self._convert_single_merged_expert_to_hf_split_experts(
                pseudo_fqn,
                tensor,
                for_checkpoint_load=kwargs.get("for_checkpoint_load", False),
                quantization=quantization,
            )
            if result is None:
                raise RuntimeError(f"Could not convert DSpark expert tensor {fqn}")
            released = [
                (_dspark_released_key(re.sub(r"^layers\.(\d+)\.", r"mtp.\1.", key)), value) for key, value in result
            ]
        else:
            released = [(_dspark_released_key(fqn), tensor)]

        exclude = kwargs.get("exclude_key_regex")
        converted: list[tuple[str, Any]] = []
        for key, value in released:
            if exclude and re.match(exclude, key):
                continue
            quantized = re.fullmatch(
                r"mtp\.\d+\.(?:attn\.(?:wq_a|wq_b|wkv|wo_a|wo_b)"
                r"|main_proj|ffn\.(?:shared_experts|experts\.\d+)\.w[123])\.weight",
                key,
            )
            if quantization and quantized:
                converted.extend(DeepseekV41StateDictAdapter._quantized_load_targets(key, value))
            else:
                converted.append((key, value))
        return converted

    def forced_hf_dtype_mapping(self, state_dict: dict[str, Any]) -> dict[str, str]:
        """Return released draft keys whose trained values must remain FP32."""
        return {
            _dspark_released_key(key): "float32"
            for key, value in state_dict.items()
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32
        }

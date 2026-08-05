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

from __future__ import annotations

import logging
import re
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import (
    create_scale_inv_for_weight,
    dequantize_from_fp8,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

logger = logging.getLogger(__name__)

NON_QUANTIZED_KEY_PATTERNS = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "norm.weight",
    "lm_head.weight",
    "embed_tokens.weight",
    "mlp.gate.weight",
    "self_attn.o_proj.weight",
    "attention_sink_bias",
]


def _should_quantize_key(key: str) -> bool:
    if not key.endswith(".weight"):
        return False
    return not any(pattern in key for pattern in NON_QUANTIZED_KEY_PATTERNS)


class MiMoV2StateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Convert MiMo-V2.5-Pro HF checkpoints to Automodel's grouped MoE layout.

    HF stores routed experts as split per-expert projections:
    ``mlp.experts.{E}.{gate,up,down}_proj.weight``.  Automodel groups those
    into ``gate_and_up_projs`` and ``down_projs`` so EP can shard experts
    without materialising every expert on every rank.

    MiMo-V2.5-Pro stores fused QKV projections as TP-interleaved shards. The
    adapter dequantizes each checkpoint shard independently, then restores the
    canonical ``[Q, K, V]`` row layout expected by the model.
    """

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        del kwargs
        for key in hf_state_dict.keys():
            if ".mlp.experts." in key and key.endswith(".weight"):
                self._uses_model_prefix = key.startswith("model.")
                break
        hf_state_dict = self._dequantize(hf_state_dict)
        return self._from_hf_w_merged_experts(hf_state_dict, device_mesh)

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert Automodel state_dict to the HF MiMo-V2.5-Pro layout.

        Note: The ``quantization`` parameter is accepted for interface
        compatibility but is **ignored**. MiMo-V2.5-Pro is distributed as an
        FP8 HF checkpoint, so this adapter always emits FP8 weights plus
        ``_scale_inv`` companions for keys that match ``_should_quantize_key``,
        regardless of the caller's preference.
        """
        hf_state_dict: dict[str, Any] = {}
        for fqn, tensor in state_dict.items():
            for key, value in self.convert_single_tensor_to_hf(
                fqn,
                tensor,
                exclude_key_regex=exclude_key_regex,
                quantization=quantization,
                **kwargs,
            ):
                hf_state_dict[key] = value
        return hf_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        result = expert_result if expert_result is not None else [(fqn, tensor)]

        if exclude_key_regex:
            result = [(key, value) for key, value in result if not re.match(exclude_key_regex, key)]

        quantized_result: list[tuple[str, Any]] = []
        for key, value in result:
            if _should_quantize_key(key):
                quantized = value.to(dtype=torch.float8_e4m3fn)
                quantized_result.append((key, quantized))
                quantized_result.append((key + "_scale_inv", create_scale_inv_for_weight(quantized)))
            else:
                quantized_result.append((key, value))
        return quantized_result

    def _dequantize(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        scale_inv_keys: list[str] = []
        dequantized_count = 0
        for key in list(state_dict.keys()):
            if not key.endswith(".weight"):
                continue
            scale_key = key + "_scale_inv"
            if scale_key not in state_dict:
                continue
            scale_inv = state_dict[scale_key]
            if key.endswith(".self_attn.qkv_proj.weight"):
                state_dict[key] = self._dequantize_interleaved_qkv(state_dict[key], scale_inv, key)
            else:
                state_dict[key] = dequantize_from_fp8(
                    state_dict[key],
                    scale_inv,
                    dtype=self.dtype,
                    name=key,
                )
            scale_inv_keys.append(scale_key)
            dequantized_count += 1

        for key in scale_inv_keys:
            state_dict.pop(key, None)

        logger.debug("[MiMo V2.5-Pro FP8 Dequant] Dequantized %s weights", dequantized_count)
        return state_dict

    def _dequantize_interleaved_qkv(
        self,
        weight: torch.Tensor,
        scale_inv: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        """Dequantize and canonicalize a TP-interleaved fused QKV projection.

        Args:
            weight: Tensor of shape [interleaved_qkv, hidden]. Axis 0 stores
                checkpoint TP shards, each laid out as [Q_shard, K_shard,
                V_shard]. A DTensor may shard either axis; its placements and
                global shape are preserved in the returned tensor.
            scale_inv: Tensor of shape [tp * scale_rows_per_shard,
                scale_columns]. Each checkpoint TP shard owns an independent
                128x128 FP8 scale grid.
            key: Fully qualified weight name containing the decoder layer index.

        Returns:
            Tensor of shape [q_rows + k_rows + v_rows, hidden] in canonical
            [Q, K, V] row order. DTensor inputs retain their original mesh and
            placements.
        """

        match = re.search(r"layers\.(\d+)\.", key)
        if match is None:
            raise ValueError(f"Cannot determine MiMo layer index from {key}")
        layer_idx = int(match.group(1))
        is_swa = bool(self.config.hybrid_layer_pattern[layer_idx])

        ckpt_tp = int(self.config.num_key_value_heads)
        if is_swa:
            num_heads = int(self.config.swa_num_attention_heads)
            num_kv_heads = int(self.config.swa_num_key_value_heads)
            head_dim = int(self.config.swa_head_dim)
            v_head_dim = int(self.config.swa_v_head_dim)
        else:
            num_heads = int(self.config.num_attention_heads)
            num_kv_heads = int(self.config.num_key_value_heads)
            head_dim = int(self.config.head_dim)
            v_head_dim = int(self.config.v_head_dim)

        q_per_shard = (num_heads // ckpt_tp) * head_dim
        k_per_shard = max(1, num_kv_heads // ckpt_tp) * head_dim
        v_per_shard = max(1, num_kv_heads // ckpt_tp) * v_head_dim
        rows_per_shard = q_per_shard + k_per_shard + v_per_shard
        scale_rows_per_shard = (rows_per_shard + 127) // 128

        if weight.shape[0] != ckpt_tp * rows_per_shard:
            raise ValueError(
                f"{key} has {weight.shape[0]} rows, expected {ckpt_tp * rows_per_shard} "
                f"for {ckpt_tp} interleaved checkpoint shards"
            )

        distributed = isinstance(weight, DTensor)
        local_weight = weight.to_local() if distributed else weight
        full_scale = scale_inv.full_tensor() if isinstance(scale_inv, DTensor) else scale_inv
        full_scale = full_scale.to(device=local_weight.device)
        expected_scale_shape = (ckpt_tp * scale_rows_per_shard, (weight.shape[1] + 127) // 128)
        if full_scale.shape != expected_scale_shape:
            raise ValueError(f"{key} scale_inv has shape {full_scale.shape}, expected {expected_scale_shape}")

        if distributed:
            _, global_offset = compute_local_shape_and_global_offset(
                weight.shape,
                weight.device_mesh,
                weight.placements,
            )
            row_offset, col_offset = global_offset
        else:
            row_offset, col_offset = 0, 0

        global_rows = torch.arange(local_weight.shape[0], device=local_weight.device) + row_offset
        scale_rows = (global_rows // rows_per_shard) * scale_rows_per_shard
        scale_rows = scale_rows + (global_rows % rows_per_shard) // 128
        scale_cols = (torch.arange(local_weight.shape[1], device=local_weight.device) + col_offset) // 128
        scales = full_scale[scale_rows[:, None], scale_cols[None, :]]
        local_dequant = (local_weight.float() * scales).to(self.dtype)

        if distributed:
            raw_dequant = DTensor.from_local(
                local_dequant,
                weight.device_mesh,
                weight.placements,
                shape=weight.shape,
                stride=weight.stride(),
            ).full_tensor()
        else:
            raw_dequant = local_dequant

        shards = raw_dequant.chunk(ckpt_tp, dim=0)
        canonical = torch.cat(
            [shard[:q_per_shard] for shard in shards]
            + [shard[q_per_shard : q_per_shard + k_per_shard] for shard in shards]
            + [shard[q_per_shard + k_per_shard :] for shard in shards],
            dim=0,
        )

        if not distributed:
            return canonical
        local_shape = local_weight.shape
        canonical_local = canonical[
            row_offset : row_offset + local_shape[0],
            col_offset : col_offset + local_shape[1],
        ].contiguous()
        return DTensor.from_local(
            canonical_local,
            weight.device_mesh,
            weight.placements,
            shape=weight.shape,
            stride=weight.stride(),
        )

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

    MiMo-V2.5-Pro uses fused QKV (``self_attn.qkv_proj.weight``), so attention
    projection keys require no renaming — they pass through unchanged.
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
            state_dict[key] = dequantize_from_fp8(
                state_dict[key],
                state_dict[scale_key],
                dtype=self.dtype,
                name=key,
            )
            scale_inv_keys.append(scale_key)
            dequantized_count += 1

        for key in scale_inv_keys:
            state_dict.pop(key, None)

        logger.debug("[MiMo V2.5-Pro FP8 Dequant] Dequantized %s weights", dequantized_count)
        return state_dict

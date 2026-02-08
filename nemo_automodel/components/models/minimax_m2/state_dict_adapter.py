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

import re
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import (
    BLOCK_SIZE,
    create_scale_inv_for_weight,
    dequantize_from_fp8,
)
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

NON_QUANTIZED_KEY_PATTERNS = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "norm.weight",
    "lm_head.weight",
    "embed_tokens.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "block_sparse_moe.gate.weight",
    "mlp.gate.weight",
]


def should_quantize_key(key: str) -> bool:
    if not key.endswith(".weight"):
        return False
    return not any(pattern in key for pattern in NON_QUANTIZED_KEY_PATTERNS)


class MiniMaxM2StateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Convert between MiniMax-M2.1 HF checkpoints and native grouped-expert format."""

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True

    @property
    def _expert_path_segment(self) -> str:
        return "mlp.experts"

    def _dequantize(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        scale_inv_keys = []
        for key, weight in state_dict.items():
            if key.endswith(".weight") and key + "_scale_inv" in state_dict:
                scale_inv = state_dict[key + "_scale_inv"]
                state_dict[key] = dequantize_from_fp8(weight, scale_inv, dtype=self.dtype, name=key)
                scale_inv_keys.append(key + "_scale_inv")

        for key in scale_inv_keys:
            state_dict.pop(key, None)

        return state_dict

    def _hf_key_to_native(self, key: str) -> str:
        key = key.replace(".block_sparse_moe.gate.weight", ".mlp.gate.weight")
        key = key.replace(".block_sparse_moe.e_score_correction_bias", ".mlp.gate.e_score_correction_bias")
        key = re.sub(r"\.block_sparse_moe\.experts\.(\d+)\.w1\.weight$", r".mlp.experts.\1.gate_proj.weight", key)
        key = re.sub(r"\.block_sparse_moe\.experts\.(\d+)\.w3\.weight$", r".mlp.experts.\1.up_proj.weight", key)
        key = re.sub(r"\.block_sparse_moe\.experts\.(\d+)\.w2\.weight$", r".mlp.experts.\1.down_proj.weight", key)
        return key

    def _native_key_to_hf(self, key: str) -> str:
        key = re.sub(r"\.mlp\.experts\.(\d+)\.gate_proj\.weight$", r".block_sparse_moe.experts.\1.w1.weight", key)
        key = re.sub(r"\.mlp\.experts\.(\d+)\.up_proj\.weight$", r".block_sparse_moe.experts.\1.w3.weight", key)
        key = re.sub(r"\.mlp\.experts\.(\d+)\.down_proj\.weight$", r".block_sparse_moe.experts.\1.w2.weight", key)
        key = key.replace(".mlp.gate.weight", ".block_sparse_moe.gate.weight")
        key = key.replace(".mlp.gate.e_score_correction_bias", ".block_sparse_moe.e_score_correction_bias")
        return key

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        hf_state_dict = {}
        for fqn, tensor in state_dict.items():
            converted_tensors = self.convert_single_tensor_to_hf(
                fqn,
                tensor,
                exclude_key_regex=exclude_key_regex,
                quantization=quantization,
                **kwargs,
            )
            for key, value in converted_tensors:
                hf_state_dict[key] = value

        return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        # Detect model prefix from key layout.
        for key in hf_state_dict.keys():
            if ".block_sparse_moe.experts." in key and key.endswith(".weight"):
                self._uses_model_prefix = key.startswith("model.")
                break

        dequantized = self._dequantize(dict(hf_state_dict))
        remapped = {self._hf_key_to_native(k): v for k, v in dequantized.items()}
        return self._from_hf_w_merged_experts(remapped, device_mesh)

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        quantization = kwargs.get("quantization", False)
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if expert_result is not None:
            result = [(self._native_key_to_hf(k), v) for k, v in expert_result]
        else:
            result = [(self._native_key_to_hf(fqn), tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        if quantization:
            quantized_result = []
            for key, value in result:
                if should_quantize_key(key):
                    value = value.to(dtype=torch.float8_e4m3fn)
                    weight_scale_inv = create_scale_inv_for_weight(value, block_size=BLOCK_SIZE)
                    quantized_result.append((key, value))
                    quantized_result.append((key + "_scale_inv", weight_scale_inv))
                else:
                    quantized_result.append((key, value))
            return quantized_result

        return result

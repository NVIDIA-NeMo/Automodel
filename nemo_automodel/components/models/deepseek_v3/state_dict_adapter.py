# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
from transformers import DeepseekV3Config

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoEStateDictMixin
from nemo_automodel.components.moe.utils import BackendConfig


class DeepSeekV3StateDictAdapter(MoEStateDictMixin, StateDictAdapter):
    def __init__(
        self,
        config: DeepseekV3Config,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True
        self._had_scale_inv_tensors = False
        self.from_hf_map = {
            "model.layers.{}.mlp.experts.{}.gate_proj.weight": "model.layers.{}.mlp.experts.gate_projs",
            "model.layers.{}.mlp.experts.{}.up_proj.weight": "model.layers.{}.mlp.experts.up_projs",
            "model.layers.{}.mlp.experts.{}.down_proj.weight": "model.layers.{}.mlp.experts.down_projs",
        }

    def _dequantize(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        scale_inv_keys = []
        for key, weight in state_dict.items():
            if key.endswith(".weight") and key + "_scale_inv" in state_dict:
                scale_inv = state_dict[key + "_scale_inv"]
                dequantized_weight = dequantize_from_fp8(weight, scale_inv, dtype=self.dtype)
                state_dict[key] = dequantized_weight
                scale_inv_keys.append(key + "_scale_inv")

        for key in scale_inv_keys:
            state_dict.pop(key)

        return state_dict

    def _add_quantization_scale_inv_tensors(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        non_quantized_keys = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "norm.weight",
            "lm_head.weight",
            "embed_tokens.weight",
            "mlp.gate.weight",
        ]

        weight_scale_inv_state_dict = {}
        for key, value in state_dict.items():
            if key.endswith(".weight") and not any(
                non_quantized_key in key for non_quantized_key in non_quantized_keys
            ):
                expected_scale_shape = calculate_scale_shape(value)
                weight_scale_inv_state_dict[key + "_scale_inv"] = torch.ones(expected_scale_shape, dtype=self.dtype)

        state_dict.update(weight_scale_inv_state_dict)
        return state_dict

    def to_hf(self, state_dict: dict[str, Any], exclude_key_regex: Optional[str] = None) -> dict[str, Any]:
        """Convert from native model state dict to HuggingFace format.
        Automatically detects format based on backend.enable_deepep configuration.
        """
        if self.backend.enable_deepep:
            hf_state_dict = self._to_hf_deepep(state_dict)
        else:
            hf_state_dict = self._to_hf_grouped_experts(state_dict)

        if exclude_key_regex:
            hf_state_dict = {k: v for k, v in hf_state_dict.items() if not re.match(exclude_key_regex, k)}

        if self._had_scale_inv_tensors:
            return self._add_quantization_scale_inv_tensors(hf_state_dict)
        else:
            return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        target_format: str = "auto",
    ) -> dict[str, Any]:
        """Convert HF checkpoint to native format.
        - Dequantize FP8 tensors if scale_inv buffers are provided
        - Aggregate per-expert weights into grouped tensors
        - If device_mesh is provided, only load experts needed for the current rank
        """
        for key in hf_state_dict.keys():
            if ".mlp.experts." in key and key.endswith(".weight"):
                self._uses_model_prefix = key.startswith("model.")
            if key.endswith("_scale_inv"):
                self._had_scale_inv_tensors = True

        hf_state_dict = self._dequantize(hf_state_dict)

        if target_format == "auto":
            actual_target_format = "deepep" if self.backend.enable_deepep else "grouped_experts"
        else:
            if target_format not in ["grouped_experts", "deepep"]:
                raise ValueError(f"target_format must be 'auto', 'grouped_experts' or 'deepep', got '{target_format}'")
            actual_target_format = target_format

        if actual_target_format == "deepep":
            return self._from_hf_deepep(hf_state_dict, device_mesh)
        else:
            return self._from_hf_grouped_experts(hf_state_dict, device_mesh)


def dequantize_from_fp8(
    weight: torch.Tensor, scale_inv: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Minimal FP8 dequantization: cast to dtype and divide by inverse scale.
    Broadcasts scale_inv over the last dimension of weight.
    """
    w = weight.to(dtype)
    s = scale_inv.to(dtype)
    # Ensure broadcast shape: append singleton dims to scale_inv to match weight
    if s.ndim < w.ndim:
        expand_shape = list(s.shape) + [1] * (w.ndim - s.ndim)
        s = s.view(*expand_shape)
    return w / s


def calculate_scale_shape(weight: torch.Tensor) -> tuple[int, ...]:
    """
    Compute expected shape for per-row inverse scales.
    - 2D [out, in] -> [out, 1]
    - 3D [N, out, in] -> [N, out, 1]
    Fallback: last dim collapsed to 1
    """
    if weight.ndim == 2:
        return (weight.shape[0], 1)
    if weight.ndim == 3:
        return (weight.shape[0], weight.shape[1], 1)
    shape = list(weight.shape)
    if len(shape) > 0:
        shape[-1] = 1
    return tuple(shape)

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
from transformers import GptOssConfig

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoEStateDictMixin
from nemo_automodel.components.moe.utils import BackendConfig

class GptOssStateDictAdapter(MoEStateDictMixin, StateDictAdapter):
    def __init__(
        self,
        config: GptOssConfig,
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
            "model.layers.{}.mlp.experts.gate_up_proj": "model.layers.{}.mlp.experts.gate_and_up_projs",
            "model.layers.{}.mlp.experts.gate_up_proj_bias": "model.layers.{}.mlp.experts.gate_and_up_bias",
            "model.layers.{}.mlp.experts.down_proj": "model.layers.{}.mlp.experts.down_projs",
            "model.layers.{}.mlp.experts.down_proj_bias": "model.layers.{}.mlp.experts.down_bias",
            "model.layers.{}.mlp.router.weight": "model.layers.{}.mlp.gate.weight",
            "model.layers.{}.mlp.router.bias": "model.layers.{}.mlp.gate.bias",
        }

    # def _dequantize(self, state_dict: dict[str, Any]) -> dict[str, Any]:
    #     scale_inv_keys = []
    #     for key, weight in state_dict.items():
    #         if key.endswith(".weight") and key + "_scale_inv" in state_dict:
    #             scale_inv = state_dict[key + "_scale_inv"]
    #             dequantized_weight = dequantize_from_fp8(weight, scale_inv, dtype=self.dtype)
    #             state_dict[key] = dequantized_weight
    #             scale_inv_keys.append(key + "_scale_inv")

    #     for key in scale_inv_keys:
    #         state_dict.pop(key)

    #     return state_dict

    # def _add_quantization_scale_inv_tensors(self, state_dict: dict[str, Any]) -> dict[str, Any]:
    #     non_quantized_keys = [
    #         "input_layernorm.weight",
    #         "post_attention_layernorm.weight",
    #         "norm.weight",
    #         "lm_head.weight",
    #         "embed_tokens.weight",
    #         "mlp.gate.weight",
    #     ]

    #     weight_scale_inv_state_dict = {}
    #     for key, value in state_dict.items():
    #         if key.endswith(".weight") and not any(
    #             non_quantized_key in key for non_quantized_key in non_quantized_keys
    #         ):
    #             expected_scale_shape = calculate_scale_shape(value)
    #             weight_scale_inv_state_dict[key + "_scale_inv"] = torch.ones(expected_scale_shape, dtype=self.dtype)

    #     state_dict.update(weight_scale_inv_state_dict)
    #     return state_dict

    def to_hf(self, state_dict: dict[str, Any], exclude_key_regex: Optional[str] = None) -> dict[str, Any]:
        """Convert from native model state dict to HuggingFace format.
        Automatically detects format based on backend.enable_deepep configuration.
        """
        if self.backend.enable_deepep:
            to_hf_map = {v: k for k, v in self.from_hf_map.items()}
            uses_model_prefix = any(k.startswith("model.") for k in state_dict.keys())

            hf_state_dict: dict[str, Any] = {}
            for key, value in state_dict.items():
                if "experts" not in key and "gate" not in key:
                    hf_state_dict[key] = value
                    continue
                
                layer_num = re.search(r"layers\.(\d+)", key).group(1)
                renamed = False
                for src_tmpl, dst_tmpl in to_hf_map.items():
                    src_key = format_template(src_tmpl, uses_model_prefix, layer_num)
                    if key == src_key:
                        new_key = format_template(dst_tmpl, uses_model_prefix, layer_num)
                        hf_state_dict[new_key] = value
                        renamed = True
                        break

                # if renaming failed, we keep the original key as fallback
                if not renamed:
                    hf_state_dict[key] = value
        else:
            raise NotImplementedError("GroupedExperts format is not supported for GPT-OSS")

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

        # hf_state_dict = self._dequantize(hf_state_dict)

        if target_format == "auto":
            actual_target_format = "deepep" if self.backend.enable_deepep else "grouped_experts"
        else:
            if target_format not in ["grouped_experts", "deepep"]:
                raise ValueError(f"target_format must be 'auto', 'grouped_experts' or 'deepep', got '{target_format}'")
            actual_target_format = target_format

        if actual_target_format == "deepep":
            state_dict: dict[str, Any] = {}
            for key, value in hf_state_dict.items():
                if "experts" not in key and "router" not in key:
                    state_dict[key] = value
                    continue
                layer_num = re.search(r"layers\.(\d+)", key).group(1)
                renamed = False
                for src_tmpl, dst_tmpl in self.from_hf_map.items():
                    src_key = format_template(src_tmpl, self._uses_model_prefix, layer_num)
                    if key == src_key:
                        new_key = format_template(dst_tmpl, self._uses_model_prefix, layer_num)
                        state_dict[new_key] = value
                        renamed = True
                        break

                # if renaming failed, we keep the original key as fallback
                if not renamed:
                    state_dict[key] = value

            return state_dict

        else:
            raise NotImplementedError("GroupedExperts format is not supported for GPT-OSS")


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

def format_template(template: str, uses_model_prefix: bool, layer_str: str) -> str:
    return (template if uses_model_prefix else template.replace("model.", "", 1)).format(layer_str)
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

import logging
import re
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import (
    BLOCK_SIZE,
    create_scale_inv_for_weight,
    dequantize_from_fp8,
)
from nemo_automodel.components.models.glm4_moe.state_dict_adapter import Glm4MoeStateDictAdapter

logger = logging.getLogger(__name__)

# GLM-5.3 keeps these tensors in BF16 while storing the remaining matrix
# weights as 128x128 blockwise FP8 with ``weight_scale_inv`` siblings.
NON_QUANTIZED_KEY_PATTERNS = [
    "norm.weight",
    "lm_head.weight",
    "embed_tokens.weight",
    "mlp.gate.weight",
    "eh_proj.weight",
    "indexer.weights_proj.weight",
]


def should_quantize_key(key: str) -> bool:
    """Return whether ``key`` has a blockwise-FP8 scale in GLM checkpoints."""
    if not key.endswith(".weight") or ".lora_" in key:
        return False
    return not any(pattern in key for pattern in NON_QUANTIZED_KEY_PATTERNS)


class GlmMoeDsaStateDictAdapter(Glm4MoeStateDictAdapter):
    """Converts between HF GLM-MoE-DSA checkpoints and native format.

    Extends Glm4MoeStateDictAdapter with handling for the DSA indexer weights
    that should not be quantized (k_norm, weights_proj).
    """

    _supports_write_through_checkpoint_load = False

    _indexer_non_quantized_keys = [
        "indexer.k_norm.weight",
        "indexer.k_norm.bias",
        "indexer.weights_proj.weight",
    ]

    def _uses_blockwise_fp8_checkpoint(self) -> bool:
        quantization_config = getattr(self.config, "quantization_config", None)
        if isinstance(quantization_config, dict):
            quant_method = quantization_config.get("quant_method")
            weight_block_size = quantization_config.get("weight_block_size")
        else:
            quant_method = getattr(quantization_config, "quant_method", None)
            weight_block_size = getattr(quantization_config, "weight_block_size", None)

        if quant_method != "fp8" or not isinstance(weight_block_size, (list, tuple)):
            return False
        return tuple(weight_block_size) == (BLOCK_SIZE, BLOCK_SIZE)

    def _dequantize(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        scale_inv_keys = []
        dequantized_count = 0
        for key, weight in state_dict.items():
            scale_key = key + "_scale_inv"
            if key.endswith(".weight") and scale_key in state_dict:
                state_dict[key] = dequantize_from_fp8(
                    weight,
                    state_dict[scale_key],
                    dtype=self.dtype,
                    name=key,
                )
                scale_inv_keys.append(scale_key)
                dequantized_count += 1

        for key in scale_inv_keys:
            state_dict.pop(key)

        logger.debug(
            "[GLM FP8 Dequant] Dequantized %d weights and removed %d scale tensors",
            dequantized_count,
            len(scale_inv_keys),
        )
        return state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Dequantize blockwise FP8 tensors before merging per-expert weights."""
        hf_state_dict = self._dequantize(hf_state_dict)
        return super().from_hf(hf_state_dict, device_mesh=device_mesh, **kwargs)

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        quantization = kwargs.get("quantization", False) and self._uses_blockwise_fp8_checkpoint()
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        expert_kwargs = {**kwargs, "quantization": quantization} if "quantization" in kwargs else kwargs
        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **expert_kwargs)
        if expert_result is not None:
            result = expert_result
        else:
            result = [(fqn, tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        if quantization:
            quantized_result = []
            for key, value in result:
                if should_quantize_key(key):
                    value = value.to(dtype=torch.float8_e4m3fn)
                    quantized_result.append((key, value))
                    quantized_result.append(
                        (key + "_scale_inv", create_scale_inv_for_weight(value, block_size=BLOCK_SIZE))
                    )
                else:
                    quantized_result.append((key, value))
            return quantized_result

        return result

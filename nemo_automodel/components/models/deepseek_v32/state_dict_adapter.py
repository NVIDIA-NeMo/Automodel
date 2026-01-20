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

"""State dict adapter for DeepSeek V3.2.

Extends DeepSeekV3StateDictAdapter with mappings for the new Indexer weights.
"""

import re
from typing import Any

import torch

from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import (
    BLOCK_SIZE,
    DeepSeekV3StateDictAdapter,
    calculate_scale_shape,
)


class DeepSeekV32StateDictAdapter(DeepSeekV3StateDictAdapter):
    """State dict adapter for DeepSeek V3.2.

    Extends the V3 adapter with support for the new Indexer weights:
    - self_attn.indexer.wq_b.weight
    - self_attn.indexer.wk.weight
    - self_attn.indexer.k_norm.weight (LayerNorm)
    - self_attn.indexer.k_norm.bias (LayerNorm)
    - self_attn.indexer.weights_proj.weight

    The indexer weights use the same naming convention between HF and native format,
    so no special key mapping is needed. The main difference is handling the
    k_norm LayerNorm which should not be quantized.
    """

    # Base non-quantized keys from V3
    _base_non_quantized_keys = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "norm.weight",
        "lm_head.weight",
        "embed_tokens.weight",
        "mlp.gate.weight",
    ]

    # V3.2 indexer LayerNorm keys that should not be quantized
    _indexer_non_quantized_keys = [
        "indexer.k_norm.weight",
        "indexer.k_norm.bias",
    ]

    @property
    def _non_quantized_keys(self) -> list[str]:
        """Get the full list of non-quantized keys including indexer keys."""
        return self._base_non_quantized_keys + self._indexer_non_quantized_keys

    def _add_quantization_scale_inv_tensors(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Add quantization scale tensors, handling indexer-specific keys."""
        weight_scale_inv_state_dict = {}
        for key, value in state_dict.items():
            if key.endswith(".weight") and not any(
                non_quantized_key in key for non_quantized_key in self._non_quantized_keys
            ):
                value = value.to(dtype=torch.float8_e4m3fn)
                state_dict[key] = value
                expected_scale_shape = calculate_scale_shape(value, BLOCK_SIZE)
                weight_scale_inv_state_dict[key + "_scale_inv"] = torch.ones(
                    expected_scale_shape, dtype=torch.float32, device=value.device
                )

        state_dict.update(weight_scale_inv_state_dict)
        return state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single tensor from native format to HuggingFace format.

        Handles both standard V3 tensors and V3.2 indexer tensors, ensuring
        indexer LayerNorm weights are not quantized.
        """
        quantization = kwargs.get("quantization", False)
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if expert_result is not None:
            result = expert_result
        else:
            result = [(fqn, tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        if quantization:
            quantized_result = []
            for key, value in result:
                if key.endswith(".weight") and not any(
                    non_quantized_key in key for non_quantized_key in self._non_quantized_keys
                ):
                    value = value.to(dtype=torch.float8_e4m3fn)
                    expected_scale_shape = calculate_scale_shape(value, BLOCK_SIZE)
                    weight_scale_inv = torch.ones(expected_scale_shape, dtype=torch.float32, device=value.device)
                    quantized_result.append((key, value))
                    quantized_result.append((key + "_scale_inv", weight_scale_inv))
                else:
                    quantized_result.append((key, value))
            return quantized_result

        return result

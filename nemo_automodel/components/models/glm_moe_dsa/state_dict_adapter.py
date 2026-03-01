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
from typing import Any

import torch

from nemo_automodel.components.models.glm4_moe.state_dict_adapter import Glm4MoeStateDictAdapter


class GlmMoeDsaStateDictAdapter(Glm4MoeStateDictAdapter):
    """Converts between HF GLM-MoE-DSA checkpoints and native format.

    Extends Glm4MoeStateDictAdapter with handling for the DSA indexer weights
    that should not be quantized (k_norm, weights_proj).
    """

    _indexer_non_quantized_keys = [
        "indexer.k_norm.weight",
        "indexer.k_norm.bias",
        "indexer.weights_proj.weight",
    ]

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
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
                    non_quantized_key in key for non_quantized_key in self._indexer_non_quantized_keys
                ):
                    value = value.to(dtype=torch.float8_e4m3fn)
                    quantized_result.append((key, value))
                else:
                    quantized_result.append((key, value))
            return quantized_result

        return result

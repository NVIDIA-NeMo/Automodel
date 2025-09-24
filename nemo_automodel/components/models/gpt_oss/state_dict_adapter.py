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
from nemo_automodel.components.moe.utils import BackendConfig


class GPTOSSStateDictAdapter(StateDictAdapter):
    def __init__(
        self,
        config: GptOssConfig,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True

        # Key mapping from HF GPT OSS format to internal format
        self.hf_to_internal_map = {
            # Router mapping
            "mlp.router.weight": "mlp.gate.weight",
            "mlp.router.bias": "mlp.gate.bias",
            # Expert projection mappings (when enable_deepep=True, use gate_and_up_projs)
            "mlp.experts.gate_up_proj": "mlp.experts.gate_and_up_projs",
            # "mlp.experts.gate_up_proj_bias": "mlp.experts.gate_and_up_bias" if backend.enable_deepep else None,
            "mlp.experts.down_proj": "mlp.experts.down_projs",
            # "mlp.experts.down_proj_bias": "mlp.experts.down_bias",
        }

        # Reverse mapping for to_hf conversion
        self.internal_to_hf_map = {v: k for k, v in self.hf_to_internal_map.items() if v is not None}

    # replace _apply_key_mapping with leaf-aware replacement
    def _apply_key_mapping(self, state_dict: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            for pattern, replacement in mapping.items():
                if replacement is not None and key.endswith(pattern):
                    new_key = key[: -len(pattern)] + replacement
                    break
            new_state_dict[new_key] = value
        return new_state_dict

    def to_hf(
        self, state_dict: dict[str, Any], exclude_key_regex: Optional[str] = None, quantization: bool = False, **kwargs
    ) -> dict[str, Any]:
        """Convert from native model state dict to HuggingFace format."""
        hf_state_dict = dict(state_dict)
        hf_state_dict = self._apply_key_mapping(hf_state_dict, self.internal_to_hf_map)

        # Apply exclude regex if provided
        if exclude_key_regex:
            hf_state_dict = {k: v for k, v in hf_state_dict.items() if not re.match(exclude_key_regex, k)}

        return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert HF checkpoint to native format.
        - Apply key mappings from HF to internal format
        - Split gate_up_proj weights if enable_deepep is False
        - Keep fused weights if enable_deepep is True
        """
        # Detect model prefix usage
        for key in hf_state_dict.keys():
            if key.startswith("model."):
                self._uses_model_prefix = True
                break

        native_state_dict = dict(hf_state_dict)
        native_state_dict = self._apply_key_mapping(native_state_dict, self.hf_to_internal_map)

        return native_state_dict

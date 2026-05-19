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
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

logger = logging.getLogger(__name__)


class Ernie4_5StateDictAdapter(StateDictAdapter):
    """Passthrough adapter for dense ERNIE 4.5 checkpoints."""

    def __init__(self, config: Any):
        self.config = config

    def from_hf(self, hf_state_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        return dict(hf_state_dict)

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        if exclude_key_regex is None:
            return dict(state_dict)
        return {key: value for key, value in state_dict.items() if not re.match(exclude_key_regex, key)}

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        exclude_key_regex = kwargs.get("exclude_key_regex", None)
        if exclude_key_regex and re.match(exclude_key_regex, fqn):
            return []
        return [(fqn, tensor)]


class Ernie4_5_MoeStateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Convert ERNIE 4.5 MoE HF checkpoints to AutoModel grouped-expert format."""

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

    def _hf_key_to_native(self, key: str) -> str:
        return key.replace(".mlp.moe_statics.e_score_correction_bias", ".mlp.gate.e_score_correction_bias")

    def _native_key_to_hf(self, key: str) -> str:
        return key.replace(".mlp.gate.e_score_correction_bias", ".mlp.moe_statics.e_score_correction_bias")

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        for key in list(hf_state_dict.keys()):
            self._uses_model_prefix = key.startswith("model.")
            if key.startswith("model.mtp_"):
                hf_state_dict.pop(key)
                continue
            new_key = self._hf_key_to_native(key)
            if new_key != key:
                value = hf_state_dict.pop(key)
                if new_key.endswith(".mlp.gate.e_score_correction_bias") and value.ndim == 2 and value.shape[0] == 1:
                    value = value.squeeze(0)
                hf_state_dict[new_key] = value

        state_dict = self._from_hf_w_merged_experts(hf_state_dict, device_mesh)
        return state_dict

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

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        exclude_key_regex = kwargs.get("exclude_key_regex", None)
        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if expert_result is not None:
            result = [(self._native_key_to_hf(key), value) for key, value in expert_result]
        else:
            key = self._native_key_to_hf(fqn)
            value = tensor
            if key.endswith(".mlp.moe_statics.e_score_correction_bias") and hasattr(value, "ndim") and value.ndim == 1:
                value = value.unsqueeze(0)
            result = [(key, value)]

        if exclude_key_regex:
            result = [(key, value) for key, value in result if not re.match(exclude_key_regex, key)]
        return result

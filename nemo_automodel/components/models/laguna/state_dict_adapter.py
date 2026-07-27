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

import re
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

_HF_TO_NATIVE_RENAMES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\.mlp\.shared_expert\."), ".mlp.shared_experts."),
    (re.compile(r"\.mlp\.experts\.e_score_correction_bias$"), ".mlp.gate.e_score_correction_bias"),
)
_NATIVE_TO_HF_RENAMES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\.mlp\.shared_experts\."), ".mlp.shared_expert."),
    (re.compile(r"\.mlp\.gate\.e_score_correction_bias$"), ".mlp.experts.e_score_correction_bias"),
)


def _apply_renames(key: str, renames: tuple[tuple[re.Pattern[str], str], ...]) -> str:
    for pattern, replacement in renames:
        key, count = pattern.subn(replacement, key)
        if count:
            break
    return key


class LagunaStateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Convert Laguna HF checkpoints to Automodel's grouped-MoE layout."""

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
        normalized: dict[str, Any] = {}
        for key, value in hf_state_dict.items():
            if ".mlp.experts." in key and key.endswith(".weight"):
                self._uses_model_prefix = key.startswith("model.")
            normalized[_apply_renames(key, _HF_TO_NATIVE_RENAMES)] = value
        return self._from_hf_w_merged_experts(normalized, device_mesh)

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        hf_state_dict: dict[str, Any] = {}
        for fqn, tensor in state_dict.items():
            converted_tensors = self.convert_single_tensor_to_hf(
                fqn,
                tensor,
                exclude_key_regex=exclude_key_regex,
                **kwargs,
            )
            for key, value in converted_tensors:
                hf_state_dict[key] = value
        return hf_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        exclude_key_regex = kwargs.get("exclude_key_regex", None)
        expert_split = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        pairs = expert_split if expert_split is not None else [(fqn, tensor)]

        out: list[tuple[str, Any]] = []
        for key, value in pairs:
            renamed = _apply_renames(key, _NATIVE_TO_HF_RENAMES)
            if exclude_key_regex and re.match(exclude_key_regex, renamed):
                continue
            out.append((renamed, value))
        return out


__all__ = ["LagunaStateDictAdapter"]

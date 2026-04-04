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

"""State dict adapter for Afmoe HF checkpoints.

Handles conversion between HF per-expert format and NeMo grouped-expert format,
plus key renaming for the router gate and expert bias.

HF key format:
  model.layers.{L}.mlp.router.gate.weight       -> model.layers.{L}.mlp.gate.weight
  model.layers.{L}.mlp.expert_bias               -> model.layers.{L}.mlp.gate.e_score_correction_bias
  model.layers.{L}.mlp.experts.{E}.gate_proj.weight  -> (stacked into gate_and_up_projs)
  model.layers.{L}.mlp.experts.{E}.up_proj.weight    -> (stacked into gate_and_up_projs)
  model.layers.{L}.mlp.experts.{E}.down_proj.weight  -> (stacked into down_projs)

Other keys (attention projections, norms, shared experts, dense MLP) pass through unchanged.
"""

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

# Bidirectional key renaming rules: (hf_pattern, nemo_pattern)
_KEY_RENAMES_HF_TO_NEMO = [
    (".mlp.router.gate.weight", ".mlp.gate.weight"),
    (".mlp.expert_bias", ".mlp.gate.e_score_correction_bias"),
]


class AfmoeStateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Converts between HF Afmoe checkpoints and NeMo grouped-experts native format."""

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

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional[DeviceMesh] = None,
        **kwargs,
    ) -> dict[str, Any]:
        # Detect whether HF checkpoints use the "model." prefix
        for key in hf_state_dict.keys():
            if ".mlp.experts." in key and key.endswith(".weight"):
                self._uses_model_prefix = key.startswith("model.")
                break

        # Rename HF keys to NeMo keys before expert merging
        renamed = {}
        for key, value in list(hf_state_dict.items()):
            new_key = key
            for hf_pat, nemo_pat in _KEY_RENAMES_HF_TO_NEMO:
                if hf_pat in new_key:
                    new_key = new_key.replace(hf_pat, nemo_pat)
                    break
            renamed[new_key] = value
        hf_state_dict = renamed

        return self._from_hf_w_merged_experts(hf_state_dict, device_mesh)

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
                fqn, tensor, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
            )
            for key, value in converted_tensors:
                hf_state_dict[key] = value

        # Rename NeMo keys back to HF keys
        renamed = {}
        for key, value in hf_state_dict.items():
            new_key = key
            for hf_pat, nemo_pat in _KEY_RENAMES_HF_TO_NEMO:
                if nemo_pat in new_key:
                    new_key = new_key.replace(nemo_pat, hf_pat)
                    break
            renamed[new_key] = value

        return renamed

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if expert_result is not None:
            result = expert_result
        else:
            result = [(fqn, tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        return result

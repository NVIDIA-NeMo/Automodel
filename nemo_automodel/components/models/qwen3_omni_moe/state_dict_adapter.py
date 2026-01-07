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
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin
from nemo_automodel.components.moe.utils import BackendConfig

logger = logging.getLogger(__name__)


class Qwen3OmniMoeStateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Converts between HF Qwen3OmniMoe checkpoints and grouped-experts native format."""

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
        self._uses_thinker_prefix = True

    def to_hf(
        self, state_dict: dict[str, Any], exclude_key_regex: Optional[str] = None, quantization: bool = False, **kwargs
    ) -> dict[str, Any]:
        hf_state_dict = self._to_hf_w_split_experts(state_dict)

        if self._uses_thinker_prefix:
            hf_state_dict_with_prefix = {}
            for key, value in hf_state_dict.items():
                new_key = "thinker." + key
                hf_state_dict_with_prefix[new_key] = value
            hf_state_dict = hf_state_dict_with_prefix

        if exclude_key_regex:
            import re

            hf_state_dict = {k: v for k, v in hf_state_dict.items() if not re.match(exclude_key_regex, k)}
        return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        for key in hf_state_dict.keys():
            if ".mlp.experts." in key and key.endswith(".weight"):
                self._uses_thinker_prefix = key.startswith("thinker.")
                self._uses_model_prefix = "model." in key
                break

        # Remove thinker prefix if present to match our internal format
        if self._uses_thinker_prefix:
            hf_state_dict_no_prefix = {}
            for key, value in hf_state_dict.items():
                if key.startswith("thinker."):
                    new_key = key[len("thinker.") :]
                    hf_state_dict_no_prefix[new_key] = value
                else:
                    hf_state_dict_no_prefix[key] = value
            hf_state_dict = hf_state_dict_no_prefix

        return self._from_hf_w_merged_experts(hf_state_dict, device_mesh)

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        converted = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if converted is None:
            converted = [(fqn, tensor)]

        if self._uses_thinker_prefix:
            converted = [(f"thinker.{key}", value) for key, value in converted]

        if exclude_key_regex:
            import re

            converted = [(key, value) for key, value in converted if not re.match(exclude_key_regex, key)]

        return converted

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
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

logger = logging.getLogger(__name__)


class Qwen3OmniMoeStateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Converts between HF Qwen3OmniMoe checkpoints and grouped-experts native format."""

    _supports_write_through_checkpoint_load = True

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
        self, state_dict: dict[str, Any], exclude_key_regex: str | None = None, quantization: bool = False, **kwargs
    ) -> dict[str, Any]:
        hf_state_dict = self._to_hf_w_split_experts(state_dict)

        if self._uses_thinker_prefix:
            hf_state_dict = {self._add_thinker_prefix(key): value for key, value in hf_state_dict.items()}

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
        # Detect the checkpoint's layout from its expert weight keys. PEFT
        # saves nest the thinker namespace inside the "base_model.model."
        # outer prefix, so check both positions, and consider every matching
        # key rather than the first: a full omni dict also carries talker
        # expert weights with no thinker namespace, and adapter dicts may mix
        # lora keys with modules_to_save-style full weights.
        expert_weight_keys = [key for key in hf_state_dict if ".mlp.experts." in key and key.endswith(".weight")]
        if expert_weight_keys:
            self._uses_thinker_prefix = any(
                key.startswith(("thinker.", "base_model.model.thinker.")) for key in expert_weight_keys
            )
            self._uses_model_prefix = any("model." in key for key in expert_weight_keys)

        # Remove thinker prefix if present to match our internal format
        if self._uses_thinker_prefix:
            hf_state_dict = {self._strip_thinker_prefix(key): value for key, value in hf_state_dict.items()}

        return self._from_hf_w_merged_experts(hf_state_dict, device_mesh)

    @staticmethod
    def _add_thinker_prefix(key: str) -> str:
        """Namespace a native key the way the HF omni checkpoint expects.

        PEFT adapter keys keep their ``base_model.model.`` outer prefix, so for
        those the ``thinker.`` namespace goes inside it — matching how PEFT
        names modules on the actual HF omni model.
        """
        if key.startswith("base_model.model."):
            return "base_model.model.thinker." + key.removeprefix("base_model.model.")
        return "thinker." + key

    @staticmethod
    def _strip_thinker_prefix(key: str) -> str:
        """Remove the omni checkpoint's ``thinker.`` namespace."""
        if key.startswith("base_model.model.thinker."):
            return "base_model.model." + key.removeprefix("base_model.model.thinker.")
        return key.removeprefix("thinker.")

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        converted = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if converted is None:
            converted = [(fqn, tensor)]

        if self._uses_thinker_prefix:
            converted = [(self._add_thinker_prefix(key), value) for key, value in converted]

        if exclude_key_regex:
            import re

            converted = [(key, value) for key, value in converted if not re.match(exclude_key_regex, key)]

        return converted

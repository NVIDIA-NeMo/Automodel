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

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

logger = logging.getLogger(__name__)


class NemotronV3StateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """State dict adapter for NemotronV3 models.

    Converts between HuggingFace checkpoint format and internal NeMo format.

    HF format uses 'backbone' prefix:
        - backbone.embed_tokens.weight
        - backbone.layers.{}.norm.weight
        - backbone.layers.{}.mixer.* (mamba/attention/moe components)
        - backbone.norm_f.weight
        - lm_head.weight

    Internal format uses 'model' prefix:
        - model.embed_tokens.weight
        - model.layers.{}.norm.weight
        - model.layers.{}.mixer.* (mamba/attention/moe components)
        - model.norm.weight
        - lm_head.weight

    For MoE layers:
        - HF: Split per-expert weights (experts.{}.up_proj.weight, experts.{}.down_proj.weight)
        - Internal: Merged expert weights (experts.gate_and_up_projs, experts.down_projs)

    NemotronV3 uses ReLU² activation (non-gated), so gate_and_up_projs has
    shape [n_experts, dim, inter_dim] instead of [n_experts, dim, 2*inter_dim].

    Note: NemotronV3 uses 'mixer' instead of 'mlp' in layer paths.
    """

    def __init__(
        self,
        config,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True

        # Mapping for expert weights (HF split → internal merged)
        self.from_hf_map = {
            "model.layers.{}.mixer.experts.{}.up_proj.weight": "model.layers.{}.mixer.experts.gate_and_up_projs",
            "model.layers.{}.mixer.experts.{}.down_proj.weight": "model.layers.{}.mixer.experts.down_projs",
        }

    @property
    def _hf_prefix(self) -> str:
        """NemotronV3 HF format uses 'backbone.' prefix."""
        return "backbone."

    @property
    def _expert_path_segment(self) -> str:
        """NemotronV3 uses 'mixer.experts' instead of 'mlp.experts'."""
        return "mixer.experts"

    def to_hf(self, state_dict: dict[str, Any], exclude_key_regex: Optional[str] = None, **kwargs) -> dict[str, Any]:
        """Convert from internal model state dict to HuggingFace format.

        Args:
            state_dict: Internal format state dict
            exclude_key_regex: Optional regex pattern to exclude keys
            **kwargs: Additional arguments

        Returns:
            HuggingFace format state dict
        """
        inplace = bool(kwargs.get("inplace", False))
        hf_state_dict = {}
        if inplace:
            for fqn in list(state_dict.keys()):
                tensor = state_dict.get(fqn, None)
                if tensor is None:
                    continue
                converted_tensors = self.convert_single_tensor_to_hf(
                    fqn, tensor, exclude_key_regex=exclude_key_regex, **kwargs
                )
                for key, value in converted_tensors:
                    hf_state_dict[key] = value
                keep_original = (
                    len(converted_tensors) == 1 and converted_tensors[0][0] == fqn and converted_tensors[0][1] is tensor
                )
                if not keep_original:
                    state_dict.pop(fqn, None)
        else:
            for fqn, tensor in state_dict.items():
                converted_tensors = self.convert_single_tensor_to_hf(
                    fqn, tensor, exclude_key_regex=exclude_key_regex, **kwargs
                )
                for key, value in converted_tensors:
                    hf_state_dict[key] = value

        return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert HF checkpoint to internal format.

        - Rename backbone → model
        - Rename norm_f → norm
        - Aggregate per-expert weights into grouped tensors
        - If device_mesh is provided, only load experts needed for the current rank

        Args:
            hf_state_dict: HuggingFace format state dict
            device_mesh: Optional device mesh for distributed expert loading
            **kwargs: Additional arguments

        Returns:
            Internal format state dict
        """
        inplace = bool(kwargs.get("inplace", False))
        # Detect if HF checkpoint uses 'backbone' or 'model' prefix
        for key in hf_state_dict.keys():
            if ".mixer.experts." in key and key.endswith(".weight"):
                self._uses_model_prefix = not key.startswith("backbone.")
                break

        # First, rename backbone → model and norm_f → norm
        if inplace:
            renamed_state_dict = hf_state_dict
            for key in list(renamed_state_dict.keys()):
                new_key = key
                if new_key.startswith("backbone."):
                    new_key = "model." + new_key[len("backbone.") :]
                if new_key == "model.norm_f.weight":
                    new_key = "model.norm.weight"
                # HF uses 'embeddings' but internal uses 'embed_tokens'
                if new_key == "model.embeddings.weight":
                    new_key = "model.embed_tokens.weight"

                if new_key != key:
                    renamed_state_dict[new_key] = renamed_state_dict.pop(key)
        else:
            renamed_state_dict = {}
            for key, value in hf_state_dict.items():
                new_key = key
                if new_key.startswith("backbone."):
                    new_key = "model." + new_key[len("backbone.") :]
                if new_key == "model.norm_f.weight":
                    new_key = "model.norm.weight"
                # HF uses 'embeddings' but internal uses 'embed_tokens'
                if new_key == "model.embeddings.weight":
                    new_key = "model.embed_tokens.weight"

                renamed_state_dict[new_key] = value

        # Then merge experts using the mixin method
        return self._from_hf_w_merged_experts(renamed_state_dict, device_mesh, inplace=inplace)

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single tensor from internal format to HuggingFace format.

        Args:
            fqn: Fully qualified name of the tensor in internal format
            tensor: The tensor to convert
            **kwargs: Additional arguments for conversion

        Returns:
            List of (fqn, tensor) tuples in HuggingFace format
        """
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        # Try to convert merged expert weights to split experts
        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if expert_result is not None:
            result = expert_result
        else:
            # Standard conversion: just rename keys
            new_fqn = fqn

            # Rename model → backbone
            if new_fqn.startswith("model."):
                new_fqn = "backbone." + new_fqn[len("model.") :]

            # Rename norm → norm_f
            if new_fqn == "backbone.norm.weight":
                new_fqn = "backbone.norm_f.weight"

            # Internal uses 'embed_tokens' but HF uses 'embeddings'
            if new_fqn == "backbone.embed_tokens.weight":
                new_fqn = "backbone.embeddings.weight"

            result = [(new_fqn, tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        return result

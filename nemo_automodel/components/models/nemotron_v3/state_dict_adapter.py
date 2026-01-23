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
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin
from nemo_automodel.components.moe.utils import BackendConfig

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

    def to_hf(self, state_dict: dict[str, Any], exclude_key_regex: Optional[str] = None, **kwargs) -> dict[str, Any]:
        """Convert from internal model state dict to HuggingFace format.

        Args:
            state_dict: Internal format state dict
            exclude_key_regex: Optional regex pattern to exclude keys
            **kwargs: Additional arguments

        Returns:
            HuggingFace format state dict
        """
        hf_state_dict = {}
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
        # Detect if HF checkpoint uses 'backbone' or 'model' prefix
        for key in hf_state_dict.keys():
            if ".mixer.experts." in key and key.endswith(".weight"):
                self._uses_model_prefix = not key.startswith("backbone.")
                break

        # First, rename backbone → model and norm_f → norm
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

        # Then merge experts (override mixer → mlp for mixin compatibility)
        return self._from_hf_w_merged_experts_nemotron(renamed_state_dict, device_mesh)

    def _from_hf_w_merged_experts_nemotron(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
    ) -> dict[str, Any]:
        """NemotronV3-specific version that handles .mixer.experts. instead of .mlp.experts."""
        from nemo_automodel.components.moe.state_dict_utils import (
            create_dtensor_from_local,
            get_expert_range_for_rank_from_mesh,
            get_submesh,
            is_dtensor,
            should_load_expert_for_rank,
        )

        n_experts = self.moe_config.n_routed_experts

        if device_mesh is not None:
            start_expert, end_expert = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            expected_experts_per_rank = end_expert - start_expert
            rank = (
                get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )
        else:
            start_expert, end_expert = 0, n_experts
            expected_experts_per_rank = n_experts
            rank = None

        state_dict: dict[str, Any] = {}
        expert_weights_by_layer: dict[str, dict[str, dict[int, torch.Tensor]]] = {}

        for key, value in hf_state_dict.items():
            if ".mixer.experts." in key and key.endswith(".weight"):
                # Handle: model.layers.{L}.mixer.experts.{E}.up_proj.weight
                m = re.match(r"(?:model\.)?layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.weight", key)
                if m is None:
                    state_dict[key] = value
                    continue

                layer_num, expert_num, which = m.groups()
                expert_num = int(expert_num)

                if not should_load_expert_for_rank(expert_num, device_mesh, n_experts):
                    continue

                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}

                if which == "up_proj":
                    native_key = f"model.layers.{layer_num}.mixer.experts.gate_and_up_projs"
                else:  # down_proj
                    native_key = f"model.layers.{layer_num}.mixer.experts.down_projs"

                if native_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][native_key] = {}

                if which == "up_proj":
                    expert_weights_by_layer[layer_num][native_key][expert_num] = value

                    if len(expert_weights_by_layer[layer_num][native_key]) == expected_experts_per_rank:
                        expert_ids = sorted(expert_weights_by_layer[layer_num][native_key].keys())

                        up_tensors = []
                        for expert_id in expert_ids:
                            up_weight = expert_weights_by_layer[layer_num][native_key][expert_id]

                            if is_dtensor(up_weight):
                                up_weight = up_weight.to_local()

                            up_t = up_weight.transpose(0, 1)  # [dim, inter_dim]
                            up_tensors.append(up_t)

                        stacked = torch.stack(up_tensors, dim=0)  # [n_experts, dim, inter_dim]
                        stacked = stacked.to(self.dtype)

                        dtensor = create_dtensor_from_local(stacked, device_mesh, rank)
                        state_dict[native_key] = dtensor

                else:  # down_proj
                    expert_weights_by_layer[layer_num][native_key][expert_num] = value

                    if len(expert_weights_by_layer[layer_num][native_key]) == expected_experts_per_rank:
                        expert_ids = sorted(expert_weights_by_layer[layer_num][native_key].keys())

                        ordered = []
                        for expert_id in expert_ids:
                            down_weight = expert_weights_by_layer[layer_num][native_key][expert_id]  # [dim, inter_dim]

                            if is_dtensor(down_weight):
                                down_weight = down_weight.to_local()

                            down_t = down_weight.transpose(0, 1)  # [inter_dim, dim]
                            ordered.append(down_t)

                        stacked = torch.stack(ordered, dim=0)
                        stacked = stacked.to(self.dtype)

                        dtensor = create_dtensor_from_local(stacked, device_mesh, rank)
                        state_dict[native_key] = dtensor

            else:
                state_dict[key] = value

        return state_dict

    def _convert_single_merged_expert_to_hf_split_experts(
        self, fqn: str, tensor: torch.Tensor, **kwargs
    ) -> list[tuple[str, torch.Tensor]]:
        """Convert a single merged expert tensor to split HuggingFace format.

        Overrides parent to handle .mixer.experts. instead of .mlp.experts.

        Args:
            fqn: Fully qualified name of the tensor in internal format
            tensor: The tensor to convert

        Returns:
            List of (fqn, tensor) tuples in HuggingFace format, or None if not an expert tensor
        """
        n_experts = self.moe_config.n_routed_experts
        inter_dim = self.moe_config.moe_inter_dim
        # For to_hf conversion, we always want backbone. prefix
        prefix = "backbone."

        # Handle gate_and_up_projs (maps to HF up_proj for non-gated activations like ReLU²)
        if ".mixer.experts.gate_and_up_projs" in fqn and fqn.endswith(".gate_and_up_projs"):
            layer_num = re.search(r"layers\.(\d+)", fqn).group(1)

            from nemo_automodel.components.moe.state_dict_utils import (
                is_dtensor,
                validate_dtensor_expert_sharding,
            )

            if is_dtensor(tensor):
                validate_dtensor_expert_sharding(tensor, n_experts, f"gate_and_up_projs layer {layer_num}")

            splits = self._split_experts_weights(tensor, n_experts)
            result = []
            for i, w in enumerate(splits):
                expert_id = self._last_expert_ids[i]
                # w is [dim, inter_dim], HF expects [inter_dim, dim]
                w_up = w.transpose(0, 1).contiguous()
                result.append((f"{prefix}layers.{layer_num}.mixer.experts.{expert_id}.up_proj.weight", w_up))
            return result

        # Handle down_projs
        elif (
            ".mixer.experts.down_projs" in fqn
            and fqn.endswith(".down_projs")
            and tensor.ndim == 3
            and tensor.shape[1] == inter_dim
        ):
            layer_num = re.search(r"layers\.(\d+)", fqn).group(1)

            from nemo_automodel.components.moe.state_dict_utils import (
                is_dtensor,
                validate_dtensor_expert_sharding,
            )

            if is_dtensor(tensor):
                validate_dtensor_expert_sharding(tensor, n_experts, f"down_projs layer {layer_num}")

            splits = self._split_experts_weights(tensor, n_experts)
            result = []
            for i, w in enumerate(splits):
                expert_id = self._last_expert_ids[i]
                result.append(
                    (
                        f"{prefix}layers.{layer_num}.mixer.experts.{expert_id}.down_proj.weight",
                        w.transpose(0, 1).contiguous(),
                    )
                )
            return result

        return None

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

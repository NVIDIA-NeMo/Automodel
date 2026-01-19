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
from nemo_automodel.components.moe.state_dict_utils import (
    create_dtensor_from_local,
    get_expert_range_for_rank_from_mesh,
    get_submesh,
    is_dtensor,
    should_load_expert_for_rank,
    split_experts_weights_dtensor_aware,
)
from nemo_automodel.components.moe.utils import BackendConfig

logger = logging.getLogger(__name__)


class NemotronV3StateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """State dict adapter for NemotronV3 (Nemotron-Nano) models.

    Key differences from other MoE models:
    1. Uses mixer.experts.{E} key pattern (not mlp.experts)
    2. Uses relu2 activation (only up_proj and down_proj, NO gate_proj)
    3. Shared experts use different intermediate size
    """

    def __init__(
        self,
        config,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True

    def _validate_expert_availability_relu2(
        self,
        hf_state_dict: dict[str, Any],
        n_experts: int,
        device_mesh: Optional["DeviceMesh"] = None,
    ) -> None:
        """Validate that all required experts are available for relu2 format (no gate_proj)."""
        if device_mesh is not None:
            start_expert, end_expert = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            required_experts = list(range(start_expert, end_expert))
            rank = (
                get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )
            rank_info = f" (rank {rank})"
        else:
            required_experts = list(range(n_experts))
            rank_info = ""

        # Detect key prefix pattern
        uses_model_prefix = any(key.startswith("model.") for key in hf_state_dict.keys() if ".mixer.experts." in key)
        key_prefix = "model." if uses_model_prefix else ""

        layers_with_experts = set()
        # Match pattern: layers.{L}.mixer.experts.{E}.(up_proj|down_proj).weight
        pattern = rf"{re.escape(key_prefix)}layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.weight"
        for key in hf_state_dict.keys():
            match = re.match(pattern, key)
            if match:
                layer_num = int(match.group(1))
                layers_with_experts.add(layer_num)

        if not layers_with_experts:
            return

        missing_weights = []
        # relu2: only up_proj and down_proj (no gate_proj)
        projection_types = ["up_proj", "down_proj"]

        for layer_num in layers_with_experts:
            for expert_id in required_experts:
                for proj_type in projection_types:
                    expected_key = f"{key_prefix}layers.{layer_num}.mixer.experts.{expert_id}.{proj_type}.weight"
                    if expected_key not in hf_state_dict:
                        missing_weights.append(expected_key)

        if missing_weights:
            missing_count = len(missing_weights)
            total_required = len(required_experts) * len(layers_with_experts) * len(projection_types)
            raise RuntimeError(
                f"Expert weights missing from checkpoint{rank_info}: {missing_count}/{total_required} required weights not found. "
                f"Cannot load experts - checkpoint may be incomplete or corrupted. "
                f"Layers with experts: {sorted(layers_with_experts)}, Required experts: {required_experts}. "
                f"First few missing keys: {missing_weights[:5]}"
                + (f" (and {missing_count - 5} more)" if missing_count > 5 else "")
            )

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert from native model state dict to HuggingFace format."""
        hf_state_dict = {}
        for fqn, tensor in state_dict.items():
            converted_tensors = self._convert_single_tensor_to_hf_relu2(
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
        """Convert HF checkpoint to native format for relu2 activation.

        For relu2 activation:
        - Only up_proj and down_proj (no gate_proj)
        - up_projs shape: [n_experts, dim, inter_dim]
        - down_projs shape: [n_experts, inter_dim, dim]
        """
        # Detect key prefix
        for key in hf_state_dict.keys():
            if ".mixer.experts." in key and key.endswith(".weight"):
                self._uses_model_prefix = key.startswith("model.")
                break

        return self._from_hf_w_merged_experts_relu2(hf_state_dict, device_mesh)

    def _from_hf_w_merged_experts_relu2(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
    ) -> dict[str, Any]:
        """Convert HF checkpoint to native format for relu2 (no gate_proj).

        Creates:
        - up_projs: [n_experts, dim, inter_dim]
        - down_projs: [n_experts, inter_dim, dim]
        """
        n_experts = self.moe_config.n_routed_experts

        self._validate_expert_availability_relu2(hf_state_dict, n_experts, device_mesh)

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
            # Match: (model.)?layers.{L}.mixer.experts.{E}.(up_proj|down_proj).weight
            if ".mixer.experts." in key and key.endswith(".weight"):
                m = re.match(
                    r"(?:model\.)?layers\.(\d+)\.mixer\.experts\.(\d+)\.(up_proj|down_proj)\.weight", key
                )
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
                    native_key = f"model.layers.{layer_num}.mlp.experts.up_projs"
                else:  # down_proj
                    native_key = f"model.layers.{layer_num}.mlp.experts.down_projs"

                if native_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][native_key] = {}

                expert_weights_by_layer[layer_num][native_key][expert_num] = value

                # Check if all experts for this key are collected
                if len(expert_weights_by_layer[layer_num][native_key]) == expected_experts_per_rank:
                    expert_ids = sorted(expert_weights_by_layer[layer_num][native_key].keys())

                    ordered = []
                    for expert_id in expert_ids:
                        weight = expert_weights_by_layer[layer_num][native_key][expert_id]

                        # Extract local tensor if input is already a DTensor
                        if is_dtensor(weight):
                            weight = weight.to_local()

                        # Transpose: [inter_dim, dim] -> [dim, inter_dim] for up_proj
                        # or [dim, inter_dim] -> [inter_dim, dim] for down_proj
                        weight_t = weight.transpose(0, 1)
                        ordered.append(weight_t)

                    stacked = torch.stack(ordered, dim=0)
                    stacked = stacked.to(self.dtype)

                    dtensor = create_dtensor_from_local(stacked, device_mesh, rank)
                    state_dict[native_key] = dtensor

            else:
                if not key.endswith("_scale_inv"):
                    state_dict[key] = value

        return state_dict

    def _convert_single_tensor_to_hf_relu2(
        self, fqn: str, tensor: torch.Tensor, **kwargs
    ) -> list[tuple[str, torch.Tensor]]:
        """Convert a single tensor from native format to HuggingFace format for relu2.

        Handles:
        - up_projs -> individual expert up_proj weights
        - down_projs -> individual expert down_proj weights
        """
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        if exclude_key_regex and re.match(exclude_key_regex, fqn):
            return []

        n_experts = self.moe_config.n_routed_experts
        inter_dim = self.moe_config.moe_inter_dim
        prefix = "model." if self._uses_model_prefix else ""

        # Handle up_projs: [n_experts, dim, inter_dim] -> individual up_proj weights
        if ".mlp.experts.up_projs" in fqn and fqn.endswith(".up_projs"):
            layer_num = re.search(r"layers\.(\d+)", fqn).group(1)

            if is_dtensor(tensor):
                from nemo_automodel.components.moe.state_dict_utils import validate_dtensor_expert_sharding
                validate_dtensor_expert_sharding(tensor, n_experts, f"up_projs layer {layer_num}")

            splits = self._split_experts_weights(tensor, n_experts)
            result = []
            for i, w in enumerate(splits):
                expert_id = self._last_expert_ids[i]
                # Transpose back: [dim, inter_dim] -> [inter_dim, dim]
                w_transposed = w.transpose(0, 1).contiguous()
                result.append((f"{prefix}layers.{layer_num}.mixer.experts.{expert_id}.up_proj.weight", w_transposed))
            return result

        # Handle down_projs: [n_experts, inter_dim, dim] -> individual down_proj weights
        elif ".mlp.experts.down_projs" in fqn and fqn.endswith(".down_projs"):
            layer_num = re.search(r"layers\.(\d+)", fqn).group(1)

            if is_dtensor(tensor):
                from nemo_automodel.components.moe.state_dict_utils import validate_dtensor_expert_sharding
                validate_dtensor_expert_sharding(tensor, n_experts, f"down_projs layer {layer_num}")

            splits = self._split_experts_weights(tensor, n_experts)
            result = []
            for i, w in enumerate(splits):
                expert_id = self._last_expert_ids[i]
                # Transpose back: [inter_dim, dim] -> [dim, inter_dim]
                w_transposed = w.transpose(0, 1).contiguous()
                result.append((f"{prefix}layers.{layer_num}.mixer.experts.{expert_id}.down_proj.weight", w_transposed))
            return result

        return [(fqn, tensor)]

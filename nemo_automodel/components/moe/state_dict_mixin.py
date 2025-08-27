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

from nemo_automodel.components.moe.state_dict_utils import (
    create_dtensor_from_local,
    get_expert_range_for_rank_from_mesh,
    is_dtensor,
    should_load_expert_for_rank,
    split_experts_weights_dtensor_aware,
    validate_dtensor_expert_sharding,
)


class MoEStateDictMixin:
    """Mixin class providing MoE state dict conversion utilities.

    This mixin provides methods for:
    - Expert parallelism calculations (ranges, assignment)
    - Format conversion between HuggingFace and native formats
    - Both GroupedExperts and DeepEP format support
    - DTensor-aware expert loading and conversion

    Can be used by any MoE model that needs expert parallelism and format conversion.
    """

    # These attributes must be set by subclasses in their __init__ method:
    # - self.moe_config: MoE configuration object with expert settings
    # - self.config: Model configuration object
    # - self.backend: Backend configuration object

    def _validate_expert_availability(
        self,
        hf_state_dict: dict[str, Any],
        n_experts: int,
        device_mesh: Optional["DeviceMesh"] = None,
    ) -> None:
        """Validate that all required experts are available in the HF state dict before loading.
        Only validates experts needed for the current rank and layers present in the state dict.

        Args:
            hf_state_dict: HuggingFace format state dict
            n_experts: Total number of experts
            device_mesh: Optional device mesh for expert parallelism

        Raises:
            RuntimeError: If required expert weights are missing from the checkpoint
        """
        if device_mesh is not None:
            start_expert, end_expert = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            required_experts = list(range(start_expert, end_expert))
            rank = device_mesh["ep"].get_rank() if "ep" in device_mesh.mesh_dim_names else device_mesh.get_rank()
            rank_info = f" (rank {rank})"
        else:
            required_experts = list(range(n_experts))
            rank_info = ""

        uses_model_prefix = any(key.startswith("model.") for key in hf_state_dict.keys() if ".mlp.experts." in key)
        key_prefix = "model." if uses_model_prefix else ""

        layers_with_experts = set()
        pattern = rf"{re.escape(key_prefix)}layers\.(\d+)\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight"
        for key in hf_state_dict.keys():
            match = re.match(pattern, key)
            if match:
                layer_num = int(match.group(1))
                layers_with_experts.add(layer_num)

        if not layers_with_experts:
            return

        missing_weights = []
        projection_types = ["gate_proj", "up_proj", "down_proj"]

        for layer_num in layers_with_experts:
            for expert_id in required_experts:
                for proj_type in projection_types:
                    expected_key = f"{key_prefix}layers.{layer_num}.mlp.experts.{expert_id}.{proj_type}.weight"
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

    def _split_experts_weights(self, weight: torch.Tensor, n_experts: int) -> list[torch.Tensor]:
        """Split grouped expert weights into individual expert weights.
        For grouped expert weights with shape [n_experts, ...], split into n_experts tensors each with shape [...].
        Supports both regular tensors and DTensors.
        """
        if is_dtensor(weight):
            split_weights, expert_ids = split_experts_weights_dtensor_aware(weight, n_experts)
            self._last_expert_ids = expert_ids
            return split_weights
        else:
            if weight.shape[0] != n_experts:
                raise ValueError(f"Expected first dimension to be {n_experts}, got {weight.shape[0]}")

            split_weights = []
            expert_ids = []
            for i in range(n_experts):
                expert_weight = weight[i]  # Shape: [...] (expert dimension removed)
                split_weights.append(expert_weight)
                expert_ids.append(i)

            self._last_expert_ids = expert_ids
            return split_weights

    def _concatenate_expert_weights(
        self, expert_weights_by_layer: dict[str, Any], n_experts: int
    ) -> Optional[torch.Tensor]:
        """Concatenate the weights of separate experts into GroupedExpert weights.

        Args:
            expert_weights_by_layer: Nested dict structure containing expert weights
            n_experts: Total number of experts expected

        Returns:
            Stacked tensor if all experts are available for a layer, None otherwise
        """
        for layer, abstract_keys in list(expert_weights_by_layer.items()):
            for abstract_key, experts in list(abstract_keys.items()):
                if len(experts) == n_experts:
                    sorted_expert_ids = sorted(experts.keys())
                    sorted_experts = [experts[i] for i in sorted_expert_ids]
                    stacked_tensor = torch.stack(sorted_experts, dim=0)

                    del expert_weights_by_layer[layer][abstract_key]
                    if not expert_weights_by_layer[layer]:
                        del expert_weights_by_layer[layer]

                    return stacked_tensor

        return None

    def _to_hf_grouped_experts(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert GroupedExperts format to HuggingFace format.
        Handles: gate_projs, up_projs, down_projs -> individual expert weights
        """

        n_experts = self.moe_config.n_routed_experts
        hf_state_dict: dict[str, Any] = {}

        for key, value in list(state_dict.items()):
            if ".mlp.experts.gate_projs" in key and key.endswith(".gate_projs"):
                # GroupedExperts: [n_experts, inter_dim, dim] -> per-expert gate_proj [inter_dim, dim]
                layer_num = re.search(r"layers\.(\d+)", key).group(1)

                if is_dtensor(value):
                    validate_dtensor_expert_sharding(value, n_experts, f"gate_projs layer {layer_num}")

                splits = self._split_experts_weights(value, n_experts)
                for i, w in enumerate(splits):
                    expert_id = self._last_expert_ids[i]
                    prefix = "model." if self._uses_model_prefix else ""
                    new_key = f"{prefix}layers.{layer_num}.mlp.experts.{expert_id}.gate_proj.weight"
                    hf_state_dict[new_key] = w
                continue

            if ".mlp.experts.up_projs" in key and key.endswith(".up_projs"):
                # GroupedExperts: [n_experts, inter_dim, dim]
                layer_num = re.search(r"layers\.(\d+)", key).group(1)

                if is_dtensor(value):
                    validate_dtensor_expert_sharding(value, n_experts, f"up_projs layer {layer_num}")

                splits = self._split_experts_weights(value, n_experts)
                for i, w in enumerate(splits):
                    expert_id = self._last_expert_ids[i]
                    prefix = "model." if self._uses_model_prefix else ""
                    new_key = f"{prefix}layers.{layer_num}.mlp.experts.{expert_id}.up_proj.weight"
                    hf_state_dict[new_key] = w
                continue

            if ".mlp.experts.down_projs" in key and key.endswith(".down_projs") and value.ndim == 3:
                # GroupedExperts: [n_experts, dim, inter_dim] -> per-expert down_proj [dim, inter_dim]
                layer_num = re.search(r"layers\.(\d+)", key).group(1)

                if is_dtensor(value):
                    validate_dtensor_expert_sharding(value, n_experts, f"down_projs layer {layer_num}")

                splits = self._split_experts_weights(value, n_experts)
                for i, w in enumerate(splits):
                    expert_id = self._last_expert_ids[i]
                    prefix = "model." if self._uses_model_prefix else ""
                    new_key = f"{prefix}layers.{layer_num}.mlp.experts.{expert_id}.down_proj.weight"
                    hf_state_dict[new_key] = w
                continue

        for key, value in state_dict.items():
            if ".mlp.experts." in key and (
                key.endswith(".gate_projs") or key.endswith(".up_projs") or key.endswith(".down_projs")
            ):
                continue
            hf_state_dict[key] = value

        return hf_state_dict

    def _to_hf_deepep(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert DeepEP format to HuggingFace format.
        Handles: gate_and_up_projs, down_projs -> individual expert weights
        """

        n_experts = self.moe_config.n_routed_experts
        inter_dim = self.moe_config.moe_inter_dim
        hf_state_dict: dict[str, Any] = {}

        for key, value in list(state_dict.items()):
            if ".mlp.experts.gate_and_up_projs" in key and key.endswith(".gate_and_up_projs"):
                # DeepEP: [n_experts, dim, 2*inter_dim] -> per-expert gate_proj/up_proj [inter_dim, dim]
                layer_num = re.search(r"layers\.(\d+)", key).group(1)

                if is_dtensor(value):
                    validate_dtensor_expert_sharding(value, n_experts, f"gate_and_up_projs layer {layer_num}")

                splits = self._split_experts_weights(value, n_experts)
                for i, w in enumerate(splits):
                    expert_id = self._last_expert_ids[i]
                    # Split concat along last dim and transpose to [inter_dim, dim]
                    w_gate = w[:, :inter_dim].transpose(0, 1).contiguous()
                    w_up = w[:, inter_dim:].transpose(0, 1).contiguous()
                    prefix = "model." if self._uses_model_prefix else ""
                    hf_state_dict[f"{prefix}layers.{layer_num}.mlp.experts.{expert_id}.gate_proj.weight"] = w_gate
                    hf_state_dict[f"{prefix}layers.{layer_num}.mlp.experts.{expert_id}.up_proj.weight"] = w_up
                continue

            if (
                ".mlp.experts.down_projs" in key
                and key.endswith(".down_projs")
                and value.ndim == 3
                and value.shape[1] == inter_dim
            ):
                # DeepEP down: [n_experts, inter_dim, dim] -> per-expert down_proj [dim, inter_dim]
                layer_num = re.search(r"layers\.(\d+)", key).group(1)

                if is_dtensor(value):
                    validate_dtensor_expert_sharding(value, n_experts, f"down_projs (DeepEP) layer {layer_num}")

                splits = self._split_experts_weights(value, n_experts)
                for i, w in enumerate(splits):
                    expert_id = self._last_expert_ids[i]
                    prefix = "model." if self._uses_model_prefix else ""
                    hf_state_dict[f"{prefix}layers.{layer_num}.mlp.experts.{expert_id}.down_proj.weight"] = w.transpose(
                        0, 1
                    ).contiguous()
                continue

        for key, value in state_dict.items():
            if ".mlp.experts." in key and (key.endswith(".gate_and_up_projs") or key.endswith(".down_projs")):
                continue
            hf_state_dict[key] = value

        return hf_state_dict

    def _from_hf_grouped_experts(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
    ) -> dict[str, Any]:
        """Convert HF checkpoint to GroupedExperts format.
        Creates separate gate_projs, up_projs, and down_projs tensors.
        """

        n_experts = self.moe_config.n_routed_experts

        self._validate_expert_availability(hf_state_dict, n_experts, device_mesh)

        if device_mesh is not None:
            start_expert, end_expert = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            expected_experts_per_rank = end_expert - start_expert
            rank = device_mesh["ep"].get_rank() if "ep" in device_mesh.mesh_dim_names else device_mesh.get_rank()
        else:
            start_expert, end_expert = 0, n_experts
            expected_experts_per_rank = n_experts
            rank = None

        state_dict: dict[str, Any] = {}
        expert_weights_by_layer: dict[str, dict[str, dict[int, torch.Tensor]]] = {}

        for key, value in hf_state_dict.items():
            if ".mlp.experts." in key and key.endswith(".weight"):
                # Handle both formats:
                # - model.layers.{L}.mlp.experts.{E}.gate_proj.weight (with model prefix)
                # - layers.{L}.mlp.experts.{E}.gate_proj.weight (without model prefix)
                m = re.match(
                    r"(?:model\.)?layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", key
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

                abstract_key = {
                    "gate_proj": "model.layers.{}.mlp.experts.gate_projs",
                    "up_proj": "model.layers.{}.mlp.experts.up_projs",
                    "down_proj": "model.layers.{}.mlp.experts.down_projs",
                }[which]

                native_key = abstract_key.format(layer_num)

                if native_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][native_key] = {}

                expert_weights_by_layer[layer_num][native_key][expert_num] = value

                if len(expert_weights_by_layer[layer_num][native_key]) == expected_experts_per_rank:
                    expert_ids = sorted(expert_weights_by_layer[layer_num][native_key].keys())
                    ordered = [expert_weights_by_layer[layer_num][native_key][i] for i in expert_ids]

                    # Shapes:
                    # gate/up: [inter_dim, dim] -> stacked [n_experts_this_rank, inter_dim, dim]
                    # down:    [dim, inter_dim] -> stacked [n_experts_this_rank, dim, inter_dim]
                    stacked = torch.stack(ordered, dim=0)
                    stacked = stacked.to(self.dtype)

                    dtensor = create_dtensor_from_local(stacked, device_mesh, rank)
                    state_dict[native_key] = dtensor

            else:
                if not key.endswith("_scale_inv"):
                    state_dict[key] = value

        return state_dict

    def _from_hf_deepep(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
    ) -> dict[str, Any]:
        """Convert HF checkpoint to DeepEP format.
        Creates combined gate_and_up_projs and transposed down_projs tensors.
        """

        n_experts = self.moe_config.n_routed_experts

        self._validate_expert_availability(hf_state_dict, n_experts, device_mesh)

        if device_mesh is not None:
            start_expert, end_expert = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            expected_experts_per_rank = end_expert - start_expert
            rank = device_mesh["ep"].get_rank() if "ep" in device_mesh.mesh_dim_names else device_mesh.get_rank()
        else:
            start_expert, end_expert = 0, n_experts
            expected_experts_per_rank = n_experts
            rank = None

        state_dict: dict[str, Any] = {}
        expert_weights_by_layer: dict[str, dict[str, dict[int, torch.Tensor]]] = {}

        for key, value in hf_state_dict.items():
            if ".mlp.experts." in key and key.endswith(".weight"):
                # Handle both formats:
                # - model.layers.{L}.mlp.experts.{E}.gate_proj.weight (with model prefix)
                # - layers.{L}.mlp.experts.{E}.gate_proj.weight (without model prefix)
                m = re.match(
                    r"(?:model\.)?layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", key
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

                if which in ["gate_proj", "up_proj"]:
                    native_key = f"model.layers.{layer_num}.mlp.experts.gate_and_up_projs"
                else:  # down_proj
                    native_key = f"model.layers.{layer_num}.mlp.experts.down_projs"

                if native_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][native_key] = {}

                if which in ["gate_proj", "up_proj"]:
                    if expert_num not in expert_weights_by_layer[layer_num][native_key]:
                        expert_weights_by_layer[layer_num][native_key][expert_num] = {}
                    expert_weights_by_layer[layer_num][native_key][expert_num][which] = value

                    if len(expert_weights_by_layer[layer_num][native_key]) == expected_experts_per_rank and all(
                        isinstance(expert_data, dict) and "gate_proj" in expert_data and "up_proj" in expert_data
                        for expert_data in expert_weights_by_layer[layer_num][native_key].values()
                    ):
                        expert_ids = sorted(expert_weights_by_layer[layer_num][native_key].keys())

                        combined_tensors = []
                        for expert_id in expert_ids:
                            expert_data = expert_weights_by_layer[layer_num][native_key][expert_id]
                            gate_weight = expert_data["gate_proj"]  # [inter_dim, dim]
                            up_weight = expert_data["up_proj"]  # [inter_dim, dim]

                            gate_t = gate_weight.transpose(0, 1)  # [dim, inter_dim]
                            up_t = up_weight.transpose(0, 1)  # [dim, inter_dim]
                            combined = torch.cat([gate_t, up_t], dim=-1)  # [dim, 2*inter_dim]
                            combined_tensors.append(combined)

                        stacked = torch.stack(combined_tensors, dim=0)
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
                            down_t = down_weight.transpose(0, 1)  # [inter_dim, dim]
                            ordered.append(down_t)

                        stacked = torch.stack(ordered, dim=0)
                        stacked = stacked.to(self.dtype)

                        dtensor = create_dtensor_from_local(stacked, device_mesh, rank)
                        state_dict[native_key] = dtensor

            else:
                if not key.endswith("_scale_inv"):
                    state_dict[key] = value

        return state_dict

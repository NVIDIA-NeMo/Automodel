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
    get_submesh,
    is_dtensor,
    should_load_expert_for_rank,
    split_experts_weights_dtensor_aware,
)


class MoESplitExpertsStateDictMixin:
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

    @property
    def _is_gated_moe(self) -> bool:
        """Check if the MoE uses gated activation (e.g., SwiGLU) or non-gated (e.g., ReLU²)."""
        from nemo_automodel.components.moe.layers import is_gated_activation

        return is_gated_activation(self.moe_config.expert_activation)

    @property
    def _hf_prefix(self) -> str:
        """Prefix for HuggingFace format keys. Override in subclass."""
        return "model." if self._uses_model_prefix else ""

    @property
    def _expert_path_segment(self) -> str:
        """Path segment for experts (e.g., 'mlp.experts' or 'mixer.experts'). Override in subclass."""
        return "mlp.experts"

    def _get_expert_hf_key_pattern(self) -> re.Pattern[str]:
        """Single regex for HF expert keys; used by bulk conversion and lazy key mapping."""
        return re.compile(
            rf"(?P<prefix>(?:model\.)?(?:language_model\.)?)layers\.(?P<layer_num>\d+)\.{re.escape(self._expert_path_segment)}\.(?P<expert_id>\d+)\.(?P<proj_type>gate_proj|up_proj|down_proj)\.weight"
        )

    def _parse_native_expert_key(self, fqn: str) -> Optional[tuple[str, str]]:
        """If fqn is an expert merged key, return (layer_num, key_type) where key_type in ('gate_and_up_projs', 'down_projs'). Else None."""
        seg = self._expert_path_segment
        if f".{seg}.gate_and_up_projs" in fqn and fqn.endswith(".gate_and_up_projs"):
            m = re.search(r"layers\.(\d+)", fqn)
            return (m.group(1), "gate_and_up_projs") if m else None
        if f".{seg}.down_projs" in fqn and fqn.endswith(".down_projs"):
            m = re.search(r"layers\.(\d+)", fqn)
            return (m.group(1), "down_projs") if m else None
        return None

    def _build_hf_key(self, layer_num: str, expert_id: int, proj_type: str, prefix: Optional[str] = None) -> str:
        """Build one HF expert key. proj_type in ('gate_proj', 'up_proj', 'down_proj').
        If prefix is None, uses self._hf_prefix (for to_hf); else uses prefix (for from_hf lookups).
        """
        p = self._hf_prefix if prefix is None else prefix
        return f"{p}layers.{layer_num}.{self._expert_path_segment}.{expert_id}.{proj_type}.weight"

    def _build_native_key(self, prefix: str, layer_num: str, proj_type: str) -> str:
        """Build native merged key. proj_type in ('gate_and_up_projs', 'down_projs')."""
        return f"{prefix}layers.{layer_num}.{self._expert_path_segment}.{proj_type}"

    def _parse_hf_expert_key(self, hf_key: str) -> Optional[tuple[str, str, int, str]]:
        """If hf_key is an expert HF key, return (prefix, layer_num, expert_id, proj_type). Else None."""
        m = self._get_expert_hf_key_pattern().match(hf_key)
        if m is None:
            return None
        prefix = m.group("prefix") or ""
        layer_num = m.group("layer_num")
        expert_id = int(m.group("expert_id"))
        proj_type = m.group("proj_type")
        return (prefix, layer_num, expert_id, proj_type)

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
            rank = (
                get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )
            rank_info = f" (rank {rank})"
        else:
            required_experts = list(range(n_experts))
            rank_info = ""

        expert_segment = self._expert_path_segment

        # Detect actual prefix from keys (handles both HF format and pre-renamed internal format)
        key_prefix = ""
        for key in hf_state_dict.keys():
            if f".{expert_segment}." in key and "layers." in key:
                key_prefix = key[: key.index("layers.")]
                break

        # Build list of all possible prefixes
        prefixes = ["model.language_model.", "model.", "language_model.", ""]
        if key_prefix and key_prefix not in prefixes:
            prefixes.insert(0, key_prefix)

        layers_with_experts: dict[int, set[str]] = {}
        # Create pattern with all prefixes
        escaped_prefixes = [re.escape(p) for p in prefixes]
        prefix_pattern = "(?P<prefix>" + "|".join(escaped_prefixes) + ")"
        pattern = (
            rf"{prefix_pattern}layers\.(\d+)\.{re.escape(expert_segment)}\.\d+\.(gate_proj|up_proj|down_proj)\.weight"
        )
        for key in hf_state_dict.keys():
            match = re.match(pattern, key)
            if match:
                prefix = match.group("prefix") or ""
                layer_num = int(match.group(2))
                layers_with_experts.setdefault(layer_num, set()).add(prefix)

        if not layers_with_experts:
            return

        missing_weights = []
        projection_types = ["gate_proj", "up_proj", "down_proj"] if self._is_gated_moe else ["up_proj", "down_proj"]

        for layer_num, prefixes in layers_with_experts.items():
            for prefix in prefixes:
                for expert_id in required_experts:
                    for proj_type in projection_types:
                        expected_key = f"{prefix}layers.{layer_num}.{expert_segment}.{expert_id}.{proj_type}.weight"
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

    def _to_hf_w_split_experts(self, state_dict: dict[str, Any], inplace: bool = False) -> dict[str, Any]:
        """Convert DeepEP format to HuggingFace format.
        Handles: gate_and_up_projs, down_projs -> individual expert weights
        """
        hf_state_dict: dict[str, Any] = {}

        if inplace:
            # Iterate keys only (not items) so we don't keep references to all tensors in a list,
            # which would defeat the purpose of inplace memory savings.
            for fqn in list(state_dict.keys()):
                tensor = state_dict.get(fqn, None)
                if tensor is None:
                    continue
                converted = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor)
                if converted is not None:
                    for key, value in converted:
                        hf_state_dict[key] = value
                    # Drop the merged expert tensor key eagerly so large tensors can be freed
                    # earlier during conversion when possible.
                    state_dict.pop(fqn, None)
                else:
                    hf_state_dict[fqn] = tensor
        else:
            for fqn, tensor in state_dict.items():
                converted = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor)
                if converted is not None:
                    for key, value in converted:
                        hf_state_dict[key] = value
                else:
                    hf_state_dict[fqn] = tensor

        return hf_state_dict

    def _from_hf_w_merged_experts(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        inplace: bool = False,
    ) -> dict[str, Any]:
        """Convert HF checkpoint to native format.

        For gated activations (SwiGLU, Quick-GEGLU):
            Creates combined gate_and_up_projs [n_experts, dim, 2*inter_dim] and
            transposed down_projs tensors.

        For non-gated activations (ReLU²):
            Creates gate_and_up_projs [n_experts, dim, inter_dim] and transposed down_projs tensors.
        """

        n_experts = self.moe_config.n_routed_experts
        is_gated = self._is_gated_moe
        expert_segment = self._expert_path_segment

        self._validate_expert_availability(hf_state_dict, n_experts, device_mesh)

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

        expert_pattern = self._get_expert_hf_key_pattern()

        if inplace:
            # Iterate keys only (not items) so we don't keep references to all tensors in a list.
            keys = list(hf_state_dict.keys())
            for key in keys:
                if key not in hf_state_dict:
                    continue
                value = hf_state_dict.pop(key)
                if f".{expert_segment}." in key and key.endswith(".weight"):
                    m = expert_pattern.match(key)
                    if m is None:
                        state_dict[key] = value
                        continue

                    prefix = m.group("prefix") or ""
                    layer_num = m.group("layer_num")
                    expert_num = int(m.group("expert_id"))
                    which = m.group("proj_type")

                    if not should_load_expert_for_rank(expert_num, device_mesh, n_experts):
                        del value
                        continue

                    if layer_num not in expert_weights_by_layer:
                        expert_weights_by_layer[layer_num] = {}

                    if which in ["gate_proj", "up_proj"]:
                        native_key = self._build_native_key(prefix, layer_num, "gate_and_up_projs")
                    else:  # down_proj
                        native_key = self._build_native_key(prefix, layer_num, "down_projs")

                    if native_key not in expert_weights_by_layer[layer_num]:
                        expert_weights_by_layer[layer_num][native_key] = {}

                    if which in ["gate_proj", "up_proj"]:
                        # Non-gated models only use up_proj, skip gate_proj
                        if not is_gated and which == "gate_proj":
                            del value
                            continue

                        # Store weight: gated uses dict for gate+up, non-gated stores tensor directly
                        if is_gated:
                            if expert_num not in expert_weights_by_layer[layer_num][native_key]:
                                expert_weights_by_layer[layer_num][native_key][expert_num] = {}
                            expert_weights_by_layer[layer_num][native_key][expert_num][which] = value
                        else:
                            expert_weights_by_layer[layer_num][native_key][expert_num] = value

                        # Check if all experts are complete
                        all_complete = len(expert_weights_by_layer[layer_num][native_key]) == expected_experts_per_rank
                        if is_gated:
                            all_complete = all_complete and all(
                                isinstance(d, dict) and "gate_proj" in d and "up_proj" in d
                                for d in expert_weights_by_layer[layer_num][native_key].values()
                            )

                        if all_complete:
                            expert_ids = sorted(expert_weights_by_layer[layer_num][native_key].keys())
                            tensors = []
                            for expert_id in expert_ids:
                                expert_data = expert_weights_by_layer[layer_num][native_key][expert_id]

                                if is_gated:
                                    gate_weight = expert_data["gate_proj"]
                                    up_weight = expert_data["up_proj"]
                                    if is_dtensor(gate_weight):
                                        gate_weight = gate_weight.to_local()
                                    if is_dtensor(up_weight):
                                        up_weight = up_weight.to_local()
                                    gate_t = gate_weight.transpose(0, 1)
                                    up_t = up_weight.transpose(0, 1)
                                    tensors.append(torch.cat([gate_t, up_t], dim=-1))
                                else:
                                    up_weight = expert_data
                                    if is_dtensor(up_weight):
                                        up_weight = up_weight.to_local()
                                    tensors.append(up_weight.transpose(0, 1))

                            stacked = torch.stack(tensors, dim=0).to(self.dtype)
                            state_dict[native_key] = create_dtensor_from_local(stacked, device_mesh, rank)
                            # Drop references to per-expert tensors as soon as we've merged this group.
                            expert_weights_by_layer[layer_num].pop(native_key, None)
                            if not expert_weights_by_layer[layer_num]:
                                expert_weights_by_layer.pop(layer_num, None)
                            del tensors, stacked

                    else:  # down_proj
                        expert_weights_by_layer[layer_num][native_key][expert_num] = value

                        if len(expert_weights_by_layer[layer_num][native_key]) == expected_experts_per_rank:
                            expert_ids = sorted(expert_weights_by_layer[layer_num][native_key].keys())

                            ordered = []
                            for expert_id in expert_ids:
                                down_weight = expert_weights_by_layer[layer_num][native_key][expert_id]  # [dim, inter_dim]

                                # Extract local tensor if input is already a DTensor
                                if is_dtensor(down_weight):
                                    down_weight = down_weight.to_local()

                                down_t = down_weight.transpose(0, 1)  # [inter_dim, dim]
                                ordered.append(down_t)

                            stacked = torch.stack(ordered, dim=0)
                            stacked = stacked.to(self.dtype)

                            dtensor = create_dtensor_from_local(stacked, device_mesh, rank)
                            state_dict[native_key] = dtensor
                            # Drop references to per-expert tensors as soon as we've merged this group.
                            expert_weights_by_layer[layer_num].pop(native_key, None)
                            if not expert_weights_by_layer[layer_num]:
                                expert_weights_by_layer.pop(layer_num, None)
                            del ordered, stacked, dtensor

                else:
                    if not key.endswith("_scale_inv"):
                        state_dict[key] = value

            return state_dict

        expert_pattern = self._get_expert_hf_key_pattern()
        for key, value in hf_state_dict.items():
            if f".{expert_segment}." in key and key.endswith(".weight"):
                m = expert_pattern.match(key)
                if m is None:
                    state_dict[key] = value
                    continue

                prefix = m.group("prefix") or ""
                layer_num = m.group("layer_num")
                expert_num = int(m.group("expert_id"))
                which = m.group("proj_type")

                if not should_load_expert_for_rank(expert_num, device_mesh, n_experts):
                    del value
                    continue

                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}

                if which in ["gate_proj", "up_proj"]:
                    native_key = self._build_native_key(prefix, layer_num, "gate_and_up_projs")
                else:  # down_proj
                    native_key = self._build_native_key(prefix, layer_num, "down_projs")

                if native_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][native_key] = {}

                if which in ["gate_proj", "up_proj"]:
                    # Non-gated models only use up_proj, skip gate_proj
                    if not is_gated and which == "gate_proj":
                        del value
                        continue

                    # Store weight: gated uses dict for gate+up, non-gated stores tensor directly
                    if is_gated:
                        if expert_num not in expert_weights_by_layer[layer_num][native_key]:
                            expert_weights_by_layer[layer_num][native_key][expert_num] = {}
                        expert_weights_by_layer[layer_num][native_key][expert_num][which] = value
                    else:
                        expert_weights_by_layer[layer_num][native_key][expert_num] = value
                    # Check if all experts are complete
                    all_complete = len(expert_weights_by_layer[layer_num][native_key]) == expected_experts_per_rank
                    if is_gated:
                        all_complete = all_complete and all(
                            isinstance(d, dict) and "gate_proj" in d and "up_proj" in d
                            for d in expert_weights_by_layer[layer_num][native_key].values()
                        )

                    if all_complete:
                        expert_ids = sorted(expert_weights_by_layer[layer_num][native_key].keys())
                        tensors = []
                        for expert_id in expert_ids:
                            expert_data = expert_weights_by_layer[layer_num][native_key][expert_id]

                            if is_gated:
                                gate_weight = expert_data["gate_proj"]
                                up_weight = expert_data["up_proj"]
                                if is_dtensor(gate_weight):
                                    gate_weight = gate_weight.to_local()
                                if is_dtensor(up_weight):
                                    up_weight = up_weight.to_local()
                                gate_t = gate_weight.transpose(0, 1)
                                up_t = up_weight.transpose(0, 1)
                                tensors.append(torch.cat([gate_t, up_t], dim=-1))
                            else:
                                up_weight = expert_data
                                if is_dtensor(up_weight):
                                    up_weight = up_weight.to_local()
                                tensors.append(up_weight.transpose(0, 1))

                        stacked = torch.stack(tensors, dim=0).to(self.dtype)
                        state_dict[native_key] = create_dtensor_from_local(stacked, device_mesh, rank)
                        # Drop references to per-expert tensors as soon as we've merged this group.
                        expert_weights_by_layer[layer_num].pop(native_key, None)
                        if not expert_weights_by_layer[layer_num]:
                            expert_weights_by_layer.pop(layer_num, None)
                        del tensors, stacked

                else:  # down_proj
                    expert_weights_by_layer[layer_num][native_key][expert_num] = value

                    if len(expert_weights_by_layer[layer_num][native_key]) == expected_experts_per_rank:
                        expert_ids = sorted(expert_weights_by_layer[layer_num][native_key].keys())

                        ordered = []
                        for expert_id in expert_ids:
                            down_weight = expert_weights_by_layer[layer_num][native_key][expert_id]  # [dim, inter_dim]

                            # Extract local tensor if input is already a DTensor
                            if is_dtensor(down_weight):
                                down_weight = down_weight.to_local()

                            down_t = down_weight.transpose(0, 1)  # [inter_dim, dim]
                            ordered.append(down_t)

                        stacked = torch.stack(ordered, dim=0)
                        stacked = stacked.to(self.dtype)

                        dtensor = create_dtensor_from_local(stacked, device_mesh, rank)
                        state_dict[native_key] = dtensor
                        # Drop references to per-expert tensors as soon as we've merged this group.
                        expert_weights_by_layer[layer_num].pop(native_key, None)
                        if not expert_weights_by_layer[layer_num]:
                            expert_weights_by_layer.pop(layer_num, None)
                        del ordered, stacked, dtensor

            else:
                if not key.endswith("_scale_inv"):
                    state_dict[key] = value

        return state_dict

    def _convert_single_merged_expert_to_hf_split_experts(
        self, fqn: str, tensor: torch.Tensor, **kwargs
    ) -> list[tuple[str, torch.Tensor]]:
        """Convert a single merged expert tensor from native format to split HuggingFace format.

        Args:
            fqn: Fully qualified name of the tensor in native format
            tensor: The tensor to convert

        Returns:
            List of (fqn, tensor) tuples in HuggingFace format, or None if not an expert tensor
        """
        n_experts = self.moe_config.n_routed_experts
        inter_dim = self.moe_config.moe_inter_dim
        parsed = self._parse_native_expert_key(fqn)
        if parsed is None:
            return None
        layer_num, key_type = parsed

        if key_type == "gate_and_up_projs":
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
                if self._is_gated_moe:
                    w_gate = w[:, :inter_dim].transpose(0, 1)
                    w_up = w[:, inter_dim:].transpose(0, 1)
                    result.append((self._build_hf_key(layer_num, expert_id, "gate_proj"), w_gate))
                    result.append((self._build_hf_key(layer_num, expert_id, "up_proj"), w_up))
                else:
                    w_up = w.transpose(0, 1)
                    result.append((self._build_hf_key(layer_num, expert_id, "up_proj"), w_up))
            return result

        if key_type == "down_projs" and tensor.ndim == 3 and tensor.shape[1] == inter_dim:
            from nemo_automodel.components.moe.state_dict_utils import (
                is_dtensor,
                validate_dtensor_expert_sharding,
            )

            if is_dtensor(tensor):
                validate_dtensor_expert_sharding(tensor, n_experts, f"down_projs (DeepEP) layer {layer_num}")

            splits = self._split_experts_weights(tensor, n_experts)
            result = []
            for i, w in enumerate(splits):
                expert_id = self._last_expert_ids[i]
                result.append((self._build_hf_key(layer_num, expert_id, "down_proj"), w.transpose(0, 1)))
            return result

        return None

    # --- Lazy / JIT conversion: key mapping and single-key tensor access ---

    def get_hf_keys_for_native_key(self, fqn: str) -> Optional[list[str]]:
        """Return the list of HF key names for this native key, without converting tensors.
        Returns None if this key is not an expert merged key (passthrough).
        """
        parsed = self._parse_native_expert_key(fqn)
        if parsed is None:
            return None
        layer_num, key_type = parsed
        n_experts = self.moe_config.n_routed_experts
        keys = []
        if key_type == "gate_and_up_projs":
            for e in range(n_experts):
                if self._is_gated_moe:
                    keys.append(self._build_hf_key(layer_num, e, "gate_proj"))
                    keys.append(self._build_hf_key(layer_num, e, "up_proj"))
                else:
                    keys.append(self._build_hf_key(layer_num, e, "up_proj"))
        else:  # down_projs
            for e in range(n_experts):
                keys.append(self._build_hf_key(layer_num, e, "down_proj"))
        return keys

    def get_native_key_for_hf_key(self, hf_key: str) -> Optional[str]:
        """Return the native key that produces this HF key, or None if passthrough."""
        if not isinstance(hf_key, str):
            return None
        parsed = self._parse_hf_expert_key(hf_key)
        if parsed is None:
            return None
        prefix, layer_num, _, proj_type = parsed
        if proj_type in ("gate_proj", "up_proj"):
            return self._build_native_key(prefix, layer_num, "gate_and_up_projs")
        return self._build_native_key(prefix, layer_num, "down_projs")

    def get_tensor_for_hf_key(self, native_fqn: str, tensor: torch.Tensor, hf_key: str) -> torch.Tensor:
        """Return the single tensor (view) for this HF key. native_fqn must be the native key for hf_key."""
        parsed = self._parse_hf_expert_key(hf_key)
        if parsed is None:
            raise KeyError(hf_key)
        _, layer_num, expert_id, proj_type = parsed
        native_parsed = self._parse_native_expert_key(native_fqn)
        if native_parsed is None:
            raise KeyError(hf_key)
        _, key_type = native_parsed
        inter_dim = self.moe_config.moe_inter_dim
        w = tensor[expert_id]

        if key_type == "gate_and_up_projs":
            if self._is_gated_moe:
                if proj_type == "gate_proj":
                    return w[:, :inter_dim].transpose(0, 1)
                if proj_type == "up_proj":
                    return w[:, inter_dim:].transpose(0, 1)
            else:
                return w.transpose(0, 1)
        if key_type == "down_projs":
            return w.transpose(0, 1)
        raise KeyError(hf_key)

    def _get_prefix_from_hf_expert_keys(self, hf_state_dict: dict[str, Any]) -> str:
        """Infer HF key prefix from any expert key in the dict. Uses shared pattern."""
        for k in hf_state_dict:
            parsed = self._parse_hf_expert_key(k)
            if parsed is not None:
                return parsed[0]  # prefix
        return ""

    def get_merged_tensor_for_native_key(
        self,
        native_key: str,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
    ) -> Optional[Any]:
        """Pop HF keys that form this native key from hf_state_dict, merge incrementally, return the native tensor.
        Returns None if native_key is not an expert merged key (caller should pass through).
        Merges one expert at a time to keep peak memory ~1x.
        """
        parsed = self._parse_native_expert_key(native_key)
        if parsed is None:
            return None
        layer_num, key_type = parsed
        n_experts = self.moe_config.n_routed_experts
        is_gated = self._is_gated_moe

        if device_mesh is not None:
            start_expert, end_expert = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            expected_experts = list(range(start_expert, end_expert))
        else:
            expected_experts = list(range(n_experts))

        rank = None
        if device_mesh is not None:
            rank = (
                get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )
        prefix = self._get_prefix_from_hf_expert_keys(hf_state_dict)

        if key_type == "gate_and_up_projs":
            dim = self.moe_config.dim
            inter_dim = self.moe_config.moe_inter_dim
            if is_gated:
                out_shape = (len(expected_experts), dim, 2 * inter_dim)
                full_size = len(expected_experts) * dim * (2 * inter_dim)
            else:
                out_shape = (len(expected_experts), dim, inter_dim)
                full_size = len(expected_experts) * dim * inter_dim
            first_expert_id = sorted(expected_experts)[0]
            gate_key = self._build_hf_key(layer_num, first_expert_id, "gate_proj", prefix=prefix)
            up_key = self._build_hf_key(layer_num, first_expert_id, "up_proj", prefix=prefix)
            if is_gated and gate_key in hf_state_dict and up_key in hf_state_dict and device_mesh is None:
                first_t = hf_state_dict.pop(gate_key)
                if is_dtensor(first_t):
                    first_t = first_t.to_local()
                st = first_t.untyped_storage()
                offset = first_t.storage_offset()
                # Round-trip: HF tensors are views of the original native tensor; storage spans full buffer.
                if st.size() >= full_size + offset:
                    strides = (dim * (2 * inter_dim), 2 * inter_dim, 1) if is_gated else (dim * inter_dim, inter_dim, 1)
                    out = torch.as_strided(first_t, out_shape, strides, offset).to(self.dtype)
                    for expert_id in sorted(expected_experts):
                        for k in (
                            self._build_hf_key(layer_num, expert_id, "gate_proj", prefix=prefix),
                            self._build_hf_key(layer_num, expert_id, "up_proj", prefix=prefix),
                        ):
                            if k in hf_state_dict:
                                del hf_state_dict[k]
                    return out
                hf_state_dict[gate_key] = first_t
            device = hf_state_dict[up_key].device
            stacked = torch.empty(out_shape, device=device, dtype=self.dtype)
            for idx, expert_id in enumerate(sorted(expected_experts)):
                gate_key = self._build_hf_key(layer_num, expert_id, "gate_proj", prefix=prefix)
                up_key = self._build_hf_key(layer_num, expert_id, "up_proj", prefix=prefix)
                if not is_gated:
                    if gate_key in hf_state_dict:
                        del hf_state_dict[gate_key]
                    if up_key not in hf_state_dict:
                        return None
                    up_weight = hf_state_dict.pop(up_key)
                    if is_dtensor(up_weight):
                        up_weight = up_weight.to_local()
                    stacked[idx].copy_(up_weight.transpose(0, 1))
                    del up_weight
                    continue
                if gate_key not in hf_state_dict or up_key not in hf_state_dict:
                    return None
                gate_weight = hf_state_dict.pop(gate_key)
                up_weight = hf_state_dict.pop(up_key)
                if is_dtensor(gate_weight):
                    gate_weight = gate_weight.to_local()
                if is_dtensor(up_weight):
                    up_weight = up_weight.to_local()
                gate_t = gate_weight.transpose(0, 1)
                up_t = up_weight.transpose(0, 1)
                stacked[idx, :, :inter_dim].copy_(gate_t)
                stacked[idx, :, inter_dim:].copy_(up_t)
                del gate_weight, up_weight, gate_t, up_t
            out = create_dtensor_from_local(stacked, device_mesh, rank)
            del stacked
            return out

        if key_type == "down_projs":
            inter_dim = self.moe_config.moe_inter_dim
            dim = self.moe_config.dim
            down_shape = (len(expected_experts), inter_dim, dim)
            full_size = len(expected_experts) * inter_dim * dim
            first_down_key = self._build_hf_key(layer_num, sorted(expected_experts)[0], "down_proj", prefix=prefix)
            if first_down_key not in hf_state_dict:
                return None
            if device_mesh is None:
                first_down = hf_state_dict.pop(first_down_key)
                if is_dtensor(first_down):
                    first_down = first_down.to_local()
                if first_down.untyped_storage().size() == full_size:
                    strides = (inter_dim * dim, dim, 1)
                    out = torch.as_strided(first_down, down_shape, strides, 0).to(self.dtype)
                    for expert_id in sorted(expected_experts):
                        k = self._build_hf_key(layer_num, expert_id, "down_proj", prefix=prefix)
                        if k in hf_state_dict:
                            del hf_state_dict[k]
                    return out
                hf_state_dict[first_down_key] = first_down
            device = hf_state_dict[first_down_key].device
            stacked = torch.empty(down_shape, device=device, dtype=self.dtype)
            for idx, expert_id in enumerate(sorted(expected_experts)):
                down_key = self._build_hf_key(layer_num, expert_id, "down_proj", prefix=prefix)
                if down_key not in hf_state_dict:
                    return None
                down_weight = hf_state_dict.pop(down_key)
                if is_dtensor(down_weight):
                    down_weight = down_weight.to_local()
                stacked[idx].copy_(down_weight.transpose(0, 1))
                del down_weight
            out = create_dtensor_from_local(stacked, device_mesh, rank)
            del stacked
            return out

        return None

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

    def _convert_lora_expert_to_hf(
        self,
        fqn: str,
        tensor: torch.Tensor,
        n_experts: int,
        inter_dim: int,
        expert_segment: str,
    ) -> list[tuple[str, torch.Tensor]]:
        """Convert a grouped MoE expert LoRA tensor to per-expert HF PEFT format.

        Handles the four LoRA parameter types produced by GroupedExpertsLoRA /
        GroupedExpertsDeepEPLoRA and converts them to per-expert ``lora_A.weight``
        / ``lora_B.weight`` keys that HF PEFT understands.

        The prefix (e.g. ``base_model.model.model.``) is preserved from the
        incoming *fqn* so that both PEFT and FFT save paths work correctly.
        """
        match = re.search(r"(.*)layers\.(\d+)\.", fqn)
        if not match:
            return None
        fqn_prefix = match.group(1)
        layer_num = match.group(2)
        suffix = fqn.rsplit(".", 1)[-1]

        splits = self._split_experts_weights(tensor, n_experts)
        result: list[tuple[str, torch.Tensor]] = []

        for i, w in enumerate(splits):
            expert_id = self._last_expert_ids[i]
            base = f"{fqn_prefix}layers.{layer_num}.{expert_segment}.{expert_id}"

            if suffix == "lora_gate_and_up_A":
                # [dim, lora_dim] -> [lora_dim, dim] (nn.Linear convention)
                w_t = w.transpose(0, 1).contiguous()
                if self._is_gated_moe:
                    result.append((f"{base}.gate_proj.lora_A.weight", w_t))
                    result.append((f"{base}.up_proj.lora_A.weight", w_t.clone()))
                else:
                    result.append((f"{base}.up_proj.lora_A.weight", w_t))

            elif suffix == "lora_gate_and_up_B":
                # [lora_dim, 2*inter] (gated) or [lora_dim, inter] (non-gated)
                if self._is_gated_moe:
                    w_gate = w[:, :inter_dim].transpose(0, 1).contiguous()
                    w_up = w[:, inter_dim:].transpose(0, 1).contiguous()
                    result.append((f"{base}.gate_proj.lora_B.weight", w_gate))
                    result.append((f"{base}.up_proj.lora_B.weight", w_up))
                else:
                    result.append((f"{base}.up_proj.lora_B.weight", w.transpose(0, 1).contiguous()))

            elif suffix == "lora_down_A":
                # [inter_dim, lora_dim] -> [lora_dim, inter_dim]
                result.append((f"{base}.down_proj.lora_A.weight", w.transpose(0, 1).contiguous()))

            elif suffix == "lora_down_B":
                # [lora_dim, dim] -> [dim, lora_dim]
                result.append((f"{base}.down_proj.lora_B.weight", w.transpose(0, 1).contiguous()))

        return result

    def _recombine_lora_expert_keys(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Recombine per-expert HF LoRA keys back to grouped MoE LoRA format.

        This is the reverse of ``_convert_lora_expert_to_hf``.  It detects
        per-expert LoRA keys (e.g.
        ``layers.0.mlp.experts.0.gate_proj.lora_A.weight``) and recombines
        them into the grouped tensors expected by GroupedExpertsLoRA /
        GroupedExpertsDeepEPLoRA (e.g. ``layers.0.mlp.experts.lora_gate_and_up_A``).
        """
        expert_segment = re.escape(self._expert_path_segment)
        n_experts = self.moe_config.n_routed_experts

        lora_pattern = re.compile(
            rf"(?P<prefix>.*)layers\.(?P<layer>\d+)\.{expert_segment}\."
            rf"(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.(?P<lora>lora_[AB])\.weight"
        )

        # Group: (prefix, layer, proj, lora) -> {expert_id: tensor}
        lora_groups: dict[tuple, dict[int, torch.Tensor]] = {}
        consumed_keys: set[str] = set()

        for key, value in state_dict.items():
            m = lora_pattern.match(key)
            if m:
                group_key = (m.group("prefix"), m.group("layer"), m.group("proj"), m.group("lora"))
                lora_groups.setdefault(group_key, {})[int(m.group("expert"))] = value
                consumed_keys.add(key)

        if not consumed_keys:
            return state_dict

        result = {k: v for k, v in state_dict.items() if k not in consumed_keys}
        processed: set[tuple] = set()

        for (prefix, layer, proj, lora), experts in lora_groups.items():
            group_id = (prefix, layer, proj, lora)
            if group_id in processed:
                continue
            processed.add(group_id)

            if len(experts) != n_experts:
                for eid, t in experts.items():
                    orig_seg = self._expert_path_segment
                    result[f"{prefix}layers.{layer}.{orig_seg}.{eid}.{proj}.{lora}.weight"] = t
                continue

            sorted_ids = sorted(experts.keys())
            base_key = f"{prefix}layers.{layer}.{self._expert_path_segment}"

            if proj in ("gate_proj", "up_proj") and lora == "lora_A":
                if self._is_gated_moe and proj == "up_proj":
                    continue  # gate_proj.lora_A already produces this (deduplicate)
                tensors = [experts[eid].transpose(0, 1).contiguous() for eid in sorted_ids]
                result[f"{base_key}.lora_gate_and_up_A"] = torch.stack(tensors, dim=0)
                if self._is_gated_moe:
                    processed.add((prefix, layer, "up_proj", "lora_A"))

            elif proj in ("gate_proj", "up_proj") and lora == "lora_B":
                if self._is_gated_moe and proj == "up_proj":
                    continue  # handled by gate_proj.lora_B below
                if self._is_gated_moe:
                    up_key = (prefix, layer, "up_proj", "lora_B")
                    up_experts = lora_groups.get(up_key, {})
                    if len(up_experts) != n_experts:
                        for eid, t in experts.items():
                            orig_seg = self._expert_path_segment
                            result[f"{prefix}layers.{layer}.{orig_seg}.{eid}.{proj}.{lora}.weight"] = t
                        continue
                    gate_ts = [experts[eid].transpose(0, 1).contiguous() for eid in sorted_ids]
                    up_ts = [up_experts[eid].transpose(0, 1).contiguous() for eid in sorted_ids]
                    combined = torch.cat([torch.stack(gate_ts, dim=0), torch.stack(up_ts, dim=0)], dim=-1)
                    result[f"{base_key}.lora_gate_and_up_B"] = combined
                    processed.add(up_key)
                else:
                    tensors = [experts[eid].transpose(0, 1).contiguous() for eid in sorted_ids]
                    result[f"{base_key}.lora_gate_and_up_B"] = torch.stack(tensors, dim=0)

            elif proj == "down_proj":
                native_suffix = "lora_down_A" if lora == "lora_A" else "lora_down_B"
                tensors = [experts[eid].transpose(0, 1).contiguous() for eid in sorted_ids]
                result[f"{base_key}.{native_suffix}"] = torch.stack(tensors, dim=0)

        return result

    def _to_hf_w_split_experts(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert DeepEP format to HuggingFace format.
        Handles: gate_and_up_projs, down_projs -> individual expert weights
        """
        hf_state_dict: dict[str, Any] = {}

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

        # Handle both formats:
        # - model.layers.{L}.{expert_segment}.{E}.gate_proj.weight (with model prefix)
        # - language_model.layers.{L}.{expert_segment}.{E}.gate_proj.weight (with language_model prefix)
        # - layers.{L}.{expert_segment}.{E}.gate_proj.weight (without model prefix)
        expert_pattern = re.compile(
            rf"(?P<prefix>(?:model\.)?(?:language_model\.)?)layers\.(\d+)\.{re.escape(expert_segment)}\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
        )

        for key, value in hf_state_dict.items():
            if f".{expert_segment}." in key and key.endswith(".weight"):
                m = expert_pattern.match(key)
                if m is None:
                    state_dict[key] = value
                    continue

                prefix = m.group("prefix") or ""
                layer_num, expert_num, which = m.group(2), m.group(3), m.group(4)
                expert_num = int(expert_num)

                if not should_load_expert_for_rank(expert_num, device_mesh, n_experts):
                    continue

                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}

                if which in ["gate_proj", "up_proj"]:
                    native_key = f"{prefix}layers.{layer_num}.{expert_segment}.gate_and_up_projs"
                else:  # down_proj
                    native_key = f"{prefix}layers.{layer_num}.{expert_segment}.down_projs"

                if native_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][native_key] = {}

                if which in ["gate_proj", "up_proj"]:
                    # Non-gated models only use up_proj, skip gate_proj
                    if not is_gated and which == "gate_proj":
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

            else:
                if not key.endswith("_scale_inv"):
                    state_dict[key] = value

        # Recombine any per-expert HF LoRA keys back to grouped format
        state_dict = self._recombine_lora_expert_keys(state_dict)

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
        prefix = self._hf_prefix
        expert_segment = self._expert_path_segment

        if f".{expert_segment}.gate_and_up_projs" in fqn and fqn.endswith(".gate_and_up_projs"):
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
                if self._is_gated_moe:
                    # Gated: split into gate_proj and up_proj
                    w_gate = w[:, :inter_dim].transpose(0, 1).contiguous()
                    w_up = w[:, inter_dim:].transpose(0, 1).contiguous()
                    result.append((f"{prefix}layers.{layer_num}.{expert_segment}.{expert_id}.gate_proj.weight", w_gate))
                    result.append((f"{prefix}layers.{layer_num}.{expert_segment}.{expert_id}.up_proj.weight", w_up))
                else:
                    # Non-gated: only up_proj (tensor is [dim, inter_dim], not [dim, 2*inter_dim])
                    w_up = w.transpose(0, 1).contiguous()
                    result.append((f"{prefix}layers.{layer_num}.{expert_segment}.{expert_id}.up_proj.weight", w_up))
            return result

        elif (
            f".{expert_segment}.down_projs" in fqn
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
                validate_dtensor_expert_sharding(tensor, n_experts, f"down_projs (DeepEP) layer {layer_num}")

            splits = self._split_experts_weights(tensor, n_experts)
            result = []
            for i, w in enumerate(splits):
                expert_id = self._last_expert_ids[i]
                result.append(
                    (
                        f"{prefix}layers.{layer_num}.{expert_segment}.{expert_id}.down_proj.weight",
                        w.transpose(0, 1).contiguous(),
                    )
                )
            return result

        # MoE expert LoRA keys: split grouped 3-D adapter tensors into per-expert
        # HF-PEFT-compatible keys so that AutoPeftModelForCausalLM can load & merge.
        _LORA_EXPERT_SUFFIXES = ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B")
        for suffix in _LORA_EXPERT_SUFFIXES:
            if f".{expert_segment}.{suffix}" in fqn and fqn.endswith(f".{suffix}"):
                return self._convert_lora_expert_to_hf(fqn, tensor, n_experts, inter_dim, expert_segment)

        return None

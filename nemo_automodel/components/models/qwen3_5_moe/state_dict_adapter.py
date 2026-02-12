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

"""State-dict adapter for Qwen3.5-MoE.

HF Qwen3.5-MoE stores expert weights as **aggregated 3-D tensors**:

    model.language_model.layers.{L}.mlp.experts.gate_up_proj   # [n_experts, 2*moe_inter, hidden]
    model.language_model.layers.{L}.mlp.experts.down_proj      # [n_experts, hidden, moe_inter]

NeMo uses a different naming convention **and transposed layout** (x @ weight):

    model.language_model.layers.{L}.mlp.experts.gate_and_up_projs  # [n_experts, hidden, 2*moe_inter]
    model.language_model.layers.{L}.mlp.experts.down_projs         # [n_experts, moe_inter, hidden]

Both expert tensors require `.transpose(1, 2)` when converting between formats.

Additionally, the shared expert uses singular in HF and plural in NeMo:

    HF:   .mlp.shared_expert.{gate,up,down}_proj.weight
    NeMo: .mlp.shared_experts.{gate,up,down}_proj.weight

All other keys (attention, linear_attn/GatedDeltaNet, norms, embeddings, lm_head,
vision encoder) pass through unchanged.
"""

import re
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe import state_dict_utils
from nemo_automodel.components.moe.layers import MoEConfig


class Qwen3_5MoeStateDictAdapter(StateDictAdapter):
    """Converts between HF Qwen3.5-MoE checkpoints and the NeMo native format.

    Handles:
      1. Aggregated expert weight renaming (gate_up_proj ↔ gate_and_up_projs)
      2. Shared expert key mapping (shared_expert ↔ shared_experts)
      3. Expert-parallel sharding when a device mesh is provided
    """

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

        self.hf_to_internal_map = {
            ".mlp.shared_expert.": ".mlp.shared_experts.",
        }
        self.internal_to_hf_map = {v: k for k, v in self.hf_to_internal_map.items()}

    def _apply_key_mapping(self, state_dict: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
        """Apply key substring mappings to state dict keys."""
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            for pattern, replacement in mapping.items():
                if pattern in key:
                    new_key = new_key.replace(pattern, replacement)
                    break
            new_state_dict[new_key] = value
        return new_state_dict

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        self._uses_model_prefix = any(key.startswith("model.") for key in state_dict.keys())
        prefix = "model." if self._uses_model_prefix else ""
        hf_state_dict: dict[str, Any] = {}
        device_mesh: Optional["DeviceMesh"] = kwargs.get("device_mesh")

        for fqn, tensor in state_dict.items():
            # --- Routed expert tensors: gather across EP ranks if needed ---
            if ".mlp.experts.gate_and_up_projs" in fqn or ".mlp.experts.down_projs" in fqn:
                layer_num = re.search(r"layers\.(\d+)", fqn).group(1)
                which = "gate_up_proj" if "gate_and_up_projs" in fqn else "down_proj"

                if device_mesh is not None:
                    n_experts = self.moe_config.n_routed_experts
                    global_tensor = torch.zeros(
                        (n_experts, tensor.shape[1], tensor.shape[2]),
                        dtype=self.dtype,
                        device="cpu",
                    )

                    if state_dict_utils.is_dtensor(tensor):
                        split_weights, expert_ids = state_dict_utils.split_experts_weights_dtensor_aware(
                            tensor, n_experts
                        )
                    else:
                        start_expert, end_expert = state_dict_utils.get_expert_range_for_rank_from_mesh(
                            device_mesh, n_experts
                        )
                        split_weights = [tensor[i].to(self.dtype).cpu() for i in range(tensor.shape[0])]
                        expert_ids = list(range(start_expert, end_expert))

                    if dist.is_initialized() and "ep" in device_mesh.mesh_dim_names:
                        try:
                            ep_dim = device_mesh.mesh_dim_names.index("ep")
                            ep_group = device_mesh.get_group(ep_dim)
                        except Exception:
                            ep_group = None

                        if ep_group is not None:
                            payload = (expert_ids, [w.cpu() for w in split_weights])
                            gathered: list[tuple[list[int], list[torch.Tensor]]] = [None] * dist.get_world_size(
                                ep_group
                            )
                            dist.all_gather_object(gathered, payload, group=ep_group)
                            for ids, weights in gathered:
                                for eid, w in zip(ids, weights):
                                    global_tensor[eid].copy_(w.to(self.dtype).cpu())
                        else:
                            for weight, expert_id in zip(split_weights, expert_ids):
                                global_tensor[expert_id].copy_(weight.to(self.dtype).cpu())
                    else:
                        for weight, expert_id in zip(split_weights, expert_ids):
                            global_tensor[expert_id].copy_(weight.to(self.dtype).cpu())

                    del split_weights, expert_ids
                    # NeMo layout is transposed relative to HF, so transpose(1,2) back
                    global_tensor = global_tensor.transpose(1, 2).contiguous()
                    key = f"{prefix}language_model.layers.{layer_num}.mlp.experts.{which}"
                    hf_state_dict[key] = global_tensor
                    del global_tensor
                else:
                    converted = self.convert_single_tensor_to_hf(
                        fqn, tensor, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
                    )
                    for key, value in converted:
                        hf_state_dict[key] = value
            else:
                converted = self.convert_single_tensor_to_hf(
                    fqn, tensor, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
                )
                for key, value in converted:
                    hf_state_dict[key] = value

        if exclude_key_regex:
            hf_state_dict = {k: v for k, v in hf_state_dict.items() if not re.match(exclude_key_regex, k)}

        return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        # Detect model prefix convention
        expert_keys = [
            key for key in hf_state_dict.keys() if ".mlp.experts.gate_up_proj" in key or ".mlp.experts.down_proj" in key
        ]
        if not expert_keys:
            raise RuntimeError("Expected aggregated expert weights (gate_up_proj / down_proj) in the checkpoint.")
        self._uses_model_prefix = any(key.startswith("model.") for key in expert_keys)
        model_prefix = "model." if self._uses_model_prefix else ""

        n_experts = self.moe_config.n_routed_experts
        if device_mesh is not None:
            start_expert, end_expert = state_dict_utils.get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            rank = (
                state_dict_utils.get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )
        else:
            start_expert, end_expert = 0, n_experts
            rank = None

        state_dict: dict[str, Any] = {}
        for key, value in hf_state_dict.items():
            # --- Aggregated expert tensors ---
            match = re.match(
                r"(model\.)?language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$",
                key,
            )
            if match:
                _, layer_num, which = match.groups()
                # HF layout is transposed relative to NeMo (x @ weight), so transpose(1,2)
                local_tensor = value[start_expert:end_expert].transpose(1, 2).to(self.dtype)
                native_key = f"{model_prefix}language_model.layers.{layer_num}.mlp.experts."
                native_key += "gate_and_up_projs" if which == "gate_up_proj" else "down_projs"
                state_dict[native_key] = state_dict_utils.create_dtensor_from_local(local_tensor, device_mesh, rank)
                continue

            # Skip quantization scale keys
            if key.endswith("_scale_inv"):
                continue

            # --- Shared expert key mapping (shared_expert → shared_experts) ---
            mapped_key = key
            for pattern, replacement in self.hf_to_internal_map.items():
                if pattern in mapped_key:
                    mapped_key = mapped_key.replace(pattern, replacement)
                    break

            # Ensure consistent prefix
            if key.startswith("model."):
                state_dict[mapped_key] = value
            else:
                state_dict[f"{model_prefix}{mapped_key}" if not mapped_key.startswith("model.") else mapped_key] = value

        return state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single native tensor back to HF format."""
        prefix = "model." if self._uses_model_prefix else ""
        exclude_key_regex = kwargs.get("exclude_key_regex")

        if ".mlp.experts.gate_and_up_projs" in fqn:
            layer_num = re.search(r"layers\.(\d+)", fqn).group(1)
            key = f"{prefix}language_model.layers.{layer_num}.mlp.experts.gate_up_proj"
            if state_dict_utils.is_dtensor(tensor):
                tensor = tensor.to_local()
            # NeMo layout is transposed relative to HF, so transpose(1,2) back
            result = [(key, tensor.transpose(1, 2).contiguous().to(self.dtype))]
        elif ".mlp.experts.down_projs" in fqn:
            layer_num = re.search(r"layers\.(\d+)", fqn).group(1)
            key = f"{prefix}language_model.layers.{layer_num}.mlp.experts.down_proj"
            if state_dict_utils.is_dtensor(tensor):
                tensor = tensor.to_local()
            # NeMo layout is transposed relative to HF, so transpose(1,2) back
            result = [(key, tensor.transpose(1, 2).contiguous().to(self.dtype))]
        else:
            result = [(fqn, tensor)]

        # Apply shared_experts → shared_expert reverse mapping
        mapped_result = []
        for key, value in result:
            new_key = key
            for pattern, replacement in self.internal_to_hf_map.items():
                if pattern in key:
                    new_key = new_key.replace(pattern, replacement)
                    break
            mapped_result.append((new_key, value))

        if exclude_key_regex:
            mapped_result = [(k, v) for k, v in mapped_result if not re.match(exclude_key_regex, k)]

        return mapped_result

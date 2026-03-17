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

"""State-dict adapter for Gemma4 MoE.

HF Gemma4 MoE stores expert weights as **separate 3-D parameter tensors**:

    layers.{L}.moe.gate_proj          # [n_experts, hidden_size, expert_inter_size]
    layers.{L}.moe.up_proj            # [n_experts, hidden_size, expert_inter_size]
    layers.{L}.moe.down_proj          # [n_experts, expert_inter_size, hidden_size]
    layers.{L}.moe.per_expert_scale   # [n_experts]

NeMo uses concatenated layout (same matrix orientation, no transpose needed):

    layers.{L}.moe.experts.gate_and_up_projs  # [n_experts, hidden_size, 2*expert_inter_size]
    layers.{L}.moe.experts.down_projs         # [n_experts, expert_inter_size, hidden_size]

Additionally, the Gemma4 router is mapped to the NeMo Gemma4Gate:

    HF:   .router.proj.weight / .router.scale
    NeMo: .moe.gate.proj.weight / .moe.gate.scale

The per_expert_scale is absorbed into down_projs during from_hf.  When
saving back to HF, per_expert_scale is emitted as ones (scale already baked
into the weights).
"""

import re
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe import state_dict_utils
from nemo_automodel.components.moe.layers import MoEConfig


class Gemma4MoEStateDictAdapter(StateDictAdapter):
    """Converts between HF Gemma4 MoE checkpoints and the NeMo format.

    Handles:
      1. Expert weight concatenation (gate_proj + up_proj -> gate_and_up_projs)
      2. per_expert_scale absorption into down_projs
      3. Router key remapping (router.* -> moe.gate.*)
      4. Expert-parallel sharding when a device mesh is provided
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

    # ------------------------------------------------------------------
    # HF -> NeMo
    # ------------------------------------------------------------------
    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional[DeviceMesh] = None,
        **kwargs,
    ) -> dict[str, Any]:
        self._uses_model_prefix = any(key.startswith("model.") for key in hf_state_dict)
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

        # Collect MoE expert tensors per layer for combined processing
        expert_buffers: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        state_dict: dict[str, Any] = {}

        _EXPERT_KEYS = {"gate_proj", "up_proj", "down_proj", "per_expert_scale"}

        for key, value in hf_state_dict.items():
            # --- Router keys: router.{proj.weight,scale} -> moe.gate.{proj.weight,scale} ---
            router_match = re.search(r"(layers\.\d+)\.router\.(proj\.weight|scale)$", key)
            if router_match:
                layer_path = router_match.group(1)
                router_attr = router_match.group(2)
                new_key = key.replace(f"{layer_path}.router.{router_attr}", f"{layer_path}.moe.gate.{router_attr}")
                state_dict[new_key] = value
                continue

            # --- Expert weight keys ---
            expert_match = re.search(r"(layers\.\d+)\.moe\.(gate_proj|up_proj|down_proj|per_expert_scale)$", key)
            if expert_match:
                layer_path = expert_match.group(1)
                weight_name = expert_match.group(2)
                expert_buffers[layer_path][weight_name] = value
                continue

            # --- Pass-through keys ---
            state_dict[key] = value

        # Process collected expert weights per layer
        for layer_path, tensors in expert_buffers.items():
            missing = _EXPERT_KEYS - tensors.keys()
            if missing:
                raise RuntimeError(
                    f"Incomplete expert weights for {layer_path}: missing {missing}. "
                    f"Available keys: {list(tensors.keys())}"
                )

            gate_proj = tensors["gate_proj"]           # [E, hidden, inter]
            up_proj = tensors["up_proj"]               # [E, hidden, inter]
            down_proj = tensors["down_proj"]            # [E, inter, hidden]
            per_expert_scale = tensors["per_expert_scale"]  # [E]

            # Concatenate gate + up projections (no transpose needed)
            gate_and_up = torch.cat([gate_proj, up_proj], dim=-1)  # [E, hidden, 2*inter]

            # Absorb per_expert_scale into down_projs
            down = down_proj * per_expert_scale[:, None, None]  # [E, inter, hidden]

            # Slice for EP
            gate_and_up_local = gate_and_up[start_expert:end_expert].to(self.dtype)
            down_local = down[start_expert:end_expert].to(self.dtype)

            prefix = f"{model_prefix}language_model.{layer_path}"
            state_dict[f"{prefix}.moe.experts.gate_and_up_projs"] = (
                state_dict_utils.create_dtensor_from_local(gate_and_up_local, device_mesh, rank)
            )
            state_dict[f"{prefix}.moe.experts.down_projs"] = (
                state_dict_utils.create_dtensor_from_local(down_local, device_mesh, rank)
            )

        return state_dict

    # ------------------------------------------------------------------
    # NeMo -> HF
    # ------------------------------------------------------------------
    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        self._uses_model_prefix = any(key.startswith("model.") for key in state_dict)
        prefix = "model." if self._uses_model_prefix else ""
        device_mesh: Optional[DeviceMesh] = kwargs.get("device_mesh")
        n_experts = self.moe_config.n_routed_experts

        hf_state_dict: dict[str, Any] = {}

        for fqn, tensor in state_dict.items():
            # --- Router keys ---
            gate_match = re.search(r"(layers\.\d+)\.moe\.gate\.(proj\.weight|scale)$", fqn)
            if gate_match:
                layer_path = gate_match.group(1)
                gate_attr = gate_match.group(2)
                hf_key = fqn.replace(f"{layer_path}.moe.gate.{gate_attr}", f"{layer_path}.router.{gate_attr}")
                hf_state_dict[hf_key] = tensor
                continue

            # --- Expert: gate_and_up_projs -> gate_proj + up_proj ---
            if ".moe.experts.gate_and_up_projs" in fqn:
                layer_num = re.search(r"layers\.(\d+)", fqn).group(1)
                global_tensor = self._gather_expert_tensor(tensor, device_mesh, n_experts)
                half = global_tensor.shape[-1] // 2
                layer_prefix = f"{prefix}language_model.layers.{layer_num}"
                hf_state_dict[f"{layer_prefix}.moe.gate_proj"] = global_tensor[..., :half].contiguous()
                hf_state_dict[f"{layer_prefix}.moe.up_proj"] = global_tensor[..., half:].contiguous()
                continue

            # --- Expert: down_projs -> down_proj + per_expert_scale ---
            if ".moe.experts.down_projs" in fqn:
                layer_num = re.search(r"layers\.(\d+)", fqn).group(1)
                global_tensor = self._gather_expert_tensor(tensor, device_mesh, n_experts)
                layer_prefix = f"{prefix}language_model.layers.{layer_num}"
                hf_state_dict[f"{layer_prefix}.moe.down_proj"] = global_tensor.contiguous()
                hf_state_dict[f"{layer_prefix}.moe.per_expert_scale"] = torch.ones(
                    n_experts, dtype=self.dtype
                )
                continue

            # --- Pass-through ---
            hf_state_dict[fqn] = tensor

        if exclude_key_regex:
            hf_state_dict = {k: v for k, v in hf_state_dict.items() if not re.match(exclude_key_regex, k)}

        return hf_state_dict

    def _gather_expert_tensor(
        self,
        tensor: torch.Tensor,
        device_mesh: Optional[DeviceMesh],
        n_experts: int,
    ) -> torch.Tensor:
        """Gather EP-sharded expert tensor across ranks into a full tensor."""
        if device_mesh is None:
            if state_dict_utils.is_dtensor(tensor):
                return tensor.to_local()
            return tensor

        global_tensor = torch.zeros(
            (n_experts, tensor.shape[1] if not state_dict_utils.is_dtensor(tensor) else tensor.to_local().shape[1],
             tensor.shape[2] if not state_dict_utils.is_dtensor(tensor) else tensor.to_local().shape[2]),
            dtype=self.dtype,
            device="cpu",
        )

        if state_dict_utils.is_dtensor(tensor):
            split_weights, expert_ids = state_dict_utils.split_experts_weights_dtensor_aware(tensor, n_experts)
        else:
            start_expert, end_expert = state_dict_utils.get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
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
                gathered: list[tuple[list[int], list[torch.Tensor]]] = [None] * dist.get_world_size(ep_group)
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
        return global_tensor

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single native tensor back to HF format (used by to_hf for non-expert keys)."""
        return [(fqn, tensor)]

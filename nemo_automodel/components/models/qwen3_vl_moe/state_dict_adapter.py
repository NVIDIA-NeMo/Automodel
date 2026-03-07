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
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe import state_dict_utils
from nemo_automodel.components.moe.config import MoEConfig


class Qwen3VLMoeStateDictAdapter(StateDictAdapter):
    """Converts between HF Qwen3-VL checkpoints and grouped-experts native format.
    Qwen3-VL HF have aggregated expert weights across all experts.
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
            if ".mlp.experts.gate_and_up_projs" in fqn or ".mlp.experts.down_projs" in fqn:
                layer_num = re.search(r"layers\.(\d+)", fqn).group(1)
                which = "gate_up_proj" if "gate_and_up_projs" in fqn else "down_proj"
                if device_mesh is not None:
                    n_experts = self.moe_config.n_routed_experts
                    # Aggregate this layer's expert tensor only for the current key, then free temps.
                    global_tensor = torch.zeros(
                        (n_experts, tensor.shape[1], tensor.shape[2]), dtype=self.dtype, device="cpu"
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

                    # If distributed is initialized and we have an ep dimension, gather all slices.
                    if dist.is_initialized() and "ep" in device_mesh.mesh_dim_names:
                        try:
                            ep_dim = device_mesh.mesh_dim_names.index("ep")
                            ep_group = device_mesh.get_group(ep_dim)
                        except Exception:
                            ep_group = None

                        if ep_group is not None:
                            # Materialize any DTensors to plain CPU tensors before pickling via all_gather_object.
                            #
                            # In multi-node runs ep_shard_size > 1 (= dp_cp_size / ep_size, e.g. 8 on 64 GPUs
                            # with pp=2, ep=4), so expert weights are sharded on BOTH ep (Shard(0)) and
                            # ep_shard (Shard(1)).  After split_experts_weights_dtensor_aware each element of
                            # split_weights is still a DTensor sharded along the ep_shard dimension whose
                            # local slice is only 1/ep_shard_size of the full weight row.
                            #
                            # - to_local()  → returns only the local ep_shard slice  → shape mismatch on copy_
                            # - .cpu()      → keeps the DTensor wrapper              → mixed-tensor error on copy_
                            # - full_tensor() → all-gathers ALL shard dims → full plain CPU tensor  ✓
                            plain_weights = [
                                w.full_tensor().cpu() if state_dict_utils.is_dtensor(w) else w.cpu()
                                for w in split_weights
                            ]
                            payload = (expert_ids, plain_weights)
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
                    del split_weights
                    del expert_ids

                    key = f"{prefix}language_model.layers.{layer_num}.mlp.experts.{which}"
                    hf_state_dict[key] = global_tensor
                    del global_tensor
                else:
                    converted_tensors = self.convert_single_tensor_to_hf(
                        fqn, tensor, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
                    )
                    for key, value in converted_tensors:
                        hf_state_dict[key] = value
            else:
                converted_tensors = self.convert_single_tensor_to_hf(
                    fqn, tensor, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
                )
                for key, value in converted_tensors:
                    hf_state_dict[key] = value

        if exclude_key_regex:
            import re as _re

            hf_state_dict = {k: v for k, v in hf_state_dict.items() if not _re.match(exclude_key_regex, k)}

        return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
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

        # Pre-compute ep_shard slice parameters once for all expert keys.
        # In multi-node runs ep_shard_size > 1: FSDP shards expert weights along dim 1.
        # from_hf must provide that dim-1 shard to create_dtensor_from_local so the
        # resulting DTensor has the correct global shape [n_experts, full_inter, hidden].
        ep_shard_rank = 0
        ep_shard_size = 1
        if device_mesh is not None and "ep_shard" in device_mesh.mesh_dim_names:
            ep_shard_sub = state_dict_utils.get_submesh(device_mesh, ("ep_shard",))
            if ep_shard_sub.size() > 1:
                ep_shard_rank = ep_shard_sub.get_local_rank()
                ep_shard_size = ep_shard_sub.size()

        state_dict: dict[str, Any] = {}
        for key, value in hf_state_dict.items():
            match = re.match(
                r"(model\.)?language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$",
                key,
            )
            if match:
                _, layer_num, which = match.groups()
                tensor = value
                local_tensor = tensor[start_expert:end_expert].to(self.dtype)
                # Also slice along dim 1 for ep_shard: FSDP shards expert dim 1 across ep_shard ranks.
                if ep_shard_size > 1:
                    assert local_tensor.shape[1] % ep_shard_size == 0, (
                        f"Expert dim 1 ({local_tensor.shape[1]}) must be divisible by ep_shard_size ({ep_shard_size})"
                    )
                    chunk = local_tensor.shape[1] // ep_shard_size
                    local_tensor = local_tensor[:, ep_shard_rank * chunk : (ep_shard_rank + 1) * chunk, :]
                native_key = f"{model_prefix}language_model.layers.{layer_num}.mlp.experts."
                native_key += "gate_and_up_projs" if which == "gate_up_proj" else "down_projs"
                state_dict[native_key] = state_dict_utils.create_dtensor_from_local(local_tensor, device_mesh, rank)
                continue

            if key.endswith("_scale_inv"):
                continue

            # Preserve non-expert tensors, ensuring the same model prefix convention.
            if key.startswith("model."):
                state_dict[key] = value
            else:
                state_dict[f"{model_prefix}{key}"] = value

        return state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single native tensor back to the aggregated HF format."""
        prefix = "model." if self._uses_model_prefix else ""
        exclude_key_regex = kwargs.get("exclude_key_regex")

        if ".mlp.experts.gate_and_up_projs" in fqn:
            layer_num = re.search(r"layers\.(\d+)", fqn).group(1)
            key = f"{prefix}language_model.layers.{layer_num}.mlp.experts.gate_up_proj"
            if state_dict_utils.is_dtensor(tensor):
                tensor = tensor.to_local()
            result = [(key, tensor.to(self.dtype))]
        elif ".mlp.experts.down_projs" in fqn:
            layer_num = re.search(r"layers\.(\d+)", fqn).group(1)
            key = f"{prefix}language_model.layers.{layer_num}.mlp.experts.down_proj"
            if state_dict_utils.is_dtensor(tensor):
                tensor = tensor.to_local()
            result = [(key, tensor.to(self.dtype))]
        else:
            result = [(fqn, tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]
        return result

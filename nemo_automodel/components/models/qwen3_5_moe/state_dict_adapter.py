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

All other keys (attention, linear_attn/GatedDeltaNet, norms, embeddings, vision
encoder) pass through unchanged. The HF VLM checkpoint stores the language
model head as ``model.lm_head`` while Automodel registers it on the outer model
as ``lm_head``.
"""

import re
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.gated_delta_net_fp32 import (
    forced_gated_delta_net_fp32_dtype_mapping,
    route_fp32_holder_key,
    strip_fp32_holder_key,
    upcast_gated_delta_net_fp32_state_tensor,
)
from nemo_automodel.components.models.qwen3_5.state_dict_adapter import (
    map_qwen3_5_mtp_from_hf_key,
    map_qwen3_5_mtp_to_hf_key,
)
from nemo_automodel.components.moe import state_dict_utils
from nemo_automodel.components.moe.layers import MoEConfig


def _strip_fp32_params(key: str) -> str:
    """Strip the fp32 holder segment from GDN state-dict keys."""
    return strip_fp32_holder_key(key)


def _route_fp32_params(key: str) -> str:
    """Route bare GDN fp32 params into the holder used by the native module."""
    return route_fp32_holder_key(key)


class Qwen3_5MoeStateDictAdapter(StateDictAdapter):
    """Converts between HF Qwen3.5-MoE checkpoints and the NeMo native format.

    HF Qwen3.5-MoE stores expert weights as **aggregated 3-D tensors**:

        model.language_model.layers.{L}.mlp.experts.gate_up_proj   # [n_experts, 2*moe_inter, hidden]
        model.language_model.layers.{L}.mlp.experts.down_proj      # [n_experts, hidden, moe_inter]

    NeMo uses a different naming convention **and transposed layout** (x @ weight):

        model.language_model.layers.{L}.mlp.experts.gate_and_up_projs  # [n_experts, hidden, 2*moe_inter]
        model.language_model.layers.{L}.mlp.experts.down_projs         # [n_experts, moe_inter, hidden]

    Both expert tensors require `.transpose(1, 2)` when converting between formats.

    Loading paths:
      DCP path:  to_hf renames+transposes native→HF, DCP loads into DTensors,
                 from_hf renames+transposes HF→native. DTensors pass through.
      Init path: from_hf receives plain tensors from safetensors, slices to local EP
                 shard, transposes, and wraps in DTensor via create_dtensor_from_local.

    Additionally, the shared expert uses singular in HF and plural in NeMo:

        HF:   .mlp.shared_expert.{gate,up,down}_proj.weight
        NeMo: .mlp.shared_experts.{gate,up,down}_proj.weight
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
        self, state_dict: dict[str, Any], exclude_key_regex: Optional[str] = None, quantization: bool = False, **kwargs
    ) -> dict[str, Any]:
        """Rename native keys to HF keys and transpose expert tensors. No comms needed."""
        hf_state_dict: dict[str, Any] = {}
        for fqn, tensor in state_dict.items():
            for key, value in self.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=exclude_key_regex):
                hf_state_dict[key] = value
        return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Rename HF keys to native keys and transpose expert tensors.

        DTensors (DCP path): rename + transpose, no slicing — DCP handles sharding.
        Plain tensors (init path): slice to local EP shard, transpose, create DTensor.
        """
        self._uses_model_prefix = any(key.startswith("model.") for key in hf_state_dict if not key.startswith("mtp."))
        model_prefix = "model." if self._uses_model_prefix else ""

        n_experts = self.moe_config.n_routed_experts

        # Pre-compute EP slicing params (only used for plain tensor path)
        start_expert, end_expert, rank = 0, n_experts, None
        ep_shard_rank, ep_shard_size = 0, 1
        if device_mesh is not None:
            start_expert, end_expert = state_dict_utils.get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            rank = (
                state_dict_utils.get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )
            if "ep_shard" in device_mesh.mesh_dim_names:
                ep_shard_sub = state_dict_utils.get_submesh(device_mesh, ("ep_shard",))
                if ep_shard_sub.size() > 1:
                    ep_shard_rank = ep_shard_sub.get_local_rank()
                    ep_shard_size = ep_shard_sub.size()

        state_dict: dict[str, Any] = {}
        mtp_expert_parts: dict[str, dict[str, dict[int, torch.Tensor]]] = {}

        def store_native_key(native_key: str, tensor: Any) -> None:
            native_key = route_fp32_holder_key(native_key)
            state_dict[native_key] = upcast_gated_delta_net_fp32_state_tensor(native_key, tensor)

        for key, value in hf_state_dict.items():
            mapped_mtp_key = map_qwen3_5_mtp_from_hf_key(key)
            if mapped_mtp_key != key:
                store_native_key(mapped_mtp_key, value)
                continue

            match = re.match(
                r"(?:model\.)?language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$",
                key,
            )
            mtp_match = re.match(r"mtp\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$", key)
            if match or mtp_match:
                active_match = match or mtp_match
                layer_num = active_match.group(1)
                which = active_match.group(2)
                if mtp_match:
                    native_key = f"mtp.layers.{layer_num}.mlp.experts."
                else:
                    native_key = f"{model_prefix}language_model.layers.{layer_num}.mlp.experts."
                native_key += "gate_and_up_projs" if which == "gate_up_proj" else "down_projs"

                if state_dict_utils.is_dtensor(value):
                    # DCP path: already sharded DTensor — rename + transpose.
                    state_dict[native_key] = value.transpose(1, 2)
                else:
                    # Init path: plain tensor — slice to local EP shard, transpose.
                    local_tensor = value[start_expert:end_expert].transpose(1, 2).to(self.dtype)
                    if ep_shard_size > 1:
                        assert local_tensor.shape[1] % ep_shard_size == 0
                        chunk = local_tensor.shape[1] // ep_shard_size
                        local_tensor = local_tensor[:, ep_shard_rank * chunk : (ep_shard_rank + 1) * chunk, :]
                    state_dict[native_key] = state_dict_utils.create_dtensor_from_local(local_tensor, device_mesh, rank)
                continue

            mtp_split_match = re.match(
                r"mtp\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$",
                key,
            )
            if mtp_split_match:
                layer_num = mtp_split_match.group(1)
                expert_num = int(mtp_split_match.group(2))
                which = mtp_split_match.group(3)
                if not state_dict_utils.should_load_expert_for_rank(expert_num, device_mesh, n_experts):
                    continue
                parts = mtp_expert_parts.setdefault(layer_num, {"gate_proj": {}, "up_proj": {}, "down_proj": {}})
                parts[which][expert_num] = value
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

            if mapped_key.startswith("mtp."):
                store_native_key(mapped_key, value)
            elif mapped_key.startswith("model.lm_head."):
                store_native_key(mapped_key.removeprefix("model."), value)
            elif mapped_key.startswith("lm_head."):
                store_native_key(mapped_key, value)
            elif key.startswith("model."):
                store_native_key(mapped_key, value)
            else:
                store_native_key(
                    f"{model_prefix}{mapped_key}" if not mapped_key.startswith("model.") else mapped_key, value
                )

        for layer_num, parts in mtp_expert_parts.items():
            expert_ids = sorted(set(parts["gate_proj"]) | set(parts["up_proj"]) | set(parts["down_proj"]))
            gate_up_tensors = []
            down_tensors = []
            for expert_id in expert_ids:
                if expert_id not in parts["gate_proj"] or expert_id not in parts["up_proj"]:
                    raise RuntimeError(f"Missing gate/up MTP expert weights for layer {layer_num}, expert {expert_id}")
                if expert_id not in parts["down_proj"]:
                    raise RuntimeError(f"Missing down MTP expert weight for layer {layer_num}, expert {expert_id}")
                gate_t = parts["gate_proj"][expert_id].transpose(0, 1)
                up_t = parts["up_proj"][expert_id].transpose(0, 1)
                down_t = parts["down_proj"][expert_id].transpose(0, 1)
                gate_up_tensors.append(torch.cat((gate_t, up_t), dim=1))
                down_tensors.append(down_t)

            gate_up_tensor = torch.stack(gate_up_tensors, dim=0).to(self.dtype)
            down_tensor = torch.stack(down_tensors, dim=0).to(self.dtype)
            if ep_shard_size > 1:
                for tensor_name, local_tensor in (("gate_and_up_projs", gate_up_tensor), ("down_projs", down_tensor)):
                    if local_tensor.shape[1] % ep_shard_size != 0:
                        raise ValueError(
                            f"MTP {tensor_name} dim 1 ({local_tensor.shape[1]}) is not divisible by "
                            f"ep_shard_size={ep_shard_size}"
                        )
                gate_chunk = gate_up_tensor.shape[1] // ep_shard_size
                down_chunk = down_tensor.shape[1] // ep_shard_size
                gate_up_tensor = gate_up_tensor[:, ep_shard_rank * gate_chunk : (ep_shard_rank + 1) * gate_chunk, :]
                down_tensor = down_tensor[:, ep_shard_rank * down_chunk : (ep_shard_rank + 1) * down_chunk, :]

            state_dict[f"mtp.layers.{layer_num}.mlp.experts.gate_and_up_projs"] = (
                state_dict_utils.create_dtensor_from_local(gate_up_tensor, device_mesh, rank)
            )
            state_dict[f"mtp.layers.{layer_num}.mlp.experts.down_projs"] = state_dict_utils.create_dtensor_from_local(
                down_tensor, device_mesh, rank
            )

        return state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Rename a single native key to HF format and transpose expert tensors."""
        exclude_key_regex = kwargs.get("exclude_key_regex")

        new_fqn = fqn
        value = tensor
        # MTP experts use the SAME grouped layout as the main decoder layers in the
        # HF checkpoint (e.g. ``mtp.layers.0.mlp.experts.{gate_up_proj,down_proj}``),
        # so they fall through to the generic grouped handling below. Splitting them
        # into per-expert keys here produced destinations like
        # ``mtp.layers.0.mlp.experts.224.down_proj.weight`` that don't exist in the
        # checkpoint, raising "Missing key in checkpoint state_dict" on load under
        # expert parallelism (AM-442).
        if ".mlp.experts.gate_and_up_projs" in fqn:
            new_fqn = fqn.replace(".mlp.experts.gate_and_up_projs", ".mlp.experts.gate_up_proj")
            value = tensor.transpose(1, 2)
        elif ".mlp.experts.down_projs" in fqn:
            new_fqn = fqn.replace(".mlp.experts.down_projs", ".mlp.experts.down_proj")
            value = tensor.transpose(1, 2)

        # Apply shared_experts → shared_expert reverse mapping
        for pattern, replacement in self.internal_to_hf_map.items():
            if pattern in new_fqn:
                new_fqn = new_fqn.replace(pattern, replacement)
                break

        # Hide the GatedDeltaNet fp32 holder wrapping so saved checkpoints keep the
        # bare HF key (``...linear_attn.A_log`` instead of
        # ``...linear_attn._fp32_params.A_log``) and stay directly HF-loadable.
        new_fqn = strip_fp32_holder_key(new_fqn)

        new_fqn = map_qwen3_5_mtp_to_hf_key(new_fqn)
        value = upcast_gated_delta_net_fp32_state_tensor(new_fqn, value)

        if exclude_key_regex and re.match(exclude_key_regex, new_fqn):
            return []
        return [(new_fqn, value)]

    def forced_hf_dtype_mapping(self, state_dict: dict[str, Any]) -> dict[str, str]:
        """Return HF export dtype overrides for intrinsically-fp32 GDN tensors."""
        return forced_gated_delta_net_fp32_dtype_mapping(state_dict)

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""State-dict adapter between HF Inkling checkpoints and the grouped-experts format.

Only the routed-expert tensors differ between the two formats; every other key
(router, shared experts, attention, short convolutions, norms, vision/audio
towers, embeddings, lm_head) is identical and passes through unchanged.

HF stacked layout (per sparse decoder layer ``L``)::

    model.language_model.layers.{L}.mlp.experts.gate_up_proj   [n_experts, 2*inter, dim]
    model.language_model.layers.{L}.mlp.experts.down_proj      [n_experts, dim, inter]

Native grouped-experts layout (transposed on the last two dims)::

    model.language_model.layers.{L}.mlp.experts.gate_and_up_projs  [n_experts, dim, 2*inter]
    model.language_model.layers.{L}.mlp.experts.down_projs         [n_experts, inter, dim]
"""

import re
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe import state_dict_utils
from nemo_automodel.components.moe.config import MoEConfig

_HF_EXPERT_RE = re.compile(r"(?:model\.)?language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")


class InklingStateDictAdapter(StateDictAdapter):
    """Convert Inkling routed-expert weights between HF and native grouped layouts."""

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.float32,
    ) -> None:
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
        """Rename native keys to HF keys, transposing the routed-expert tensors."""
        hf_state_dict = {}
        for fqn, tensor in state_dict.items():
            for key, value in self.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=exclude_key_regex):
                hf_state_dict[key] = value
        return hf_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Rename (and transpose) a single native key to HF format."""
        exclude_key_regex = kwargs.get("exclude_key_regex")

        new_fqn = fqn
        value = tensor
        if ".mlp.experts.gate_and_up_projs" in fqn:
            new_fqn = fqn.replace(".mlp.experts.gate_and_up_projs", ".mlp.experts.gate_up_proj")
            value = _transpose_last2(tensor)
        elif ".mlp.experts.down_projs" in fqn:
            new_fqn = fqn.replace(".mlp.experts.down_projs", ".mlp.experts.down_proj")
            value = _transpose_last2(tensor)

        if exclude_key_regex and re.match(exclude_key_regex, new_fqn):
            return []
        return [(new_fqn, value)]

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional[DeviceMesh] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Rename HF keys to native keys, transposing (and EP-slicing) routed experts."""
        self._uses_model_prefix = any(key.startswith("model.") for key in hf_state_dict)
        model_prefix = "model." if self._uses_model_prefix else ""

        n_experts = self.moe_config.n_routed_experts

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
        for key, value in hf_state_dict.items():
            match = _HF_EXPERT_RE.match(key)
            if match:
                layer_num = match.group(1)
                which = match.group(2)
                native_key = f"{model_prefix}language_model.layers.{layer_num}.mlp.experts."
                native_key += "gate_and_up_projs" if which == "gate_up_proj" else "down_projs"

                if state_dict_utils.is_dtensor(value):
                    # DCP path: already a sharded DTensor in native layout — rename only.
                    state_dict[native_key] = value
                else:
                    # Init path: plain HF tensor. Transpose to native layout, then
                    # slice to the local EP shard on the expert dimension.
                    local_tensor = _transpose_last2(value).to(self.dtype)
                    local_tensor = local_tensor[start_expert:end_expert].contiguous()
                    if ep_shard_size > 1:
                        assert local_tensor.shape[1] % ep_shard_size == 0
                        chunk = local_tensor.shape[1] // ep_shard_size
                        local_tensor = local_tensor[:, ep_shard_rank * chunk : (ep_shard_rank + 1) * chunk, :]
                    if device_mesh is not None:
                        local_tensor = state_dict_utils.create_dtensor_from_local(local_tensor, device_mesh, rank)
                    state_dict[native_key] = local_tensor
                continue

            if key.endswith("_scale_inv"):
                continue

            if key.startswith("model.") or key.startswith("lm_head."):
                state_dict[key] = value
            else:
                state_dict[f"{model_prefix}{key}"] = value

        return state_dict


def _transpose_last2(tensor: Any) -> Any:
    """Transpose the last two dims of a stacked expert tensor, keeping it contiguous."""
    return tensor.transpose(-1, -2).contiguous()

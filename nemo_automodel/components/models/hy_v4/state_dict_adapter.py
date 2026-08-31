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

"""HY V4 checkpoint conversion.

The public checkpoint already uses AutoModel-compatible names for every
non-expert tensor.  Routed experts are grouped in both formats but store their
matrix axes in opposite orders::

    HF gate_up_proj:            [experts, 2 * intermediate, hidden]
    native gate_and_up_projs:   [experts, hidden, 2 * intermediate]
    HF down_proj:               [experts, hidden, intermediate]
    native down_projs:          [experts, intermediate, hidden]

This adapter keeps the transpose as a view on DCP/write-through loads and
slices plain checkpoint tensors before wrapping the local EP shard.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe import state_dict_utils
from nemo_automodel.components.moe.config import MoEConfig

__all__ = ["HyV4StateDictAdapter"]

_HF_GATE_UP = ".mlp.experts.gate_up_proj"
_HF_DOWN = ".mlp.experts.down_proj"
_NATIVE_GATE_UP = ".mlp.experts.gate_and_up_projs"
_NATIVE_DOWN = ".mlp.experts.down_projs"
_HF_SINK = ".self_attn.learnable_sink_param"
_NATIVE_SINK = ".self_attn.learnable_sink_param.weight"


class HyV4StateDictAdapter(StateDictAdapter):
    """Convert HY V4's grouped Hugging Face experts to AutoModel layout."""

    _supports_write_through_checkpoint_load = True
    _supports_checkpoint_load_without_full_copy = True

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True

    @property
    def supports_write_through_checkpoint_load(self) -> bool:
        """Return whether the configured grouped-expert backend preserves views."""
        return self._supports_write_through_checkpoint_load and self._expert_backend_preserves_checkpoint_views

    @property
    def supports_checkpoint_load_without_full_copy(self) -> bool:
        """Return whether DCP can load without allocating grouped expert copies."""
        return self._supports_checkpoint_load_without_full_copy and self._expert_backend_preserves_checkpoint_views

    @property
    def _expert_backend_preserves_checkpoint_views(self) -> bool:
        """Return whether expert storage matches the validated grouped tensor layout."""
        return self.backend.experts not in {"te", "mok"} and self.backend.dispatcher != "mok"

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Expose native weights through checkpoint-layout aliases.

        Grouped expert tensors are transposed views between native
        ``[experts, input, output]`` and checkpoint ``[experts, output, input]``
        layouts. Non-expert tensors and the FP32 sink remain identity aliases.
        """
        output: dict[str, Any] = {}
        for fqn, tensor in state_dict.items():
            for key, value in self.convert_single_tensor_to_hf(
                fqn,
                tensor,
                exclude_key_regex=exclude_key_regex,
                **kwargs,
            ):
                output[key] = value
        return output

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convert checkpoint tensors to native grouped-expert layouts.

        Args:
            hf_state_dict: Checkpoint mapping. Expert tensors use
                ``[experts, output, input]``; other tensor layouts are unchanged.
            device_mesh: Optional EP/EP-shard mesh used to select local experts
                and the sharded native output axis.
            **kwargs: Reserved checkpoint-loader arguments.

        Returns:
            Native mapping. DCP DTensors and unsharded tensors are represented
            by transpose/identity views whenever the backend capability is true.
        """
        del kwargs
        n_experts = int(self.moe_config.n_routed_experts)
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
                ep_shard_mesh = state_dict_utils.get_submesh(device_mesh, ("ep_shard",))
                ep_shard_rank = ep_shard_mesh.get_local_rank()
                ep_shard_size = ep_shard_mesh.size()

        output: dict[str, Any] = {}
        for key, tensor in hf_state_dict.items():
            if _HF_GATE_UP in key:
                native_key = key.replace(_HF_GATE_UP, _NATIVE_GATE_UP)
            elif _HF_DOWN in key:
                native_key = key.replace(_HF_DOWN, _NATIVE_DOWN)
            elif key.endswith(_HF_SINK):
                output[key[: -len(_HF_SINK)] + _NATIVE_SINK] = tensor
                continue
            else:
                output[key] = tensor
                continue

            # DCP loaded the HF transpose view directly into native parameter
            # storage. Transposing it back restores the original DTensor layout
            # without a model-sized allocation.
            if state_dict_utils.is_dtensor(tensor):
                output[native_key] = tensor.transpose(-2, -1)
                continue

            local_tensor = tensor[start_expert:end_expert].transpose(-2, -1)
            if ep_shard_size > 1:
                if local_tensor.shape[1] % ep_shard_size:
                    raise ValueError(
                        f"Cannot shard {native_key} dimension {local_tensor.shape[1]} over ep_shard={ep_shard_size}."
                    )
                shard_width = local_tensor.shape[1] // ep_shard_size
                shard_start = ep_shard_rank * shard_width
                local_tensor = local_tensor[:, shard_start : shard_start + shard_width, :]
            local_tensor = local_tensor.to(dtype=self.dtype)
            output[native_key] = state_dict_utils.create_dtensor_from_local(local_tensor, device_mesh, rank)
        return output

    def convert_single_tensor_to_hf(
        self,
        fqn: str,
        tensor: Any,
        **kwargs: Any,
    ) -> list[tuple[str, Any]]:
        """Map one native tensor to its checkpoint key and layout view.

        Args:
            fqn: Native fully qualified tensor name.
            tensor: Native tensor. Grouped experts use
                ``[experts, input, output]``; other layouts are unchanged.
            **kwargs: May contain an ``exclude_key_regex`` filter.

        Returns:
            Zero or one checkpoint key/tensor pair. Expert outputs are
            transposed views and all other outputs alias ``tensor``.
        """
        exclude_key_regex = kwargs.get("exclude_key_regex")
        if _NATIVE_GATE_UP in fqn:
            hf_fqn = fqn.replace(_NATIVE_GATE_UP, _HF_GATE_UP)
            tensor = tensor.transpose(-2, -1)
        elif _NATIVE_DOWN in fqn:
            hf_fqn = fqn.replace(_NATIVE_DOWN, _HF_DOWN)
            tensor = tensor.transpose(-2, -1)
        elif fqn.endswith(_NATIVE_SINK):
            hf_fqn = fqn[: -len(_NATIVE_SINK)] + _HF_SINK
        else:
            hf_fqn = fqn

        if exclude_key_regex and re.match(exclude_key_regex, hf_fqn):
            return []
        return [(hf_fqn, tensor)]

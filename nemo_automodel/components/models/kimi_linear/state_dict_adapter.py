# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""State-dict adapter for Kimi Linear MoE checkpoints."""

from __future__ import annotations

import re
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

_HF_TO_GENERIC_EXPERT_PROJ = {
    "w1": "gate_proj",
    "w2": "down_proj",
    "w3": "up_proj",
}
_GENERIC_TO_HF_EXPERT_PROJ = {value: key for key, value in _HF_TO_GENERIC_EXPERT_PROJ.items()}
_FP32_KEY_PARTS = ("A_log", "dt_bias", "e_score_correction_bias")
_KDA_FP32_HOLDER = re.compile(r"(\.self_attn)\._fp32_params\.")
_KDA_FP32_PARAM_NAMES = ("A_log", "dt_bias")


def _upcast_fp32_state_tensor(key: str, value: Any) -> Any:
    if isinstance(value, torch.Tensor) and any(part in key for part in _FP32_KEY_PARTS):
        return value.to(torch.float32)
    return value


def _strip_kda_fp32_holder(key: str) -> str:
    return _KDA_FP32_HOLDER.sub(r"\1.", key)


def _route_kda_fp32_holder(key: str) -> str:
    if not key.endswith(_KDA_FP32_PARAM_NAMES):
        return key
    if "._fp32_params." in key:
        return key
    if ".self_attn." not in key:
        return key
    head, tail = key.rsplit(".self_attn.", 1)
    return f"{head}.self_attn._fp32_params.{tail}"


# The decoder block calls its feed-forward ``mlp`` for both the dense and the MoE
# case, so only the MoE-owned children are renamed. Dense layers keep
# ``mlp.{gate,up,down}_proj`` on both sides and must not be touched, which is why
# these are exact path segments rather than a blanket ``mlp`` rewrite.
_MOE_CHILD_SEGMENTS = ("gate.", "shared_experts.")


def _hf_moe_key_to_native(key: str) -> str:
    """Rewrite a checkpoint MoE key onto the decoder block's ``mlp`` submodule."""
    for segment in _MOE_CHILD_SEGMENTS:
        key = key.replace(f".block_sparse_moe.{segment}", f".mlp.{segment}")
    return key


def _native_moe_key_to_hf(key: str) -> str:
    """Inverse of :func:`_hf_moe_key_to_native`."""
    for segment in _MOE_CHILD_SEGMENTS:
        key = key.replace(f".mlp.{segment}", f".block_sparse_moe.{segment}")
    return key


class KimiLinearStateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Convert Kimi Linear HF split experts to Automodel grouped experts.

    HF stores routed experts as per-expert Kimi names:

    * ``block_sparse_moe.experts.{E}.w1.weight``: SwiGLU gate projection, shape [inter, hidden].
    * ``block_sparse_moe.experts.{E}.w3.weight``: SwiGLU up projection, shape [inter, hidden].
    * ``block_sparse_moe.experts.{E}.w2.weight``: down projection, shape [hidden, inter].

    Automodel stores grouped experts as:

    * ``mlp.experts.gate_and_up_projs`` with shape [experts, hidden, 2 * inter].
    * ``mlp.experts.down_projs`` with shape [experts, inter, hidden].

    The decoder block names both its dense and its MoE feed-forward ``mlp`` (the
    naming the custom-MoE parallelizer looks for), so the checkpoint's
    ``block_sparse_moe`` path segment is translated here in both directions.
    """

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

    @property
    def _expert_path_segment(self) -> str:
        return "mlp.experts"

    def _map_hf_expert_key_to_generic(self, key: str) -> str:
        match = re.match(
            r"(?P<prefix>(?:model\.)?layers\.\d+)\.block_sparse_moe\.experts\.(?P<expert>\d+)\."
            r"(?P<proj>w1|w2|w3)\.weight$",
            key,
        )
        if match is None:
            return key
        projection = _HF_TO_GENERIC_EXPERT_PROJ[match.group("proj")]
        return f"{match.group('prefix')}.mlp.experts.{match.group('expert')}.{projection}.weight"

    def _map_generic_expert_key_to_hf(self, key: str) -> str:
        match = re.match(
            r"(?P<prefix>(?:model\.)?layers\.\d+)\.mlp\.experts\.(?P<expert>\d+)\."
            r"(?P<proj>gate_proj|up_proj|down_proj)\.weight$",
            key,
        )
        if match is None:
            return key
        projection = _GENERIC_TO_HF_EXPERT_PROJ[match.group("proj")]
        return f"{match.group('prefix')}.block_sparse_moe.experts.{match.group('expert')}.{projection}.weight"

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        quantization: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convert Automodel native tensors to Kimi HF checkpoint keys."""
        previous_device_mesh = getattr(self, "_active_device_mesh", None)
        self._active_device_mesh = kwargs.get("device_mesh")
        hf_state_dict: dict[str, Any] = {}
        try:
            for fqn, tensor in state_dict.items():
                converted_tensors = self.convert_single_tensor_to_hf(
                    fqn,
                    tensor,
                    exclude_key_regex=exclude_key_regex,
                    quantization=quantization,
                    **kwargs,
                )
                for key, value in converted_tensors:
                    hf_state_dict[key] = value
        finally:
            self._active_device_mesh = previous_device_mesh
        return hf_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs: Any) -> list[tuple[str, Any]]:
        """Convert one Automodel tensor to one or more Kimi HF tensors.

        Args:
            fqn: Fully qualified native tensor name.
            tensor: Native tensor. Grouped routed expert tensors use [experts, hidden, 2 * inter]
                for gate/up and [experts, inter, hidden] for down.
            **kwargs: Adapter options forwarded by checkpoint save/load.

        Returns:
            HF key/tensor pairs. Split expert tensors use Kimi ``w1``/``w2``/``w3`` names.
        """
        exclude_key_regex = kwargs.get("exclude_key_regex", None)
        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        result = expert_result if expert_result is not None else [(fqn, tensor)]
        result = [(_strip_kda_fp32_holder(key), value) for key, value in result]
        result = [(self._map_generic_expert_key_to_hf(key), value) for key, value in result]
        result = [(_native_moe_key_to_hf(key), value) for key, value in result]
        if exclude_key_regex:
            result = [(key, value) for key, value in result if not re.match(exclude_key_regex, key)]
        return result

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convert Kimi HF checkpoint keys to Automodel native keys.

        Args:
            hf_state_dict: HF state dict whose routed expert tensors use split Kimi names.
            device_mesh: Optional EP/FSDP mesh used to load only local expert shards.
            **kwargs: Adapter options forwarded by checkpoint load.

        Returns:
            Native state dict with grouped routed expert tensors.
        """
        self._uses_model_prefix = any(key.startswith("model.") for key in hf_state_dict)
        generic_state_dict = {
            _route_kda_fp32_holder(_hf_moe_key_to_native(self._map_hf_expert_key_to_generic(key))): (
                _upcast_fp32_state_tensor(key, value)
            )
            for key, value in hf_state_dict.items()
        }
        return self._from_hf_w_merged_experts(generic_state_dict, device_mesh)

    def _split_experts_weights(self, weight: torch.Tensor, n_experts: int) -> list[torch.Tensor]:
        """Split grouped experts, tolerating DTensors whose mesh dim is not named ``ep``.

        Args:
            weight: Grouped routed-expert tensor of shape [experts, ...]. May be a plain tensor or a DTensor
                sharded or replicated over the expert axis; for Shard(0), the local shard covers this rank's
                expert slice.
            n_experts: Global number of routed experts.

        Returns:
            List of per-expert tensors of shape [...] for the experts local to this rank.
        """
        from torch.distributed._tensor.placement_types import Replicate, Shard

        from nemo_automodel.components.moe.state_dict_utils import get_submesh, is_dtensor

        if not is_dtensor(weight) or "ep" in weight.device_mesh.mesh_dim_names:
            return super()._split_experts_weights(weight, n_experts)

        local_tensor = weight.to_local()
        placement = weight.placements[-1] if weight.placements else None
        if isinstance(placement, Replicate):
            start_expert = 0
            local_n_experts = n_experts
        elif isinstance(placement, Shard) and placement.dim == 0:
            mesh = getattr(self, "_active_device_mesh", None)
            if mesh is not None and "ep" in mesh.mesh_dim_names:
                ep_mesh = get_submesh(mesh, ("ep",))
                mesh_rank = ep_mesh.get_local_rank()
                mesh_size = ep_mesh.size()
            else:
                mesh_rank = weight.device_mesh.get_local_rank()
                mesh_size = weight.device_mesh.size()
            experts_per_rank = n_experts // mesh_size
            remainder = n_experts % mesh_size
            if mesh_rank < remainder:
                local_n_experts = experts_per_rank + 1
                start_expert = mesh_rank * local_n_experts
            else:
                local_n_experts = experts_per_rank
                start_expert = remainder * (experts_per_rank + 1) + (mesh_rank - remainder) * experts_per_rank
        else:
            start_expert = 0
            local_n_experts = local_tensor.shape[0]

        if local_tensor.shape[0] != local_n_experts:
            raise ValueError(
                f"Expected local Kimi expert tensor first dimension to be {local_n_experts} "
                f"(experts {start_expert}:{start_expert + local_n_experts}), got {local_tensor.shape[0]}"
            )

        self._last_expert_ids = list(range(start_expert, start_expert + local_n_experts))
        return [local_tensor[i] for i in range(local_n_experts)]

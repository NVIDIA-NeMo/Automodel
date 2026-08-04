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

"""State-dict conversion for native Kimi K3 and its MXFP4 checkpoint."""

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
_FP32_KEY_PARTS = (
    "A_log",
    "dt_bias",
    "e_score_correction_bias",
    "q_conv1d.weight",
    "k_conv1d.weight",
    "v_conv1d.weight",
    "o_norm.weight",
)
_KDA_FP32_HOLDER = re.compile(r"(\.self_attn)\._fp32_params\.")
_KDA_FP32_OP_HOLDER = re.compile(r"(\.self_attn\.(?:q_conv1d|k_conv1d|v_conv1d|o_norm))\._fp32_params\.")
_KDA_FP32_OP_PARAM = re.compile(r"(\.self_attn\.(?:q_conv1d|k_conv1d|v_conv1d|o_norm))\.(weight)$")
_KDA_FP32_PARAM_NAMES = ("A_log", "dt_bias")
_MXFP4_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def dequantize_mxfp4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Decode K3 ``[out, in / 2]`` MXFP4 bytes into ``[out, in]`` weights."""
    packed = weight_packed.to_local() if hasattr(weight_packed, "to_local") else weight_packed
    scales = weight_scale.to_local() if hasattr(weight_scale, "to_local") else weight_scale
    if packed.ndim != 2 or scales.ndim != 2:
        raise ValueError(f"K3 MXFP4 tensors must be rank two, got {packed.shape=} and {scales.shape=}.")
    if packed.shape[0] != scales.shape[0] or packed.shape[1] != scales.shape[1] * 16:
        raise ValueError(f"Incompatible K3 MXFP4 shapes: {packed.shape=} and {scales.shape=}.")

    output_features, groups = scales.shape
    blocks = packed.reshape(output_features, groups, 16).to(torch.uint8)
    lookup = torch.tensor(_MXFP4_VALUES, dtype=dtype, device=blocks.device)
    unpacked = torch.empty(output_features, groups, 32, dtype=dtype, device=blocks.device)
    unpacked[..., 0::2] = lookup[(blocks & 0x0F).long()]
    unpacked[..., 1::2] = lookup[(blocks >> 4).long()]
    exponents = scales.to(torch.int32).sub(127).unsqueeze(-1)
    return torch.ldexp(unpacked, exponents).reshape(output_features, groups * 32)


def _upcast_fp32_state_tensor(key: str, value: Any) -> Any:
    if isinstance(value, torch.Tensor) and any(part in key for part in _FP32_KEY_PARTS):
        return value.to(torch.float32)
    return value


def _strip_kda_fp32_holder(key: str) -> str:
    key = _KDA_FP32_OP_HOLDER.sub(r"\1.", key)
    return _KDA_FP32_HOLDER.sub(r"\1.", key)


def _route_kda_fp32_holder(key: str) -> str:
    if "._fp32_params." in key:
        return key
    routed = _KDA_FP32_OP_PARAM.sub(r"\1._fp32_params.\2", key)
    if routed != key:
        return routed
    if not key.endswith(_KDA_FP32_PARAM_NAMES):
        return key
    if ".self_attn." not in key:
        return key
    head, tail = key.rsplit(".self_attn.", 1)
    return f"{head}.self_attn._fp32_params.{tail}"


# The decoder block calls its feed-forward ``mlp`` for both the dense and the MoE
# case, so only the MoE-owned children are renamed. Dense layers keep
# ``mlp.{gate,up,down}_proj`` on both sides and must not be touched, which is why
# these are exact path segments rather than a blanket ``mlp`` rewrite.
_MOE_CHILD_SEGMENTS = (
    "gate.",
    "shared_experts.",
    "routed_expert_down_proj.",
    "routed_expert_up_proj.",
    "routed_expert_norm.",
)


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


class KimiK3StateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Convert K3 split/packed experts to AutoModel grouped experts.

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
        self.config = getattr(config, "text_config", config)
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
        if quantization:
            self._mxfp4_load_views: dict[str, torch.Tensor] = {}
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
        split_kwargs = kwargs
        if kwargs.get("quantization", False):
            # K3's packed checkpoint must first load into compact uint8 buffers,
            # but the eventual BF16 values can still be written through views
            # into the model's grouped expert storage. Tell the generic splitter
            # to preserve those views instead of allocating contiguous BF16
            # copies, then retain each view until ``from_hf`` dequantizes it.
            split_kwargs = {**kwargs, "quantization": False}
        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **split_kwargs)
        result = expert_result if expert_result is not None else [(fqn, tensor)]
        result = [(_strip_kda_fp32_holder(key), value) for key, value in result]
        result = [(self._map_generic_expert_key_to_hf(key), value) for key, value in result]
        result = [(_native_moe_key_to_hf(key), value) for key, value in result]
        result = [(self._add_hf_text_prefix(key), value) for key, value in result]
        result = [(key, self._pad_checkpoint_a_log(key, value)) for key, value in result]
        if kwargs.get("quantization", False):
            packed_result: list[tuple[str, Any]] = []
            for key, value in result:
                if re.search(r"\.block_sparse_moe\.experts\.\d+\.w[123]\.weight$", key):
                    load_views = getattr(self, "_mxfp4_load_views", None)
                    if load_views is not None:
                        load_views[self._strip_hf_text_prefix(key)] = value
                    packed_result.extend(self._make_mxfp4_load_destinations(key, value))
                else:
                    packed_result.append((key, value))
            result = packed_result
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
        self._uses_model_prefix = True
        stripped_state_dict = {
            self._strip_hf_text_prefix(key): self._normalize_checkpoint_tensor(key, value)
            for key, value in hf_state_dict.items()
        }
        self._dequantize_packed_experts(stripped_state_dict)
        generic_state_dict = {
            _route_kda_fp32_holder(_hf_moe_key_to_native(self._map_hf_expert_key_to_generic(key))): value
            for key, value in stripped_state_dict.items()
        }
        return self._from_hf_w_merged_experts(generic_state_dict, device_mesh)

    @staticmethod
    def _add_hf_text_prefix(key: str) -> str:
        """Prefix native text keys while leaving vision/projector keys unchanged."""
        if key.startswith(("model.", "lm_head.")):
            return f"language_model.{key}"
        return key

    @staticmethod
    def _strip_hf_text_prefix(key: str) -> str:
        """Remove the K3 checkpoint's ``language_model.`` namespace."""
        return key.removeprefix("language_model.")

    def _normalize_checkpoint_tensor(self, key: str, value: Any) -> Any:
        """Remove K3's zero padding from the per-head KDA decay parameter."""
        value = _upcast_fp32_state_tensor(key, value)
        if not key.endswith(".self_attn.A_log") or not isinstance(value, torch.Tensor):
            return value
        num_heads = self.config.linear_attn_config["num_heads"]
        if value.shape[-1] < num_heads:
            raise ValueError(f"K3 {key} has {value.shape[-1]} entries, expected at least {num_heads}.")
        return value[..., :num_heads]

    def _pad_checkpoint_a_log(self, key: str, value: Any) -> Any:
        """Restore the 128-entry checkpoint storage layout for KDA ``A_log``."""
        if not key.endswith(".self_attn.A_log") or not isinstance(value, torch.Tensor):
            return value
        num_heads = self.config.linear_attn_config["num_heads"]
        checkpoint_size = ((num_heads + 127) // 128) * 128
        if value.shape[-1] == checkpoint_size:
            return value
        if value.shape[-1] != num_heads:
            raise ValueError(f"K3 {key} has {value.shape[-1]} entries, expected {num_heads}.")
        padding = value.new_zeros(*value.shape[:-1], checkpoint_size - num_heads)
        return torch.cat((value, padding), dim=-1)

    @staticmethod
    def _make_mxfp4_load_destinations(key: str, weight: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Create packed checkpoint destinations matching one plain ``[out, in]`` expert weight."""
        local = weight.to_local() if hasattr(weight, "to_local") else weight
        output_features, input_features = local.shape
        if input_features % 32:
            raise ValueError(f"K3 MXFP4 input dimension must be divisible by 32, got {input_features}.")
        return [
            (
                f"{key}_packed",
                torch.empty(
                    output_features,
                    input_features // 2,
                    dtype=torch.uint8,
                    device=local.device,
                ),
            ),
            (
                f"{key}_scale",
                torch.empty(
                    output_features,
                    input_features // 32,
                    dtype=torch.uint8,
                    device=local.device,
                ),
            ),
        ]

    def _dequantize_packed_experts(self, state_dict: dict[str, Any]) -> None:
        """Decode packed experts, writing directly into model views when available."""
        grouped: dict[str, dict[str, torch.Tensor]] = {}
        pattern = re.compile(
            r"(?P<base>model\.layers\.\d+\.block_sparse_moe\.experts\.\d+\.w[123]\.weight)"
            r"_(?P<kind>packed|scale)$"
        )
        for key in list(state_dict):
            match = pattern.fullmatch(key)
            if match is None:
                continue
            grouped.setdefault(match.group("base"), {})[match.group("kind")] = state_dict.pop(key)

        load_views = getattr(self, "_mxfp4_load_views", None) or {}
        for base in list(grouped):
            tensors = grouped.pop(base)
            if set(tensors) != {"packed", "scale"}:
                raise RuntimeError(f"Incomplete K3 MXFP4 pair for {base}: found {sorted(tensors)}.")
            decoded = dequantize_mxfp4(
                tensors["packed"],
                tensors["scale"],
                dtype=self.dtype,
            )
            destination = load_views.get(base)
            if destination is None:
                state_dict[base] = decoded
            else:
                destination.copy_(decoded)
                # Keep a zero-copy HF-shaped value in the state dict. The generic
                # MoE merger sees the registered in-place native key, counts it as
                # loaded, and skips rebuilding the grouped tensor.
                state_dict[base] = destination
            del decoded, tensors
        if hasattr(self, "_mxfp4_load_views"):
            del self._mxfp4_load_views

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

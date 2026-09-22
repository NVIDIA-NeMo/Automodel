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

from __future__ import annotations

import logging
import re
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import (
    create_scale_inv_for_weight,
    dequantize_from_fp8,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

logger = logging.getLogger(__name__)

_MXFP4_BLOCK_SIZE = 32
_FUSED_QKV_TP_SIZE = 4
_FUSED_QKV_WEIGHT = re.compile(r"^(.*\.layers\.(\d+)\.self_attn)\.qkv_proj\.weight$")
_SPLIT_Q_WEIGHT = re.compile(r"^(.*\.layers\.(\d+)\.self_attn)\.q_proj\.weight$")
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

NON_QUANTIZED_KEY_PATTERNS = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "norm.weight",
    "lm_head.weight",
    "embed_tokens.weight",
    "mlp.gate.weight",
    "self_attn.o_proj.weight",
]


def _should_quantize_key(key: str) -> bool:
    if not key.endswith(".weight"):
        return False
    if key.startswith(("visual.", "audio.")):
        return False
    return not any(pattern in key for pattern in NON_QUANTIZED_KEY_PATTERNS)


def _dequantize_mxfp4(
    packed_weight: torch.Tensor,
    scale_e8m0: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize one OCP MXFP4 matrix from the MiMo-V2.6 checkpoint.

    Args:
        packed_weight: Packed E2M1 tensor of shape [output, input / 2], with
            the earlier input element in the low nibble of each byte.
        scale_e8m0: Unsigned E8M0 block scales of shape
            [output, input / 32].
        dtype: Floating-point dtype for the returned training weight.

    Returns:
        Dequantized tensor of shape [output, input].
    """
    if packed_weight.dtype != torch.uint8 or scale_e8m0.dtype != torch.uint8:
        raise TypeError(
            "MiMo-V2.6 MXFP4 weights and scales must both use uint8 storage, "
            f"got {packed_weight.dtype} and {scale_e8m0.dtype}"
        )
    unpacked_input = packed_weight.shape[-1] * 2
    expected_scale_shape = (*packed_weight.shape[:-1], unpacked_input // _MXFP4_BLOCK_SIZE)
    if unpacked_input % _MXFP4_BLOCK_SIZE != 0 or tuple(scale_e8m0.shape) != expected_scale_shape:
        raise ValueError(
            f"MXFP4 scale shape {tuple(scale_e8m0.shape)} does not match packed weight "
            f"{tuple(packed_weight.shape)}; expected {expected_scale_shape}"
        )

    low = packed_weight & 0x0F
    high = packed_weight >> 4
    codes = torch.stack((low, high), dim=-1).reshape(*packed_weight.shape[:-1], unpacked_input)
    magnitude_lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed_weight.device)
    magnitudes = magnitude_lut[(codes & 0x07).long()]
    signs = torch.where(codes & 0x08 != 0, -1.0, 1.0)
    values = magnitudes * signs
    scales = torch.exp2(scale_e8m0.to(torch.float32) - 127.0).repeat_interleave(_MXFP4_BLOCK_SIZE, dim=-1)
    return (values * scales).to(dtype)


def _fused_qkv_sizes(config: Any, layer_idx: int) -> tuple[int, int, int]:
    """Return per-checkpoint-shard Q, K, and V row counts for one layer."""
    is_swa = bool(config.hybrid_layer_pattern[layer_idx])
    if is_swa:
        query_heads = int(config.swa_num_attention_heads)
        key_value_heads = int(config.swa_num_key_value_heads)
        head_dim = int(config.swa_head_dim)
        value_head_dim = int(config.swa_v_head_dim)
    else:
        query_heads = int(config.num_attention_heads)
        key_value_heads = int(config.num_key_value_heads)
        head_dim = int(config.head_dim)
        value_head_dim = int(config.v_head_dim)
    if query_heads % _FUSED_QKV_TP_SIZE or key_value_heads % _FUSED_QKV_TP_SIZE:
        raise ValueError(
            f"MiMo fused QKV requires query and key/value head counts divisible by {_FUSED_QKV_TP_SIZE}, "
            f"got {query_heads} and {key_value_heads} at layer {layer_idx}"
        )
    return (
        query_heads // _FUSED_QKV_TP_SIZE * head_dim,
        key_value_heads // _FUSED_QKV_TP_SIZE * head_dim,
        key_value_heads // _FUSED_QKV_TP_SIZE * value_head_dim,
    )


def _split_fused_qkv(
    weight: torch.Tensor,
    scale_inv: torch.Tensor | None,
    *,
    config: Any,
    layer_idx: int,
    dtype: torch.dtype,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert MiMo's TP4-interleaved fused QKV matrix to split projections.

    Args:
        weight: Fused tensor of shape [4 * (q_rows + k_rows + v_rows), hidden].
        scale_inv: Optional FP8 inverse scales of shape
            [4 * ceil((q_rows + k_rows + v_rows) / 128), hidden / 128].
        config: MiMo model configuration.
        layer_idx: Decoder layer index used to select full or sliding dimensions.
        dtype: Floating-point dtype for dequantized weights.
        name: Checkpoint key used in diagnostics.

    Returns:
        Query, key, and value weights with shapes [q_total, hidden],
        [k_total, hidden], and [v_total, hidden].
    """
    q_rows, k_rows, v_rows = _fused_qkv_sizes(config, layer_idx)
    rows_per_shard = q_rows + k_rows + v_rows
    expected_rows = _FUSED_QKV_TP_SIZE * rows_per_shard
    hidden_size = int(config.hidden_size)
    if weight.ndim != 2 or tuple(weight.shape) != (expected_rows, hidden_size):
        raise ValueError(
            f"{name} has shape {tuple(weight.shape)}; expected "
            f"[{expected_rows}, {hidden_size}] for TP{_FUSED_QKV_TP_SIZE}-interleaved QKV"
        )

    weight_shards = weight.split(rows_per_shard, dim=0)
    if scale_inv is None:
        if weight.dtype == torch.float8_e4m3fn:
            raise ValueError(f"{name} is FP8 but its weight_scale_inv tensor is missing")
        dequantized_shards = weight_shards
    else:
        if scale_inv.dtype != torch.float32:
            raise TypeError(f"{name}_scale_inv must be float32, got {scale_inv.dtype}")
        scale_rows_per_shard = (rows_per_shard + 127) // 128
        expected_scale_rows = _FUSED_QKV_TP_SIZE * scale_rows_per_shard
        expected_scale_columns = (hidden_size + 127) // 128
        if scale_inv.ndim != 2 or tuple(scale_inv.shape) != (expected_scale_rows, expected_scale_columns):
            raise ValueError(
                f"{name}_scale_inv has shape {tuple(scale_inv.shape)}; expected "
                f"[{expected_scale_rows}, {expected_scale_columns}]"
            )
        scale_shards = scale_inv.split(scale_rows_per_shard, dim=0)
        dequantized_shards = tuple(
            dequantize_from_fp8(shard, scale, dtype=dtype, name=f"{name}.tp{shard_idx}")
            for shard_idx, (shard, scale) in enumerate(zip(weight_shards, scale_shards))
        )

    queries, keys, values = [], [], []
    for shard in dequantized_shards:
        query, key, value = shard.split((q_rows, k_rows, v_rows), dim=0)
        queries.append(query)
        keys.append(key)
        values.append(value)
    return torch.cat(queries), torch.cat(keys), torch.cat(values)


class MiMoV2FlashStateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Convert MiMo-V2-Flash HF checkpoints to Automodel's grouped MoE layout.

    HF stores routed experts as split per-expert projections:
    ``mlp.experts.{E}.{gate,up,down}_proj.weight``.  Automodel groups those
    into ``gate_and_up_projs`` and ``down_projs`` so EP can shard experts
    without materializing every expert on every rank.
    """

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True

    @property
    def _uses_fused_qkv_checkpoint(self) -> bool:
        """Whether checkpoint tensors use MiMo-V2.6 fused QKV and MXFP4 experts."""
        return getattr(self.config, "attention_projection_layout", "split") == "fused_qkv"

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        del kwargs
        for key in hf_state_dict.keys():
            if ".mlp.experts." in key and key.endswith(".weight"):
                self._uses_model_prefix = key.startswith("model.")
                break
        hf_state_dict = self._convert_fused_qkv(hf_state_dict)
        hf_state_dict = self._dequantize_mxfp4_experts(hf_state_dict)
        hf_state_dict = self._dequantize(hf_state_dict)
        return self._from_hf_w_merged_experts(hf_state_dict, device_mesh)

    def _convert_fused_qkv(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Replace checkpoint fused QKV tensors with native split projections.

        Args:
            state_dict: Checkpoint tensor mapping. Fused QKV weights have shape
                [4 * local_qkv_rows, hidden], and optional scale tensors have
                shape [4 * local_scale_rows, hidden_blocks].

        Returns:
            The mutated mapping with split query, key, and value weights.
        """
        for key in list(state_dict):
            match = _FUSED_QKV_WEIGHT.match(key)
            if match is None:
                continue
            scale_key = key + "_scale_inv"
            scale_inv = state_dict.get(scale_key)
            query, key_weight, value = _split_fused_qkv(
                state_dict[key],
                scale_inv,
                config=self.config,
                layer_idx=int(match.group(2)),
                dtype=self.dtype,
                name=key,
            )
            prefix = match.group(1)
            load_views = getattr(self, "_fused_qkv_load_views", {})
            destination = load_views.get(prefix)
            if destination is None:
                state_dict[f"{prefix}.q_proj.weight"] = query
                state_dict[f"{prefix}.k_proj.weight"] = key_weight
                state_dict[f"{prefix}.v_proj.weight"] = value
            else:
                native_tensors, local_views = destination
                with torch.no_grad():
                    for native_tensor, local_view, converted in zip(
                        native_tensors,
                        local_views,
                        (query, key_weight, value),
                    ):
                        self._copy_converted_to_native(native_tensor, local_view, converted)
                state_dict[f"{prefix}.q_proj.weight"] = native_tensors[0]
                state_dict[f"{prefix}.k_proj.weight"] = native_tensors[1]
                state_dict[f"{prefix}.v_proj.weight"] = native_tensors[2]
            state_dict.pop(key)
            state_dict.pop(scale_key, None)
        if hasattr(self, "_fused_qkv_load_views"):
            del self._fused_qkv_load_views
        return state_dict

    def _dequantize_mxfp4_experts(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Expand packed expert weights into the native floating-point layout.

        Args:
            state_dict: Checkpoint tensor mapping. MXFP4 expert weights have
                shape [output, input / 2], and their ``weight_scale`` tensors
                have shape [output, input / 32].

        Returns:
            The mutated mapping with floating-point expert matrices.
        """
        scale_keys = []
        expert_weights = {key for key in state_dict if ".mlp.experts." in key and key.endswith(".weight")}
        orphan_scales = [
            key
            for key in state_dict
            if ".mlp.experts." in key
            and key.endswith(".weight_scale")
            and key.removesuffix("_scale") not in expert_weights
        ]
        if orphan_scales:
            raise ValueError(f"MXFP4 expert scales are missing weights: {sorted(orphan_scales)[:3]}")
        for key in list(state_dict):
            if ".mlp.experts." not in key or not key.endswith(".weight"):
                continue
            weight = state_dict[key]
            scale_key = key + "_scale"
            if weight.dtype != torch.uint8:
                continue
            if scale_key not in state_dict:
                raise ValueError(f"MXFP4 expert weight {key} is missing {scale_key}")
            packed_local = self._local_tensor(weight)
            scale_local = self._local_tensor(state_dict[scale_key])
            decoded = _dequantize_mxfp4(packed_local, scale_local, dtype=self.dtype)
            destination = getattr(self, "_mxfp4_load_views", {}).get(key)
            if destination is None:
                state_dict[key] = decoded
            else:
                with torch.no_grad():
                    self._local_tensor(destination).copy_(decoded)
                state_dict[key] = destination
            del decoded
            scale_keys.append(scale_key)
        for key in scale_keys:
            state_dict.pop(key)
        if scale_keys:
            logger.debug("[MiMo MXFP4 Dequant] Dequantized %s expert weights", len(scale_keys))
        if hasattr(self, "_mxfp4_load_views"):
            del self._mxfp4_load_views
        return state_dict

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert native tensors to MiMo checkpoint keys and load destinations."""
        load_mode = quantization and kwargs.get("for_checkpoint_load", False)
        if quantization and self._uses_fused_qkv_checkpoint and not load_mode:
            raise ValueError("MiMo-V2.6 quantized conversion is only supported for checkpoint loading")
        fused_qkv_keys: set[str] = set()
        if load_mode and self._uses_fused_qkv_checkpoint:
            self._mxfp4_load_views: dict[str, torch.Tensor] = {}
            self._fused_qkv_load_views: dict[
                str,
                tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
            ] = {}

        hf_state_dict: dict[str, Any] = {}
        if load_mode and self._uses_fused_qkv_checkpoint:
            for fqn, query in state_dict.items():
                match = _SPLIT_Q_WEIGHT.match(fqn)
                if match is None:
                    continue
                prefix = match.group(1)
                key_fqn = f"{prefix}.k_proj.weight"
                value_fqn = f"{prefix}.v_proj.weight"
                if key_fqn not in state_dict or value_fqn not in state_dict:
                    raise KeyError(f"Incomplete native QKV projection group at {prefix}")
                key_weight = state_dict[key_fqn]
                value = state_dict[value_fqn]
                checkpoint_tensors = self._make_fused_qkv_load_destinations(
                    prefix,
                    int(match.group(2)),
                    query,
                    key_weight,
                    value,
                )
                hf_state_dict.update(checkpoint_tensors)
                fused_qkv_keys.update((fqn, key_fqn, value_fqn))

        for fqn, tensor in state_dict.items():
            if fqn in fused_qkv_keys:
                continue
            converted_tensors = self.convert_single_tensor_to_hf(
                fqn,
                tensor,
                exclude_key_regex=exclude_key_regex,
                quantization=quantization,
                **kwargs,
            )
            for key, value in converted_tensors:
                hf_state_dict[key] = value
        return hf_state_dict

    @staticmethod
    def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Return the rank-local storage view for a Tensor or DTensor."""
        return tensor.to_local() if hasattr(tensor, "to_local") else tensor

    @staticmethod
    def _copy_converted_to_native(
        native_tensor: torch.Tensor,
        local_view: torch.Tensor,
        converted: torch.Tensor,
    ) -> None:
        """Copy a full checkpoint-layout conversion into local model storage.

        Args:
            native_tensor: Native Tensor or DTensor with the global projection shape.
            local_view: Rank-local model storage for ``native_tensor``.
            converted: Full deinterleaved projection in checkpoint dtype conversion output.
        """
        if hasattr(native_tensor, "device_mesh") and hasattr(native_tensor, "placements"):
            from torch.distributed.tensor import distribute_tensor

            distributed = distribute_tensor(
                converted,
                native_tensor.device_mesh,
                native_tensor.placements,
                src_data_rank=None,
            )
            local_view.copy_(distributed.to_local())
            return
        local_view.copy_(converted)

    def _make_fused_qkv_load_destinations(
        self,
        prefix: str,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Create V2.6 fused-QKV checkpoint buffers for three native weights.

        Args:
            prefix: Projection prefix ending in ``self_attn``.
            layer_idx: Decoder layer index selecting full or sliding shapes.
            query: Native query weight of shape [query_rows, hidden].
            key: Native key weight of shape [key_rows, hidden].
            value: Native value weight of shape [value_rows, hidden].

        Returns:
            Fused FP8 weight and FP32 inverse-scale destinations using the exact
            checkpoint names and shapes.
        """
        local_views = tuple(self._local_tensor(tensor) for tensor in (query, key, value))
        q_rows, k_rows, v_rows = _fused_qkv_sizes(self.config, layer_idx)
        expected_shapes = (
            (_FUSED_QKV_TP_SIZE * q_rows, int(self.config.hidden_size)),
            (_FUSED_QKV_TP_SIZE * k_rows, int(self.config.hidden_size)),
            (_FUSED_QKV_TP_SIZE * v_rows, int(self.config.hidden_size)),
        )
        actual_shapes = tuple(tuple(tensor.shape) for tensor in (query, key, value))
        if actual_shapes != expected_shapes:
            raise ValueError(f"Native QKV shapes at {prefix} are {actual_shapes}; expected {expected_shapes}")
        local_devices = {tensor.device for tensor in local_views}
        if len(local_devices) != 1:
            raise ValueError(f"Native QKV local views at {prefix} span devices {sorted(map(str, local_devices))}")

        rows_per_shard = q_rows + k_rows + v_rows
        fused_rows = _FUSED_QKV_TP_SIZE * rows_per_shard
        hidden_size = int(self.config.hidden_size)
        scale_rows = _FUSED_QKV_TP_SIZE * ((rows_per_shard + 127) // 128)
        scale_columns = (hidden_size + 127) // 128
        device = local_views[0].device
        self._fused_qkv_load_views[prefix] = ((query, key, value), local_views)
        weight_key = f"{prefix}.qkv_proj.weight"
        return {
            weight_key: torch.empty((fused_rows, hidden_size), dtype=torch.float8_e4m3fn, device=device),
            f"{weight_key}_scale_inv": torch.empty((scale_rows, scale_columns), dtype=torch.float32, device=device),
        }

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        split_kwargs = kwargs
        load_mode = kwargs.get("quantization", False) and kwargs.get("for_checkpoint_load", False)
        if load_mode and self._uses_fused_qkv_checkpoint:
            split_kwargs = {**kwargs, "quantization": False}
        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **split_kwargs)
        result = expert_result if expert_result is not None else [(fqn, tensor)]

        if exclude_key_regex:
            result = [(key, value) for key, value in result if not re.match(exclude_key_regex, key)]

        if load_mode and self._uses_fused_qkv_checkpoint:
            packed_result: list[tuple[str, Any]] = []
            for key, value in result:
                if re.search(r"\.mlp\.experts\.\d+\.(?:gate|up|down)_proj\.weight$", key):
                    load_views = getattr(self, "_mxfp4_load_views", None)
                    if load_views is not None:
                        load_views[key] = value
                    packed_result.extend(self._make_mxfp4_load_destinations(key, value))
                else:
                    packed_result.append((key, value))
            result = packed_result

        quantized_result = []
        for key, value in result:
            if kwargs.get("quantization", False) and _should_quantize_key(key) and value.dtype != torch.uint8:
                quantized = value.to(dtype=torch.float8_e4m3fn)
                quantized_result.append((key, quantized))
                quantized_result.append((key + "_scale_inv", self._create_scale_inv_for_hf_key(key, quantized)))
            else:
                quantized_result.append((key, value))
        return quantized_result

    def _make_mxfp4_load_destinations(self, key: str, weight: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """Create packed MXFP4 destinations for one logical expert matrix.

        Args:
            key: Checkpoint expert weight key.
            weight: Logical expert view of shape [output, input].

        Returns:
            Packed E2M1 weight [output, input / 2] and E8M0 scales
            [output, input / 32].
        """
        if weight.ndim != 2 or weight.shape[1] % _MXFP4_BLOCK_SIZE:
            raise ValueError(
                f"MiMo MXFP4 expert weight must be rank two with input divisible by 32, got {weight.shape}"
            )
        local = self._local_tensor(weight)
        if local.shape[1] % _MXFP4_BLOCK_SIZE:
            raise ValueError(
                "MiMo MXFP4 local expert shard input must be divisible by 32, "
                f"got global {tuple(weight.shape)} and local {tuple(local.shape)}"
            )
        output_features, input_features = local.shape
        packed = torch.empty((output_features, input_features // 2), dtype=torch.uint8, device=local.device)
        scale = torch.empty(
            (output_features, input_features // _MXFP4_BLOCK_SIZE),
            dtype=torch.uint8,
            device=local.device,
        )
        if hasattr(weight, "device_mesh") and hasattr(weight, "placements"):
            from torch.distributed.tensor import DTensor

            # DCP validates global checkpoint shapes before reading. Preserve
            # the per-expert DTensor spec after compressing its local shard so
            # an ep_shard/CP placement remains visible to the planner.
            packed = DTensor.from_local(packed, weight.device_mesh, weight.placements, run_check=False)
            scale = DTensor.from_local(scale, weight.device_mesh, weight.placements, run_check=False)
        return [
            (key, packed),
            (f"{key}_scale", scale),
        ]

    def _create_scale_inv_for_hf_key(self, key: str, weight: torch.Tensor) -> torch.Tensor:
        scale_inv = create_scale_inv_for_weight(weight)
        full_k_rows = int(self.config.num_key_value_heads) * int(self.config.head_dim)
        if key.endswith(".self_attn.k_proj.weight") and weight.shape[0] == full_k_rows:
            padded_block_rows = 8
            if scale_inv.shape[0] < padded_block_rows:
                pad = torch.ones(
                    (padded_block_rows - scale_inv.shape[0], scale_inv.shape[1]),
                    dtype=scale_inv.dtype,
                    device=scale_inv.device,
                )
                scale_inv = torch.cat([scale_inv, pad], dim=0)
        return scale_inv

    def _dequantize(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        scale_inv_keys = []
        dequantized_count = 0
        for key in list(state_dict.keys()):
            if not key.endswith(".weight"):
                continue
            scale_key = key + "_scale_inv"
            if scale_key not in state_dict:
                continue
            state_dict[key] = dequantize_from_fp8(
                state_dict[key],
                state_dict[scale_key],
                dtype=self.dtype,
                name=key,
            )
            scale_inv_keys.append(scale_key)
            dequantized_count += 1

        for key in scale_inv_keys:
            state_dict.pop(key, None)

        logger.debug("[MiMo FP8 Dequant] Dequantized %s weights", dequantized_count)
        return state_dict

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

import logging
import re
from typing import Any, Optional

import torch

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

LOGGER = logging.getLogger(__name__)


def dequantize_int4(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_shape: torch.Tensor,
    group_size: int = 32,
    device: str = "cuda",
) -> torch.Tensor:
    """Dequantize INT4 packed weights to bfloat16.

    Extracts local tensors from DTensors before unpacking (bitwise ops don't work on DTensor).
    Both weight_packed and weight_scale should have matching sharding so .to_local() gives
    corresponding slices automatically.

    Args:
        weight_packed: INT4 packed weights [out_features, in_features // 8], may be DTensor
        weight_scale: Per-group scales [out_features, num_groups], should be DTensor with same sharding
        weight_shape: Original shape [2], stores global dimensions
        group_size: Elements per scale group (default 32)
        device: Target device for computation
    """

    is_packed_dtensor = hasattr(weight_packed, "device_mesh")
    is_scale_dtensor = hasattr(weight_scale, "device_mesh")

    if is_packed_dtensor:
        weight_packed = weight_packed.to_local()

    if is_scale_dtensor:
        weight_scale = weight_scale.to_local()

    local_out, local_packed_in = weight_packed.shape
    local_in = local_packed_in * 8  # 8 INT4 values per int32

    use_cuda = device == "cuda" and torch.cuda.is_available()

    if use_cuda:
        weight_packed = weight_packed.cuda()
        weight_scale = weight_scale.cuda()

    # Unpack INT4: [out, packed_in] -> [out, packed_in, 8] -> [out, in_features]
    shifts = torch.arange(8, device=weight_packed.device) * 4

    packed_unsqueezed = weight_packed.unsqueeze(-1)
    unpacked = ((packed_unsqueezed >> shifts) & 0xF).float()
    unpacked = unpacked.reshape(local_out, local_in)

    # Convert unsigned 4-bit (0-15) to signed (-8 to 7) using OFFSET BINARY
    # This matches compressed-tensors library which packs as: value + 8
    # So unpack as: value - 8
    unpacked = unpacked - 8

    # Apply scale - both are now local tensors with corresponding slices
    scale = weight_scale.float()
    if scale.ndim == 1:
        local_num_groups = scale.numel() // local_out
        scale = scale.view(local_out, local_num_groups)
    else:
        scale = scale.view(local_out, -1)

    local_num_groups = scale.shape[1]
    elements_per_group = local_in // local_num_groups

    # repeat_interleave expands [local_out, local_num_groups] -> [local_out, local_in]
    scale_expanded = scale.repeat_interleave(elements_per_group, dim=1)

    if scale_expanded.shape[1] < local_in:
        # Pad if needed
        scale_expanded = torch.nn.functional.pad(
            scale_expanded, (0, local_in - scale_expanded.shape[1]), value=scale_expanded[:, -1:].mean()
        )
    scale_expanded = scale_expanded[:, :local_in]
    result = unpacked * scale_expanded

    result = result.to(torch.bfloat16)

    return result


def quantize_to_int4(
    weight: torch.Tensor,
    group_size: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize bfloat16/float16 weights to INT4 packed format.

    Returns:
        weight_packed: INT4 values packed into int32 (8 values per int32)
        weight_scale: Per-group scale factors (float16)
        weight_shape: Original tensor shape (int64)
    """
    out_features, in_features = weight.shape
    weight_shape = torch.tensor([out_features, in_features], dtype=torch.int64)

    # Convert to float32 for computation
    w = weight.float()

    # Compute per-group scales (group along in_features dimension)
    num_groups = (in_features + group_size - 1) // group_size
    w_grouped = w.view(out_features, num_groups, -1)

    # Symmetric quantization: scale = max(abs(w)) / 7
    group_max = w_grouped.abs().amax(dim=-1)
    scale = group_max / 7.0
    scale = scale.clamp(min=1e-10)  # Avoid division by zero

    # Quantize: w_q = round(w / scale), clamp to [-8, 7]
    scale_expanded = scale.unsqueeze(-1).expand_as(w_grouped)
    w_q = (w_grouped / scale_expanded).round().clamp(-8, 7)

    # Convert signed [-8, 7] to unsigned [0, 15] using offset binary
    w_q = w_q.view(out_features, -1)[:, :in_features]
    w_q = (w_q + 8).to(torch.uint8)

    # Pack 8 INT4 values into each int32 along the in_features dimension
    # HF format: [out_features, in_features//8] - 2D packed tensor
    assert in_features % 8 == 0, f"in_features must be divisible by 8, got {in_features}"

    # Reshape to [out_features, in_features//8, 8] for packing along dim 1
    w_q_grouped = w_q.view(out_features, in_features // 8, 8).to(torch.int32)

    # Pack 8 nibbles into 1 int32
    packed = torch.zeros(out_features, in_features // 8, dtype=torch.int32, device=weight.device)
    for i in range(8):
        packed |= (w_q_grouped[:, :, i] & 0xF) << (i * 4)

    weight_packed = packed
    weight_scale = scale.to(torch.float16)

    return weight_packed, weight_scale, weight_shape


class KimiK25VLStateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """State dict adapter for KimiK25VL checkpoints."""

    def __init__(self, config, moe_config: MoEConfig, backend: BackendConfig, dtype: torch.dtype = torch.float32):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self.llm_adapter = DeepSeekV3StateDictAdapter(config.text_config, moe_config, backend, dtype)
        self._last_expected_hf_keys: set[str] | None = None
        self._uses_model_prefix = True
        self._quant_shapes_cache: dict[str, tuple] | None = None

    def to_hf(
        self, state_dict: dict[str, Any], exclude_key_regex: Optional[str] = None, quantization: bool = False, **kwargs
    ) -> dict[str, Any]:
        """Convert from native model state dict to HuggingFace format.

        If quantization=True, expert weights are quantized to INT4.
        """
        hf_state_dict = {}
        for fqn, tensor in state_dict.items():
            converted_tensors = self.convert_single_tensor_to_hf(
                fqn, tensor, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
            )
            for key, value in converted_tensors:
                hf_state_dict[key] = value

        return hf_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single tensor from native format to HuggingFace format.

        Args:
            fqn: Fully qualified name of the tensor in native format
            tensor: The tensor to convert
            **kwargs: Additional arguments for conversion

        Returns:
            List of (fqn, tensor) tuples in HuggingFace format
        """
        quantization = kwargs.get("quantization", False)
        quantization = True
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        if fqn.startswith("model."):
            fqn = fqn[6:]

        if fqn.startswith("multi_modal_projector."):
            fqn = fqn.replace("multi_modal_projector.", "mm_projector.")
            fqn = fqn.replace(".linear_1.", ".proj.0.")
            fqn = fqn.replace(".linear_2.", ".proj.2.")

        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if expert_result is not None:
            result = [("language_model." + k, v) for k, v in expert_result]
        else:
            result = [(fqn, tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        if quantization:
            quantized_result = []
            device_mesh = kwargs.get("device_mesh", None)

            for key, value in result:
                if self._is_quantized_expert_key(key):
                    from torch.distributed.tensor import DTensor, Shard

                    is_dtensor = hasattr(value, "_local_tensor") or str(type(value).__name__) == "DTensor"
                    base = key[:-7] if key.endswith(".weight") else key

                    local_tensor = (
                        value._local_tensor
                        if is_dtensor and hasattr(value, "_local_tensor")
                        else (value.to_local() if is_dtensor else value)
                    )
                    is_meta = local_tensor.is_meta

                    if is_meta:
                        out_features, in_features = local_tensor.shape
                        group_size = 32
                        packed = torch.empty(out_features, in_features // 8, dtype=torch.int32, device="meta")
                        scale = torch.empty(out_features, in_features // group_size, dtype=torch.float16, device="meta")
                        shape = torch.tensor([value.shape[0], value.shape[1]], dtype=torch.int64)
                    else:
                        packed, scale, shape = quantize_to_int4(local_tensor)
                        packed = packed.to(local_tensor.device)
                        scale = scale.to(local_tensor.device)
                        shape = shape.to(local_tensor.device)

                    if is_dtensor and not is_meta:
                        placements = value.placements if hasattr(value, "placements") else [Shard(0)]
                        mesh = value.device_mesh if hasattr(value, "device_mesh") else device_mesh
                        if mesh is not None:
                            packed = DTensor.from_local(packed, mesh, placements)
                            scale = DTensor.from_local(scale, mesh, placements)

                    quantized_result.append((f"{base}.weight_packed", packed))
                    quantized_result.append((f"{base}.weight_scale", scale))
                    quantized_result.append((f"{base}.weight_shape", shape))
                else:
                    quantized_result.append((key, value))
            return quantized_result

        return result

    def _is_quantized_expert_key(self, key: str) -> bool:
        if "mlp.experts." in key and ".weight" in key:
            if "shared_experts" in key:
                return False
            if ".layers.0." in key:
                return False
            return True
        return False

    def _expand_quantized_keys(self, state_dict: dict) -> dict:
        """Expand expert 'weight' keys to INT4 triplets: *_packed/*_scale/*_shape.

        MoE expert weights are known to be INT4 quantized in the HF checkpoint.
        """
        result = {}

        for key, value in state_dict.items():
            if self._is_quantized_expert_key(key):
                base = key[:-7] if key.endswith(".weight") else key
                # INT4 quantized: expand to packed/scale/shape triplet with placeholder tensors
                result[f"{base}.weight_packed"] = torch.empty(1, dtype=torch.int32)
                result[f"{base}.weight_scale"] = torch.empty(1, dtype=torch.float16)
                result[f"{base}.weight_shape"] = torch.empty(2, dtype=torch.int64)
            else:
                result[key] = value

        return result

    def from_hf(self, state_dict: dict, **kwargs) -> dict:
        """Convert HF checkpoint state dict to model format.

        This handles INT4 dequantization: *_packed/*_scale/*_shape -> weight
        """
        native_state_dict = {}

        # First pass: collect quantized weight triplets and dequantize
        quant_pat = re.compile(r"^(.+)\.(weight|bias)_(packed|scale|shape)$")
        quant_groups = {}  # base_key -> {packed, scale, shape}
        processed_keys = set()

        for key in state_dict.keys():
            m = quant_pat.match(key)
            if m:
                base = f"{m.group(1)}.{m.group(2)}"
                suffix = m.group(3)
                if base not in quant_groups:
                    quant_groups[base] = {}
                quant_groups[base][suffix] = key

        def dequant_one(base_and_parts):
            base, parts = base_and_parts

            # Check if keys exist in state_dict
            packed_exists = parts["packed"] in state_dict
            scale_exists = parts["scale"] in state_dict
            shape_exists = parts["shape"] in state_dict

            if not all([packed_exists, scale_exists, shape_exists]):
                return base, None, []

            packed_tensor = state_dict[parts["packed"]]
            scale_tensor = state_dict[parts["scale"]]
            shape_tensor = state_dict[parts["shape"]]
            weight = dequantize_int4(
                packed_tensor,
                scale_tensor,
                shape_tensor,
                device="cuda",
            )
            return base, weight, list(parts.values())

        valid_groups = [
            (base, parts)
            for base, parts in quant_groups.items()
            if all(p in parts for p in ["packed", "scale", "shape"])
        ]

        dequantized = {}

        for i, (base, parts) in enumerate(valid_groups):
            base_result, weight, keys = dequant_one((base, parts))
            if weight is not None:
                dequantized[base_result] = weight
                processed_keys.update(keys)

        # Merge dequantized weights back into state_dict view
        effective_state_dict = {k: v for k, v in state_dict.items() if k not in processed_keys}
        effective_state_dict.update(dequantized)

        llm_keys = {}

        for key, value in effective_state_dict.items():
            if key.startswith("language_model.model."):
                llm_key = key.replace("language_model.model.", "model.")
                llm_keys[llm_key] = value
            elif key.startswith("language_model.lm_head."):
                native_key = key.replace("language_model.lm_head.", "lm_head.")
                native_state_dict[native_key] = value
            elif key.startswith("mm_projector."):
                # Map HF mm_projector keys to our multi_modal_projector structure
                # mm_projector.proj.0.* -> multi_modal_projector.linear_1.*
                # mm_projector.proj.2.* -> multi_modal_projector.linear_2.*
                # mm_projector.pre_norm.* -> multi_modal_projector.pre_norm.*
                native_key = key.replace("mm_projector.", "multi_modal_projector.")
                native_key = native_key.replace(".proj.0.", ".linear_1.")
                native_key = native_key.replace(".proj.2.", ".linear_2.")
                native_state_dict["model." + native_key] = value
            elif key.startswith(("vision_tower.", "multi_modal_projector.", "language_model.")):
                native_state_dict["model." + key] = value
            else:
                native_state_dict[key] = value

        if llm_keys:
            # Check if these are individual expert keys that need fusion
            expert_llm_keys = [k for k in llm_keys.keys() if "experts." in k and ".weight" in k]
            if expert_llm_keys:
                for k in expert_llm_keys[:6]:
                    v = llm_keys[k]
            # Count unique expert IDs in keys
            expert_ids = set()
            for k in llm_keys.keys():
                m = re.search(r"experts\.(\d+)\.", k)
                if m:
                    expert_ids.add(int(m.group(1)))

            converted_llm = self.llm_adapter.from_hf(llm_keys, **kwargs)
            for k, v in converted_llm.items():
                native_state_dict[k.replace("model.", "model.language_model.model.")] = v

        target_dtype = self.dtype if self.dtype != torch.float32 else torch.bfloat16
        for key in native_state_dict:
            if isinstance(native_state_dict[key], torch.Tensor) and native_state_dict[key].is_floating_point():
                native_state_dict[key] = native_state_dict[key].to(target_dtype)

        return native_state_dict

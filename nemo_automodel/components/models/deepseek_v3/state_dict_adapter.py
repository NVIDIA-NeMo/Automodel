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
from torch.distributed.device_mesh import DeviceMesh
from transformers import DeepseekV3Config

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin
from nemo_automodel.components.moe.state_dict_utils import is_dtensor

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

logger = logging.getLogger(__name__)

# Fixed block size of 128x128 as specified in https://arxiv.org/pdf/2412.19437
BLOCK_SIZE = 128

if _TRITON_AVAILABLE:

    @triton.jit
    def _weight_dequant_kernel(
        x_ptr,
        s_ptr,
        y_ptr,
        M,
        N,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn, mask=mask).to(tl.float32)
        s = tl.load(s_ptr + pid_m * stride_sm + pid_n * stride_sn)
        y = x * s
        tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn, y, mask=mask)


class DeepSeekV3StateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    def __init__(
        self,
        config: DeepseekV3Config,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True
        self.from_hf_map = {
            "model.layers.{}.mlp.experts.{}.gate_proj.weight": "model.layers.{}.mlp.experts.gate_projs",
            "model.layers.{}.mlp.experts.{}.up_proj.weight": "model.layers.{}.mlp.experts.up_projs",
            "model.layers.{}.mlp.experts.{}.down_proj.weight": "model.layers.{}.mlp.experts.down_projs",
        }

    def _dequantize(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        scale_inv_keys = []
        for key, weight in state_dict.items():
            if key.endswith(".weight") and key + "_scale_inv" in state_dict:
                scale_inv = state_dict[key + "_scale_inv"]
                dequantized_weight = dequantize_from_fp8(weight, scale_inv, dtype=self.dtype)
                state_dict[key] = dequantized_weight
                scale_inv_keys.append(key + "_scale_inv")

        for key in scale_inv_keys:
            state_dict.pop(key)

        return state_dict

    def _add_quantization_scale_inv_tensors(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        non_quantized_keys = [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "norm.weight",
            "lm_head.weight",
            "embed_tokens.weight",
            "mlp.gate.weight",
        ]

        weight_scale_inv_state_dict = {}
        for key, value in state_dict.items():
            if key.endswith(".weight") and not any(
                non_quantized_key in key for non_quantized_key in non_quantized_keys
            ):
                value = value.to(dtype=torch.float8_e4m3fn)
                state_dict[key] = value
                expected_scale_shape = calculate_scale_shape(value)
                # Create scale_inv on the same device as the weight to avoid device mismatch during dequantization
                weight_scale_inv_state_dict[key + "_scale_inv"] = torch.ones(
                    expected_scale_shape, dtype=torch.float32, device=value.device
                )

        state_dict.update(weight_scale_inv_state_dict)
        return state_dict

    def to_hf(
        self, state_dict: dict[str, Any], exclude_key_regex: Optional[str] = None, quantization: bool = False, **kwargs
    ) -> dict[str, Any]:
        """Convert from native model state dict to HuggingFace format.
        Automatically detects format based on backend.enable_deepep configuration.
        """
        hf_state_dict = {}
        for fqn, tensor in state_dict.items():
            converted_tensors = self.convert_single_tensor_to_hf(
                fqn, tensor, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
            )
            for key, value in converted_tensors:
                hf_state_dict[key] = value

        return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert HF checkpoint to native format.
        - Dequantize FP8 tensors if scale_inv buffers are provided
        - Aggregate per-expert weights into grouped tensors
        - If device_mesh is provided, only load experts needed for the current rank
        """
        for key in hf_state_dict.keys():
            if ".mlp.experts." in key and key.endswith(".weight"):
                self._uses_model_prefix = key.startswith("model.")

        hf_state_dict = self._dequantize(hf_state_dict)
        return self._from_hf_w_merged_experts(hf_state_dict, device_mesh)

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
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if expert_result is not None:
            result = expert_result
        else:
            result = [(fqn, tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        if quantization:
            quantized_result = []
            for key, value in result:
                if key.endswith(".weight") and not any(
                    non_quantized_key in key
                    for non_quantized_key in [
                        "input_layernorm.weight",
                        "post_attention_layernorm.weight",
                        "norm.weight",
                        "lm_head.weight",
                        "embed_tokens.weight",
                        "mlp.gate.weight",
                    ]
                ):
                    value = value.to(dtype=torch.float8_e4m3fn)
                    expected_scale_shape = calculate_scale_shape(value)
                    weight_scale_inv = torch.ones(expected_scale_shape, dtype=torch.float32, device=value.device)
                    quantized_result.append((key, value))
                    quantized_result.append((key + "_scale_inv", weight_scale_inv))
                else:
                    quantized_result.append((key, value))
            return quantized_result

        return result


def calculate_scale_shape(weight: torch.Tensor, BLOCK_SIZE: int = BLOCK_SIZE) -> torch.Size:
    # Calculate the scale tensor shape
    orig_shape = weight.shape

    # Calculate number of blocks needed
    block_rows = (orig_shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_cols = (orig_shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE

    return torch.Size((block_rows, block_cols))


def _dequantize_with_torch(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    dtype: torch.dtype,
    block_size: int,
) -> torch.Tensor:
    float_weight = weight.to(torch.float32)
    orig_shape = weight.shape
    block_rows = (orig_shape[0] + block_size - 1) // block_size
    block_cols = (orig_shape[1] + block_size - 1) // block_size

    # NOTE: When processing large models on-the-fly, misalignment between block boundaries
    # and DTensor local shape partitioning can lead to silent numerical inaccuracies.
    dequantized = float_weight.detach().clone().to(dtype=dtype)

    for i in range(block_rows):
        row_start = i * block_size
        row_end = min(row_start + block_size, orig_shape[0])

        for j in range(block_cols):
            col_start = j * block_size
            col_end = min(col_start + block_size, orig_shape[1])

            block = float_weight[row_start:row_end, col_start:col_end]
            scale = scale_inv[i, j]
            block = block * scale

            block_converted = block.to(dtype=torch.float32)
            dequantized[row_start:row_end, col_start:col_end] = block_converted

    return dequantized


def _dequantize_with_triton(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    dtype: torch.dtype,
    block_size: int,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for dequantization.")

    m, n = weight.shape
    output = torch.empty((m, n), device=weight.device, dtype=dtype)
    grid = (triton.cdiv(m, block_size), triton.cdiv(n, block_size))
    _weight_dequant_kernel[grid](
        weight,
        scale_inv,
        output,
        m,
        n,
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        scale_inv.stride(0),
        scale_inv.stride(1),
        BLOCK_SIZE=block_size,
    )
    return output


def dequantize_from_fp8(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    dtype=torch.bfloat16,
    BLOCK_SIZE: int = BLOCK_SIZE,
) -> torch.Tensor:
    weight_is_dtensor = is_dtensor(weight)
    scale_is_dtensor = is_dtensor(scale_inv)

    weight_local = weight.to_local() if weight_is_dtensor else weight
    scale_local = scale_inv.to_local() if scale_is_dtensor else scale_inv

    expected_scale_shape = calculate_scale_shape(weight_local, BLOCK_SIZE)
    if scale_local.shape != expected_scale_shape:
        logger.warning(f"scale_inv shape {scale_local.shape} doesn't match expected shape {expected_scale_shape}")

    scale_local = scale_local.to(device=weight_local.device)
    if not weight_local.is_contiguous():
        weight_local = weight_local.contiguous()
    if not scale_local.is_contiguous():
        scale_local = scale_local.contiguous()

    use_triton = (
        _TRITON_AVAILABLE
        and weight_local.is_cuda
        and scale_local.is_cuda
        and weight_local.dim() == 2
        and scale_local.dim() == 2
    )

    if use_triton:
        try:
            dequantized_local = _dequantize_with_triton(weight_local, scale_local, dtype, BLOCK_SIZE)
        except Exception as exc:
            logger.warning(f"Triton dequant failed ({exc}). Falling back to torch.")
            dequantized_local = _dequantize_with_torch(weight_local, scale_local, dtype, BLOCK_SIZE)
    else:
        dequantized_local = _dequantize_with_torch(weight_local, scale_local, dtype, BLOCK_SIZE)

    if weight_is_dtensor:
        from torch.distributed._tensor import DTensor

        return DTensor.from_local(dequantized_local, weight.device_mesh, weight.placements)

    return dequantized_local

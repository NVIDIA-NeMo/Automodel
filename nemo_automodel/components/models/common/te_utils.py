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

import importlib.util
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from nemo_automodel.shared.utils import dtype_from_str

HAVE_TE = importlib.util.find_spec("transformer_engine") is not None
HAVE_DEEP_EP = importlib.util.find_spec("deep_ep") is not None


def is_tensor_unallocated(x: torch.Tensor) -> bool:
    """Check if tensor is unallocated (meta tensor, fake tensor, etc.).

    TE kernels don't support meta tensors, fake tensors, or unallocated tensors.
    This helper detects such cases for fallback handling.

    Args:
        x: Tensor to check

    Returns:
        True if tensor is unallocated or cannot be accessed
    """
    try:
        return x.data_ptr() == 0 or x.numel() == 0
    except Exception:
        return True


@dataclass(kw_only=True)
class BackendConfig:
    attn: Literal["te", "sdpa", "flex"] = "te" if HAVE_TE and torch.cuda.is_available() else "sdpa"
    linear: Literal["torch", "te"] = "te" if HAVE_TE and torch.cuda.is_available() else "torch"
    rms_norm: Literal["torch", "te"] = "te" if HAVE_TE and torch.cuda.is_available() else "torch"
    rope_fusion: bool = HAVE_TE and torch.cuda.is_available()
    enable_deepep: bool = HAVE_DEEP_EP
    fake_balanced_gate: bool = False
    enable_hf_state_dict_adapter: bool = True
    enable_fsdp_optimizations: bool = False
    gate_precision: str | torch.dtype | None = None

    def __post_init__(self):
        if isinstance(self.gate_precision, str):
            self.gate_precision = dtype_from_str(self.gate_precision, default=None)


class TENormWrapper(nn.Module):
    """Wrapper for TransformerEngine RMSNorm to handle meta tensors."""

    def __init__(self, normalized_shape, eps, device, params_dtype):
        super().__init__()
        from transformer_engine.pytorch.module.rmsnorm import RMSNorm as TransformerEngineRMSNorm

        te_norm = TransformerEngineRMSNorm(
            normalized_shape=normalized_shape, eps=eps, device=device, params_dtype=params_dtype
        )
        torch_norm = nn.RMSNorm(normalized_shape, eps=eps, device=device, dtype=params_dtype)

        # Share parameters
        self.weight = te_norm.weight
        torch_norm.weight = self.weight

        # Use object.__setattr__ to prevent submodules from being registered in self._modules.
        # This ensures that the state_dict keys remain flat (e.g., 'weight' instead of 'te_norm.weight').
        object.__setattr__(self, "te_norm", te_norm)
        object.__setattr__(self, "torch_norm", torch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_tensor_unallocated(x):
            # Shape inference only - return empty tensor with same shape
            return torch.empty_like(x)

        # Re-sync shared weights in case they were replaced during materialization
        # (e.g., by to_empty_parameters_only which replaces _parameters entries)
        te_norm = object.__getattribute__(self, "te_norm")
        if te_norm.weight is not self.weight:
            te_norm.weight = self.weight

        return te_norm(x)


class TELinearWrapper(nn.Module):
    """Wrapper for TransformerEngine Linear to handle meta tensors."""

    def __init__(self, in_features, out_features, bias, device, params_dtype):
        super().__init__()
        from transformer_engine.pytorch.module.linear import Linear as TransformerEngineLinear

        te_linear = TransformerEngineLinear(
            in_features, out_features, bias=bias, device=device, params_dtype=params_dtype
        )
        torch_linear = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=params_dtype)

        # Share parameters
        self.weight = te_linear.weight
        torch_linear.weight = self.weight
        if bias:
            self.bias = te_linear.bias
            torch_linear.bias = self.bias

        # Use object.__setattr__ to prevent submodules from being registered in self._modules.
        object.__setattr__(self, "te_linear", te_linear)
        object.__setattr__(self, "torch_linear", torch_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if is_tensor_unallocated(x):
            # Shape inference only - return empty tensor with correct output shape
            out_shape = x.shape[:-1] + (self.weight.shape[0],)
            return torch.empty(out_shape, dtype=x.dtype, device=x.device)

        # Re-sync shared weights in case they were replaced during materialization
        te_linear = object.__getattribute__(self, "te_linear")
        if te_linear.weight is not self.weight:
            te_linear.weight = self.weight
            if hasattr(self, "bias") and self.bias is not None:
                te_linear.bias = self.bias

        return te_linear(x)


def initialize_rms_norm_module(
    rms_norm_impl: str,
    dim: int,
    eps: float = 1e-5,
    device: torch.device | str = "meta",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    if rms_norm_impl == "te":
        return TENormWrapper(normalized_shape=dim, eps=eps, device=device, params_dtype=dtype)
    elif rms_norm_impl == "torch":
        rms_norm_module = nn.RMSNorm(dim, eps=eps, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported RMSNorm implementation: {rms_norm_impl}")
    return rms_norm_module


def initialize_linear_module(
    linear_impl: str,
    in_features: int,
    out_features: int,
    bias: bool = False,
    device: torch.device | str = "meta",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    if linear_impl == "torch":
        return nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    elif linear_impl == "te":
        return TELinearWrapper(in_features, out_features, bias=bias, device=device, params_dtype=dtype)
    else:
        raise ValueError(f"Unsupported Linear implementation: {linear_impl}")

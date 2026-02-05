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


def is_tensor_unallocated(tensor: torch.Tensor) -> bool:
    """Check if tensor is unallocated (meta tensor, fake tensor, etc.).

    TE kernels don't support meta tensors, fake tensors, or unallocated tensors.
    This helper detects such cases for fallback handling.

    Args:
        tensor: Tensor to check

    Returns:
        True if tensor is unallocated or cannot be accessed
    """
    try:
        return tensor.data_ptr() == 0 or tensor.numel() == 0
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


def initialize_rms_norm_module(
    rms_norm_impl: str,
    dim: int,
    eps: float = 1e-5,
    device: torch.device | str = "meta",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Initialize RMSNorm module with the specified backend.

    For TE backend, creates TE module directly on meta device (following GroupedExpertsTE pattern).
    Call reset_parameters() to materialize weights.

    Args:
        rms_norm_impl: Backend implementation ("te" or "torch")
        dim: Normalized dimension
        eps: Epsilon for numerical stability
        device: Device to create module on (meta for TE to support lazy init)
        dtype: Parameter dtype

    Returns:
        RMSNorm module
    """
    if rms_norm_impl == "te":
        from transformer_engine.pytorch.module.rmsnorm import RMSNorm as TransformerEngineRMSNorm

        return TransformerEngineRMSNorm(normalized_shape=dim, eps=eps, device=device, params_dtype=dtype)
    elif rms_norm_impl == "torch":
        return nn.RMSNorm(dim, eps=eps, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported RMSNorm implementation: {rms_norm_impl}")


def initialize_linear_module(
    linear_impl: str,
    in_features: int,
    out_features: int,
    bias: bool = False,
    device: torch.device | str = "meta",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Initialize Linear module with the specified backend.

    For TE backend, creates TE module directly on meta device (following GroupedExpertsTE pattern).
    Call reset_parameters() to materialize weights.

    Args:
        linear_impl: Backend implementation ("te" or "torch")
        in_features: Input features
        out_features: Output features
        bias: Whether to use bias
        device: Device to create module on (meta for TE to support lazy init)
        dtype: Parameter dtype

    Returns:
        Linear module
    """
    if linear_impl == "torch":
        return nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    elif linear_impl == "te":
        from transformer_engine.pytorch.module.linear import Linear as TransformerEngineLinear

        # Create TE module directly on meta device (same as GroupedExpertsTE)
        return TransformerEngineLinear(
            in_features=in_features, out_features=out_features, bias=bias, device=device, params_dtype=dtype
        )
    else:
        raise ValueError(f"Unsupported Linear implementation: {linear_impl}")


def _patch_te_modules():
    """Patch TE modules to handle unallocated tensors for PP shape inference."""
    from transformer_engine.pytorch.module.linear import Linear as TELinear
    from transformer_engine.pytorch.module.rmsnorm import RMSNorm as TERMSNorm

    _original_rmsnorm_forward = TERMSNorm.forward
    _original_linear_forward = TELinear.forward

    def _patched_rmsnorm_forward(self, x):
        is_unallocated = is_tensor_unallocated(x)
        if is_unallocated:
            return torch.empty_like(x)
        return _original_rmsnorm_forward(self, x)

    def _patched_linear_forward(self, x):
        is_unallocated = is_tensor_unallocated(x)
        if is_unallocated:
            out_shape = x.shape[:-1] + (self.weight.shape[0],)
            return torch.empty(out_shape, dtype=x.dtype, device=x.device)
        return _original_linear_forward(self, x)

    TERMSNorm.forward = _patched_rmsnorm_forward
    TELinear.forward = _patched_linear_forward


# Apply TE patches automatically if transformer_engine is available
if HAVE_TE:
    _patch_te_modules()


__all__ = [
    "BackendConfig",
    "initialize_linear_module",
    "initialize_rms_norm_module",
]

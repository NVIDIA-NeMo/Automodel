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
import warnings
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from nemo_automodel.components.loggers.log_utils import logger
from nemo_automodel.shared.utils import dtype_from_str

HAVE_TE = importlib.util.find_spec("transformer_engine") is not None
HAVE_DEEP_EP = importlib.util.find_spec("deep_ep") is not None
HAVE_GMM = importlib.util.find_spec("grouped_gemm") is not None


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
    experts: Literal["torch", "te", "gmm"] = "gmm" if HAVE_GMM and torch.cuda.is_available() else "torch"
    dispatcher: Literal["torch", "deepep"] = "deepep" if HAVE_DEEP_EP and torch.cuda.is_available() else "torch"
    enable_deepep: bool | None = None  # Deprecated: use dispatcher="deepep" instead
    fake_balanced_gate: bool = False
    enable_hf_state_dict_adapter: bool = True
    enable_fsdp_optimizations: bool = False
    gate_precision: str | torch.dtype | None = None

    def __post_init__(self):
        if isinstance(self.gate_precision, str):
            self.gate_precision = dtype_from_str(self.gate_precision, default=None)

        # Handle deprecated enable_deepep parameter
        if self.enable_deepep is not None:
            warnings.warn(
                "enable_deepep is deprecated and will be removed in a future release. "
                "Use experts='gmm' and dispatcher='deepep' instead of enable_deepep=True, "
                "or dispatcher='torch' instead of enable_deepep=False.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.enable_deepep:
                self.experts = "gmm"
                self.dispatcher = "deepep"
            else:
                self.dispatcher = "torch"
            # Clear the deprecated field after conversion
            self.enable_deepep = None

        # Backward compatibility
        if (self.experts == "gmm" or self.experts == "te") and self.dispatcher != "deepep":
            logger.info(
                f"experts='{self.experts}' requires dispatcher='deepep', but got dispatcher='{self.dispatcher}'. "
                "Setting both to torch."
            )
            self.dispatcher = "torch"
            self.experts = "torch"


def initialize_rms_norm_module(
    rms_norm_impl: str,
    dim: int,
    eps: float = 1e-5,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Initialize RMSNorm module with the specified backend.

    For TE backend, creates TE module directly on specified device.
    Call reset_parameters() to materialize weights if created on meta device.

    Args:
        rms_norm_impl: Backend implementation ("te" or "torch")
        dim: Normalized dimension
        eps: Epsilon for numerical stability
        device: Device to create module on (None uses PyTorch default, typically CPU)
        dtype: Parameter dtype

    Returns:
        RMSNorm module
    """
    if rms_norm_impl == "te":
        from transformer_engine.pytorch.module.rmsnorm import RMSNorm as TransformerEngineRMSNorm

        _patch_te_modules()
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
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Initialize Linear module with the specified backend.

    For TE backend, creates TE module directly on specified device.
    Call reset_parameters() to materialize weights if created on meta device.

    Args:
        linear_impl: Backend implementation ("te" or "torch")
        in_features: Input features
        out_features: Output features
        bias: Whether to use bias
        device: Device to create module on (None uses PyTorch default, typically CPU)
        dtype: Parameter dtype

    Returns:
        Linear module
    """
    if linear_impl == "torch":
        return nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    elif linear_impl == "te":
        from transformer_engine.pytorch.module.linear import Linear as TransformerEngineLinear

        _patch_te_modules()
        # Create TE module directly on meta device (same as GroupedExpertsTE)
        return TransformerEngineLinear(
            in_features=in_features, out_features=out_features, bias=bias, device=device, params_dtype=dtype
        )
    else:
        raise ValueError(f"Unsupported Linear implementation: {linear_impl}")


def _make_lazy_te_patcher():
    """Return a callable that patches TE modules exactly once.

    Uses a closure instead of module-level global state to track whether the
    patch has already been applied.  The actual ``transformer_engine`` import
    is deferred until the first call so that importing this module never
    triggers heavy native-library loads (flash-attn, CUDA kernels, etc.).
    """
    patched = False

    def _patch():
        nonlocal patched
        if patched:
            return
        patched = True

        from transformer_engine.pytorch.module.linear import Linear as TELinear
        from transformer_engine.pytorch.module.rmsnorm import RMSNorm as TERMSNorm

        _original_rmsnorm_forward = TERMSNorm.forward
        _original_linear_forward = TELinear.forward

        def _patched_rmsnorm_forward(self, x):
            if is_tensor_unallocated(x):
                return torch.empty_like(x)
            return _original_rmsnorm_forward(self, x)

        def _patched_linear_forward(self, x):
            if is_tensor_unallocated(x):
                out_shape = x.shape[:-1] + (self.weight.shape[0],)
                return torch.empty(out_shape, dtype=x.dtype, device=x.device)
            return _original_linear_forward(self, x)

        TERMSNorm.forward = _patched_rmsnorm_forward
        TELinear.forward = _patched_linear_forward

    return _patch


_patch_te_modules = _make_lazy_te_patcher()


__all__ = [
    "BackendConfig",
    "initialize_linear_module",
    "initialize_rms_norm_module",
]

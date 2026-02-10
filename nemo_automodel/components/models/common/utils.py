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
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

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
class TEFp8Config:
    """Configuration for Transformer Engine FP8 quantization.

    When present (not None) in BackendConfig, FP8 is enabled.
    The ``recipe`` field accepts either a string shorthand (``"current"`` or ``"block"``)
    or a pre-built TE recipe object (e.g. ``Float8CurrentScaling(fp8_dpa=True)``).
    """

    recipe: Literal["current", "block"] | Any = "current"


def build_fp8_recipe(config: TEFp8Config):
    """Build and return the TE FP8 recipe object from a :class:`TEFp8Config`.

    If ``config.recipe`` is already a TE recipe object (e.g. ``Float8CurrentScaling(...)``),
    it is returned directly.  String values ``"current"`` and ``"block"`` are
    mapped to the corresponding TE recipe class.
    """
    if not HAVE_TE:
        return None

    # Pass through pre-built recipe objects directly
    if not isinstance(config.recipe, str):
        return config.recipe

    from transformer_engine.common.recipe import Float8BlockScaling, Float8CurrentScaling

    if config.recipe == "block":
        return Float8BlockScaling()
    return Float8CurrentScaling()


def maybe_te_fp8_autocast(config: TEFp8Config | None):
    """Return a TE FP8 autocast context manager, or :func:`nullcontext` if disabled.

    Args:
        config: FP8 configuration, or ``None`` to disable.
    """
    if config is None or not HAVE_TE:
        return nullcontext()
    from transformer_engine.pytorch.quantization import autocast as te_autocast

    return te_autocast(enabled=True, recipe=build_fp8_recipe(config))


@dataclass(kw_only=True)
class BackendConfig:
    attn: Literal["te", "sdpa", "flex"] = "te" if HAVE_TE and torch.cuda.is_available() else "sdpa"
    linear: Literal["torch", "te"] = "te" if HAVE_TE and torch.cuda.is_available() else "torch"
    rms_norm: Literal["torch", "te"] = "te" if HAVE_TE and torch.cuda.is_available() else "torch"
    rope_fusion: bool = HAVE_TE and torch.cuda.is_available()
    experts: Literal["torch", "te", "gmm"] = "torch"
    dispatcher: Literal["torch", "deepep"] = "torch"
    enable_deepep: bool | None = None  # Deprecated: use dispatcher="deepep" instead
    fake_balanced_gate: bool = False
    enable_hf_state_dict_adapter: bool = True
    enable_fsdp_optimizations: bool = False
    te_fp8: TEFp8Config | None = None
    gate_precision: str | torch.dtype | None = None

    def __post_init__(self):
        # Normalize te_fp8: dict -> TEFp8Config, None stays None
        if isinstance(self.te_fp8, dict):
            self.te_fp8 = TEFp8Config(**self.te_fp8)

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

        # TE and GMM experts require DeepEP dispatcher
        if self.experts in ("te", "gmm") and self.dispatcher != "deepep":
            raise ValueError(
                f"experts='{self.experts}' requires dispatcher='deepep', but got dispatcher='{self.dispatcher}'"
            )

        # FP8 requires at least one TE backend (applies to all TE modules: Linear, GroupedLinear, RMSNorm)
        if self.te_fp8 is not None and self.linear != "te" and self.experts != "te":
            raise ValueError(
                "te_fp8 requires at least one TE backend "
                f"(linear='te' or experts='te'), but got linear='{self.linear}', experts='{self.experts}'"
            )


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
        FP8CachingLinear = _make_fp8_caching_linear()

        return FP8CachingLinear(
            in_features=in_features, out_features=out_features, bias=bias, device=device, params_dtype=dtype
        )
    else:
        raise ValueError(f"Unsupported Linear implementation: {linear_impl}")


def _patch_te_modules():
    """Patch TE modules so unallocated tensors short-circuit to empty outputs.

    TE kernels don't support meta/fake tensors.  During PP shape inference the
    runtime may pass such tensors through TE modules; this patch detects them
    and returns empty tensors of the correct shape instead of crashing.
    """
    from transformer_engine.pytorch.module.linear import Linear as TELinear
    from transformer_engine.pytorch.module.rmsnorm import RMSNorm as TERMSNorm

    _original_rmsnorm_forward = TERMSNorm.forward
    _original_linear_forward = TELinear.forward

    def _patched_rmsnorm_forward(self, x):
        if is_tensor_unallocated(x):
            return torch.empty_like(x)
        return _original_rmsnorm_forward(self, x)

    def _patched_linear_forward(self, x, **kwargs):
        if is_tensor_unallocated(x):
            out_shape = x.shape[:-1] + (self.weight.shape[0],)
            return torch.empty(out_shape, dtype=x.dtype, device=x.device)
        return _original_linear_forward(self, x, **kwargs)

    TERMSNorm.forward = _patched_rmsnorm_forward
    TELinear.forward = _patched_linear_forward


# Apply TE patches automatically if transformer_engine is available
if HAVE_TE:
    _patch_te_modules()


# ---------------------------------------------------------------------------
#  FP8 weight-caching subclasses
#
#  TE Linear and GroupedLinear accept ``is_first_microbatch`` to control FP8
#  weight caching during gradient accumulation.  Rather than relying on
#  external state, these subclasses auto-detect weight updates by tracking
#  ``weight._version`` (incremented by every in-place op such as
#  ``optimizer.step()``).  When the version changes the module re-quantises;
#  otherwise it reuses the cached FP8 weights.
# ---------------------------------------------------------------------------


def _make_fp8_caching_linear():
    """Build and return the FP8CachingLinear class (deferred TE import)."""
    from transformer_engine.pytorch.module.linear import Linear as TELinear

    class FP8CachingLinear(TELinear):
        """TE Linear with automatic ``is_first_microbatch`` management.

        Tracks ``self.weight._version`` to detect optimizer updates and passes
        ``is_first_microbatch=True`` on the first forward after a weight change,
        ``False`` on subsequent forwards.  Callers may still override explicitly.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._prev_weight_version: int = -1

        def forward(self, x: torch.Tensor, is_first_microbatch: bool | None = None, **kwargs: Any) -> torch.Tensor:
            if is_first_microbatch is None:
                v = self.weight._version
                is_first_microbatch = v != self._prev_weight_version
                self._prev_weight_version = v
            return super().forward(x, is_first_microbatch=is_first_microbatch, **kwargs)

    return FP8CachingLinear


def _make_fp8_caching_grouped_linear():
    """Build and return the FP8CachingGroupedLinear class (deferred TE import)."""
    from transformer_engine.pytorch.module.grouped_linear import GroupedLinear as TEGroupedLinear

    class FP8CachingGroupedLinear(TEGroupedLinear):
        """TE GroupedLinear with automatic ``is_first_microbatch`` management.

        Uses ``self.weight0._version`` as a sentinel (all expert weights are
        updated in the same optimizer step).
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._prev_weight_version: int = -1

        def forward(self, inp: torch.Tensor, m_splits: list[int], is_first_microbatch: bool | None = None):
            if is_first_microbatch is None:
                v = self.weight0._version
                is_first_microbatch = v != self._prev_weight_version
                self._prev_weight_version = v
            return super().forward(inp, m_splits, is_first_microbatch=is_first_microbatch)

    return FP8CachingGroupedLinear


__all__ = [
    "BackendConfig",
    "TEFp8Config",
    "_make_fp8_caching_grouped_linear",
    "_make_fp8_caching_linear",
    "build_fp8_recipe",
    "initialize_linear_module",
    "initialize_rms_norm_module",
    "maybe_te_fp8_autocast",
]

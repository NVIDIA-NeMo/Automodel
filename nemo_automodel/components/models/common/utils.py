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
import logging
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn

logger = logging.getLogger(__name__)
from nemo_automodel.shared.utils import dtype_from_str

HAVE_TE = importlib.util.find_spec("transformer_engine") is not None
HAVE_DEEP_EP = importlib.util.find_spec("deep_ep") is not None
HAVE_GMM = importlib.util.find_spec("grouped_gemm") is not None

# ---------------------------------------------------------------------------
#  Global state flags for training coordination
#  Set by training utility functions, read by TE module patches and MoE modules.
# ---------------------------------------------------------------------------

IS_OPTIM_STEP = False
IS_FIRST_MICROBATCH: bool | None = None


def set_is_optim_step(value: bool) -> None:
    """Set the global IS_OPTIM_STEP flag.

    Args:
        value: Whether we are in an optimization step.
    """
    global IS_OPTIM_STEP
    IS_OPTIM_STEP = value


def get_is_optim_step() -> bool:
    """Get the global IS_OPTIM_STEP flag.

    Returns:
        Whether we are in an optimization step.
    """
    return IS_OPTIM_STEP


def set_is_first_microbatch(value: bool | None) -> None:
    """Set the global IS_FIRST_MICROBATCH flag for FP8 weight caching.

    Args:
        value: True for first microbatch (quantize+cache), False for subsequent
               (use cached), None to disable caching.
    """
    global IS_FIRST_MICROBATCH
    IS_FIRST_MICROBATCH = value


def get_is_first_microbatch() -> bool | None:
    """Get the global IS_FIRST_MICROBATCH flag.

    Returns:
        True/False/None indicating microbatch position for FP8 weight caching.
    """
    return IS_FIRST_MICROBATCH


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

    def build_recipe(self):
        """Build and return the TE FP8 recipe object.

        If ``recipe`` is already a TE recipe object (e.g. ``Float8CurrentScaling(...)``),
        it is returned directly.  String values ``"current"`` and ``"block"`` are
        mapped to the corresponding TE recipe class.
        """
        # Pass through pre-built recipe objects directly
        if not isinstance(self.recipe, str):
            return self.recipe

        if not HAVE_TE:
            return None

        from transformer_engine.common.recipe import Float8BlockScaling, Float8CurrentScaling

        if self.recipe == "block":
            return Float8BlockScaling()
        return Float8CurrentScaling()

    def maybe_te_autocast(self):
        """Return te_autocast context manager for FP8."""
        if not HAVE_TE:
            return nullcontext()
        from transformer_engine.pytorch.quantization import autocast as te_autocast

        return te_autocast(enabled=True, recipe=self.build_recipe())


@dataclass(kw_only=True)
class BackendConfig:
    """Backend configuration for model components.

    Attributes:
        attn: Attention backend ("te", "sdpa", or "flex").
        linear: Linear layer backend ("torch" or "te").
        rms_norm: RMSNorm backend ("torch" or "te").
        rope_fusion: Whether to use fused RoPE (requires TE).
        experts: MoE expert GEMM backend. "torch" uses per-expert loop,
            "te" uses TE GroupedLinear, "gmm" uses grouped_gemm.ops.gmm,
            "torch_mm" uses torch._grouped_mm.
        dispatcher: MoE token dispatcher. "torch" uses DTensor all-gather/reduce-scatter,
            "deepep" uses DeepEP for token dispatch.
        enable_deepep: Deprecated. Use dispatcher="deepep" and experts="gmm" instead.
        fake_balanced_gate: If True, replace the learned Gate with FakeBalancedGate
            that assigns tokens to experts without learned routing weights.
        fake_gate_noise: Noise level [0, 1] for FakeBalancedGate. When > 0, uses
            biased topk selection seeded from the input content so routing varies
            dynamically across training steps (like real Gate) while remaining
            deterministic for activation checkpointing recompute (same input = same
            routing). Only used when fake_balanced_gate=True.
        enable_hf_state_dict_adapter: Whether to enable HuggingFace state dict adapter.
        enable_fsdp_optimizations: Whether to enable FSDP2 optimizations.
        gate_precision: Optional dtype override for the gate computation. Accepts
            torch.dtype or string (e.g., "torch.float32", "float32").
    """

    attn: Literal["te", "sdpa", "flex"] = "te" if HAVE_TE and torch.cuda.is_available() else "sdpa"
    linear: Literal["torch", "te"] = "te" if HAVE_TE and torch.cuda.is_available() else "torch"
    rms_norm: Literal["torch", "torch_fp32", "te"] = "te" if HAVE_TE and torch.cuda.is_available() else "torch"
    rope_fusion: bool = HAVE_TE and torch.cuda.is_available()
    experts: Literal["torch", "te", "gmm", "torch_mm"] = "torch_mm" if torch.cuda.is_available() else "torch"
    dispatcher: Literal["torch", "deepep"] = "deepep" if HAVE_DEEP_EP and torch.cuda.is_available() else "torch"
    enable_deepep: bool | None = None  # Deprecated: use dispatcher="deepep" instead
    fake_balanced_gate: bool = False
    # Approximate max/mean load ratios (64 experts, top-8, 4096 tokens):
    # 0.0→1.00x, 0.1→~1.2x, 0.3→~1.6x, 0.5→~2.0x, 1.0→~2.8x.
    fake_gate_noise: float = 0.0
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

        # Backward compatibility
        if self.experts in ("te", "gmm") and self.dispatcher != "deepep":
            if (
                torch.distributed.is_initialized() and torch.distributed.get_rank() == 0
            ) or not torch.distributed.is_initialized():
                logger.info(
                    f"experts='{self.experts}' requires dispatcher='deepep', but got dispatcher='{self.dispatcher}'. "
                    "Setting dispatcher to torch and experts to torch_mm."
                )
            self.dispatcher = "torch"
            self.experts = "torch_mm"

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
        rms_norm_impl: Backend implementation ("te", "torch", or "torch_fp32")
            - "te": Transformer Engine fused RMSNorm kernel
            - "torch": PyTorch native nn.RMSNorm (computes in input dtype)
            - "torch_fp32": Float32 input upcast RMSNorm
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
    elif rms_norm_impl == "torch_fp32":
        # LlamaRMSNorm reference: generic fp32-upcast implementation for accuracy matching
        from transformers.models.llama.modeling_llama import LlamaRMSNorm as Float32RMSNorm

        return Float32RMSNorm(dim, eps=eps).to(device=device, dtype=dtype)
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

    Two patches are applied:
    1. Unallocated tensor handling: TE kernels don't support meta/fake tensors,
       so we short-circuit with empty tensors for PP shape inference.
    2. is_first_microbatch injection: Reads the global IS_FIRST_MICROBATCH flag and
       passes it to TE Linear/GroupedLinear for FP8 weight caching during
       gradient accumulation (quantize on first microbatch, reuse cached on rest).
    """
    patched = False

    def _patch():
        nonlocal patched
        if patched:
            return
        patched = True

        from transformer_engine.pytorch.module.grouped_linear import GroupedLinear as TEGroupedLinear
        from transformer_engine.pytorch.module.linear import Linear as TELinear
        from transformer_engine.pytorch.module.rmsnorm import RMSNorm as TERMSNorm

        _original_rmsnorm_forward = TERMSNorm.forward
        _original_linear_forward = TELinear.forward
        _original_grouped_linear_forward = TEGroupedLinear.forward

        def _patched_rmsnorm_forward(self, x):
            if is_tensor_unallocated(x):
                return torch.empty_like(x)
            return _original_rmsnorm_forward(self, x)

        def _patched_linear_forward(self, x, is_first_microbatch=None, **kwargs):
            if is_tensor_unallocated(x):
                out_shape = x.shape[:-1] + (self.weight.shape[0],)
                return torch.empty(out_shape, dtype=x.dtype, device=x.device)
            if is_first_microbatch is None:
                is_first_microbatch = get_is_first_microbatch()
            return _original_linear_forward(self, x, is_first_microbatch=is_first_microbatch, **kwargs)

        def _patched_grouped_linear_forward(self, inp, m_splits, is_first_microbatch=None):
            if is_first_microbatch is None:
                is_first_microbatch = get_is_first_microbatch()
            return _original_grouped_linear_forward(self, inp, m_splits, is_first_microbatch=is_first_microbatch)

        TERMSNorm.forward = _patched_rmsnorm_forward
        TELinear.forward = _patched_linear_forward
        TEGroupedLinear.forward = _patched_grouped_linear_forward

    return _patch


_patch_te_modules = _make_lazy_te_patcher()


def get_rope_config(config) -> tuple[float, dict, float]:
    """Extract rope configuration from ``config.rope_parameters``.

    Args:
        config: A HuggingFace model config object.

    Returns:
        Tuple of (rope_theta, rope_parameters, partial_rotary_factor).
    """
    rope_parameters = config.rope_parameters
    rope_theta = rope_parameters["rope_theta"]
    partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
    return rope_theta, rope_parameters, partial_rotary_factor


__all__ = [
    "BackendConfig",
    "TEFp8Config",
    "get_is_first_microbatch",
    "get_is_optim_step",
    "get_rope_config",
    "initialize_linear_module",
    "initialize_rms_norm_module",
    "set_is_first_microbatch",
    "set_is_optim_step",
]

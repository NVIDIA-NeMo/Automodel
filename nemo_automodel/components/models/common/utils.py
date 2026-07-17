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
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.shared.import_utils import safe_import_from
from nemo_automodel.shared.utils import dtype_from_str

logger = logging.getLogger(__name__)

HAVE_TE = importlib.util.find_spec("transformer_engine") is not None
HAVE_DEEP_EP = importlib.util.find_spec("deep_ep") is not None
HAVE_UCCL_EP = importlib.util.find_spec("uccl") is not None or importlib.util.find_spec("ep") is not None
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
    The ``recipe`` field accepts either a string shorthand (``"current"``, ``"block"``,
    or ``"mxfp8"``) or a pre-built TE recipe object (e.g. ``Float8CurrentScaling(fp8_dpa=True)``).

    ``"mxfp8"`` selects TE's :class:`MXFP8BlockScaling` recipe (e4m3 data + e8m0 block
    scales). Unlike torchao's MXFP8 grouped GEMM, TE's MXFP8 backward is mature (no
    e8m0-overflow NaN), which is why GPT-OSS experts (grouped + bias) use the
    ``experts="te"`` path with this recipe instead of ``experts="torch_mm_mxfp8"``.
    """

    recipe: Literal["current", "block", "mxfp8"] | Any = "current"

    def build_recipe(self):
        """Build and return the TE FP8 recipe object.

        If ``recipe`` is already a TE recipe object (e.g. ``Float8CurrentScaling(...)``),
        it is returned directly.  String values ``"current"``, ``"block"``, and
        ``"mxfp8"`` are mapped to the corresponding TE recipe class.
        """
        if not HAVE_TE:
            return None

        # Pass through pre-built recipe objects directly
        if not isinstance(self.recipe, str):
            return self.recipe

        from transformer_engine.common.recipe import Float8BlockScaling, Float8CurrentScaling

        if self.recipe == "mxfp8":
            try:
                from transformer_engine.common.recipe import MXFP8BlockScaling
            except ImportError as e:  # TE too old for MXFP8 (added ~v2.3)
                raise ImportError(
                    "te_fp8.recipe='mxfp8' requires transformer_engine.common.recipe.MXFP8BlockScaling "
                    "(TE >= ~2.3). The installed TE does not provide it; rebuild on a newer TE image."
                ) from e
            logger.warning("te_fp8.recipe='mxfp8': using TE MXFP8BlockScaling (mature MXFP8 backward).")
            return MXFP8BlockScaling()
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
        attn: Attention backend ("te", "sdpa", "flex", "eager", or "tilelang").
            For DeepSeek V4, "tilelang" enables the TileLang sparse attention,
            indexer, and Sinkhorn kernels together.
        linear: Linear layer backend ("torch", "te", or "quack").
        rms_norm: RMSNorm backend ("torch", "torch_fp32", "te", or "quack").
        rope: Rotary embedding backend ("torch" or "quack"). QuACK is currently
            integrated for Llama-family rotary embeddings.
        rope_fusion: Whether to use fused RoPE (requires TE).
        experts: MoE expert GEMM backend. "torch" uses per-expert loop,
            "te" uses TE GroupedLinear, "gmm" uses grouped_gemm.ops.gmm,
            "torch_mm" uses torch._grouped_mm, "torch_mm_mxfp8" uses torch._grouped_mm
            dispatch but routes the expert grouped GEMMs through torchao's MXFP8
            scaled grouped GEMM (training-only; GB200/sm_100+ with torchao installed,
            else falls back to torch._grouped_mm at runtime).
        dispatcher: MoE token dispatcher. "torch" uses DTensor all-gather/reduce-scatter,
            "deepep" uses DeepEP for token dispatch,
            "uccl_ep" uses UCCL-EP for token dispatch across heterogeneous GPUs and NICs.
        dispatcher_share_token_dispatcher: Whether flex token dispatchers share a communication
            manager instance across MoE layers.
        dispatcher_async_dispatch: Whether DeepEP/UCCL-EP dispatch should return asynchronously
            and allocate dispatched tensors on the communication stream.
        enable_deepep: Removed and ignored. Logs a warning if set; configure "dispatcher"
            and "experts" explicitly instead.
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
        compile_attn: torch.compile(fullgraph) the attention module's forward — both the
            DeepSeek-V3 MLA and standard GQA attention (e.g. Qwen3-MoE) honor it. Requires
            attn="sdpa", linear="torch", rms_norm="torch", rope_fusion=False.
    """

    attn: Literal["te", "sdpa", "flex", "eager", "tilelang"] = "te" if HAVE_TE and torch.cuda.is_available() else "sdpa"
    linear: Literal["torch", "te", "quack"] = "te" if HAVE_TE and torch.cuda.is_available() else "torch"
    rms_norm: Literal["torch", "torch_fp32", "te", "quack"] = "torch_fp32"
    rope: Literal["torch", "quack"] = "torch"
    rope_fusion: bool = HAVE_TE and torch.cuda.is_available()
    experts: Literal["torch", "te", "gmm", "torch_mm", "torch_mm_mxfp8"] = (
        "torch_mm" if torch.cuda.is_available() else "torch"
    )
    dispatcher: Literal["torch", "deepep", "hybridep", "uccl_ep"] = (
        "deepep"
        if HAVE_DEEP_EP and torch.cuda.is_available()
        else "uccl_ep"
        if HAVE_UCCL_EP and torch.cuda.is_available()
        else "torch"
    )
    dispatcher_num_sms: int = 20
    dispatcher_share_token_dispatcher: bool = True
    dispatcher_async_dispatch: bool = False
    enable_deepep: bool | None = None  # Removed: ignored with a warning; set dispatcher/experts explicitly
    fake_balanced_gate: bool = False
    # Approximate max/mean load ratios (64 experts, top-8, 4096 tokens):
    # 0.0→1.00x, 0.1→~1.2x, 0.3→~1.6x, 0.5→~2.0x, 1.0→~2.8x.
    fake_gate_noise: float = 0.0
    enable_hf_state_dict_adapter: bool = True
    enable_fsdp_optimizations: bool = False
    te_fp8: TEFp8Config | None = None
    gate_precision: str | torch.dtype | None = None
    # When True, torch.compile(fullgraph=True) the attention module's forward to fuse its many
    # small ops (projections, RoPE, reshapes, SDPA). Applies to both the DeepSeek-V3 MLA and
    # standard GQA attention (e.g. Qwen3-MoE). The compiled region must contain no TE
    # custom-autograd submodules (TE's fused attention/Linear/RMSNorm are black boxes that
    # fullgraph can't trace), so it requires attn="sdpa", linear="torch", rms_norm="torch",
    # rope_fusion=False. Default False.
    compile_attn: bool = False

    def __post_init__(self):
        # TEMPORARY: force TE fused RoPE off globally. The fused kernel computes cos/sin
        # in fp32 in-kernel while HF/vLLM rotate with bf16 tables, breaking logprob parity
        # in some models. See #3027. This is the one chokepoint every BackendConfig passes
        # through, so it also overrides an explicit rope_fusion=True from a recipe/config.
        if self.rope_fusion:
            logger.warning("rope_fusion is temporarily force-disabled globally (see #3027).")
        self.rope_fusion = False

        # Normalize te_fp8: dict -> TEFp8Config, None stays None
        if isinstance(self.te_fp8, dict):
            self.te_fp8 = TEFp8Config(**self.te_fp8)

        if isinstance(self.gate_precision, str):
            self.gate_precision = dtype_from_str(self.gate_precision, default=None)

        # enable_deepep was removed. It is no longer honored; warn (once, on rank 0) if a stale
        # config still sets it so the user migrates to explicit dispatcher/experts. The field is
        # retained only so loading an old config does not crash this kw_only dataclass.
        if self.enable_deepep is not None:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                logger.warning(
                    "enable_deepep is no longer supported and is ignored. "
                    "Set 'dispatcher' (deepep/hybridep/torch) and 'experts' explicitly instead. "
                    "Previously enable_deepep=True was equivalent to experts=gmm + dispatcher=deepep, "
                    "and enable_deepep=False to dispatcher=torch."
                )
            self.enable_deepep = None

        # Backward compatibility
        if self.experts in ("te", "gmm") and self.dispatcher not in ("deepep", "hybridep", "uccl_ep"):
            if (
                torch.distributed.is_initialized() and torch.distributed.get_rank() == 0
            ) or not torch.distributed.is_initialized():
                logger.info(
                    f"experts='{self.experts}' requires dispatcher='deepep' or 'uccl_ep', "
                    f"but got dispatcher='{self.dispatcher}'. "
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


@torch.compile(fullgraph=True, dynamic=True)
def _float32_rms_norm_fwd(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Compiled fp32 RMSNorm forward — standalone function to minimize dynamo guards."""
    input_dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (weight * x).to(input_dtype)


class Float32RMSNorm(nn.Module):
    """RMSNorm with explicit fp32 computation for training stability.

    Weights stay in the model's dtype (e.g. bf16) for FSDP2 compatibility.
    Inputs are upcast to fp32, norm is computed in fp32, and the output
    is cast back to the original input dtype.
    """

    def __init__(self, dim, eps=1e-5, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x):
        return _float32_rms_norm_fwd(x, self.weight, self.eps)


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
        rms_norm_impl: Backend implementation ("te", "torch", "torch_fp32", or "quack")
            - "te": Transformer Engine fused RMSNorm kernel
            - "torch": PyTorch native nn.RMSNorm (computes in input dtype)
            - "torch_fp32": torch.compiled fp32 RMSNorm for training stability
            - "quack": QuACK CuTe DSL RMSNorm kernel
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
        return Float32RMSNorm(dim, eps=eps, device=device, dtype=dtype)
    elif rms_norm_impl == "quack":
        available, quack_rms_norm = safe_import_from(
            "quack.rmsnorm",
            "QuackRMSNorm",
            msg="rms_norm='quack' requires the 'quack-kernels' package. Install nemo-automodel[cuda].",
        )
        if not available:
            raise ImportError("rms_norm='quack' requires the 'quack-kernels' package. Install nemo-automodel[cuda].")
        return quack_rms_norm(dim, eps=eps, device=device, dtype=dtype)
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
        linear_impl: Backend implementation ("te", "torch", or "quack")
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
    elif linear_impl == "quack":
        available, quack_linear = safe_import_from(
            "quack.linear",
            "Linear",
            msg="linear='quack' requires the 'quack-kernels' package. Install nemo-automodel[cuda].",
        )
        if not available:
            raise ImportError("linear='quack' requires the 'quack-kernels' package. Install nemo-automodel[cuda].")
        return quack_linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
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
            # Skip the unallocated-tensor short-circuit during torch.compile tracing:
            # fake tensors used by inductor have data_ptr()==0 but are NOT unallocated --
            # returning empty_like here produces a leaf with no grad_fn, breaking AOT autograd.
            if is_tensor_unallocated(x) and not torch.compiler.is_compiling():
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


def cast_model_to_dtype(
    model: nn.Module, dtype: torch.dtype = torch.bfloat16, skip_modules: tuple[str, ...] = ()
) -> None:
    """Cast model parameters to the target dtype, keeping fp32 modules in full precision.

    Respects ``_keep_in_fp32_modules`` / ``_keep_in_fp32_modules_strict`` on
    the model (the same attributes HuggingFace transformers uses).

    Uses ``nn.Module.to()`` which is safe for both plain tensors and DTensors
    (FSDP2 sharded parameters).  When the model is already FSDP2-sharded
    (parameters are DTensors), strict fp32 modules are restored to fp32 because
    they are expected to be isolated as uniform fp32 FSDP units. Non-strict fp32
    hints only restore matching buffers, since their parameters may share an
    FSDP unit with lower-precision parameters.

    Args:
        model: The model whose parameters should be cast.
        dtype: Target dtype (e.g. ``torch.bfloat16``).
        skip_modules: Names of immediate submodules to leave entirely untouched
            (kept at their current dtype). Unlike the ``_keep_in_fp32_modules``
            restore path, these are *detached* during the cast so ``model.to()``
            never visits them — the only reliable way to preserve an fp32
            parameter once it is FSDP2-sharded (post-shard ``.data`` reassignment
            does not stick). The caller must guarantee each skipped submodule is
            its own dtype-uniform FSDP group (e.g. Qwen3.5's ``_fp32_params``
            holder, sharded separately in fp32), so leaving it fp32 cannot break
            FSDP's uniform-dtype rule.
    """
    fp32_keywords = _get_fp32_module_keywords(model)
    strict_fp32_keywords = _get_strict_fp32_module_keywords(model)
    has_dtensor_params = _has_dtensor_params(model)

    if has_dtensor_params:
        fp32_snapshots = _snapshot_fp32_tensors(
            model,
            parameter_keywords=strict_fp32_keywords,
            buffer_keywords=fp32_keywords,
        )
    else:
        fp32_snapshots = _snapshot_fp32_tensors(
            model,
            parameter_keywords=fp32_keywords,
            buffer_keywords=fp32_keywords,
        )

    # Detach skip_modules so ``model.to(dtype)`` does not descend into them. This
    # preserves their exact dtype (e.g. fp32 master weights) through the cast.
    detached: list[tuple[nn.Module, str, nn.Module]] = []
    if skip_modules:
        for _, parent in model.named_modules():
            for child_name, child in list(parent._modules.items()):
                if child is not None and child_name in skip_modules:
                    detached.append((parent, child_name, child))
                    parent._modules[child_name] = None

    try:
        model.to(dtype)
    finally:
        for parent, child_name, child in detached:
            parent._modules[child_name] = child

    if fp32_keywords:
        if has_dtensor_params:
            if strict_fp32_keywords:
                _restore_fp32_tensor_snapshots(
                    model,
                    parameter_snapshots=fp32_snapshots[0],
                    buffer_snapshots={},
                )

            buffer_only_keywords = [kw for kw in fp32_keywords if kw not in strict_fp32_keywords]
            if buffer_only_keywords:
                logger.warning(
                    "Model parameters are DTensors (FSDP2) — skipping fp32 parameter "
                    "restoration for non-strict keywords=%s. Only buffers will be restored to fp32. "
                    "FSDP2 requires uniform dtype within each parameter group.",
                    buffer_only_keywords,
                )
            _restore_fp32_tensor_snapshots(
                model,
                parameter_snapshots={},
                buffer_snapshots=fp32_snapshots[1],
            )
        else:
            _restore_fp32_tensor_snapshots(
                model,
                parameter_snapshots=fp32_snapshots[0],
                buffer_snapshots=fp32_snapshots[1],
            )


@contextmanager
def yield_fp32_model(model: nn.Module, restore_dtype: torch.dtype | None = None):
    """Run a block with the model temporarily in fp32, then cast it to ``restore_dtype``.

    On entry the whole model is cast to fp32; on exit it is cast to ``restore_dtype``
    (which defaults to the model's pre-context floating-point dtype, so by default the
    original dtype is restored). The exit cast is a no-op when the target is already fp32.

    The motivating use is from-scratch weight initialization. Sampling a random init directly
    in a reduced-precision dtype (e.g. bf16) distorts the init's variance/mean schedule: bf16's
    8-bit mantissa quantizes the small init magnitudes and biases the truncation/scaling
    arithmetic used by ``normal_`` / ``trunc_normal_``. In a deep residual stack this compounds
    and produces genuinely huge gradients on the first optimization steps of from-scratch
    pretraining (flat / diverging loss). Sampling in fp32 and then casting back avoids this while
    keeping reduced-precision storage: the round-to-bf16 of a correct fp32 sample is an unbiased
    per-element perturbation that preserves the init statistics. Wrap the body of a model's
    ``initialize_weights`` to keep that round-trip in one place.

    Works whether or not the model is already FSDP2-sharded: both casts are *uniform* whole-model
    casts, so FSDP2's invariant that every parameter in a group shares one dtype is preserved. In
    the AutoModel pipeline ``initialize_weights`` actually runs after sharding (via
    ``checkpointer.initialize_model_weights``), i.e. on DTensor params, which is supported.

    ``_keep_in_fp32_modules`` / ``_keep_in_fp32_modules_strict`` handling is delegated to
    ``cast_model_to_dtype``: on an unsharded model those modules' params and buffers are restored
    to fp32 on exit; on a sharded model, strict fp32 modules are restored while non-strict modules
    only have their buffers restored.

    Args:
        model: The model to run in fp32 within the context.
        restore_dtype: The dtype to cast the model to on exit. Defaults to the model's current
            floating-point dtype (captured before the fp32 cast), i.e. the original dtype.

    Yields:
        The same ``model``, now in fp32.

    Example:
        >>> with yield_fp32_model(self, dtype):
        ...     self.model.init_weights(buffer_device=buffer_device)
    """
    if restore_dtype is None:
        restore_dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.float32)
    cast_model_to_dtype(model, torch.float32)
    try:
        yield model
    finally:
        cast_model_to_dtype(model, restore_dtype)


def _get_strict_fp32_module_keywords(model: nn.Module) -> list[str]:
    val = getattr(model, "_keep_in_fp32_modules_strict", None)
    if not isinstance(val, (list, set, tuple)):
        return []
    return list(dict.fromkeys(val))


def _get_fp32_module_keywords(model: nn.Module) -> list[str]:
    """Collect module name patterns that must remain in fp32.

    Reads ``_keep_in_fp32_modules`` and ``_keep_in_fp32_modules_strict``
    from the model (the same attributes HuggingFace transformers uses).

    Args:
        model: The model to inspect.

    Returns:
        De-duplicated list of module-name keywords to keep in fp32.
    """
    keywords: list[str] = []
    for attr in ("_keep_in_fp32_modules_strict", "_keep_in_fp32_modules"):
        val = getattr(model, attr, None)
        # HuggingFace's PreTrainedModel.__init__ normalizes a class-level
        # list[str] into an instance-level set[str], so accept both (and tuple).
        if isinstance(val, (list, set, tuple)):
            keywords.extend(val)

    # de-duplicate while preserving order
    return list(dict.fromkeys(keywords))


def _has_dtensor_params(model: nn.Module) -> bool:
    """Check if any model parameter is a DTensor (FSDP2 sharded)."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False
    return any(isinstance(p, DTensor) for p in model.parameters())


def _snapshot_fp32_tensors(
    model: nn.Module,
    *,
    parameter_keywords: list[str],
    buffer_keywords: list[str],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Clone fp32-preserved tensors before a broad dtype cast.

    Casting ``fp32 -> bf16 -> fp32`` restores the dtype but not the original
    values. Snapshot the matching tensors first so strict fp32 state such as
    router correction bias or recurrent-decay parameters is restored exactly.
    """
    parameter_snapshots = {
        name: param.detach().to(torch.float32).clone()
        for name, param in model.named_parameters()
        if param.is_floating_point() and any(keyword in name for keyword in parameter_keywords)
    }
    buffer_snapshots = {
        name: buf.detach().to(torch.float32).clone()
        for name, buf in model.named_buffers(remove_duplicate=False)
        if buf.is_floating_point() and any(keyword in name for keyword in buffer_keywords)
    }
    return parameter_snapshots, buffer_snapshots


def _restore_fp32_tensor_snapshots(
    model: nn.Module,
    *,
    parameter_snapshots: dict[str, torch.Tensor],
    buffer_snapshots: dict[str, torch.Tensor],
) -> None:
    """Restore fp32-preserved tensors from pre-cast snapshots."""
    named_parameters = dict(model.named_parameters())
    for name, snapshot in parameter_snapshots.items():
        param = named_parameters.get(name)
        if param is None:
            continue
        param.data = snapshot.to(dtype=torch.float32)

    for name, snapshot in buffer_snapshots.items():
        # ActivationWrapper forwards __getattr__ / __setattr__ to the wrapped
        # module. Try the literal FQN first for buffers registered on the wrapped
        # leaf, then strip the wrapper alias as a fallback for forwarded names.
        candidate_names = [name]
        stripped_name = name.replace("._checkpoint_wrapped_module", "")
        if stripped_name != name:
            candidate_names.append(stripped_name)

        for candidate_name in candidate_names:
            module_name, _, buffer_name = candidate_name.rpartition(".")
            try:
                module = model.get_submodule(module_name) if module_name else model
            except AttributeError:
                continue
            if buffer_name not in module._buffers:
                continue
            module._buffers[buffer_name] = snapshot.to(dtype=torch.float32)
            break


def _restore_fp32_modules(model: nn.Module, fp32_keywords: list[str]) -> None:
    """Cast modules or individual tensors matching *fp32_keywords* back to float32.

    Only safe for unsharded models (plain tensors). FSDP2 requires uniform
    dtype within each parameter group, so this must not be called on DTensor-sharded
    models. Keywords may name modules (for example ``norm``) or individual
    parameters (for example ``attn_hc.fn``), matching HuggingFace's strict fp32
    module declarations.

    Args:
        model: The model (already cast to the target dtype).
        fp32_keywords: Substrings matched against dot-separated module names.
    """
    for name, module in model.named_modules():
        if any(kw in name for kw in fp32_keywords):
            module.to(torch.float32)
    for name, param in model.named_parameters():
        if any(kw in name for kw in fp32_keywords):
            param.data = param.data.to(torch.float32)
    for name, buf in model.named_buffers():
        if any(kw in name for kw in fp32_keywords):
            module_name, _, buffer_name = name.rpartition(".")
            module = model.get_submodule(module_name) if module_name else model
            module._buffers[buffer_name] = buf.to(torch.float32)


def _restore_fp32_buffers(model: nn.Module, fp32_keywords: list[str]) -> None:
    """Cast only matching buffers (not parameters) back to float32.

    Safe for FSDP2-sharded models because buffers are plain tensors, not
    DTensors managed by FSDP2.

    Args:
        model: The model (already cast to the target dtype).
        fp32_keywords: Substrings matched against dot-separated module names.
    """
    for name, module in model.named_modules():
        if any(kw in name for kw in fp32_keywords):
            for buf_name, buf in module.named_buffers(recurse=False):
                module._buffers[buf_name] = buf.to(torch.float32)
    for name, buf in model.named_buffers():
        if any(kw in name for kw in fp32_keywords):
            module_name, _, buffer_name = name.rpartition(".")
            module = model.get_submodule(module_name) if module_name else model
            module._buffers[buffer_name] = buf.to(torch.float32)


def compute_lm_head_logits(
    lm_head: nn.Module | None,
    hidden_states: torch.Tensor,
    logits_to_keep: int | torch.Tensor = 0,
    is_thd: bool = False,
    fp32_lm_head: bool = False,
    output_hidden_states: bool = False,
) -> CausalLMOutputWithPast:
    """Project hidden states through ``lm_head`` and wrap the result.

    Centralizes the lm_head projection and output packaging shared by every
    custom ``*ForCausalLM`` / ``*ForConditionalGeneration`` ``forward()``. The
    returned ``CausalLMOutputWithPast`` carries the projected ``logits`` and,
    when requested, the final ``hidden_states``; callers that also need ``loss``,
    ``past_key_values``, etc. read ``.logits`` and build their own output.

    - ``lm_head is None`` (e.g. a non-final pipeline-parallel stage that does not
      own the head): ``hidden_states`` is passed through as ``logits`` so the
      next stage receives it.
    - ``logits_to_keep == 0`` (training default): every position is projected.
      The full range is deliberately *not* sliced, because ``slice(0, None)`` on
      a DTensor is unsupported (it raises on the ``aten.alias`` op under tensor
      parallel with sequence parallelism).
    - ``logits_to_keep`` as a positive int or a tensor of indices: only the
      requested positions are projected. Both 2D ``[T, H]`` (THD/packed) and 3D
      ``[B, S, H]`` (BSHD) hidden states are handled.
    - ``is_thd``: THD/packed inputs yield 2D ``[T, V]`` logits; the leading batch
      dim is restored (``unsqueeze(0)`` -> ``[1, T, V]``) so downstream code sees
      a uniform ``[B, S, V]`` layout. The same restoration is applied to the
      ``hidden_states`` field. Only applied while the tensor is still 2D, so an
      ``inputs_embeds`` path that already produced ``[1, T, *]`` is left
      untouched.
    - ``fp32_lm_head``: run the projection in fp32 and cast the logits back to
      the input dtype. Used by models whose ``lm_head.weight`` has been promoted
      to fp32 (e.g. via the MoE ``lm_head_precision`` setting). The matmul goes
      through ``lm_head`` (``nn.Linear``, DTensor-aware under FSDP2) rather than
      ``F.linear`` so DTensor redistribution is preserved.
    - ``output_hidden_states``: when set, the (full-sequence, THD-restored)
      ``hidden_states`` are attached to the output so the fused cross-entropy
      path can recompute logits over every position; otherwise the field is
      ``None``.

    Args:
        lm_head: The language-model head module, or ``None`` on a pipeline stage
            that does not own it.
        hidden_states: Final hidden states, shaped ``[T, H]`` or ``[B, S, H]``.
        logits_to_keep: ``0`` to project every position; a positive int to keep
            the last ``N`` positions; or a tensor of position indices.
        is_thd: Whether the inputs were THD/packed; if so, a 2D logits (and
            hidden-states) result is unsqueezed back to a leading batch dim of 1.
        fp32_lm_head: Project in fp32 and cast the result back to the input
            dtype. Ignored when ``lm_head`` is ``None``.
        output_hidden_states: Attach the final hidden states to the output.

    Returns:
        A ``CausalLMOutputWithPast`` whose ``logits`` are the projected logits
        (or ``hidden_states`` unchanged when ``lm_head`` is ``None``) and whose
        ``hidden_states`` are the final hidden states when ``output_hidden_states``
        is set, else ``None``.
    """
    if lm_head is None:
        logits = hidden_states
    else:
        if isinstance(logits_to_keep, int) and logits_to_keep == 0:
            sliced = hidden_states
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            if hidden_states.dim() == 2:
                sliced = hidden_states[slice_indices, :]
            else:
                sliced = hidden_states[:, slice_indices, :]
        if fp32_lm_head:
            logits = lm_head(sliced.float()).to(hidden_states.dtype)
        else:
            logits = lm_head(sliced)
    if is_thd and logits.dim() == 2:
        logits = logits.unsqueeze(0)

    hidden_out = None
    if output_hidden_states:
        hidden_out = hidden_states.unsqueeze(0) if is_thd and hidden_states.dim() == 2 else hidden_states

    return CausalLMOutputWithPast(logits=logits, hidden_states=hidden_out)


def cast_frozen_modules_to_compute_dtype(model: nn.Module, compute_dtype: torch.dtype | None) -> None:
    """Cast the floating-point tensors of frozen submodules to ``compute_dtype``.

    When parameters are stored in fp32 (the fp32-master-weights pattern) while compute runs
    in bf16, a fully frozen submodule -- such as a frozen vision tower -- can still produce
    fp32 values that flow into bf16 trainable modules and raise a dtype mismatch in the next
    matmul. This walks each maximal fully-frozen submodule and casts its parameters and
    buffers to ``compute_dtype``, handling the two tensor kinds differently:

    * **Parameters** are cast only when they are plain (unsharded) tensors. Sharded (DTensor)
      params are left as-is: FSDP all-gathers them to the compute dtype during forward, and
      changing a sharded param's dtype in place would desync FSDP's flat-parameter and
      ``orig_dtype`` bookkeeping.
    * **Buffers** are always cast. Buffers are never sharded, so they stay in their stored
      dtype regardless of the wrapper; an fp32 buffer (for example a standardization
      constant) used in a forward op promotes the surrounding bf16 activations to fp32.

    Tensors whose qualified name matches ``_keep_in_fp32_modules`` or
    ``_keep_in_fp32_modules_strict`` are left in fp32. The function is a no-op when
    ``compute_dtype`` is None and for tensors already in ``compute_dtype``. Frozen modules are
    never updated, so casting them does not affect training accuracy.

    Args:
        model: The model, already materialized, checkpoint-loaded, and sharded.
        compute_dtype: The compute dtype (``mp_policy.param_dtype``); None disables the cast.
    """
    if compute_dtype is None:
        return

    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = ()

    fp32_keywords = _get_fp32_module_keywords(model)

    def _is_fp32_pinned(name: str) -> bool:
        return any(kw in name for kw in fp32_keywords)

    # ``named_modules`` yields parents before children, so the first fully-frozen subtree
    # we accept is always maximal; descendants are skipped via the ancestor check below.
    # Sharded subtrees are included so their buffers get cast; their params are skipped per-tensor.
    selected: list[str] = []
    for name, module in model.named_modules():
        params = list(module.parameters(recurse=True))
        if not params:
            continue
        if any(p.requires_grad for p in params):
            continue
        if any(name == anc or name.startswith(anc + ".") for anc in selected):
            continue
        selected.append(name)

    for name in selected:
        module = model.get_submodule(name) if name else model
        prefix = f"{name}." if name else ""
        # Parameters: cast plain (unsharded) floats; leave sharded (DTensor) params to FSDP.
        for param_name, param in module.named_parameters():
            full_name = prefix + param_name
            if _is_fp32_pinned(full_name):
                continue
            if DTensor and isinstance(param, DTensor):
                continue
            if param.is_floating_point() and param.dtype != compute_dtype:
                param.data = param.data.to(compute_dtype)
        # Buffers: never FSDP-managed (always plain tensors), so always safe to cast.
        for buffer_name, buf in module.named_buffers():
            full_name = prefix + buffer_name
            if _is_fp32_pinned(full_name):
                continue
            if buf.is_floating_point() and buf.dtype != compute_dtype:
                owner_name, _, leaf = buffer_name.rpartition(".")
                owner = module.get_submodule(owner_name) if owner_name else module
                owner._buffers[leaf] = buf.to(compute_dtype)


__all__ = [
    "BackendConfig",
    "Float32RMSNorm",
    "TEFp8Config",
    "cast_frozen_modules_to_compute_dtype",
    "cast_model_to_dtype",
    "compute_lm_head_logits",
    "get_is_first_microbatch",
    "get_is_optim_step",
    "get_rope_config",
    "initialize_linear_module",
    "initialize_rms_norm_module",
    "set_is_first_microbatch",
    "set_is_optim_step",
]

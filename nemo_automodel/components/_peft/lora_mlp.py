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

# Recompute SwiGLU in backward follows NVIDIA Megatron-Core's SwiGLUFunction; the fused LoRA-MLP and
# the in-place backward buffer reuse follow Unsloth's LoRA_MLP (github.com/unslothai/unsloth). Both Apache-2.0.

"""Fused LoRA SwiGLU MLP.

Applying LoRA per ``nn.Linear`` makes standard autograd save every MLP intermediate
(``gate_out``, ``up_out``, ``silu_out`` and the down-projection input). For a SwiGLU MLP
that is several ``tokens x intermediate`` tensors per layer. Fusing gate+up+down+SwiGLU into a
single autograd ``Function`` lets us save only ``(x, gate_out, up_out)`` and **recompute** the
SwiGLU activation and the down-projection input during the backward pass, which roughly halves
MLP activation memory at equal speed.

The SwiGLU activation/gradient are computed by elementwise Triton kernels (``_swiglu_fwd`` /
``_swiglu_bwd``) so no separate ``sigmoid``/``silu``/``mul`` activation buffers are materialized;
a pure-torch fallback is used when Triton is unavailable. The matmuls stay on cuBLAS.
"""

import torch
import torch.nn.functional as F
from packaging import version
from torch.distributed.tensor import DTensor

from nemo_automodel.shared.import_utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = bool(version.parse(triton.__version__) >= version.parse("2.0.0"))
except ImportError:  # pragma: no cover
    HAVE_TRITON = False
if not HAVE_TRITON:  # pragma: no cover
    from unittest.mock import MagicMock

    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()


@triton.jit
def _swiglu_fwd_kernel(e_ptr, g_ptr, h_ptr, n_elements, BLOCK: tl.constexpr):
    """h = silu(e) * g, elementwise. Avoids a separate silu activation buffer."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    e = tl.load(e_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    h = (e * tl.sigmoid(e)) * g
    tl.store(h_ptr + offs, h.to(h_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _swiglu_bwd_kernel(dh_ptr, e_ptr, g_ptr, n_elements, BLOCK: tl.constexpr):
    """In-place SwiGLU backward (h = silu(e)*g). Reads (d_h, e, g) and overwrites the SAME three
    buffers with (h, d_g, d_e) — zero new allocations. Safe for a single backward: the saved
    ``e``/``g`` are dead afterward (no double-backward in LoRA SFT)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    dh = tl.load(dh_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    e = tl.load(e_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(e)
    f = e * sig  # silu(e)
    # in-place mapping (d_h<-h, e<-d_g, g<-d_e) adapted from Unsloth's _DWf_DW_dfg_kernel
    tl.store(dh_ptr + offs, (f * g).to(dh_ptr.dtype.element_ty), mask=mask)  # d_h buffer <- h
    tl.store(e_ptr + offs, (dh * f).to(e_ptr.dtype.element_ty), mask=mask)  # e buffer  <- d_g
    de = (dh * g) * (sig * (1.0 + e * (1.0 - sig)))  # d_f * silu'(e)
    tl.store(g_ptr + offs, de.to(g_ptr.dtype.element_ty), mask=mask)  # g buffer  <- d_e


def _use_triton(*tensors) -> bool:
    return HAVE_TRITON and all(t.is_cuda and t.is_contiguous() for t in tensors)


def _cast(dtype: torch.dtype, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Cast tensors to ``dtype``, preserving shape.

    Args:
        dtype: Target dtype.
        tensors: Tensors of arbitrary shape; each is returned with the same shape and device.

    Returns:
        The tensors cast to ``dtype``, in argument order. ``Tensor.to`` returns the tensor object
        itself when it already has ``dtype``, so a matching cast allocates nothing -- which also
        means a result may alias its input (and, for a reshaped gradient, the caller's ``grad_out``).
        Treat every result as read-only.
    """
    return tuple(t.to(dtype) for t in tensors)


def _like_input(grad: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """Return ``grad`` on the device and in the dtype of the input it belongs to.

    Autograd rejects a gradient whose device or dtype differs from its input's.

    * Device: under pipeline-parallel graph construction torch tracks the LoRA parameters on the meta
      device while the activations (and the grads computed from them) are on cuda, so it rejected the
      cuda grads ("invalid gradient ... expected device meta but got cuda").
    * Dtype: under mixed precision (FSDP2 ``param_dtype=bf16`` with fp32 activations) the gradients
      come out in the lower forward compute dtype while the input they belong to is fp32.

    Both are no-ops in normal single-device, uniform-dtype training.

    Args:
        grad: Gradient tensor, already in ``inp``'s shape.
        inp: The forward input this gradient belongs to; only its device and dtype are read.

    Returns:
        Tensor of ``grad``'s shape on ``inp``'s device and in ``inp``'s dtype. Returns ``grad``
        itself when both already match.
    """
    return grad.to(device=inp.device, dtype=inp.dtype)


def _swiglu_fwd(e, g):
    """h = silu(e) * g."""
    if not _use_triton(e, g):
        return F.silu(e) * g
    h = torch.empty_like(e)
    n = e.numel()
    _swiglu_fwd_kernel[(triton.cdiv(n, 1024),)](e, g, h, n, BLOCK=1024)
    return h


def _swiglu_bwd_inplace(dh, e, g):
    """In-place: overwrite (dh, e, g) with (h, d_g, d_e) for h = silu(e)*g. Returns the aliases.

    No new ``tokens x intermediate`` buffers (the Triton path), so backward activation memory stays
    at ~model size. The torch fallback computes into temporaries first, then copies in place
    (correctness over frugality on the non-Triton path)."""
    if not _use_triton(dh, e, g):
        sig = torch.sigmoid(e)
        f = e * sig
        d_g = dh * f
        d_e = (dh * g) * (sig * (1.0 + e * (1.0 - sig)))
        h = f * g
        dh.copy_(h)
        e.copy_(d_g)
        g.copy_(d_e)
        return dh, e, g
    n = dh.numel()
    _swiglu_bwd_kernel[(triton.cdiv(n, 1024),)](dh, e, g, n, BLOCK=1024)
    return dh, e, g


class LoRASwiGLUMLPFunction(torch.autograd.Function):
    """Fused ``down(silu(gate(x)) * up(x))`` with LoRA on all three projections.

    Saves only ``(x, gate_out, up_out)`` plus the (frozen) base/LoRA weights; the SwiGLU
    activation and the down-projection input are recomputed in ``backward``. Base weights are
    frozen (no gradient); LoRA ``A``/``B`` and ``x`` receive gradients.

    Linear convention is ``F.linear(x, W) = x @ W.T``:
      * ``gW``/``uW``: ``(inter, hidden)``; ``dW``: ``(hidden, inter)``
      * ``gA``/``uA``: ``(rank, hidden)``; ``gB``/``uB``: ``(inter, rank)``
      * ``dA``: ``(rank, inter)``; ``dB``: ``(hidden, rank)``
    """

    @staticmethod
    def forward(ctx, x, gW, gA, gB, gS, uW, uA, uB, uS, dW, dA, dB, dS):
        """Compute the fused SwiGLU MLP output, saving only x, gate_out, up_out for backward."""
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])

        # Build each projection in-place (base buffer += scaled LoRA delta) to avoid an extra
        # tokens x out buffer per projection. forward runs outside autograd, so this is safe.
        # ``addmm_`` is not autocast-eligible — an in-place op cannot change its output dtype — so
        # unlike the ``F.linear`` calls its operands are *not* auto-cast and must already be in the
        # base projection's dtype. They are not when the adapters are held in a different dtype than
        # the base (``lora_dtype``), which the per-linear path handles via autocast; cast to keep the
        # fused path identical to it. A no-op when the dtypes already agree.
        e = F.linear(x2, gW)  # gate_out (saved)
        e.addmm_(*_cast(e.dtype, F.linear(x2, gA) * gS, gB.t()))
        g = F.linear(x2, uW)  # up_out (saved)
        g.addmm_(*_cast(g.dtype, F.linear(x2, uA) * uS, uB.t()))
        h = _swiglu_fwd(e, g)  # down-projection input (recomputed in backward)
        out = F.linear(h, dW)
        out.addmm_(*_cast(out.dtype, F.linear(h, dA) * dS, dB.t()))

        # Save only x / gate_out / up_out; the SwiGLU activation and down-input are recomputed.
        ctx.save_for_backward(x2, e, g, gA, gB, uA, uB, dA, dB)
        ctx.bases = (gW, uW, dW)
        ctx.scales = (gS, uS, dS)
        ctx.orig_shape = orig_shape
        return out.view(*orig_shape[:-1], dW.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        """Recompute SwiGLU + down-input, then backprop to x and the six LoRA matrices.

        The SwiGLU backward overwrites the saved ``e``/``g`` and the ``d_h`` buffers in place
        (``h``, ``d_g``, ``d_e``), so the only new ``tokens x intermediate`` buffer is ``d_h``.
        Safe for single backward (no double-backward in LoRA SFT); ``grad_out`` and the LoRA
        ``A``/``B`` are never mutated.
        """
        x, e, g, gA, gB, uA, uB, dA, dB = ctx.saved_tensors
        gW, uW, dW = ctx.bases
        gS, uS, dS = ctx.scales
        dY = grad_out.reshape(-1, grad_out.shape[-1])
        needs_x = ctx.needs_input_grad[0]

        # A custom backward runs outside autocast, so nothing re-applies the forward's compute dtype
        # here. Under mixed precision (FSDP2 ``param_dtype=bf16`` with fp32 activations) the saved
        # ``x`` is fp32 while the base/LoRA weights are bf16, and the matmuls below would raise
        # "expected m1 and m2 to have the same dtype". ``e`` is a forward matmul output, so its dtype
        # *is* the forward compute dtype (the one autocast picked): run everything in it, exactly as
        # autograd would through autocast's own cast nodes, and hand each gradient back in its
        # input's dtype. Every cast is a no-op when the dtypes already agree, so the uniform-dtype
        # path allocates nothing extra; ``e``/``g`` are already in the compute dtype and stay
        # untouched so the in-place SwiGLU backward keeps writing into the saved buffers.
        compute_dtype = e.dtype
        dY, xc = _cast(compute_dtype, dY, x)
        (dW,) = _cast(compute_dtype, dW)
        gAc, gBc, uAc, uBc, dAc, dBc = _cast(compute_dtype, gA, gB, uA, uB, dA, dB)

        # Grad of the down-projection input h = silu(e)*g (does not need h itself):
        # d_h = dY@dW + dS*(dY@dB)@dA. This is the only new (N, inter) buffer in backward.
        d_P = dS * (dY @ dBc)  # (N, r)
        d_h = torch.addmm(dY @ dW, d_P, dAc)  # (N, inter)

        # Recompute SwiGLU and produce (h, d_g, d_e) in the (d_h, e, g) buffers (in place).
        h, d_g, d_e = _swiglu_bwd_inplace(d_h, e, g)

        # ---- down LoRA grads (use recomputed h): out = h@dW.T + dS*(h@dA.T)@dB.T ----
        P = F.linear(h, dAc)  # (N, r)
        d_dB = dS * (dY.t() @ P)  # (hidden, r)
        d_dA = d_P.t() @ h  # (r, inter)

        # ---- up: g = x@uW.T + uS*(x@uA.T)@uB.T ----
        Q = F.linear(xc, uAc)  # (N, r)
        d_uB = uS * (d_g.t() @ Q)  # (inter, r)
        d_Q = uS * (d_g @ uBc)  # (N, r)
        d_uA = d_Q.t() @ xc  # (r, hidden)

        # ---- gate: e = x@gW.T + gS*(x@gA.T)@gB.T ----
        R = F.linear(xc, gAc)  # (N, r)
        d_gB = gS * (d_e.t() @ R)  # (inter, r)
        d_R = gS * (d_e @ gBc)  # (N, r)
        d_gA = d_R.t() @ xc  # (r, hidden)

        d_x = None
        if needs_x:
            # gate base+lora (d_e@gW + d_R@gA) plus up base+lora (d_g@uW + d_Q@uA). gW/uW are read
            # only here, so cast them inside the branch -- they are (inter, hidden) each, and an
            # eager cast would burn that memory on every backward that does not need d_x.
            gW, uW = _cast(compute_dtype, gW, uW)
            d_x = torch.addmm(d_e @ gW, d_R, gAc)
            d_x = d_x.addmm_(d_g, uW).addmm_(d_Q, uAc).view(ctx.orig_shape)

        # Hand every gradient back on its input's device and in its input's dtype (see _like_input).
        d_gA, d_gB = _like_input(d_gA, gA), _like_input(d_gB, gB)
        d_uA, d_uB = _like_input(d_uA, uA), _like_input(d_uB, uB)
        d_dA, d_dB = _like_input(d_dA, dA), _like_input(d_dB, dB)
        if d_x is not None:
            d_x = _like_input(d_x, x)

        # order matches forward(x, gW, gA, gB, gS, uW, uA, uB, uS, dW, dA, dB, dS)
        return (d_x, None, d_gA, d_gB, None, None, d_uA, d_uB, None, None, d_dA, d_dB, None)


def _fusible(module) -> bool:
    """A LoRA linear is fusible when it is a plain materialized adapter."""
    lora_A = getattr(module, "lora_A", None)
    lora_B = getattr(module, "lora_B", None)
    if lora_A is None or lora_B is None:
        return False
    if not getattr(module, "use_memory_efficient_lora", False):
        return False
    if getattr(module, "use_dora", False):
        return False
    if getattr(module, "dropout_p", 0.0) and module.training:
        return False
    # The fused forward calls ``F.linear(x, base_weight)`` with no bias term, so a biased base
    # projection would train on silently wrong math. Models plumb this from the HF config
    # (``nn.Linear(..., bias=config.mlp_bias)``), so decline and let the per-linear path add it.
    if getattr(module, "bias", None) is not None:
        return False
    # QLoRA / quantized base weights are stored as packed buffers (e.g. bitsandbytes 4-bit
    # carries a ``quant_state`` and a flattened weight shaped like ``(1, out*in/2)`` rather than
    # a 2D ``(out_features, in_features)`` matrix). The fused path calls ``F.linear(x, base_weight)``
    # directly, which fails for a packed buffer ("mat1 and mat2 shapes cannot be multiplied"); bail
    # so the per-linear ``LinearLoRA.forward`` path (which dequantizes the base) handles it instead.
    base_w = module.weight
    if getattr(base_w, "quant_state", None) is not None or getattr(module, "quant_state", None) is not None:
        return False
    out_features = getattr(module, "out_features", None)
    in_features = getattr(module, "in_features", None)
    if out_features is not None and in_features is not None and tuple(base_w.shape) != (out_features, in_features):
        return False
    for w in (base_w, lora_A.weight, lora_B.weight):
        if isinstance(w, DTensor):
            return False
        if getattr(w, "is_meta", False):
            return False
    return True


def fused_lora_swiglu_mlp(gate, up, down, x):
    """Run ``down(silu(gate(x)) * up(x))`` through the fused LoRA autograd function.

    ``gate``/``up``/``down`` are ``LinearLoRA`` modules. Returns the MLP output, or ``None`` if the
    modules are not fusible (plain ``nn.Linear`` without LoRA, DoRA, active dropout, or
    DTensor-sharded) so the caller can fall back to the standard per-linear path.
    """
    if not (_fusible(gate) and _fusible(up) and _fusible(down)):
        return None
    return LoRASwiGLUMLPFunction.apply(
        x,
        gate.weight,
        gate.lora_A.weight,
        gate.lora_B.weight,
        gate.scale,
        up.weight,
        up.lora_A.weight,
        up.lora_B.weight,
        up.scale,
        down.weight,
        down.lora_A.weight,
        down.lora_B.weight,
        down.scale,
    )


class LoRAReLU2MLPFunction(torch.autograd.Function):
    """Fused ``down(relu(up(x)) ** 2)`` (ReLU²) with LoRA on the up/down projections.

    The non-gated counterpart of :class:`LoRASwiGLUMLPFunction` (e.g. Nemotron-H's dense MLP).
    Saves only ``(x, up_out)`` plus the frozen base/LoRA weights; the ReLU² activation (the
    down-projection input) is recomputed in ``backward``. Base weights are frozen.

    Linear convention ``F.linear(x, W) = x @ W.T``: ``uW`` ``(inter, hidden)``; ``dW``
    ``(hidden, inter)``; ``uA`` ``(rank, hidden)``; ``uB`` ``(inter, rank)``; ``dA`` ``(rank, inter)``;
    ``dB`` ``(hidden, rank)``.
    """

    @staticmethod
    def forward(ctx, x, uW, uA, uB, uS, dW, dA, dB, dS):
        """Compute the fused ReLU² MLP output, saving only x and up_out for backward."""
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        # See LoRASwiGLUMLPFunction.forward: ``addmm_`` is not autocast-eligible, so cast its operands
        # to the base projection's dtype for a ``lora_dtype`` that differs from the base weights.
        e = F.linear(x2, uW)  # up_out (saved)
        e.addmm_(*_cast(e.dtype, F.linear(x2, uA) * uS, uB.t()))
        relu_e = torch.relu(e)
        f = relu_e * relu_e  # down-projection input (recomputed in backward)
        out = F.linear(f, dW)
        out.addmm_(*_cast(out.dtype, F.linear(f, dA) * dS, dB.t()))
        ctx.save_for_backward(x2, e, uA, uB, dA, dB)
        ctx.bases = (uW, dW)
        ctx.scales = (uS, dS)
        ctx.orig_shape = orig_shape
        return out.view(*orig_shape[:-1], dW.shape[0])

    @staticmethod
    def backward(ctx, grad_out):
        """Recompute ReLU² + down-input, then backprop to x, up-LoRA, and down-LoRA."""
        x, e, uA, uB, dA, dB = ctx.saved_tensors
        uW, dW = ctx.bases
        uS, dS = ctx.scales
        dY = grad_out.reshape(-1, grad_out.shape[-1])
        needs_x = ctx.needs_input_grad[0]

        # See LoRASwiGLUMLPFunction.backward: a custom backward runs outside autocast, so recompute in
        # the forward compute dtype (``e``'s, since ``e`` is a forward matmul output) and hand every
        # gradient back in its input's dtype. All no-ops when the tensors already share a dtype.
        compute_dtype = e.dtype
        dY, xc = _cast(compute_dtype, dY, x)
        (dW,) = _cast(compute_dtype, dW)
        uAc, uBc, dAc, dBc = _cast(compute_dtype, uA, uB, dA, dB)

        relu_e = torch.relu(e)
        f = relu_e * relu_e  # recompute down-projection input

        # ---- down: out = f@dW.T + dS*(f@dA.T)@dB.T ----
        d_P = dS * (dY @ dBc)  # (N, r)
        P = F.linear(f, dAc)  # (N, r)
        d_dB = dS * (dY.t() @ P)  # (hidden, r)
        d_dA = d_P.t() @ f  # (r, inter)
        d_f = torch.addmm(dY @ dW, d_P, dAc)  # (N, inter)

        # ---- ReLU²: d(relu(e)**2)/de = 2*relu(e). Reuse d_f's buffer for d_e. ----
        d_e = d_f.mul_(2.0 * relu_e)

        # ---- up: e = x@uW.T + uS*(x@uA.T)@uB.T ----
        Q = F.linear(xc, uAc)  # (N, r)
        d_uB = uS * (d_e.t() @ Q)  # (inter, r)
        d_Q = uS * (d_e @ uBc)  # (N, r)
        d_uA = d_Q.t() @ xc  # (r, hidden)

        d_x = None
        if needs_x:
            (uW,) = _cast(compute_dtype, uW)  # read only here; see LoRASwiGLUMLPFunction.backward
            d_x = torch.addmm(d_e @ uW, d_Q, uAc).view(ctx.orig_shape)

        # Hand every gradient back on its input's device and in its input's dtype (see _like_input).
        d_uA, d_uB = _like_input(d_uA, uA), _like_input(d_uB, uB)
        d_dA, d_dB = _like_input(d_dA, dA), _like_input(d_dB, dB)
        if d_x is not None:
            d_x = _like_input(d_x, x)

        # order matches forward(x, uW, uA, uB, uS, dW, dA, dB, dS)
        return (d_x, None, d_uA, d_uB, None, None, d_dA, d_dB, None)


def fused_lora_relu2_mlp(up, down, x):
    """Run ``down(relu(up(x)) ** 2)`` through the fused LoRA autograd function.

    ``up``/``down`` are ``LinearLoRA`` modules. Returns the MLP output, or ``None`` if either is not
    fusible (plain ``nn.Linear``, DoRA, active dropout, or DTensor-sharded) for caller fallback.
    """
    if not (_fusible(up) and _fusible(down)):
        return None
    return LoRAReLU2MLPFunction.apply(
        x,
        up.weight,
        up.lora_A.weight,
        up.lora_B.weight,
        up.scale,
        down.weight,
        down.lora_A.weight,
        down.lora_B.weight,
        down.scale,
    )


def _is_silu(act_fn) -> bool:
    """Detect a SiLU/Swish activation robustly across implementations.

    Covers ``nn.SiLU``, HF's ``SiLUActivation`` / ``ACT2FN["silu"]`` (whose class is *not*
    ``nn.SiLU``), and functional ``silu`` — by verifying numerically against ``F.silu`` so a
    GeGLU/ReLU activation is never mistaken for SiLU.
    """
    if act_fn is None:
        return False
    if isinstance(act_fn, torch.nn.SiLU):
        return True
    try:
        x = torch.linspace(-3.0, 3.0, 16)
        return bool(torch.allclose(act_fn(x), F.silu(x), atol=1e-4))
    except Exception:
        return False


def _is_silu_swiglu_mlp(module) -> bool:
    """Return True if ``module`` is a SiLU-gated SwiGLU MLP with separate gate/up/down projections.

    Matches AutoModel's ``moe.layers.MLP`` (``activation == "swiglu"``) and HF-style MLPs whose
    ``act_fn`` is SiLU (e.g. Llama / Qwen / Mistral, including HF's ``SiLUActivation``). Non-gated
    (ReLU²), clamped-SwiGLU (``swiglu_limit > 0``), non-SiLU gated (e.g. GeGLU), and combined
    ``gate_up_proj`` MLPs are excluded so the fused kernel is only used where
    ``down(silu(gate(x)) * up(x))`` over separate projections is exactly correct.
    """
    if any(getattr(module, proj, None) is None for proj in ("gate_proj", "up_proj", "down_proj")):
        return False
    activation = getattr(module, "activation", None)
    if activation is not None:  # AutoModel native MLP carries an explicit activation tag
        return activation == "swiglu" and not getattr(module, "swiglu_limit", 0)
    return _is_silu(getattr(module, "act_fn", None))  # HF-style MLP


def _is_relu2_mlp(module) -> bool:
    """Return True if ``module`` is a non-gated ReLU² MLP (``up_proj``/``down_proj``, no gate).

    Matches AutoModel's ``moe.layers.MLP`` with ``activation == "relu2"`` (e.g. Nemotron-H's dense
    MLP), where the forward is ``down(relu(up(x)) ** 2)``.
    """
    if getattr(module, "up_proj", None) is None or getattr(module, "down_proj", None) is None:
        return False
    if getattr(module, "gate_proj", None) is not None:
        return False  # gated MLP is handled by the SwiGLU path
    return getattr(module, "activation", None) == "relu2"


def _projs_are_fusible(module, projs) -> bool:
    return all(_fusible(getattr(module, proj)) for proj in projs)


def _swiglu_forward(mod, orig_forward):
    def forward(x):
        out = fused_lora_swiglu_mlp(mod.gate_proj, mod.up_proj, mod.down_proj, x)
        return out if out is not None else orig_forward(x)

    return forward


def _relu2_forward(mod, orig_forward):
    def forward(x):
        out = fused_lora_relu2_mlp(mod.up_proj, mod.down_proj, x)
        return out if out is not None else orig_forward(x)

    return forward


def install_fused_lora_mlp(model) -> int:
    """Swap each LoRA-applied SwiGLU or ReLU² MLP's ``forward`` to the fused path (with fallback).

    Intended to be called by the LoRA matcher after the projections have been patched to
    ``LinearLoRA``. Handles SiLU-SwiGLU MLPs (gate/up/down) via :func:`fused_lora_swiglu_mlp` and
    non-gated ReLU² MLPs (up/down) via :func:`fused_lora_relu2_mlp`. A wrapper is installed only when
    the projections are fusible and materialized at install time. The installed ``forward`` still
    falls back to the module's original per-linear ``forward`` whenever fusion stops applying at
    runtime, for example once projections become ``DTensor`` under tensor/expert parallelism, or for
    DoRA / active dropout. This keeps the fused memory win on single-GPU and pure-DP while staying
    correct (and identical to the unfused path) under sharding and PP/meta construction.

    Returns the number of MLP modules whose ``forward`` was swapped. Idempotent.
    """
    count = 0
    for mlp in model.modules():
        if getattr(mlp, "_lora_mlp_fused", False):
            continue
        if _is_silu_swiglu_mlp(mlp) and _projs_are_fusible(mlp, ("gate_proj", "up_proj", "down_proj")):
            mlp.forward = _swiglu_forward(mlp, mlp.forward)
        elif _is_relu2_mlp(mlp) and _projs_are_fusible(mlp, ("up_proj", "down_proj")):
            mlp.forward = _relu2_forward(mlp, mlp.forward)
        else:
            continue
        mlp._lora_mlp_fused = True
        count += 1
    return count

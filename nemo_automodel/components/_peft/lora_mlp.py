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

import logging
import os

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

logger = logging.getLogger(__name__)

_DEBUG_NAN = os.environ.get("NEMO_LORA_MLP_DEBUG_NAN", "0").lower() not in ("0", "", "false", "no")
_DEBUG_NAN_SEEN: set[tuple[str, str, str]] = set()


def _debug_nonfinite(debug_name: str | None, stage: str, **tensors) -> None:
    """Log first nonfinite tensor per fused MLP stage when NEMO_LORA_MLP_DEBUG_NAN=1."""
    if not _DEBUG_NAN or debug_name is None:
        return
    for tensor_name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            continue
        finite = torch.isfinite(tensor)
        if bool(finite.all().item()):
            continue
        key = (debug_name, stage, tensor_name)
        if key in _DEBUG_NAN_SEEN:
            continue
        _DEBUG_NAN_SEEN.add(key)
        finite_count = int(finite.sum().item())
        total = tensor.numel()
        finite_abs_max = torch.nan_to_num(tensor.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).abs().max().item()
        logger.error(
            "[lora-mlp-nan] module=%s stage=%s tensor=%s shape=%s dtype=%s device=%s nonfinite=%d/%d finite_abs_max=%g",
            debug_name,
            stage,
            tensor_name,
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
            total - finite_count,
            total,
            finite_abs_max,
        )


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
    def forward(ctx, x, gW, gA, gB, gS, uW, uA, uB, uS, dW, dA, dB, dS, debug_name=None):
        """Compute the fused SwiGLU MLP output, saving only x, gate_out, up_out for backward."""
        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])

        # Build each projection in-place (base buffer += scaled LoRA delta) to avoid an extra
        # tokens x out buffer per projection. forward runs outside autograd, so this is safe.
        e = F.linear(x2, gW)  # gate_out (saved)
        e.addmm_(F.linear(x2, gA) * gS, gB.t())
        g = F.linear(x2, uW)  # up_out (saved)
        g.addmm_(F.linear(x2, uA) * uS, uB.t())
        h = _swiglu_fwd(e, g)  # down-projection input (recomputed in backward)
        out = F.linear(h, dW)
        out.addmm_(F.linear(h, dA) * dS, dB.t())
        _debug_nonfinite(debug_name, "forward", x=x2, gate=e, up=g, swiglu=h, out=out)

        # Save only x / gate_out / up_out; the SwiGLU activation and down-input are recomputed.
        ctx.save_for_backward(x2, e, g, gA, gB, uA, uB, dA, dB)
        ctx.bases = (gW, uW, dW)
        ctx.scales = (gS, uS, dS)
        ctx.orig_shape = orig_shape
        ctx.debug_name = debug_name
        ctx.has_debug_name_input = debug_name is not None
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
        debug_name = getattr(ctx, "debug_name", None)
        _debug_nonfinite(debug_name, "backward_input", grad_out=dY)

        # Grad of the down-projection input h = silu(e)*g (does not need h itself):
        # d_h = dY@dW + dS*(dY@dB)@dA. This is the only new (N, inter) buffer in backward.
        d_P = dS * (dY @ dB)  # (N, r)
        d_h = torch.addmm(dY @ dW, d_P, dA)  # (N, inter)
        _debug_nonfinite(debug_name, "backward_down_input", d_P=d_P, d_h=d_h)

        # Recompute SwiGLU and produce (h, d_g, d_e) in the (d_h, e, g) buffers (in place).
        h, d_g, d_e = _swiglu_bwd_inplace(d_h, e, g)
        _debug_nonfinite(debug_name, "backward_swiglu", h=h, d_up=d_g, d_gate=d_e)

        # ---- down LoRA grads (use recomputed h): out = h@dW.T + dS*(h@dA.T)@dB.T ----
        P = F.linear(h, dA)  # (N, r)
        d_dB = dS * (dY.t() @ P)  # (hidden, r)
        d_dA = d_P.t() @ h  # (r, inter)
        _debug_nonfinite(debug_name, "backward_down_lora", P=P, d_down_A=d_dA, d_down_B=d_dB)

        # ---- up: g = x@uW.T + uS*(x@uA.T)@uB.T ----
        Q = F.linear(x, uA)  # (N, r)
        d_uB = uS * (d_g.t() @ Q)  # (inter, r)
        d_Q = uS * (d_g @ uB)  # (N, r)
        d_uA = d_Q.t() @ x  # (r, hidden)
        _debug_nonfinite(debug_name, "backward_up_lora", Q=Q, d_Q=d_Q, d_up_A=d_uA, d_up_B=d_uB)

        # ---- gate: e = x@gW.T + gS*(x@gA.T)@gB.T ----
        R = F.linear(x, gA)  # (N, r)
        d_gB = gS * (d_e.t() @ R)  # (inter, r)
        d_R = gS * (d_e @ gB)  # (N, r)
        d_gA = d_R.t() @ x  # (r, hidden)
        _debug_nonfinite(debug_name, "backward_gate_lora", R=R, d_R=d_R, d_gate_A=d_gA, d_gate_B=d_gB)

        d_x = None
        if needs_x:
            # gate base+lora (d_e@gW + d_R@gA) plus up base+lora (d_g@uW + d_Q@uA)
            d_x = torch.addmm(d_e @ gW, d_R, gA)
            d_x = d_x.addmm_(d_g, uW).addmm_(d_Q, uA).view(ctx.orig_shape)
        _debug_nonfinite(debug_name, "backward_output", d_x=d_x)

        # Each returned gradient must live on the same device as its corresponding input. Under
        # pipeline-parallel graph construction torch tracks the LoRA parameters on the meta device
        # while the activations (and the grads computed from them) are on cuda, so it rejected the
        # cuda grads ("invalid gradient ... expected device meta but got cuda"). Move each LoRA grad
        # onto its parameter's device: a no-op in normal single-device training, and correct in the
        # PP/meta graph pass (the real gradients still flow on the cuda execution passes).
        d_gA, d_gB = d_gA.to(gA.device), d_gB.to(gB.device)
        d_uA, d_uB = d_uA.to(uA.device), d_uB.to(uB.device)
        d_dA, d_dB = d_dA.to(dA.device), d_dB.to(dB.device)
        if d_x is not None:
            d_x = d_x.to(x.device)

        # order matches forward(x, gW, gA, gB, gS, uW, uA, uB, uS, dW, dA, dB, dS)
        grads = (d_x, None, d_gA, d_gB, None, None, d_uA, d_uB, None, None, d_dA, d_dB, None)
        if getattr(ctx, "has_debug_name_input", False):
            return grads + (None,)
        return grads


def _fusible(module) -> bool:
    """A LoRA linear is fusible when it is a plain (non-DoRA, dropout-free, non-DTensor) adapter."""
    lora_A = getattr(module, "lora_A", None)
    lora_B = getattr(module, "lora_B", None)
    if lora_A is None or lora_B is None:
        return False
    if getattr(module, "use_dora", False):
        return False
    if getattr(module, "dropout_p", 0.0) and module.training:
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
    return True


def fused_lora_swiglu_mlp(gate, up, down, x, debug_name: str | None = None):
    """Run ``down(silu(gate(x)) * up(x))`` through the fused LoRA autograd function.

    ``gate``/``up``/``down`` are ``LinearLoRA`` modules. Returns the MLP output, or ``None`` if the
    modules are not fusible (plain ``nn.Linear`` without LoRA, DoRA, active dropout, or
    DTensor-sharded) so the caller can fall back to the standard per-linear path.
    """
    if not (_fusible(gate) and _fusible(up) and _fusible(down)):
        return None
    args = (
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
    if debug_name is not None:
        return LoRASwiGLUMLPFunction.apply(*args, debug_name)
    return LoRASwiGLUMLPFunction.apply(*args)


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
        e = F.linear(x2, uW)  # up_out (saved)
        e.addmm_(F.linear(x2, uA) * uS, uB.t())
        relu_e = torch.relu(e)
        f = relu_e * relu_e  # down-projection input (recomputed in backward)
        out = F.linear(f, dW)
        out.addmm_(F.linear(f, dA) * dS, dB.t())
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

        relu_e = torch.relu(e)
        f = relu_e * relu_e  # recompute down-projection input

        # ---- down: out = f@dW.T + dS*(f@dA.T)@dB.T ----
        d_P = dS * (dY @ dB)  # (N, r)
        P = F.linear(f, dA)  # (N, r)
        d_dB = dS * (dY.t() @ P)  # (hidden, r)
        d_dA = d_P.t() @ f  # (r, inter)
        d_f = torch.addmm(dY @ dW, d_P, dA)  # (N, inter)

        # ---- ReLU²: d(relu(e)**2)/de = 2*relu(e). Reuse d_f's buffer for d_e. ----
        d_e = d_f.mul_(2.0 * relu_e)

        # ---- up: e = x@uW.T + uS*(x@uA.T)@uB.T ----
        Q = F.linear(x, uA)  # (N, r)
        d_uB = uS * (d_e.t() @ Q)  # (inter, r)
        d_Q = uS * (d_e @ uB)  # (N, r)
        d_uA = d_Q.t() @ x  # (r, hidden)

        d_x = None
        if needs_x:
            d_x = torch.addmm(d_e @ uW, d_Q, uA).view(ctx.orig_shape)

        # See LoRASwiGLUMLPFunction.backward: return each LoRA grad on its parameter's device so the
        # pipeline-parallel meta graph pass (params on meta, activations on cuda) doesn't reject the
        # cuda grads. No-op in normal single-device training.
        d_uA, d_uB = d_uA.to(uA.device), d_uB.to(uB.device)
        d_dA, d_dB = d_dA.to(dA.device), d_dB.to(dB.device)
        if d_x is not None:
            d_x = d_x.to(x.device)

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


def _projs_are_lora(module, projs) -> bool:
    return all(getattr(getattr(module, proj), "lora_A", None) is not None for proj in projs)


def _swiglu_forward(mod, orig_forward, debug_name: str | None = None):
    def forward(x):
        out = fused_lora_swiglu_mlp(mod.gate_proj, mod.up_proj, mod.down_proj, x, debug_name=debug_name)
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
    non-gated ReLU² MLPs (up/down) via :func:`fused_lora_relu2_mlp`. The installed ``forward`` falls
    back to the module's original per-linear ``forward`` whenever fusion does not apply at runtime —
    notably once the projections become ``DTensor`` under tensor/expert parallelism, or for DoRA /
    active dropout. This keeps the fused memory win on single-GPU and pure-DP while staying correct
    (and identical to the unfused path) under sharding.

    Returns the number of MLP modules whose ``forward`` was swapped. Idempotent.
    """
    count = 0
    for name, mlp in model.named_modules():
        if getattr(mlp, "_lora_mlp_fused", False):
            continue
        if _is_silu_swiglu_mlp(mlp) and _projs_are_lora(mlp, ("gate_proj", "up_proj", "down_proj")):
            mlp.forward = _swiglu_forward(mlp, mlp.forward, debug_name=name if _DEBUG_NAN else None)
        elif _is_relu2_mlp(mlp) and _projs_are_lora(mlp, ("up_proj", "down_proj")):
            mlp.forward = _relu2_forward(mlp, mlp.forward)
        else:
            continue
        mlp._lora_mlp_fused = True
        count += 1
    return count

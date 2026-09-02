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

"""Kimi-K3 SiTU activation and attention-residual compute cores.

Memory- and dispatch-optimized fp32 chains extracted from ``model.py``:
the chunked bf16-saving weighted-SiTU autograd Function, the attn-res
mixing chain, and the opt-in ``torch.compile`` wrapper
(``BackendConfig.compile_situ``) shared by all of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn  # noqa: F401 - kept for parity with model.py type usage

if TYPE_CHECKING:
    from nemo_automodel.components.models.kimi_k3.model import KimiRMSNorm  # noqa: F401

# The eager fp32 path in _weighted_situ upcasts the whole
# [tokens, 2 * intermediate] projection to fp32 and lets autograd save the
# fp32 intermediates for backward; under activation checkpointing every MoE
# layer of a recompute region holds them simultaneously (multi-GiB transients
# per layer at large token counts). _WeightedSiTUFunction computes the
# identical fp32 math in row chunks and saves only the low-precision inputs,
# recomputing fp32 per chunk in backward with analytic gradients. The chain is
# elementwise per row, so the forward is bitwise-identical to the eager path.
_SITU_CHUNK_ROWS = 32768
# Engage the chunked path only for large dispatch tensors, where the memory
# saving matters; below this the backward recompute is a net compute tax.
_SITU_CHUNK_THRESHOLD = 12288


def _situ_fwd_core(
    g: torch.Tensor,
    u0: torch.Tensor,
    w: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    """Compute the fp32 SiTU chain for one chunk of rows.

    Args:
        g: fp32 gate projections of shape [rows, intermediate].
        u0: fp32 up projections of shape [rows, intermediate].
        w: fp32 routing weights broadcastable to [rows, intermediate],
            typically of shape [rows, 1].
        beta: SiTU beta applied to the gate branch.
        linear_beta: Optional bounded-linear beta applied to the up branch.

    Returns:
        fp32 tensor of shape [rows, intermediate]: ``situ(g) * up(u0) * w``.
    """
    tg = torch.tanh(g / beta)
    a = beta * tg * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u0 / linear_beta)
    else:
        u = u0
    return a * u * w


def _situ_bwd_core(
    g: torch.Tensor,
    u0: torch.Tensor,
    w: torch.Tensor,
    go: torch.Tensor,
    beta: float,
    linear_beta: float | None,
    want_drw: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Compute analytic fp32 SiTU gradients for one chunk of rows.

    Args:
        g: fp32 gate projections of shape [rows, intermediate].
        u0: fp32 up projections of shape [rows, intermediate].
        w: fp32 routing weights broadcastable to [rows, intermediate],
            typically of shape [rows, 1].
        go: fp32 upstream gradient of shape [rows, intermediate].
        beta: SiTU beta applied to the gate branch.
        linear_beta: Optional bounded-linear beta applied to the up branch.
        want_drw: Whether the routing-weight gradient reduction is needed.

    Returns:
        Tuple of fp32 tensors ``(d_g, d_u, red)`` where ``d_g`` and ``d_u``
        of shape [rows, intermediate] are the gradients w.r.t. ``g`` and
        ``u0``, and ``red`` of shape [rows, intermediate] is
        ``go * situ(g) * up(u0)`` (the routing-weight gradient before its
        broadcast reduction), or ``None`` when ``want_drw`` is False.
    """
    tg = torch.tanh(g / beta)
    sg = torch.sigmoid(g)
    a = beta * tg * sg
    da_dg = (1.0 - tg * tg) * sg + beta * tg * sg * (1.0 - sg)
    if linear_beta is not None:
        tu = torch.tanh(u0 / linear_beta)
        u = linear_beta * tu
        du_du0 = 1.0 - tu * tu
    else:
        u = u0
        du_du0 = None
    gow = go * w
    d_g = gow * u * da_dg
    d_u = gow * a if du_du0 is None else gow * a * du_du0
    red = go * (a * u) if want_drw else None
    return d_g, d_u, red


_SITU_CORES_COMPILED = False


def _compile_situ_cores() -> None:
    """Wrap the SiTU chunk cores and the attn-res core with ``torch.compile``.

    Runs once per process: the compiled functions replace the module-level
    eager cores, so every layer shares the same compiled kernels and repeated
    model construction does not recompile. Compilation itself is lazy (at
    first call). Compiled numerics are allclose to eager, not
    bitwise-identical.
    """
    global _situ_fwd_core, _situ_bwd_core, _attn_res_core, _SITU_CORES_COMPILED
    if _SITU_CORES_COMPILED:
        return
    _situ_fwd_core = torch.compile(_situ_fwd_core, dynamic=True)
    _situ_bwd_core = torch.compile(_situ_bwd_core, dynamic=True)
    _attn_res_core = torch.compile(_attn_res_core, dynamic=True)
    _SITU_CORES_COMPILED = True


def _situ_rw_is_row_aligned(gate_up: torch.Tensor, routing_weights: torch.Tensor) -> bool:
    """Return True when ``routing_weights`` carries one entry per ``gate_up`` row.

    Args:
        gate_up: Gate+up projections of shape [..., 2 * intermediate].
        routing_weights: Routing weights; row-aligned when its shape is
            [..., k] with the same leading dimensions as ``gate_up``.
    """
    return routing_weights.dim() == gate_up.dim() and routing_weights.shape[:-1] == gate_up.shape[:-1]


class _WeightedSiTUFunction(torch.autograd.Function):
    """Chunked fp32 weighted-SiTU that saves only the low-precision inputs.

    The forward computes the same fp32 chain as the eager `_weighted_situ`
    path in row chunks (bitwise-identical result); the backward recomputes
    the fp32 intermediates per chunk with analytic gradients that match
    autograd's fp32 chain, so autograd never stores full-size fp32 copies of
    the [tokens, 2 * intermediate] projections.
    """

    @staticmethod
    def forward(
        ctx: Any,
        gate_up: torch.Tensor,
        routing_weights: torch.Tensor,
        beta: float,
        linear_beta: float | None,
    ) -> torch.Tensor:
        """Apply SiTU and routing weights chunk by chunk.

        Args:
            ctx: Autograd context; saves ``gate_up`` and ``routing_weights``
                in their original (typically bf16 / fp32) dtypes.
            gate_up: Gate+up projections of shape [..., 2 * intermediate],
                gate in the first half of the last axis, up in the second.
            routing_weights: Routing weights, either row-aligned with shape
                [..., k] matching ``gate_up``'s leading dimensions (typically
                [tokens, 1]) or broadcastable against them.
            beta: SiTU beta applied to the gate branch.
            linear_beta: Optional bounded-linear beta applied to the up branch.

        Returns:
            Tensor of shape ``broadcast(gate_up.shape[:-1] + [intermediate],
            routing_weights.shape)`` in ``gate_up``'s dtype.
        """
        ctx.beta, ctx.linear_beta = beta, linear_beta
        ctx.save_for_backward(gate_up, routing_weights)
        last = gate_up.shape[-1]
        half = last // 2
        gu2 = gate_up.reshape(-1, last)
        row_aligned = _situ_rw_is_row_aligned(gate_up, routing_weights)
        rw2 = routing_weights.reshape(-1, routing_weights.shape[-1]) if row_aligned else routing_weights
        out = torch.empty((gu2.shape[0], half), dtype=gate_up.dtype, device=gate_up.device)
        for s in range(0, gu2.shape[0], _SITU_CHUNK_ROWS):
            e = min(s + _SITU_CHUNK_ROWS, gu2.shape[0])
            g = gu2[s:e, :half].float()
            u = gu2[s:e, half:].float()
            w = rw2[s:e].float() if row_aligned else routing_weights.float()
            out[s:e] = _situ_fwd_core(g, u, w, beta, linear_beta).to(gate_up.dtype)
        # Match the eager broadcast semantics exactly: the result shape is
        # broadcast(a * u, routing_weights) (e.g. 1-D gate_up x [1, 1] weights
        # -> [1, half] in the experts.py zero-token dummy path). Only
        # leading-1 expansions are legal here; a true row fan-out would change
        # the element count.
        out_shape = torch.broadcast_shapes((*gate_up.shape[:-1], half), routing_weights.shape)
        return out.reshape(out_shape)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, None, None]:
        """Recompute fp32 per chunk and return analytic gradients.

        Args:
            ctx: Autograd context holding the saved low-precision inputs.
            grad_out: Upstream gradient with the forward's output shape
                [..., intermediate].

        Returns:
            Tuple ``(d_gate_up, d_routing_weights, None, None)`` where
            ``d_gate_up`` has ``gate_up``'s shape and dtype and
            ``d_routing_weights`` has ``routing_weights``'s shape and dtype
            (or is ``None`` when no gradient is required).
        """
        gate_up, routing_weights = ctx.saved_tensors
        beta, linear_beta = ctx.beta, ctx.linear_beta
        last = gate_up.shape[-1]
        half = last // 2
        gu2 = gate_up.reshape(-1, last)
        go2 = grad_out.reshape(-1, half)
        row_aligned = _situ_rw_is_row_aligned(gate_up, routing_weights)
        rw2 = routing_weights.reshape(-1, routing_weights.shape[-1]) if row_aligned else routing_weights
        d_gu2 = torch.empty_like(gu2)
        want_drw = ctx.needs_input_grad[1]
        d_rw2 = torch.empty_like(rw2) if (want_drw and row_aligned) else None
        d_rw_acc = (
            torch.zeros(routing_weights.shape, dtype=torch.float32, device=routing_weights.device)
            if (want_drw and not row_aligned)
            else None
        )
        for s in range(0, gu2.shape[0], _SITU_CHUNK_ROWS):
            e = min(s + _SITU_CHUNK_ROWS, gu2.shape[0])
            g = gu2[s:e, :half].float()
            u0 = gu2[s:e, half:].float()
            w = rw2[s:e].float() if row_aligned else routing_weights.float()
            go = go2[s:e].float()
            d_g, d_u, red = _situ_bwd_core(g, u0, w, go, beta, linear_beta, want_drw)
            d_gu2[s:e, :half] = d_g.to(gate_up.dtype)
            d_gu2[s:e, half:] = d_u.to(gate_up.dtype)
            if want_drw:
                if row_aligned:
                    d_rw2[s:e] = red.sum_to_size(e - s, rw2.shape[-1]).to(rw2.dtype)
                else:
                    d_rw_acc += red.sum_to_size(routing_weights.shape)
        d_gate_up = d_gu2.reshape(gate_up.shape)
        if not want_drw:
            d_rw = None
        elif row_aligned:
            d_rw = d_rw2.reshape(routing_weights.shape)
        else:
            d_rw = d_rw_acc.to(routing_weights.dtype)
        return d_gate_up, d_rw, None, None


def _weighted_situ(
    gate_up: torch.Tensor,
    routing_weights: torch.Tensor,
    *,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    """Apply SiTU and routing weights to ``[tokens, 2 * intermediate]`` projections."""
    # Route only the memory-relevant case (large 2-D row-aligned dispatch
    # tensors) through the chunked custom Function; every small or irregular
    # shape (zero-token dummy paths pass 1-D tensors, 0-row probs, broadcast
    # [1, 1] weights, ...) keeps the eager implementation verbatim.
    if (
        gate_up.dim() == 2
        and routing_weights.dim() == 2
        and routing_weights.shape[0] == gate_up.shape[0]
        and routing_weights.shape[1] == 1
        and gate_up.shape[0] > _SITU_CHUNK_THRESHOLD
    ):
        return _WeightedSiTUFunction.apply(gate_up, routing_weights, beta, linear_beta)
    input_dtype = gate_up.dtype
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    activated = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (activated * up * routing_weights.float()).to(input_dtype)


def _attn_res_core(
    values: torch.Tensor,
    norm_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    variance_epsilon: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """fp32 attention-residual mixing chain.

    The weighted combine is multiply+sum rather than
    ``torch.matmul(prob[T, 1, B], values[T, B, H])``: cuBLAS dispatches that
    degenerate batched-GEMM shape to non-tensor-core fp32 kernels
    (magma_sgemmEx / gemv2 — 4.3% of busy GPU time on a 256×GB200 Kimi-K3
    profile). multiply+sum runs the same fp32 math on the elementwise/reduce
    path (identical up to fp32 accumulation order, which is below bf16
    resolution for typical shapes) and fuses cleanly under ``torch.compile``
    when ``BackendConfig.compile_situ`` is set.

    Args:
        values: Stacked residuals of shape [tokens, blocks+1, hidden] (block
            residuals concatenated with the current prefix sum along axis 1).
        norm_weight: RMSNorm weight of shape [hidden].
        proj_weight: Squeezed attn-res projection weight of shape [hidden].
        variance_epsilon: RMSNorm epsilon.
        out_dtype: dtype of the returned mixed tensor.

    Returns:
        Mixed residual of shape [tokens, hidden] in ``out_dtype``.
    """
    values_fp32 = values.float()
    variance = values_fp32.pow(2).mean(-1, keepdim=True)
    keys = values_fp32 * torch.rsqrt(variance + variance_epsilon)
    score_weight = norm_weight.float() * proj_weight.float()
    probabilities = (keys * score_weight).sum(-1).softmax(-1)
    mixed = (probabilities.unsqueeze(-1) * values_fp32).sum(dim=1)
    return mixed.to(out_dtype)


def _apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    projection: nn.Linear,
    norm: KimiRMSNorm,
) -> torch.Tensor:
    """Mix ``[tokens, hidden]`` with prior ``[tokens, blocks, hidden]`` residuals."""
    values = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    return _attn_res_core(
        values,
        norm.weight,
        projection.weight.squeeze(0),
        norm.variance_epsilon,
        values.dtype,
    )

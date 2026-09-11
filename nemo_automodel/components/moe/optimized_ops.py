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

"""Memory-optimized MoE elementwise ops extracted from ``experts.py``.

Fused Triton / chunked custom-autograd router-weight fp32 multiply: computes
the identical fp32 math in registers/row chunks and saves only low-precision
inputs, removing the full-size fp32 intermediates that otherwise pin ~7 GiB
blocks per MoE layer under activation checkpointing.
"""

from __future__ import annotations

from typing import Any

import torch

from nemo_automodel.shared.import_utils import safe_import

_HAVE_TRITON, triton = safe_import("triton")
_HAVE_TRITON_LANGUAGE, tl = safe_import("triton.language")
_TRITON_ROUTER_WEIGHT_AVAILABLE = _HAVE_TRITON and _HAVE_TRITON_LANGUAGE

_RW_CHUNK_ROWS = 8192
# Engage the optimized path only for large dispatch tensors, where the memory
# saving matters; below this the backward recompute is a net compute tax.
_RW_CHUNK_THRESHOLD = 12288

if _TRITON_ROUTER_WEIGHT_AVAILABLE:

    @triton.jit
    def _router_weight_fwd_kernel(
        x_ptr,
        probs_ptr,
        out_ptr,
        n_tokens,
        hidden_dim,
        stride_x_row,
        stride_p_row,
        stride_out_row,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused forward kernel: computes out = (x.float() * prob.float()).to(out_dtype).

        Each program processes one token row across its hidden dimension.
        The multiply is evaluated in registers in fp32, avoiding full-size fp32
        HBM allocations.
        """
        row_idx = tl.program_id(0).to(tl.int64)
        if row_idx >= n_tokens:
            return

        # Load scalar probability for this token
        prob = tl.load(probs_ptr + row_idx * stride_p_row).to(tl.float32)

        # Base pointers for row
        x_row = x_ptr + row_idx * stride_x_row
        out_row = out_ptr + row_idx * stride_out_row

        for col_offset in range(0, hidden_dim, BLOCK_SIZE):
            cols = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = cols < hidden_dim

            x_vals = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            out_vals = x_vals * prob
            tl.store(out_row + cols, out_vals, mask=mask)

    @triton.jit
    def _router_weight_bwd_kernel(
        grad_out_ptr,
        probs_ptr,
        x_ptr,
        grad_x_ptr,
        grad_p_ptr,
        n_tokens,
        hidden_dim,
        stride_go_row,
        stride_p_row,
        stride_x_row,
        stride_gx_row,
        stride_gp_row,
        HAS_GRAD_X: tl.constexpr,
        HAS_GRAD_P: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Fused backward kernel: computes grad_x and grad_p with in-register reduction.

        grad_x = (grad_out * prob).to(x.dtype)
        grad_p = sum(grad_out.float() * x.float(), dim=-1)
        """
        row_idx = tl.program_id(0).to(tl.int64)
        if row_idx >= n_tokens:
            return

        go_row = grad_out_ptr + row_idx * stride_go_row
        accum_p = 0.0

        if HAS_GRAD_X:
            prob = tl.load(probs_ptr + row_idx * stride_p_row).to(tl.float32)
            gx_row = grad_x_ptr + row_idx * stride_gx_row

        if HAS_GRAD_P:
            x_row = x_ptr + row_idx * stride_x_row

        for col_offset in range(0, hidden_dim, BLOCK_SIZE):
            cols = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = cols < hidden_dim

            go_vals = tl.load(go_row + cols, mask=mask, other=0.0).to(tl.float32)

            if HAS_GRAD_X:
                gx_vals = go_vals * prob
                tl.store(gx_row + cols, gx_vals, mask=mask)

            if HAS_GRAD_P:
                x_vals = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
                accum_p += tl.sum(go_vals * x_vals)

        if HAS_GRAD_P:
            tl.store(grad_p_ptr + row_idx * stride_gp_row, accum_p)

    class _TritonRouterWeightMulFunction(torch.autograd.Function):
        """Triton-accelerated fp32 router-weight multiply that saves only raw inputs."""

        @staticmethod
        def forward(
            ctx: Any,
            x: torch.Tensor,
            probs: torch.Tensor,
            out_dtype: torch.dtype,
            save_x: bool,
        ) -> torch.Tensor:
            """Multiply expert outputs by routing probabilities in fp32 via Triton.

            Args:
                ctx: Autograd context.
                x: Tensor of shape [tokens, hidden] containing expert outputs.
                probs: Tensor of shape [tokens, 1] containing routing probabilities.
                out_dtype: Target output dtype.
                save_x: Whether backward requires x for probs gradient calculation.

            Returns:
                Tensor of shape [tokens, hidden] and dtype out_dtype.
            """
            x = x.contiguous()
            probs = probs.contiguous()
            n_tokens, hidden_dim = x.shape
            out = torch.empty(x.shape, dtype=out_dtype, device=x.device)

            if save_x:
                ctx.save_for_backward(x, probs)
            else:
                ctx.save_for_backward(probs)
            ctx.save_x = save_x
            ctx.x_dtype = x.dtype
            ctx.probs_dtype = probs.dtype
            ctx.hidden_dim = hidden_dim

            BLOCK_SIZE = min(8192, max(32, triton.next_power_of_2(hidden_dim)))
            num_warps = 4 if hidden_dim <= 2048 else (8 if hidden_dim <= 8192 else 16)

            _router_weight_fwd_kernel[(n_tokens,)](
                x,
                probs,
                out,
                n_tokens,
                hidden_dim,
                x.stride(0),
                probs.stride(0),
                out.stride(0),
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
            return out

        @staticmethod
        def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
            """Compute fp32 gradients for the router-weight multiply via Triton.

            Args:
                ctx: Autograd context holding saved inputs.
                grad_out: Tensor of shape [tokens, hidden] containing upstream gradient.

            Returns:
                Tuple containing:
                    - grad_x: Tensor of shape [tokens, hidden] with x's dtype and device,
                      or None if x does not require gradients.
                    - grad_probs: Tensor of shape [tokens, 1] with probs's dtype and device,
                      or None if probs does not require gradients.
                    - None for out_dtype argument.
                    - None for save_x argument.
            """
            if ctx.save_x:
                x, probs = ctx.saved_tensors
            else:
                (probs,) = ctx.saved_tensors
                x = None

            grad_out = grad_out.contiguous()
            has_grad_x = bool(ctx.needs_input_grad[0])
            has_grad_p = bool(ctx.needs_input_grad[1] and x is not None)

            grad_x = None
            grad_p = None
            n_tokens = grad_out.shape[0]
            hidden_dim = ctx.hidden_dim

            if has_grad_x:
                grad_x = torch.empty(grad_out.shape, dtype=ctx.x_dtype, device=grad_out.device)
            if has_grad_p:
                grad_p = torch.empty(probs.shape, dtype=ctx.probs_dtype, device=grad_out.device)

            if has_grad_x or has_grad_p:
                BLOCK_SIZE = min(8192, max(32, triton.next_power_of_2(hidden_dim)))
                num_warps = 4 if hidden_dim <= 2048 else (8 if hidden_dim <= 8192 else 16)

                dummy_tensor = grad_out
                _router_weight_bwd_kernel[(n_tokens,)](
                    grad_out,
                    probs,
                    x if x is not None else dummy_tensor,
                    grad_x if grad_x is not None else dummy_tensor,
                    grad_p if grad_p is not None else dummy_tensor,
                    n_tokens,
                    hidden_dim,
                    grad_out.stride(0),
                    probs.stride(0),
                    x.stride(0) if x is not None else 0,
                    grad_x.stride(0) if grad_x is not None else 0,
                    grad_p.stride(0) if grad_p is not None else 0,
                    HAS_GRAD_X=has_grad_x,
                    HAS_GRAD_P=has_grad_p,
                    BLOCK_SIZE=BLOCK_SIZE,
                    num_warps=num_warps,
                )

            return grad_x, grad_p, None, None

else:
    _TritonRouterWeightMulFunction = None  # type: ignore[assignment, misc]


class _RouterWeightMulFunction(torch.autograd.Function):
    """Chunked fp32 router-weight multiply that saves only the raw inputs.

    The plain ``(x.float() * probs.float()).to(dtype)`` lets autograd keep
    full-size fp32 [tokens, hidden] intermediates alive for backward (the
    upcast input, the product, and a recompute copy under activation
    checkpointing). This Function computes the same fp32 multiply in row
    chunks and saves only the low-precision inputs. The forward is
    bitwise-identical; the backward matches autograd's fp32 chain
    (``grad_x = (g_f32 * probs_f32).to(x.dtype)``,
    ``grad_probs = (g_f32 * x_f32).sum(-1, keepdim=True)`` cast to
    ``probs.dtype``).
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        probs: torch.Tensor,
        out_dtype: torch.dtype,
        save_x: bool,
    ) -> torch.Tensor:
        """Multiply expert outputs by routing probabilities in fp32.

        Args:
            ctx: Autograd context.
            x: Tensor of shape [tokens, hidden] containing expert outputs.
            probs: Tensor of shape [tokens, 1] containing routing probabilities.
            out_dtype: Output dtype (the dispatcher's expected activation
                dtype, or fp32 for the scatter-add reduction path).
            save_x: Whether backward needs x. x is only consumed by
                the probs gradient; when probs carries no grad (e.g.
                FakeBalancedGate emits constant weights), saving it would pin
                a full-size [tokens, hidden] tensor per MoE layer across the
                activation-checkpointing backward window for nothing. Callers
                pass probs.requires_grad.

        Returns:
            Tensor of shape [tokens, hidden] and dtype out_dtype.
        """
        if save_x:
            ctx.save_for_backward(x, probs)
        else:
            ctx.save_for_backward(probs)
        ctx.save_x = save_x
        ctx.x_dtype = x.dtype
        out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
        for s in range(0, x.shape[0], _RW_CHUNK_ROWS):
            e = min(s + _RW_CHUNK_ROWS, x.shape[0])
            out[s:e] = (x[s:e].float() * probs[s:e].float()).to(out_dtype)
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None, None, None]:
        """Compute chunked fp32 gradients for the router-weight multiply.

        Args:
            ctx: Autograd context holding the saved inputs.
            grad_out: Tensor of shape [tokens, hidden] containing upstream gradient.

        Returns:
            Tuple containing:
                - grad_x: Tensor of shape [tokens, hidden] with x's dtype and device,
                  or None when x does not require gradients.
                - grad_probs: Tensor of shape [tokens, 1] with probs's dtype and device,
                  or None when probs does not require gradients.
                - None for out_dtype argument.
                - None for save_x argument.
        """
        if ctx.save_x:
            x, probs = ctx.saved_tensors
        else:
            (probs,) = ctx.saved_tensors
            x = None
        grad_x = None
        grad_p = None
        if ctx.needs_input_grad[0]:
            grad_x = torch.empty(grad_out.shape, dtype=ctx.x_dtype, device=grad_out.device)
        if ctx.needs_input_grad[1] and x is not None:
            grad_p = torch.empty_like(probs)
        for s in range(0, grad_out.shape[0], _RW_CHUNK_ROWS):
            e = min(s + _RW_CHUNK_ROWS, grad_out.shape[0])
            g = grad_out[s:e].float()
            if grad_x is not None:
                grad_x[s:e] = (g * probs[s:e].float()).to(grad_x.dtype)
            if grad_p is not None:
                grad_p[s:e] = (g * x[s:e].float()).sum(dim=-1, keepdim=True).to(probs.dtype)
        return grad_x, grad_p, None, None


def _apply_router_weight_fp32(
    output2: torch.Tensor,
    permuted_probs: torch.Tensor,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Apply routing probabilities to expert outputs with fp32 arithmetic.

    Large 2-D row-aligned inputs go through the fused Triton Function (or chunked
    fallback Function) so autograd does not retain full-size fp32 intermediates;
    every other shape keeps the plain eager multiply (bitwise-identical result).

    Args:
        output2: Tensor of shape [tokens, hidden] containing expert down-projection
            outputs on CUDA or CPU.
        permuted_probs: Tensor of shape [tokens, 1] containing routing probabilities
            broadcastable against output2.
        compute_dtype: Output tensor element dtype.

    Returns:
        Tensor of shape [tokens, hidden] and dtype compute_dtype containing
        (output2 * permuted_probs) evaluated in fp32.
    """
    if (
        output2.dim() == 2
        and permuted_probs.dim() == 2
        and permuted_probs.shape[0] == output2.shape[0]
        and permuted_probs.shape[1] == 1
        and output2.shape[0] > _RW_CHUNK_THRESHOLD
    ):
        if _TRITON_ROUTER_WEIGHT_AVAILABLE and output2.is_cuda and _TritonRouterWeightMulFunction is not None:
            return _TritonRouterWeightMulFunction.apply(
                output2, permuted_probs, compute_dtype, permuted_probs.requires_grad
            )
        return _RouterWeightMulFunction.apply(output2, permuted_probs, compute_dtype, permuted_probs.requires_grad)
    return (output2.float() * permuted_probs.float()).to(compute_dtype)

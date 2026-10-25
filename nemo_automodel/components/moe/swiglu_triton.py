# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Triton kernels for the clamped SwiGLU expert activation (DeepSeek-V4 style, ``swiglu_limit > 0``).

The routed experts apply ``silu(clamp(gate, max=L)) * clamp(up, -L, L) * probs`` to the grouped-GEMM output
``x = [gate | up]`` (bf16, ``[rows, 2 * inter]``) with fp32 routing probabilities ``probs`` (``[rows, 1]``). The
compiled PyTorch version (``experts._swiglu_clamped_deepep_compiled``) is one inductor kernel forward, but its
autograd-derived backward concatenates ``d_gate`` and ``d_up`` and ran at about a quarter of HBM bandwidth on
GB200 (0.7 ms for a 1.1 GB pass at the 16-node shape). These kernels do one pass each:

* forward: a 2-D tile per program, fp32 math, result rounded to ``x.dtype``;
* backward: one program per block of rows looping over the columns, writing ``d_gate`` / ``d_up`` straight into
  the two halves of ``grad_x`` and accumulating the per-row probability gradient in registers.

Clamp gradients follow ``torch.clamp`` (inclusive range masks on the input values). The dispatcher in
``experts.swiglu_clamped_deepep`` uses these kernels on contiguous CUDA inputs when Triton is importable and
falls back to the compiled version otherwise.
"""

from __future__ import annotations

import torch

from nemo_automodel.shared.import_utils import safe_import

_HAVE_TRITON, triton = safe_import("triton")
_HAVE_TRITON_LANGUAGE, tl = safe_import("triton.language")
HAVE_TRITON: bool = bool(_HAVE_TRITON and _HAVE_TRITON_LANGUAGE)

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_FWD_BLOCK_M = 8
_FWD_BLOCK_N = 256
_BWD_BLOCK_M = 4
_BWD_BLOCK_N = 512


if HAVE_TRITON:

    @triton.jit
    def _swiglu_clamped_fwd_kernel(
        x_ptr,
        probs_ptr,
        y_ptr,
        rows,
        inter,
        stride_x,
        stride_y,
        limit,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):  # pragma: no cover - Triton JIT executes on the GPU, outside Python tracing.
        # Skewed expert routing can make rows * stride_x exceed 2**31.
        pid_m = tl.program_id(0).to(tl.int64)
        pid_n = tl.program_id(1)
        row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_ok = row < rows
        mask = row_ok[:, None] & (col[None, :] < inter)
        x_off = row[:, None] * stride_x + col[None, :]
        # `limit` arrives as a Python float: fp32 under eager Triton, fp64 when inductor compiles the kernel
        # (torch.compile of the enclosing module) — cast once so the math and the stores stay fp32
        limit = limit.to(tl.float32)
        gate = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(x_ptr + x_off + inter, mask=mask, other=0.0).to(tl.float32)
        probs = tl.load(probs_ptr + row, mask=row_ok, other=0.0).to(tl.float32)
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
        sig = tl.sigmoid(gate)
        y = gate * sig * up * probs[:, None]
        tl.store(y_ptr + row[:, None] * stride_y + col[None, :], y.to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _swiglu_clamped_bwd_kernel(
        x_ptr,
        probs_ptr,
        grad_y_ptr,
        grad_x_ptr,
        grad_probs_ptr,
        rows,
        inter,
        stride_x,
        stride_gy,
        stride_gx,
        limit,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):  # pragma: no cover - Triton JIT executes on the GPU, outside Python tracing.
        # Promote before multiplying by the row stride, including grad_x offsets.
        pid = tl.program_id(0).to(tl.int64)
        row = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        row_ok = row < rows
        limit = limit.to(tl.float32)  # see the forward kernel: fp64 under inductor otherwise, and the loop-carried
        probs = tl.load(probs_ptr + row, mask=row_ok, other=0.0).to(tl.float32)  # `acc` would change type
        acc = tl.zeros([BLOCK_M], dtype=tl.float32)
        for start in range(0, inter, BLOCK_N):
            col = start + tl.arange(0, BLOCK_N)
            mask = row_ok[:, None] & (col[None, :] < inter)
            x_off = row[:, None] * stride_x + col[None, :]
            gate_raw = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
            up_raw = tl.load(x_ptr + x_off + inter, mask=mask, other=0.0).to(tl.float32)
            grad = tl.load(grad_y_ptr + row[:, None] * stride_gy + col[None, :], mask=mask, other=0.0).to(tl.float32)
            gate = tl.minimum(gate_raw, limit)
            up = tl.minimum(tl.maximum(up_raw, -limit), limit)
            sig = tl.sigmoid(gate)
            silu = gate * sig
            acc += tl.sum(grad * silu * up, axis=1).to(tl.float32)
            weighted = grad * probs[:, None]
            d_gate = weighted * up * (sig * (1.0 + gate * (1.0 - sig)))
            d_up = weighted * silu
            d_gate = tl.where(gate_raw <= limit, d_gate, 0.0)
            d_up = tl.where((up_raw >= -limit) & (up_raw <= limit), d_up, 0.0)
            gx_off = row[:, None] * stride_gx + col[None, :]
            tl.store(grad_x_ptr + gx_off, d_gate.to(grad_x_ptr.dtype.element_ty), mask=mask)
            tl.store(grad_x_ptr + gx_off + inter, d_up.to(grad_x_ptr.dtype.element_ty), mask=mask)
        tl.store(grad_probs_ptr + row, acc, mask=row_ok)


def uses_triton_kernel(x: torch.Tensor, probs: torch.Tensor) -> bool:
    """Whether the Triton kernels can take these inputs (contiguous 2-D CUDA ``x``, one probability per row)."""
    return (
        HAVE_TRITON
        and x.is_cuda
        and x.dim() == 2
        and x.is_contiguous()
        and x.dtype in _SUPPORTED_DTYPES
        and x.shape[1] % 2 == 0
        and probs.is_cuda
        and probs.numel() == x.shape[0]
        and probs.dtype == torch.float32
        and probs.is_contiguous()
    )


def _forward(x: torch.Tensor, probs: torch.Tensor, limit: float) -> torch.Tensor:
    rows, width = x.shape
    inter = width // 2
    y = torch.empty((rows, inter), dtype=x.dtype, device=x.device)
    if rows == 0 or inter == 0:
        return y
    grid = (triton.cdiv(rows, _FWD_BLOCK_M), triton.cdiv(inter, _FWD_BLOCK_N))
    _swiglu_clamped_fwd_kernel[grid](
        x,
        probs,
        y,
        rows,
        inter,
        x.stride(0),
        y.stride(0),
        float(limit),
        BLOCK_M=_FWD_BLOCK_M,
        BLOCK_N=_FWD_BLOCK_N,
        num_warps=4,
    )
    return y


def _backward(
    x: torch.Tensor, probs: torch.Tensor, grad_y: torch.Tensor, limit: float
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width = x.shape
    inter = width // 2
    grad_x = torch.empty_like(x)
    grad_probs = torch.empty((rows,), dtype=torch.float32, device=x.device)
    if rows == 0 or inter == 0:
        grad_probs.zero_()
        return grad_x, grad_probs.view_as(probs)
    grad_y = grad_y.contiguous()
    grid = (triton.cdiv(rows, _BWD_BLOCK_M),)
    _swiglu_clamped_bwd_kernel[grid](
        x,
        probs,
        grad_y,
        grad_x,
        grad_probs,
        rows,
        inter,
        x.stride(0),
        grad_y.stride(0),
        grad_x.stride(0),
        float(limit),
        BLOCK_M=_BWD_BLOCK_M,
        BLOCK_N=_BWD_BLOCK_N,
        num_warps=8,
    )
    return grad_x, grad_probs.view_as(probs)


class SwigluClampedTriton(torch.autograd.Function):
    """``silu(clamp(gate)) * clamp(up) * probs`` with the Triton forward and backward kernels."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, probs: torch.Tensor, limit: float) -> torch.Tensor:
        ctx.save_for_backward(x, probs)
        ctx.limit = limit
        return _forward(x, probs, limit)

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x, probs = ctx.saved_tensors
        grad_x, grad_probs = _backward(x, probs, grad_y, ctx.limit)
        return grad_x, (grad_probs if ctx.needs_input_grad[1] else None), None


def swiglu_clamped_triton(x: torch.Tensor, probs: torch.Tensor, limit: float) -> torch.Tensor:
    """Apply the clamped SwiGLU with the Triton kernels; callers check :func:`uses_triton_kernel` first."""
    return SwigluClampedTriton.apply(x, probs, limit)

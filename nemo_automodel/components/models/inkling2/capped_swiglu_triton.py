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
"""Triton kernels for the capped SwiGLU expert activation of Inkling 2.

The routed experts apply ``gate_softcap * tanh(gate / gate_softcap) * sigmoid(gate) * up_softcap *
tanh(up / up_softcap) * weight`` to the grouped-GEMM output ``gate_up`` (``[rows, 2 * inter]``, gate and up
interleaved) with one routing weight per row. Written as PyTorch operations this reads and writes the rows about
ten times forward and more backward: 48 ms and 83 ms for one million rows of 2304 columns on GB300. These
kernels do one pass each, with FP32 arithmetic, in 4.5 ms and 10 ms:

* forward: a 2-D tile per program, result rounded to ``gate_up.dtype``;
* backward: one program per block of rows looping over the columns, writing the gate and up gradients interleaved
  and summing each row's routing-weight gradient.

``layers.capped_swiglu`` uses them for contiguous 2-D CUDA inputs when Triton is importable and keeps the PyTorch
operations otherwise. See ``nemo_automodel.components.moe.swiglu_triton`` for the clamped SwiGLU counterpart.
"""

from __future__ import annotations

import torch
from torch.autograd.function import FunctionCtx

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
    def _tanh(z):  # pragma: no cover - Triton JIT executes on the GPU, outside Python tracing.
        # Triton has no tanh. 2 * sigmoid(2z) - 1 cancels for small z, so those use the series instead.
        series = z * (1.0 - z * z * (1.0 / 3.0 - z * z * (2.0 / 15.0)))
        return tl.where(tl.abs(z) < 0.05, series, 2.0 * tl.sigmoid(2.0 * z) - 1.0)

    @triton.jit
    def _capped_swiglu_fwd_kernel(
        x_ptr,
        weights_ptr,
        y_ptr,
        rows,
        inter,
        stride_x,
        stride_y,
        gate_softcap,
        up_softcap,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):  # pragma: no cover - Triton JIT executes on the GPU, outside Python tracing.
        # rows * stride_x can exceed 2**31.
        pid_m = tl.program_id(0).to(tl.int64)
        pid_n = tl.program_id(1)
        row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        col = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_ok = row < rows
        mask = row_ok[:, None] & (col[None, :] < inter)
        x_off = row[:, None] * stride_x + 2 * col[None, :]
        # The caps arrive as Python floats: FP32 under eager Triton, FP64 when inductor compiles the kernel.
        gate_softcap = gate_softcap.to(tl.float32)
        up_softcap = up_softcap.to(tl.float32)
        gate = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(x_ptr + x_off + 1, mask=mask, other=0.0).to(tl.float32)
        weights = tl.load(weights_ptr + row, mask=row_ok, other=0.0).to(tl.float32)
        capped_gate = gate_softcap * _tanh(gate / gate_softcap) * tl.sigmoid(gate)
        capped_up = up_softcap * _tanh(up / up_softcap)
        y = capped_gate * capped_up * weights[:, None]
        tl.store(y_ptr + row[:, None] * stride_y + col[None, :], y.to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _capped_swiglu_bwd_kernel(
        x_ptr,
        weights_ptr,
        grad_y_ptr,
        grad_x_ptr,
        grad_weights_ptr,
        rows,
        inter,
        stride_x,
        stride_gy,
        stride_gx,
        gate_softcap,
        up_softcap,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):  # pragma: no cover - Triton JIT executes on the GPU, outside Python tracing.
        pid = tl.program_id(0).to(tl.int64)
        row = pid * BLOCK_M + tl.arange(0, BLOCK_M)
        row_ok = row < rows
        gate_softcap = gate_softcap.to(tl.float32)
        up_softcap = up_softcap.to(tl.float32)
        weights = tl.load(weights_ptr + row, mask=row_ok, other=0.0).to(tl.float32)
        acc = tl.zeros([BLOCK_M], dtype=tl.float32)
        for start in range(0, inter, BLOCK_N):
            col = start + tl.arange(0, BLOCK_N)
            mask = row_ok[:, None] & (col[None, :] < inter)
            x_off = row[:, None] * stride_x + 2 * col[None, :]
            gate = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
            up = tl.load(x_ptr + x_off + 1, mask=mask, other=0.0).to(tl.float32)
            grad = tl.load(grad_y_ptr + row[:, None] * stride_gy + col[None, :], mask=mask, other=0.0).to(tl.float32)
            sig = tl.sigmoid(gate)
            tanh_gate = _tanh(gate / gate_softcap)
            tanh_up = _tanh(up / up_softcap)
            capped_gate = gate_softcap * tanh_gate * sig
            capped_up = up_softcap * tanh_up
            acc += tl.sum(grad * capped_gate * capped_up, axis=1).to(tl.float32)
            weighted = grad * weights[:, None]
            d_gate = weighted * capped_up * ((1.0 - tanh_gate * tanh_gate) * sig + capped_gate * (1.0 - sig))
            d_up = weighted * capped_gate * (1.0 - tanh_up * tanh_up)
            gx_off = row[:, None] * stride_gx + 2 * col[None, :]
            tl.store(grad_x_ptr + gx_off, d_gate.to(grad_x_ptr.dtype.element_ty), mask=mask)
            tl.store(grad_x_ptr + gx_off + 1, d_up.to(grad_x_ptr.dtype.element_ty), mask=mask)
        tl.store(grad_weights_ptr + row, acc.to(grad_weights_ptr.dtype.element_ty), mask=row_ok)


def uses_triton_kernel(gate_up: torch.Tensor, weights: torch.Tensor) -> bool:
    """Whether the Triton kernels can take these inputs (contiguous 2-D CUDA ``gate_up``, one weight per row)."""
    return (
        HAVE_TRITON
        and gate_up.is_cuda
        and gate_up.dim() == 2
        and gate_up.is_contiguous()
        and gate_up.dtype in _SUPPORTED_DTYPES
        and gate_up.shape[1] % 2 == 0
        and weights.is_cuda
        and weights.numel() == gate_up.shape[0]
        and weights.dtype in _SUPPORTED_DTYPES
        and weights.is_contiguous()
    )


def _forward(gate_up: torch.Tensor, weights: torch.Tensor, gate_softcap: float, up_softcap: float) -> torch.Tensor:
    rows, width = gate_up.shape
    inter = width // 2
    y = torch.empty((rows, inter), dtype=gate_up.dtype, device=gate_up.device)
    if rows == 0 or inter == 0:
        return y
    grid = (triton.cdiv(rows, _FWD_BLOCK_M), triton.cdiv(inter, _FWD_BLOCK_N))
    _capped_swiglu_fwd_kernel[grid](
        gate_up,
        weights,
        y,
        rows,
        inter,
        gate_up.stride(0),
        y.stride(0),
        float(gate_softcap),
        float(up_softcap),
        BLOCK_M=_FWD_BLOCK_M,
        BLOCK_N=_FWD_BLOCK_N,
        num_warps=4,
    )
    return y


def _backward(
    gate_up: torch.Tensor, weights: torch.Tensor, grad_y: torch.Tensor, gate_softcap: float, up_softcap: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the gradients of ``gate_up`` and ``weights``.

    Args:
        gate_up: Contiguous tensor of shape [rows, 2 * inter], gate and up interleaved.
        weights: Routing weights of shape [rows] or [rows, 1].
        grad_y: Upstream gradient of shape [rows, inter].
        gate_softcap: Positive gate tanh cap.
        up_softcap: Positive up tanh cap.

    Returns:
        Tuple of the gradient of shape [rows, 2 * inter] in gate_up's dtype and the gradient in the shape and
        dtype of weights.
    """
    rows, width = gate_up.shape
    inter = width // 2
    grad_x = torch.empty_like(gate_up)
    grad_weights = torch.empty_like(weights)
    if rows == 0 or inter == 0:
        return grad_x, grad_weights.zero_()
    grad_y = grad_y.contiguous()
    grid = (triton.cdiv(rows, _BWD_BLOCK_M),)
    _capped_swiglu_bwd_kernel[grid](
        gate_up,
        weights,
        grad_y,
        grad_x,
        grad_weights,
        rows,
        inter,
        gate_up.stride(0),
        grad_y.stride(0),
        grad_x.stride(0),
        float(gate_softcap),
        float(up_softcap),
        BLOCK_M=_BWD_BLOCK_M,
        BLOCK_N=_BWD_BLOCK_N,
        num_warps=8,
    )
    return grad_x, grad_weights


class CappedSwigluTriton(torch.autograd.Function):
    """The capped SwiGLU with the Triton forward and backward kernels."""

    @staticmethod
    def forward(
        ctx: FunctionCtx, gate_up: torch.Tensor, weights: torch.Tensor, gate_softcap: float, up_softcap: float
    ) -> torch.Tensor:
        ctx.save_for_backward(gate_up, weights)
        ctx.gate_softcap = gate_softcap
        ctx.up_softcap = up_softcap
        return _forward(gate_up, weights, gate_softcap, up_softcap)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, None, None]:
        gate_up, weights = ctx.saved_tensors
        grad_x, grad_weights = _backward(gate_up, weights, grad_y, ctx.gate_softcap, ctx.up_softcap)
        return grad_x, (grad_weights if ctx.needs_input_grad[1] else None), None, None


def capped_swiglu_triton(
    gate_up: torch.Tensor, weights: torch.Tensor, gate_softcap: float, up_softcap: float
) -> torch.Tensor:
    """Apply the capped SwiGLU with the Triton kernels; callers check :func:`uses_triton_kernel` first."""
    return CappedSwigluTriton.apply(gate_up, weights, gate_softcap, up_softcap)

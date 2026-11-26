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

"""Triton kernel for the gradient of the capped expert activation.

The compiled chain differentiates the activation in two kernels, and each of them reads the
inputs and recomputes the forward terms. This kernel reads the inputs once and writes the
interleaved gate/up gradient and the weight gradient in the same pass.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from nemo_automodel.shared.import_utils import null_decorator

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only where Triton is absent
    HAVE_TRITON = False

if not HAVE_TRITON:  # pragma: no cover
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    libdevice = MagicMock()

# Rows per program, gate/up pairs per step and warps, measured on GB200.
_LAUNCH = (8, 64, 2)


@triton.jit
def _gradient_kernel(
    GATE_UP,
    WEIGHTS,
    GRAD,
    OUT,
    WEIGHT_GRAD,
    tokens,
    intermediate,
    gate_cap,
    up_cap,
    NEED_WEIGHTS: tl.constexpr,
    R: tl.constexpr,
    W: tl.constexpr,
):  # pragma: no cover - Triton JIT executes on the GPU, outside Python tracing.
    """Differentiate R tokens of the activation, W gate/up pairs at a time, in FP32."""
    rows = tl.program_id(0) * R + tl.arange(0, R)
    row_ok = rows < tokens
    weight = tl.load(WEIGHTS + rows, mask=row_ok, other=0.0).to(tl.float32)
    base = rows.to(tl.int64) * intermediate
    total = tl.zeros([R], dtype=tl.float32)
    for start in range(0, intermediate, W):
        cols = start + tl.arange(0, W)
        mask = row_ok[:, None] & (cols < intermediate)[None, :]
        pair = 2 * base[:, None] + 2 * cols[None, :]
        gate = tl.load(GATE_UP + pair, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(GATE_UP + pair + 1, mask=mask, other=0.0).to(tl.float32)
        grad = tl.load(GRAD + base[:, None] + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        # The forward terms and the chain rule, in the order of the compiled chain.
        gate_tanh = libdevice.tanh(gate / gate_cap)
        gate_capped = gate_cap * gate_tanh
        gate_sigmoid = tl.sigmoid(gate)
        gate_value = gate_capped * gate_sigmoid
        up_tanh = libdevice.tanh(up / up_cap)
        up_value = up_cap * up_tanh
        grad_product = grad * weight[:, None]
        grad_gate_value = grad_product * up_value
        grad_up_value = grad_product * gate_value
        grad_gate = (grad_gate_value * gate_sigmoid * gate_cap) * (1 - gate_tanh * gate_tanh) / gate_cap
        grad_gate = grad_gate + (grad_gate_value * gate_capped) * (gate_sigmoid * (1 - gate_sigmoid))
        grad_up = (grad_up_value * up_cap) * (1 - up_tanh * up_tanh) / up_cap
        tl.store(OUT + pair, grad_gate.to(OUT.dtype.element_ty), mask=mask)
        tl.store(OUT + pair + 1, grad_up.to(OUT.dtype.element_ty), mask=mask)
        if NEED_WEIGHTS:
            total += tl.sum(tl.where(mask, grad * (gate_value * up_value), 0.0), axis=1)
    if NEED_WEIGHTS:
        tl.store(WEIGHT_GRAD + rows, total.to(WEIGHT_GRAD.dtype.element_ty), mask=row_ok)


def capped_swiglu_gradient(
    gate_up: torch.Tensor,
    weights: torch.Tensor,
    grad_output: torch.Tensor,
    gate_softcap: float,
    up_softcap: float,
    weights_need_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Differentiate the capped activation in FP32 from its inputs alone, on a CUDA device.

    Args:
        gate_up: Tensor of shape [tokens, 2 * intermediate], interleaved gate/up.
        weights: Tensor of shape [tokens, 1], containing selected expert weights.
        grad_output: Gradient of the activation output, shape [tokens, intermediate].
        gate_softcap: Positive gate tanh cap.
        up_softcap: Positive up tanh cap.
        weights_need_grad: Whether to return the gradient of the weights.

    Returns:
        The gradient of gate_up in its dtype and layout, and the gradient of the weights
        [tokens, 1] in their dtype (None when not needed).
    """
    tokens, doubled = gate_up.shape
    grad_gate_up = torch.empty_like(gate_up, memory_format=torch.contiguous_format)
    grad_weights = torch.empty_like(weights, memory_format=torch.contiguous_format) if weights_need_grad else None
    if gate_up.numel() == 0:
        return grad_gate_up, grad_weights
    block_r, block_w, warps = _LAUNCH
    _gradient_kernel[(triton.cdiv(tokens, block_r),)](
        gate_up.contiguous(),
        weights.contiguous(),
        grad_output.contiguous(),
        grad_gate_up,
        grad_gate_up if grad_weights is None else grad_weights,
        tokens,
        doubled // 2,
        gate_softcap,
        up_softcap,
        NEED_WEIGHTS=weights_need_grad,
        R=block_r,
        W=block_w,
        num_warps=warps,
    )
    return grad_gate_up, grad_weights

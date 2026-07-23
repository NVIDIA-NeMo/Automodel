# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Fused soft-target cross-entropy Triton kernel for EAGLE-3 training."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from nemo_automodel.components.loss.triton.te_cross_entropy import MAX_FUSED_SIZE, element_mul_kernel
from nemo_automodel.shared.import_utils import MISSING_TRITON_MSG, null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    triton.autotune = null_decorator
    triton.heuristics = null_decorator
    tl = MagicMock()


@triton.jit
def _soft_ce_forward_kernel(
    X_ptr,
    X_stride,
    P_ptr,
    P_stride,
    Loss_ptr,
    LSE_ptr,
    Sump_ptr,
    Mask_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """CE = -dot(p, x) + (max + log(sum_exp)) * sum(p), single-pass online softmax.

    ``sum(p)`` is saved per row so the backward can form the exact gradient
    ``sum(p) * softmax(x) - p`` for arbitrary (not necessarily normalized) targets.
    """
    row_id = tl.program_id(0).to(tl.int64)

    if tl.load(Mask_ptr + row_id) == 0.0:
        tl.store(Loss_ptr + row_id, 0.0)
        tl.store(LSE_ptr + row_id, 0.0)
        tl.store(Sump_ptr + row_id, 0.0)
        return

    X_row_ptr = X_ptr + row_id * X_stride
    P_row_ptr = P_ptr + row_id * P_stride

    m = float("-inf")
    d = 0.0
    dot_px = 0.0
    sum_p = 0.0

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        valid = offsets < n_cols

        x_block = tl.load(X_row_ptr + offsets, mask=valid, other=float("-inf")).to(tl.float32)
        p_block = tl.load(P_row_ptr + offsets, mask=valid, other=0.0).to(tl.float32)

        # Online softmax: rescale running sum when max updates
        block_max = tl.max(x_block)
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x_block - m_new))
        m = m_new

        # tl.where avoids 0 * (-inf) = NaN at padding positions
        dot_px += tl.sum(tl.where(valid, p_block * x_block, 0.0))
        sum_p += tl.sum(p_block)

    lse = m + tl.log(d)
    tl.store(Loss_ptr + row_id, -dot_px + lse * sum_p)
    tl.store(LSE_ptr + row_id, lse)
    tl.store(Sump_ptr + row_id, sum_p)


@triton.jit
def _soft_ce_backward_kernel(
    X_ptr,
    X_stride,
    P_ptr,
    P_stride,
    LSE_ptr,
    Sump_ptr,
    Mask_ptr,
    InvCount_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """grad = (sum_p * softmax(x) - p) / valid_count, single-pass using saved lse / sum_p.

    ``sum_p`` (the per-row target mass, == 1 for a normalized distribution) is the
    derivative of the forward's ``lse * sum(p)`` term and keeps the gradient exact
    for arbitrary targets.
    """
    row_id = tl.program_id(0).to(tl.int64)

    if tl.load(Mask_ptr + row_id) == 0.0:
        X_row_ptr = X_ptr + row_id * X_stride
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            tl.store(X_row_ptr + offsets, 0.0, mask=offsets < n_cols)
        return

    X_row_ptr = X_ptr + row_id * X_stride
    P_row_ptr = P_ptr + row_id * P_stride
    lse = tl.load(LSE_ptr + row_id).to(tl.float32)
    sum_p = tl.load(Sump_ptr + row_id).to(tl.float32)
    inv_count = tl.load(InvCount_ptr).to(tl.float32)

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        valid = offsets < n_cols
        x_block = tl.load(X_row_ptr + offsets, mask=valid, other=float("-inf"))
        out_dtype = x_block.dtype
        x_block = x_block.to(tl.float32)
        p_block = tl.load(P_row_ptr + offsets, mask=valid, other=0.0).to(tl.float32)

        grad_block = (sum_p * tl.exp(x_block - lse) - p_block) * inv_count
        tl.store(X_row_ptr + offsets, grad_block.to(out_dtype), mask=valid)


def _select_num_warps(V: int) -> int:
    if V <= 1024:
        return 4
    elif V <= 4096:
        return 8
    elif V <= 16384:
        return 16
    return 32


def _launch_config(V: int) -> tuple[int, int]:
    """Return the (BLOCK_SIZE, num_warps) launch config for a vocab size of ``V``."""
    return min(MAX_FUSED_SIZE, triton.next_power_of_2(V)), _select_num_warps(V)


class SoftCrossEntropyFunction(torch.autograd.Function):
    """Autograd wrapper around the fused Triton soft cross-entropy kernels."""

    @staticmethod
    def forward(ctx, logits, target_probs, valid_mask, valid_count):
        B, S, V = logits.shape
        n_rows = B * S

        logits_2d = logits.view(n_rows, V)
        if logits_2d.stride(-1) != 1:
            logits_2d = logits_2d.contiguous()
        target_probs_2d = target_probs.view(n_rows, V)
        if target_probs_2d.stride(-1) != 1:
            target_probs_2d = target_probs_2d.contiguous()

        mask_1d = valid_mask.view(n_rows)
        loss_1d = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        lse_1d = torch.empty(n_rows, dtype=torch.float32, device=logits.device)
        sump_1d = torch.empty(n_rows, dtype=torch.float32, device=logits.device)

        BLOCK_SIZE, num_warps = _launch_config(V)

        _soft_ce_forward_kernel[(n_rows,)](
            logits_2d,
            logits_2d.stride(0),
            target_probs_2d,
            target_probs_2d.stride(0),
            loss_1d,
            lse_1d,
            sump_1d,
            mask_1d,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        scalar_loss = loss_1d.sum() / valid_count
        inv_valid_count = torch.reciprocal(valid_count)

        ctx.save_for_backward(logits_2d.detach(), target_probs_2d, mask_1d, lse_1d, sump_1d, inv_valid_count)
        ctx.shape = (B, S, V)
        return scalar_loss

    @staticmethod
    def backward(ctx, grad_output):
        logits_2d, target_probs_2d, mask_1d, lse_1d, sump_1d, inv_valid_count = ctx.saved_tensors
        B, S, V = ctx.shape
        n_rows = B * S

        BLOCK_SIZE, num_warps = _launch_config(V)

        _soft_ce_backward_kernel[(n_rows,)](
            logits_2d,
            logits_2d.stride(0),
            target_probs_2d,
            target_probs_2d.stride(0),
            lse_1d,
            sump_1d,
            mask_1d,
            inv_valid_count,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        # Scale the in-place gradient by the incoming grad_output. Launched
        # unconditionally: the elementwise multiply is cheap and reading
        # grad_output on-device avoids a per-step ``.item()`` host sync.
        element_mul_kernel[(n_rows,)](
            logits_2d,
            logits_2d.stride(0),
            grad_output,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

        return logits_2d.view(B, S, V), None, None, None


def fused_soft_cross_entropy(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    position_mask: torch.Tensor,
) -> torch.Tensor:
    """Fused Triton soft cross-entropy, drop-in replacement for the PyTorch path."""
    if not HAVE_TRITON:
        raise ImportError(MISSING_TRITON_MSG)
    valid_mask = position_mask.squeeze(-1).float()
    valid_count = valid_mask.sum().clamp_min(1.0)
    return SoftCrossEntropyFunction.apply(logits, target_probs, valid_mask, valid_count)

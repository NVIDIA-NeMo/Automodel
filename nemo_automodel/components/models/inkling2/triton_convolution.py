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


"""Triton kernels for the static short convolution.

Every tap reads the sequence at another offset. A kernel that walks the output in one
dimension reads the input once per tap; these kernels work on tiles of several positions
by several channels, so the taps of a tile read the same rows.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from nemo_automodel.shared.import_utils import null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only where Triton is absent
    HAVE_TRITON = False

if not HAVE_TRITON:  # pragma: no cover
    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()

# Tile (positions, channels) and warps of both kernels, measured on GB200.
_LAUNCH = (16, 128, 4)


@triton.jit
def _lagged_tile(X, VALID, PREFIX, rows, cols, row_ok, col_ok, b, T, C, lag, first_row, HALO: tl.constexpr):
    """Return the masked input ``lag`` positions before ``rows`` in FP32, [positions, channels].

    Positions before the sequence read the exchanged history ``PREFIX`` [batch, HALO, channels].
    """
    source = rows - lag
    own = row_ok & (source >= 0)
    offset = (b * T + source).to(tl.int64) * C
    value = tl.load(X + offset[:, None] + cols[None, :], mask=own[:, None] & col_ok[None, :], other=0.0)
    keep = tl.load(VALID + b * T + source, mask=own, other=0)
    value = value.to(tl.float32) * keep.to(tl.float32)[:, None]
    if HALO > 0:
        # Only the first tile of a sequence has positions before the sequence.
        if first_row < lag:
            early = row_ok & (source < 0)
            history = (b * HALO + HALO + source).to(tl.int64) * C
            before = tl.load(
                PREFIX + history[:, None] + cols[None, :], mask=early[:, None] & col_ok[None, :], other=0.0
            )
            value = tl.where(early[:, None], before, value)
    return value


@triton.jit
def _forward_kernel(
    X,
    VALID,
    POS,
    TAPS,
    PREFIX,
    OUT,
    T,
    C,
    scale,
    WIDTH: tl.constexpr,
    R: tl.constexpr,
    W: tl.constexpr,
):
    b = tl.program_id(2)
    first_row = tl.program_id(0) * R
    rows = first_row + tl.arange(0, R)
    cols = tl.program_id(1) * W + tl.arange(0, W)
    row_ok = rows < T
    col_ok = cols < C
    position = tl.load(POS + b * T + rows, mask=row_ok, other=0)
    wide = _lagged_tile(X, VALID, PREFIX, rows, cols, row_ok, col_ok, b, T, C, 0, first_row, WIDTH - 1)
    # Add the taps oldest first in FP32, the order of a grouped convolution. A tap that
    # would read before its document's first token contributes zero.
    filtered = tl.zeros([R, W], tl.float32)
    for slot in tl.static_range(WIDTH):
        lag = WIDTH - 1 - slot
        tap = tl.load(TAPS + cols * WIDTH + slot, mask=col_ok, other=0.0)
        if lag == 0:
            term = wide * tap[None, :]
        else:
            term = (
                _lagged_tile(X, VALID, PREFIX, rows, cols, row_ok, col_ok, b, T, C, lag, first_row, WIDTH - 1)
                * tap[None, :]
            )
            term = tl.where((position < lag)[:, None], 0.0, term)
        if slot == 0:
            filtered = term
        else:
            filtered = filtered + term
    keep = tl.load(VALID + b * T + rows, mask=row_ok, other=0)
    out = ((wide + filtered) * scale).to(OUT.dtype.element_ty) * keep.to(OUT.dtype.element_ty)[:, None]
    offset = (b * T + rows).to(tl.int64) * C
    tl.store(OUT + offset[:, None] + cols[None, :], out, mask=row_ok[:, None] & col_ok[None, :])


@triton.jit
def _backward_kernel(
    G,
    X,
    VALID,
    POS,
    TAPS,
    PREFIX,
    DX,
    PART,
    T,
    C,
    TILES,
    scale,
    WIDTH: tl.constexpr,
    R: tl.constexpr,
    W: tl.constexpr,
):
    b = tl.program_id(2)
    tile = tl.program_id(0)
    first_row = tile * R
    rows = first_row + tl.arange(0, R)
    cols = tl.program_id(1) * W + tl.arange(0, W)
    row_ok = rows < T
    col_ok = cols < C
    cell_ok = row_ok[:, None] & col_ok[None, :]
    offset = (b * T + rows).to(tl.int64) * C
    position = tl.load(POS + b * T + rows, mask=row_ok, other=0)
    keep = tl.load(VALID + b * T + rows, mask=row_ok, other=0)
    # Gradient of the FP32 sum that the output rounds.
    flow = tl.load(G + offset[:, None] + cols[None, :], mask=cell_ok, other=0.0)
    flow = (flow * keep.to(flow.dtype)[:, None]).to(tl.float32) * scale
    part = ((b * TILES + tile) * WIDTH).to(tl.int64) * C
    tap = tl.load(TAPS + cols * WIDTH + (WIDTH - 1), mask=col_ok, other=0.0)
    grad_wide = flow + flow * tap[None, :]
    wide = _lagged_tile(X, VALID, PREFIX, rows, cols, row_ok, col_ok, b, T, C, 0, first_row, WIDTH - 1)
    tl.store(PART + part + (WIDTH - 1) * C + cols, tl.sum(flow * wide, axis=0), mask=col_ok)
    for slot in tl.static_range(WIDTH - 1):
        lag = WIDTH - 1 - slot
        tap = tl.load(TAPS + cols * WIDTH + slot, mask=col_ok, other=0.0)
        # The output ``lag`` positions later read these positions through tap ``slot``.
        later = rows + lag
        later_ok = row_ok & (later < T)
        later_offset = (b * T + later).to(tl.int64) * C
        later_flow = tl.load(
            G + later_offset[:, None] + cols[None, :], mask=later_ok[:, None] & col_ok[None, :], other=0.0
        )
        later_keep = tl.load(VALID + b * T + later, mask=later_ok, other=0)
        later_flow = (later_flow * later_keep.to(later_flow.dtype)[:, None]).to(tl.float32) * scale
        later_position = tl.load(POS + b * T + later, mask=later_ok, other=0)
        reads = later_ok & (later_position >= lag)
        grad_wide = grad_wide + tl.where(reads[:, None], later_flow * tap[None, :], 0.0)
        # The tap's own gradient: this tile's outputs times the input ``lag`` positions earlier.
        product = flow * _lagged_tile(X, VALID, PREFIX, rows, cols, row_ok, col_ok, b, T, C, lag, first_row, WIDTH - 1)
        product = tl.where((position < lag)[:, None], 0.0, product)
        tl.store(PART + part + slot * C + cols, tl.sum(product, axis=0), mask=col_ok)
    grad_x = grad_wide.to(DX.dtype.element_ty) * keep.to(DX.dtype.element_ty)[:, None]
    tl.store(DX + offset[:, None] + cols[None, :], grad_x, mask=cell_ok)


class _StaticConvolution(torch.autograd.Function):
    """Static short convolution with an explicit derivative; the saved tensors are the inputs."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        valid: torch.Tensor,
        positions: torch.Tensor,
        weight: torch.Tensor,
        prefix: torch.Tensor,
        width: int,
    ) -> torch.Tensor:
        x = x.contiguous()
        valid = valid.contiguous()
        positions = positions.to(torch.int64).contiguous()
        prefix = prefix.contiguous()
        taps = weight.float()[:, 0].contiguous()
        ctx.save_for_backward(x, valid, positions, weight, prefix)
        ctx.width = width
        batch, sequence, channels = x.shape
        block_r, block_w, warps = _LAUNCH
        out = torch.empty_like(x)
        _forward_kernel[(triton.cdiv(sequence, block_r), triton.cdiv(channels, block_w), batch)](
            x,
            valid,
            positions,
            taps,
            prefix,
            out,
            sequence,
            channels,
            (width + 1) ** -0.5,
            WIDTH=width,
            R=block_r,
            W=block_w,
            num_warps=warps,
        )
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, torch.Tensor, torch.Tensor | None, None]:
        x, valid, positions, weight, prefix = ctx.saved_tensors
        width = ctx.width
        halo = width - 1
        batch, sequence, channels = x.shape
        block_r, block_w, warps = _LAUNCH
        tiles = triton.cdiv(sequence, block_r)
        scale = (width + 1) ** -0.5
        taps = weight.float()[:, 0].contiguous()
        grad = grad.contiguous()
        grad_x = torch.empty_like(x)
        # Every tile writes its share of the tap gradients; the shares are summed afterwards.
        partial = torch.empty(batch * tiles, width, channels, dtype=torch.float32, device=x.device)
        _backward_kernel[(tiles, triton.cdiv(channels, block_w), batch)](
            grad,
            x,
            valid,
            positions,
            taps,
            prefix,
            grad_x,
            partial,
            sequence,
            channels,
            tiles,
            scale,
            WIDTH=width,
            R=block_r,
            W=block_w,
            num_warps=warps,
        )
        grad_weight = partial.sum(0).t().unsqueeze(1).to(weight.dtype)
        grad_prefix = None
        if ctx.needs_input_grad[4]:
            # History row ``row`` feeds position ``row - slot`` through tap ``slot``.
            head = min(halo, sequence)
            flow = (grad[:, :head] * valid[:, :head, None]).float() * scale
            rows = []
            for row in range(halo):
                total = torch.zeros(batch, channels, dtype=torch.float32, device=x.device)
                for slot in range(row + 1):
                    position = row - slot
                    if position < head:
                        term = flow[:, position] * taps[:, slot]
                        total = total + term.masked_fill(positions[:, position, None] < halo - slot, 0)
                rows.append(total)
            grad_prefix = torch.stack(rows, dim=1) if rows else torch.zeros_like(prefix)
        return grad_x, None, None, grad_weight, grad_prefix, None


def static_convolution(
    x: torch.Tensor,
    valid: torch.Tensor,
    positions: torch.Tensor,
    weight: torch.Tensor,
    prefix: torch.Tensor,
    width: int,
) -> torch.Tensor:
    """Apply static causal taps that stop at each document's first token, on a CUDA device.

    Args:
        x: Tensor of shape [batch, sequence, channels].
        valid: Boolean tensor of shape [batch, sequence], true for real tokens.
        positions: Logical document positions [batch, sequence].
        weight: Taps [channels, 1, width] in conv1d order: oldest first, current position last.
        prefix: FP32 history [batch, width - 1, channels] before the first position.
        width: Number of taps.

    Returns:
        Tensor of shape [batch, sequence, channels], in x's dtype; padding is zero. The inputs
        are not mutated.
    """
    return _StaticConvolution.apply(x, valid, positions, weight, prefix, width)

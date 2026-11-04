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

"""Fused local Muown row operations; Newton-Schulz and collectives stay in Muown.

Only contiguous FP32 local tensors with complete input-feature axes use these
kernels. Matrix batches and output-neuron shards are independent. Public DTensor
ownership and autograd version counters remain the optimizer's responsibility.
"""

from __future__ import annotations

import torch
from torch import Tensor

from nemo_automodel.shared.import_utils import null_decorator, safe_import

_HAS_TRITON, triton = safe_import("triton")
_HAS_LANGUAGE, tl = safe_import("triton.language")
HAVE_TRITON = _HAS_TRITON and _HAS_LANGUAGE


@(triton.jit if HAVE_TRITON else null_decorator)
def _prepare_kernel(
    W,
    G,
    N,
    Grad,
    Mom,
    D,
    U,
    GG,
    R: tl.constexpr,
    INPUTS: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    MU: tl.constexpr,
    EPS: tl.constexpr,
    NESTEROV: tl.constexpr,
    BR: tl.constexpr,
    BI: tl.constexpr,
):
    batch = tl.program_id(1)
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, BI)
    ridx = batch * R + row
    if TRANSPOSED:
        offsets = batch * R * INPUTS + col[None, :] * R + row[:, None]
    else:
        offsets = batch * R * INPUTS + row[:, None] * INPUTS + col[None, :]
    mask = (row[:, None] < R) & (col[None, :] < INPUTS)
    weight = tl.load(W + offsets, mask, 0)
    grad = tl.load(Grad + offsets, mask, 0)
    magnitude = tl.load(G + ridx, row < R, 1)
    magnitude = tl.where(magnitude < 0, -tl.maximum(tl.abs(magnitude), EPS), tl.maximum(tl.abs(magnitude), EPS))
    norm = tl.load(N + ridx, row < R, 1)
    unit = weight / magnitude[:, None]
    direction = unit * norm[:, None]
    magnitude_grad = tl.sum(grad * unit, axis=1)
    direction_grad = (magnitude / norm)[:, None] * (grad - unit * magnitude_grad[:, None])
    momentum = tl.load(Mom + offsets, mask, 0)
    momentum = momentum * MU + direction_grad
    if NESTEROV:
        update = tl.fma(momentum, MU, direction_grad)
    else:
        update = momentum
    tl.store(Mom + offsets, momentum, mask)
    tl.store(D + offsets, direction, mask)
    tl.store(U + offsets, update, mask)
    tl.store(GG + ridx, magnitude_grad, row < R)


@(triton.jit if HAVE_TRITON else null_decorator)
def _finish_kernel(
    W,
    G,
    N,
    M,
    V,
    D,
    U,
    GG,
    R: tl.constexpr,
    INPUTS: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    LR,
    DIRECTION_LR,
    DECAY,
    B1: tl.constexpr,
    B2: tl.constexpr,
    BC1,
    BC2,
    EPS: tl.constexpr,
    WD: tl.constexpr,
    BR: tl.constexpr,
    BI: tl.constexpr,
):
    batch = tl.program_id(1)
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, BI)
    ridx = batch * R + row
    if TRANSPOSED:
        offsets = batch * R * INPUTS + col[None, :] * R + row[:, None]
    else:
        offsets = batch * R * INPUTS + row[:, None] * INPUTS + col[None, :]
    mask = (row[:, None] < R) & (col[None, :] < INPUTS)
    direction = tl.load(D + offsets, mask, 0)
    update = tl.load(U + offsets, mask, 0).to(tl.float32)
    direction = tl.fma(update, DIRECTION_LR, direction)
    magnitude_grad = tl.load(GG + ridx, row < R, 0)
    magnitude = tl.load(G + ridx, row < R, 0)
    m = tl.load(M + ridx, row < R, 0) * B1
    m = tl.fma(magnitude_grad, 1 - B1, m)
    v = tl.load(V + ridx, row < R, 0) * B2
    v = v + (1 - B2) * magnitude_grad * magnitude_grad
    denominator = tl.sqrt(v / BC2) + EPS
    magnitude = tl.fma((m / BC1) / denominator, -LR, magnitude)
    norm = tl.sqrt(tl.sum(direction * direction, axis=1))
    weight = magnitude[:, None] * (direction / tl.maximum(norm, EPS)[:, None])
    if WD != 0.0:
        old_weight = tl.load(W + offsets, mask, 0)
        weight = tl.fma(old_weight, DECAY, weight)
        magnitude = tl.sqrt(tl.sum(weight * weight, axis=1))
        magnitude = tl.maximum(magnitude, EPS)
    tl.store(W + offsets, weight, mask)
    tl.store(G + ridx, magnitude, row < R)
    tl.store(N + ridx, tl.maximum(norm, EPS), row < R)
    tl.store(M + ridx, m, row < R)
    tl.store(V + ridx, v, row < R)


@(triton.jit if HAVE_TRITON else null_decorator)
def _magnitude_grad_kernel(
    W, G, Grad, GG, R: tl.constexpr, INPUTS: tl.constexpr, EPS: tl.constexpr, BR: tl.constexpr, BI: tl.constexpr
):
    batch = tl.program_id(1)
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    col = tl.arange(0, BI)
    offsets = batch * R * INPUTS + col[None, :] * R + row[:, None]
    mask = (row[:, None] < R) & (col[None, :] < INPUTS)
    weight = tl.load(W + offsets, mask, 0)
    grad = tl.load(Grad + offsets, mask, 0)
    g = tl.load(G + batch * R + row, row < R, 1)
    g = tl.where(g < 0, -tl.maximum(tl.abs(g), EPS), tl.maximum(tl.abs(g), EPS))
    gg = tl.sum(grad * (weight / g[:, None]), axis=1)
    tl.store(GG + batch * R + row, gg, row < R)


@(triton.jit if HAVE_TRITON else null_decorator)
def _direction_kernel(
    W,
    G,
    N,
    Grad,
    Mom,
    D,
    U,
    GG,
    SIZE: tl.constexpr,
    R: tl.constexpr,
    INPUTS: tl.constexpr,
    MU: tl.constexpr,
    EPS: tl.constexpr,
    NESTEROV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < SIZE
    row = (offset // (R * INPUTS)) * R + offset % R
    weight = tl.load(W + offset, mask, 0)
    grad = tl.load(Grad + offset, mask, 0)
    g = tl.load(G + row, mask, 1)
    g = tl.where(g < 0, -tl.maximum(tl.abs(g), EPS), tl.maximum(tl.abs(g), EPS))
    norm = tl.load(N + row, mask, 1)
    gg = tl.load(GG + row, mask, 0)
    unit = weight / g
    direction = unit * norm
    dg = (g / norm) * (grad - unit * gg)
    mom = tl.load(Mom + offset, mask, 0) * MU + dg
    if NESTEROV:
        update = tl.fma(mom, MU, dg)
    else:
        update = mom
    tl.store(Mom + offset, mom, mask)
    tl.store(D + offset, direction, mask)
    tl.store(U + offset, update, mask)


def _supported(tensors: tuple[Tensor, ...], transposed: bool) -> bool:
    """Check local dtype/strides and the bounded row-reduction footprint."""
    weight = tensors[0]
    inputs = weight.shape[-2 if transposed else -1]
    # Bound register pressure. Wider rows retain the existing Torch path.
    return (
        HAVE_TRITON
        and 0 < weight.numel() < 2**31
        and weight.numel() // (weight.shape[-2] * weight.shape[-1]) <= 65535
        and inputs <= (8192 if transposed else 16384)
        and all(t.is_cuda and t.dtype == torch.float32 and t.is_contiguous() for t in tensors)
    )


def _prepare(
    weight: Tensor,
    gradient: Tensor,
    momentum: Tensor,
    magnitude: Tensor,
    norm: Tensor,
    *,
    transposed: bool,
    mu: float,
    epsilon: float,
    nesterov: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Prepare complete local rows and mutate momentum.

    Args:
        weight: Contiguous FP32 CUDA [..., output, input], or [..., input,
            output] when transposed. Leading dimensions batch independent
            matrices; the input axis must be complete on this rank.
        gradient: Same shape/dtype/device/layout as weight, preserved.
        momentum: Same shape/dtype/device/layout as weight, updated in place.
        magnitude: Contiguous FP32 tensor with the input axis reduced to 1.
        norm: Same shape/layout as magnitude; direction norms.
        transposed: Whether output neurons occupy the final physical axis.
        mu: Momentum coefficient.
        epsilon: Signed magnitude division guard.
        nesterov: Whether to use Nesterov momentum.

    Returns:
        Independent direction (FP32), update (BF16), and magnitude-gradient
        (FP32) buffers. The first two match weight; the last matches magnitude.
    """
    rows, inputs = weight.shape[-2:]
    if transposed:
        rows, inputs = inputs, rows
    batches = weight.numel() // (rows * inputs)
    direction = torch.empty_like(weight)
    update = torch.empty_like(weight, dtype=torch.bfloat16)
    magnitude_grad = torch.empty_like(magnitude)
    if transposed:
        # Splitting the reduction from the pointwise update keeps the latter's
        # accesses contiguous without retaining a large tile of live values.
        row_tile = 32 if inputs <= 1024 else (16 if inputs <= 2048 else 8)
        _magnitude_grad_kernel[(triton.cdiv(rows, row_tile), batches)](
            weight,
            magnitude,
            gradient,
            magnitude_grad,
            rows,
            inputs,
            epsilon,
            row_tile,
            triton.next_power_of_2(inputs),
            num_warps=8,
            enable_fp_fusion=False,
        )
        _direction_kernel[(triton.cdiv(weight.numel(), 2048),)](
            weight,
            magnitude,
            norm,
            gradient,
            momentum,
            direction,
            update,
            magnitude_grad,
            weight.numel(),
            rows,
            inputs,
            mu,
            epsilon,
            nesterov,
            2048,
            num_warps=4,
            enable_fp_fusion=False,
        )
    else:
        _prepare_kernel[(rows, batches)](
            weight,
            magnitude,
            norm,
            gradient,
            momentum,
            direction,
            update,
            magnitude_grad,
            rows,
            inputs,
            False,
            mu,
            epsilon,
            nesterov,
            1,
            triton.next_power_of_2(inputs),
            num_warps=8 if inputs > 4096 else 4,
            enable_fp_fusion=False,
        )
    return direction, update, magnitude_grad


def _finish(
    weight: Tensor,
    magnitude: Tensor,
    norm: Tensor,
    first_moment: Tensor,
    second_moment: Tensor,
    direction: Tensor,
    update: Tensor,
    magnitude_grad: Tensor,
    *,
    transposed: bool,
    lr: float,
    scale: float,
    beta1: float,
    beta2: float,
    step: int,
    epsilon: float,
    weight_decay: float,
) -> None:
    """Apply direction/magnitude updates, normalize and decay local weights.

    Tensor inputs follow _prepare's contiguous local-row contract. Update is
    BF16 with weight's shape; direction is FP32 with that shape. All other
    tensors are FP32 with magnitude's row shape. Weight, magnitude, norm and
    both moments are mutated in place; direction/update/gradients are preserved.
    Scale must be computed from the GLOBAL matrix shape, including on FSDP
    output shards. LR and bias corrections are runtime scalars.
    """
    rows, inputs = weight.shape[-2:]
    if transposed:
        rows, inputs = inputs, rows
    batches = weight.numel() // (rows * inputs)
    # Bound each transposed tile as the input width grows. A fixed 16-row
    # tile spills heavily at MiMo's 4096/6144 input widths.
    row_tile = min(16, 32768 // triton.next_power_of_2(inputs)) if transposed else 1
    warps = 16 if transposed and inputs > 4096 else (8 if transposed or inputs > 4096 else 4)
    _finish_kernel[(triton.cdiv(rows, row_tile), batches)](
        weight,
        magnitude,
        norm,
        first_moment,
        second_moment,
        direction,
        update,
        magnitude_grad,
        rows,
        inputs,
        transposed,
        lr,
        -lr * scale,
        -lr * weight_decay,
        beta1,
        beta2,
        1 - beta1**step,
        1 - beta2**step,
        epsilon,
        weight_decay,
        row_tile,
        triton.next_power_of_2(inputs),
        num_warps=warps,
        enable_fp_fusion=False,
    )

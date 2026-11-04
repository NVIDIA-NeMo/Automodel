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

"""Hand-written Triton kernels for the Kimi-K3 SiTU activation.

The SiTU chain (``beta * tanh(g / beta) * sigmoid(g) * linear_beta *
tanh(u / linear_beta) * w``) is a pure elementwise function of the gate/up
halves of one ``[rows, 2 * intermediate]`` projection plus one routing weight
per row. ``torch.compile`` (``BackendConfig.compile_situ``) fuses it into a
single kernel, but the generated 1-D pointwise kernels pay for two things on
every element: 64-bit ``div``/``mod`` to recover the row index for the
row-broadcast routing weight (the pointwise index space is flattened), and,
in the backward, both ``cat`` branches evaluated under ``tl.where`` (twelve
masked loads and four ``tanh`` per output element). On GB200 that leaves the
backward ~10x above its bandwidth bound (17 ms per 262k x 6144 bf16 call on a
2-node K3 profile).

These kernels tile the problem in 2-D (rows x columns) instead: the row index
is a per-tile 64-bit multiply, the column index stays 32-bit, each transcendental
is evaluated once, ``d_gate`` and ``d_up`` are stored straight into the two
halves of the output, and the routing-weight gradient (``sum_cols(go * situ(g)
* up(u))``) is reduced inside the backward kernel while the tile is in
registers. The fp32 math and its operation order are identical to
``situ._situ_fwd_core`` / ``situ._situ_bwd_core``; only the fp32 accumulation
order of the routing-weight reduction differs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from nemo_automodel.shared.import_utils import null_decorator

try:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra import libdevice as _libdevice
    except ImportError:  # pragma: no cover - older Triton layout
        from triton.language.extra.cuda import libdevice as _libdevice

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - exercised only where Triton is absent
    HAVE_TRITON = False

if not HAVE_TRITON:  # pragma: no cover
    triton = MagicMock()
    triton.jit = null_decorator
    triton.autotune = null_decorator
    tl = MagicMock()
    _libdevice = MagicMock()


def _tile_configs() -> list:
    """Autotune candidates: ~16 fp32 values per thread keeps the backward below the register cap."""
    if not HAVE_TRITON:  # pragma: no cover
        return []
    return [
        triton.Config({"BLOCK_R": 8, "BLOCK_C": 256}, num_warps=4),
        triton.Config({"BLOCK_R": 4, "BLOCK_C": 512}, num_warps=4),
        triton.Config({"BLOCK_R": 16, "BLOCK_C": 128}, num_warps=4),
        triton.Config({"BLOCK_R": 8, "BLOCK_C": 512}, num_warps=8),
        triton.Config({"BLOCK_R": 16, "BLOCK_C": 256}, num_warps=8),
    ]


@triton.jit
def _tanh_fast(x):
    """MUFU.TANH (``tanh.approx.f32``, sm_75+): one SFU op, |rel err| <= 2^-11 — below bf16 resolution."""
    return tl.inline_asm_elementwise("tanh.approx.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _rcp_fast(x):
    """``rcp.approx.ftz.f32`` on an fp32 block of any shape (one SFU op; results below 2^-126 flush to zero)."""
    return tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", "=r,r", [x], dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _situ_tanh(x, FAST: tl.constexpr):
    """tanh of an fp32 block of any shape: the SFU approximation when ``FAST``, libdevice otherwise."""
    if FAST:
        return _tanh_fast(x)
    else:
        return _libdevice.tanh(x)


@triton.jit
def _situ_sigmoid(x, FAST: tl.constexpr):
    """sigmoid of an fp32 block of any shape: SFU exp2 + approximate reciprocal when ``FAST``, ``tl.sigmoid`` otherwise."""
    if FAST:
        # 1 / (1 + 2^(-x * log2 e)): the exact path already lowers its exp to the SFU (ex2.approx), so what the fast
        # variant removes is tl.sigmoid's IEEE divide sequence. The kernels sit at 43-52 % of HBM bandwidth because
        # they are bound by the transcendental/ALU work, not by memory.
        return _rcp_fast(1.0 + tl.exp2(x * -1.4426950408889634))
    else:
        return tl.sigmoid(x)


@triton.autotune(configs=_tile_configs(), key=["half"])
@triton.jit
def _situ_fwd_kernel(
    gu_ptr,
    rw_ptr,
    out_ptr,
    n_rows,
    half,
    stride_gu,
    stride_out,
    beta,
    linear_beta,
    HAS_RW: tl.constexpr,
    HAS_LINEAR: tl.constexpr,
    FAST_MATH: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """out[r, c] = beta * tanh(g / beta) * sigmoid(g) * up(u) * w[r] for one (rows x cols) tile."""
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    rmask = rows < n_rows
    cmask = cols < half
    mask = rmask[:, None] & cmask[None, :]
    rows64 = rows.to(tl.int64)
    gu_off = rows64[:, None] * stride_gu + cols[None, :]
    g = tl.load(gu_ptr + gu_off, mask=mask, other=0.0).to(tl.float32)
    u0 = tl.load(gu_ptr + gu_off + half, mask=mask, other=0.0).to(tl.float32)
    tg = _situ_tanh(g / beta, FAST_MATH)
    a = beta * tg * _situ_sigmoid(g, FAST_MATH)
    if HAS_LINEAR:
        u = linear_beta * _situ_tanh(u0 / linear_beta, FAST_MATH)
    else:
        u = u0
    out = a * u
    if HAS_RW:
        w = tl.load(rw_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
        out = out * w[:, None]
    out_off = rows64[:, None] * stride_out + cols[None, :]
    tl.store(out_ptr + out_off, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=_tile_configs(), key=["half"])
@triton.jit
def _situ_bwd_kernel(
    gu_ptr,
    rw_ptr,
    go_ptr,
    dgu_ptr,
    drw_ptr,
    n_rows,
    half,
    stride_gu,
    stride_go,
    stride_dgu,
    beta,
    linear_beta,
    HAS_RW: tl.constexpr,
    HAS_LINEAR: tl.constexpr,
    WANT_DRW: tl.constexpr,
    FAST_MATH: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Analytic SiTU gradients for BLOCK_R full rows; the routing-weight gradient is reduced in-kernel."""
    pid_r = tl.program_id(0)
    rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    rmask = rows < n_rows
    rows64 = rows.to(tl.int64)
    if HAS_RW:
        w = tl.load(rw_ptr + rows, mask=rmask, other=0.0).to(tl.float32)
    acc = tl.zeros([BLOCK_R], dtype=tl.float32)
    for c0 in range(0, half, BLOCK_C):
        cols = c0 + tl.arange(0, BLOCK_C)
        cmask = cols < half
        mask = rmask[:, None] & cmask[None, :]
        gu_off = rows64[:, None] * stride_gu + cols[None, :]
        g = tl.load(gu_ptr + gu_off, mask=mask, other=0.0).to(tl.float32)
        u0 = tl.load(gu_ptr + gu_off + half, mask=mask, other=0.0).to(tl.float32)
        go = tl.load(go_ptr + rows64[:, None] * stride_go + cols[None, :], mask=mask, other=0.0).to(tl.float32)
        tg = _situ_tanh(g / beta, FAST_MATH)
        sg = _situ_sigmoid(g, FAST_MATH)
        a = beta * tg * sg
        da_dg = (1.0 - tg * tg) * sg + beta * tg * sg * (1.0 - sg)
        if HAS_RW:
            gow = go * w[:, None]
        else:
            gow = go
        if HAS_LINEAR:
            tu = _situ_tanh(u0 / linear_beta, FAST_MATH)
            u = linear_beta * tu
            d_u = gow * a * (1.0 - tu * tu)
        else:
            u = u0
            d_u = gow * a
        d_g = gow * u * da_dg
        dgu_off = rows64[:, None] * stride_dgu + cols[None, :]
        tl.store(dgu_ptr + dgu_off, d_g.to(dgu_ptr.dtype.element_ty), mask=mask)
        tl.store(dgu_ptr + dgu_off + half, d_u.to(dgu_ptr.dtype.element_ty), mask=mask)
        if WANT_DRW:
            red = go * (a * u)
            acc += tl.sum(tl.where(mask, red, 0.0), axis=1)
    if WANT_DRW:
        tl.store(drw_ptr + rows, acc.to(drw_ptr.dtype.element_ty), mask=rmask)


def _check_2d_rows(name: str, t: torch.Tensor) -> None:
    if t.dim() != 2 or t.stride(1) != 1:
        raise ValueError(f"{name} must be a 2-D tensor with unit stride along the last axis, got {tuple(t.shape)}")


def situ_fwd_triton(
    gate_up2: torch.Tensor,
    routing_weights2: torch.Tensor | None,
    beta: float,
    linear_beta: float | None,
    *,
    fast_math: bool = False,
) -> torch.Tensor:
    """Weighted (or dense) SiTU forward on ``[rows, 2 * intermediate]`` projections.

    Args:
        gate_up2: Gate+up projections of shape [rows, 2 * intermediate] on a CUDA
            device, unit stride along the last axis; gate in the first half.
        routing_weights2: Optional routing weights of shape [rows, 1] (any float
            dtype, contiguous), or None for the dense activation.
        beta: SiTU beta applied to the gate branch.
        linear_beta: Optional bounded-linear beta applied to the up branch.
        fast_math: SFU ``tanh.approx`` / ``exp2`` / ``rcp.approx`` instead of libdevice tanh and an IEEE divide
            (``KimiK3TextConfig.situ_backend = "triton_fast_math"``); at most one bf16 ulp from the exact chain.

    Returns:
        Tensor of shape [rows, intermediate] in ``gate_up2``'s dtype.
    """
    _check_2d_rows("gate_up2", gate_up2)
    n_rows, last = gate_up2.shape
    half = last // 2
    out = torch.empty((n_rows, half), dtype=gate_up2.dtype, device=gate_up2.device)
    if n_rows == 0 or half == 0:
        return out
    has_rw = routing_weights2 is not None
    if has_rw:
        if routing_weights2.shape != (n_rows, 1) or not routing_weights2.is_contiguous():
            raise ValueError(f"routing_weights2 must be a contiguous [{n_rows}, 1] tensor")
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_R"]), triton.cdiv(half, meta["BLOCK_C"]))  # noqa: E731
    _situ_fwd_kernel[grid](
        gate_up2,
        routing_weights2 if has_rw else gate_up2,
        out,
        n_rows,
        half,
        gate_up2.stride(0),
        out.stride(0),
        float(beta),
        float(linear_beta) if linear_beta is not None else 0.0,
        HAS_RW=has_rw,
        HAS_LINEAR=linear_beta is not None,
        FAST_MATH=bool(fast_math),
    )
    return out


def situ_bwd_triton(
    gate_up2: torch.Tensor,
    routing_weights2: torch.Tensor | None,
    grad_out2: torch.Tensor,
    beta: float,
    linear_beta: float | None,
    want_drw: bool,
    *,
    fast_math: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Weighted (or dense) SiTU backward.

    Args:
        gate_up2: Saved gate+up projections of shape [rows, 2 * intermediate].
        routing_weights2: Saved routing weights of shape [rows, 1], or None (dense).
        grad_out2: Upstream gradient of shape [rows, intermediate], unit last stride.
        beta: SiTU beta applied to the gate branch.
        linear_beta: Optional bounded-linear beta applied to the up branch.
        want_drw: Whether to reduce the routing-weight gradient (requires weights).
        fast_math: Same SFU variants as in :func:`situ_fwd_triton`; the autograd Functions in ``situ.py`` pass
            the setting their forward used.

    Returns:
        ``(d_gate_up2, d_routing_weights2)`` in the inputs' dtypes; the second
        entry is None when ``want_drw`` is False or there are no weights.
    """
    _check_2d_rows("gate_up2", gate_up2)
    _check_2d_rows("grad_out2", grad_out2)
    n_rows, last = gate_up2.shape
    half = last // 2
    if grad_out2.shape != (n_rows, half):
        raise ValueError(f"grad_out2 must have shape [{n_rows}, {half}], got {tuple(grad_out2.shape)}")
    has_rw = routing_weights2 is not None
    want_drw = bool(want_drw and has_rw)
    d_gu = torch.empty_like(gate_up2)
    d_rw = torch.empty_like(routing_weights2) if want_drw else None
    if n_rows == 0 or half == 0:
        if d_rw is not None:
            d_rw.zero_()
        return d_gu, d_rw
    if has_rw and (routing_weights2.shape != (n_rows, 1) or not routing_weights2.is_contiguous()):
        raise ValueError(f"routing_weights2 must be a contiguous [{n_rows}, 1] tensor")
    grid = lambda meta: (triton.cdiv(n_rows, meta["BLOCK_R"]),)  # noqa: E731
    _situ_bwd_kernel[grid](
        gate_up2,
        routing_weights2 if has_rw else gate_up2,
        grad_out2,
        d_gu,
        d_rw if want_drw else d_gu,
        n_rows,
        half,
        gate_up2.stride(0),
        grad_out2.stride(0),
        d_gu.stride(0),
        float(beta),
        float(linear_beta) if linear_beta is not None else 0.0,
        HAS_RW=has_rw,
        HAS_LINEAR=linear_beta is not None,
        WANT_DRW=want_drw,
        FAST_MATH=bool(fast_math),
    )
    return d_gu, d_rw

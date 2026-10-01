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

"""Hand-written Triton kernels for the Kimi-K3 attention-residual mix.

The mix (``situ._attn_res_core``) scores every entry of ``[tokens, k+1, hidden]``
(the ``k`` block residuals plus the current prefix sum) with an RMS-normalised
dot product against ``norm_weight * proj_weight``, softmaxes the ``k+1`` scores
per token and returns the probability-weighted sum of the entries. The eager
chain materialises ``torch.cat((block_residual, prefix_sum))`` and an fp32
upcast of it, and the ``torch.compile`` version (``BackendConfig.compile_situ``)
lowers the row-broadcast multiply and the entry-axis reduction into 1-D
pointwise / reduction kernels that run far below HBM bandwidth (on a 256-GPU
K3 profile the two named kernels cost 761 us and 390 us per call at stage 0,
where ``k`` is at most 1; ``k`` grows to 8 on the last pipeline stage).

These kernels handle one token per program and stream the hidden row in
``C``-wide chunks (autotuned 512-4096 columns x 4-8 warps), so the register
footprint stays small and many programs share an SM. The forward streams every
entry once for the two row reductions (sum of squares, dot with the score
weight) with elementwise accumulators reduced once per entry, softmaxes the
scores in registers and streams the entries a second time for the weighted sum;
no concatenation, no fp32 copy, bf16 in / bf16 out with fp32 math. The
per-token probabilities, inverse RMS and dot products (``3 x (k+1)`` fp32
values) are saved for the backward, which runs as two kernels: a per-token
coefficient kernel (``<grad, entry>`` reductions and the softmax backward) and
a ``[tokens / BWD_ROWS, chunks]`` gradient kernel that writes the analytic input
gradients and accumulates the score-weight gradient per (row block, chunk) into
a partial buffer that is summed in torch (deterministic, no atomics).

The fp32 math matches ``_attn_res_core`` up to fp32 accumulation order (the
reference multiplies each element by the inverse RMS before its dot-product
reduction; the kernel reduces first and scales the sum), which is below bf16
resolution for the shapes this is used at.
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
    triton.autotune = null_decorator
    tl = MagicMock()

# Largest number of block-residual entries (excluding the prefix sum) the kernels
# accept; Kimi-K3 (93 layers, block size 12) reaches 8.
MAX_ENTRIES = 16
# Largest hidden size the launch wrappers accept (the row is streamed in chunks, so this
# only bounds the per-token statistics layout).
MAX_HIDDEN = 16384
# Tokens per backward gradient program; each program writes one score-weight-gradient partial row.
BWD_ROWS = 8


def _chunk_configs() -> list:
    """Autotune candidates: chunk width x warps (register footprint vs loads in flight)."""
    if not HAVE_TRITON:  # pragma: no cover
        return []
    return [
        triton.Config({"C": 1024}, num_warps=4),
        triton.Config({"C": 2048}, num_warps=4),
        triton.Config({"C": 2048}, num_warps=8),
        triton.Config({"C": 4096}, num_warps=8),
        triton.Config({"C": 512}, num_warps=4),
    ]


@triton.autotune(configs=_chunk_configs(), key=["T", "H", "NB"])
@triton.jit
def _attn_res_fwd_kernel(
    br_ptr,
    ps_ptr,
    nw_ptr,
    pw_ptr,
    out_ptr,
    p_ptr,
    r_ptr,
    a_ptr,
    T,
    eps,
    s_br_t,
    s_br_k,
    s_ps_t,
    s_out_t,
    s_st_t,
    H: tl.constexpr,
    NB: tl.constexpr,
    NB_PAD: tl.constexpr,
    C: tl.constexpr,
):
    """One token per program: scores, softmax and weighted sum over NB block entries + the prefix row."""
    t = tl.program_id(0).to(tl.int64)
    offs_b = tl.arange(0, NB_PAD)
    valid = offs_b <= NB
    offs_c = tl.arange(0, C)
    br_row = br_ptr + t * s_br_t
    ps_row = ps_ptr + t * s_ps_t

    # Pass 1: per-entry sum of squares and score-weight dot, accumulated elementwise per chunk and
    # reduced once per entry.
    ss = tl.zeros([NB_PAD], dtype=tl.float32)
    dot = tl.zeros([NB_PAD], dtype=tl.float32)
    for j in tl.static_range(NB + 1):
        if j < NB:
            row = br_row + j * s_br_k
        else:
            row = ps_row
        acc_ss = tl.zeros([C], dtype=tl.float32)
        acc_dot = tl.zeros([C], dtype=tl.float32)
        for c0 in range(0, H, C):
            offs = c0 + offs_c
            m = offs < H
            x = tl.load(row + offs, mask=m, other=0.0).to(tl.float32)
            swc = tl.load(nw_ptr + offs, mask=m, other=0.0).to(tl.float32) * tl.load(
                pw_ptr + offs, mask=m, other=0.0
            ).to(tl.float32)
            acc_ss += x * x
            acc_dot += x * swc
        ss = tl.where(offs_b == j, tl.sum(acc_ss, 0), ss)
        dot = tl.where(offs_b == j, tl.sum(acc_dot, 0), dot)

    r = tl.rsqrt(ss / H + eps)
    s = tl.where(valid, dot * r, float("-inf"))
    mx = tl.max(s, 0)
    e = tl.where(valid, tl.exp(s - mx), 0.0)
    p = e / tl.sum(e, 0)

    # Pass 2: probability-weighted sum, one chunk at a time over all entries.
    for c0 in range(0, H, C):
        offs = c0 + offs_c
        m = offs < H
        acc = tl.zeros([C], dtype=tl.float32)
        for j in tl.static_range(NB + 1):
            if j < NB:
                row = br_row + j * s_br_k
            else:
                row = ps_row
            x = tl.load(row + offs, mask=m, other=0.0).to(tl.float32)
            pj = tl.sum(tl.where(offs_b == j, p, 0.0), 0)
            acc += pj * x
        tl.store(out_ptr + t * s_out_t + offs, acc.to(out_ptr.dtype.element_ty), mask=m)

    st = t * s_st_t
    tl.store(p_ptr + st + offs_b, p, mask=valid)
    tl.store(r_ptr + st + offs_b, r, mask=valid)
    tl.store(a_ptr + st + offs_b, dot, mask=valid)


@triton.autotune(configs=_chunk_configs(), key=["T", "H", "NB"])
@triton.jit
def _attn_res_bwd_coef_kernel(
    br_ptr,
    ps_ptr,
    g_ptr,
    p_ptr,
    r_ptr,
    a_ptr,
    csw_ptr,
    cx_ptr,
    T,
    s_br_t,
    s_br_k,
    s_ps_t,
    s_g_t,
    s_st_t,
    H: tl.constexpr,
    NB: tl.constexpr,
    NB_PAD: tl.constexpr,
    C: tl.constexpr,
):
    """One token per program: <g, x_j> per entry, then the softmax-backward coefficients.

    With p = softmax(s), s_j = r_j * a_j, r_j = rsqrt(mean(x_j^2) + eps), a_j = <x_j, sw>:
    dp_j = <g, x_j>, ds_j = p_j * (dp_j - sum_i p_i dp_i). Writes c_sw_j = ds_j * r_j (the
    coefficient on the score weight) and c_x_j = -ds_j * a_j * r_j^3 / H (the coefficient on x_j),
    so that dx_j = p_j * g + c_sw_j * sw + c_x_j * x_j.
    """
    t = tl.program_id(0).to(tl.int64)
    offs_b = tl.arange(0, NB_PAD)
    valid = offs_b <= NB
    offs_c = tl.arange(0, C)
    br_row = br_ptr + t * s_br_t
    ps_row = ps_ptr + t * s_ps_t
    g_row = g_ptr + t * s_g_t
    st = t * s_st_t

    dp = tl.zeros([NB_PAD], dtype=tl.float32)
    for j in tl.static_range(NB + 1):
        if j < NB:
            row = br_row + j * s_br_k
        else:
            row = ps_row
        acc = tl.zeros([C], dtype=tl.float32)
        for c0 in range(0, H, C):
            offs = c0 + offs_c
            m = offs < H
            x = tl.load(row + offs, mask=m, other=0.0).to(tl.float32)
            g = tl.load(g_row + offs, mask=m, other=0.0).to(tl.float32)
            acc += x * g
        dp = tl.where(offs_b == j, tl.sum(acc, 0), dp)

    p = tl.load(p_ptr + st + offs_b, mask=valid, other=0.0)
    r = tl.load(r_ptr + st + offs_b, mask=valid, other=0.0)
    a = tl.load(a_ptr + st + offs_b, mask=valid, other=0.0)
    ds = p * (dp - tl.sum(p * dp, 0))
    tl.store(csw_ptr + st + offs_b, ds * r, mask=valid)
    tl.store(cx_ptr + st + offs_b, -(ds * a * r * r * r) / H, mask=valid)


@triton.autotune(configs=_chunk_configs(), key=["T", "H", "NB"])
@triton.jit
def _attn_res_bwd_grad_kernel(
    br_ptr,
    ps_ptr,
    nw_ptr,
    pw_ptr,
    g_ptr,
    p_ptr,
    csw_ptr,
    cx_ptr,
    dbr_ptr,
    dps_ptr,
    dsw_ptr,
    T,
    s_br_t,
    s_br_k,
    s_ps_t,
    s_g_t,
    s_st_t,
    s_dbr_t,
    s_dbr_k,
    s_dps_t,
    H: tl.constexpr,
    NB: tl.constexpr,
    NB_PAD: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
):
    """Program (row block, chunk): dx_j = p_j g + c_sw_j sw + c_x_j x_j for R tokens on one C-wide chunk.

    Accumulates the score-weight gradient of its chunk over its R tokens and writes it to
    dsw[row block, chunk] (deterministic partials, summed by the caller).
    """
    pid_t = tl.program_id(0).to(tl.int64)
    c0 = tl.program_id(1) * C
    offs = c0 + tl.arange(0, C)
    m_c = offs < H
    offs_b = tl.arange(0, NB_PAD)
    valid = offs_b <= NB
    sw = tl.load(nw_ptr + offs, mask=m_c, other=0.0).to(tl.float32) * tl.load(pw_ptr + offs, mask=m_c, other=0.0).to(
        tl.float32
    )
    dsw_acc = tl.zeros([C], dtype=tl.float32)

    for i in range(R):
        t = pid_t * R + i
        row_ok = t < T
        m = m_c & row_ok
        m_b = valid & row_ok
        st = t * s_st_t
        p = tl.load(p_ptr + st + offs_b, mask=m_b, other=0.0)
        csw = tl.load(csw_ptr + st + offs_b, mask=m_b, other=0.0)
        cx = tl.load(cx_ptr + st + offs_b, mask=m_b, other=0.0)
        g = tl.load(g_ptr + t * s_g_t + offs, mask=m, other=0.0).to(tl.float32)
        br_row = br_ptr + t * s_br_t
        dbr_row = dbr_ptr + t * s_dbr_t
        for j in tl.static_range(NB + 1):
            if j < NB:
                x = tl.load(br_row + j * s_br_k + offs, mask=m, other=0.0).to(tl.float32)
            else:
                x = tl.load(ps_ptr + t * s_ps_t + offs, mask=m, other=0.0).to(tl.float32)
            pj = tl.sum(tl.where(offs_b == j, p, 0.0), 0)
            csw_j = tl.sum(tl.where(offs_b == j, csw, 0.0), 0)
            cx_j = tl.sum(tl.where(offs_b == j, cx, 0.0), 0)
            dx = pj * g + csw_j * sw + cx_j * x
            if j < NB:
                tl.store(dbr_row + j * s_dbr_k + offs, dx.to(dbr_ptr.dtype.element_ty), mask=m)
            else:
                tl.store(dps_ptr + t * s_dps_t + offs, dx.to(dps_ptr.dtype.element_ty), mask=m)
            dsw_acc += csw_j * x

    tl.store(dsw_ptr + pid_t * H + offs, dsw_acc, mask=m_c)


def _check_inputs(
    prefix_sum: torch.Tensor, block_residual: torch.Tensor, norm_weight: torch.Tensor, proj_weight: torch.Tensor
):
    if prefix_sum.dim() != 2 or block_residual.dim() != 3:
        raise ValueError("attn_res_triton expects prefix_sum [tokens, hidden] and block_residual [tokens, k, hidden]")
    tokens, hidden = prefix_sum.shape
    if block_residual.shape[0] != tokens or block_residual.shape[2] != hidden:
        raise ValueError(f"attn_res_triton shape mismatch: {tuple(prefix_sum.shape)} vs {tuple(block_residual.shape)}")
    if block_residual.shape[1] > MAX_ENTRIES:
        raise ValueError(f"attn_res_triton supports at most {MAX_ENTRIES} block entries, got {block_residual.shape[1]}")
    if hidden > MAX_HIDDEN:
        raise ValueError(f"attn_res_triton supports hidden <= {MAX_HIDDEN}, got {hidden}")
    if norm_weight.shape != (hidden,) or proj_weight.shape != (hidden,):
        raise ValueError("attn_res_triton expects 1-D norm_weight / proj_weight of length hidden")
    for name, t in (("prefix_sum", prefix_sum), ("block_residual", block_residual)):
        if t.stride(-1) != 1:
            raise ValueError(f"attn_res_triton needs a contiguous last dim for {name}")


def _entries_pad(entries: int) -> int:
    return max(2, triton.next_power_of_2(entries + 1))


def attn_res_fwd_triton(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    norm_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused attention-residual mix forward.

    Args:
        prefix_sum: ``[tokens, hidden]`` current residual stream (any float dtype; sets the output dtype).
        block_residual: ``[tokens, k, hidden]`` prior block starts, ``k <= MAX_ENTRIES`` (may be 0).
        norm_weight: ``[hidden]`` RMSNorm weight of the score norm.
        proj_weight: ``[hidden]`` squeezed ``[1, hidden]`` projection weight.
        eps: RMSNorm epsilon.

    Returns:
        ``(mixed [tokens, hidden] in prefix_sum's dtype, stats [3, tokens, k+1] fp32)`` where the stats
        hold the per-entry probabilities, inverse RMS and score-weight dot products for the backward.
    """
    _check_inputs(prefix_sum, block_residual, norm_weight, proj_weight)
    tokens, hidden = prefix_sum.shape
    entries = block_residual.shape[1]
    norm_weight = norm_weight.contiguous()
    proj_weight = proj_weight.contiguous()
    out = torch.empty_like(prefix_sum)
    stats = torch.empty((3, tokens, entries + 1), dtype=torch.float32, device=prefix_sum.device)
    if tokens == 0:
        return out, stats
    _attn_res_fwd_kernel[(tokens,)](
        block_residual,
        prefix_sum,
        norm_weight,
        proj_weight,
        out,
        stats[0],
        stats[1],
        stats[2],
        tokens,
        float(eps),
        block_residual.stride(0),
        block_residual.stride(1),
        prefix_sum.stride(0),
        out.stride(0),
        stats.stride(1),
        H=hidden,
        NB=entries,
        NB_PAD=_entries_pad(entries),
    )
    return out, stats


def attn_res_bwd_triton(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    norm_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    grad_out: torch.Tensor,
    stats: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused attention-residual mix backward.

    Args:
        prefix_sum, block_residual, norm_weight, proj_weight: the forward inputs.
        grad_out: ``[tokens, hidden]`` gradient of the mixed output (contiguous last dim).
        stats: the forward's ``[3, tokens, k+1]`` fp32 statistics.

    Returns:
        ``(d_prefix_sum, d_block_residual, d_score_weight)``: the first two in their inputs' dtypes, the
        third the fp32 ``[hidden]`` gradient of ``norm_weight * proj_weight`` (chain it to the two weights
        in the caller).
    """
    _check_inputs(prefix_sum, block_residual, norm_weight, proj_weight)
    tokens, hidden = prefix_sum.shape
    entries = block_residual.shape[1]
    if grad_out.shape != prefix_sum.shape or grad_out.stride(-1) != 1:
        raise ValueError("attn_res_triton backward needs grad_out shaped like prefix_sum with a contiguous last dim")
    norm_weight = norm_weight.contiguous()
    proj_weight = proj_weight.contiguous()
    d_prefix = torch.empty_like(prefix_sum)
    d_block = torch.empty_like(block_residual)
    if tokens == 0:
        return d_prefix, d_block, torch.zeros(hidden, dtype=torch.float32, device=prefix_sum.device)
    nb_pad = _entries_pad(entries)
    coef = torch.empty((2, tokens, entries + 1), dtype=torch.float32, device=prefix_sum.device)
    _attn_res_bwd_coef_kernel[(tokens,)](
        block_residual,
        prefix_sum,
        grad_out,
        stats[0],
        stats[1],
        stats[2],
        coef[0],
        coef[1],
        tokens,
        block_residual.stride(0),
        block_residual.stride(1),
        prefix_sum.stride(0),
        grad_out.stride(0),
        stats.stride(1),
        H=hidden,
        NB=entries,
        NB_PAD=nb_pad,
    )
    programs = triton.cdiv(tokens, BWD_ROWS)
    dsw_partial = torch.empty((programs, hidden), dtype=torch.float32, device=prefix_sum.device)
    _attn_res_bwd_grad_kernel[lambda meta: (programs, triton.cdiv(hidden, meta["C"]))](
        block_residual,
        prefix_sum,
        norm_weight,
        proj_weight,
        grad_out,
        stats[0],
        coef[0],
        coef[1],
        d_block,
        d_prefix,
        dsw_partial,
        tokens,
        block_residual.stride(0),
        block_residual.stride(1),
        prefix_sum.stride(0),
        grad_out.stride(0),
        stats.stride(1),
        d_block.stride(0),
        d_block.stride(1),
        d_prefix.stride(0),
        H=hidden,
        NB=entries,
        NB_PAD=nb_pad,
        R=BWD_ROWS,
    )
    return d_prefix, d_block, dsw_partial.sum(dim=0)

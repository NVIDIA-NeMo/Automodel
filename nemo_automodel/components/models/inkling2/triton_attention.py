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


"""Triton attention with learned relative logits, a per-query sink and document-local causal keys.

One forward kernel and two backward kernels (queries; keys and values) compute, for every
query, a softmax over the keys of its own document up to its own position (inside a sliding
window in local layers) plus one sink of value zero. A relative logit, indexed by the
distance between query and key, is added to each score. The relative logits are read as
tiles from a per-query profile stored in key order, so a tile load is affine in both the
query and the key index, and their gradient is written back the same way.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import torch
from torch.nn import functional as F
from torch.utils._python_dispatch import _disable_current_modes

from nemo_automodel.shared.import_utils import null_decorator
from nemo_automodel.shared.recompute_replay import RecomputeReplay

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

# Launch shapes (query block, key block, warps, stages), measured on GB200 at head_dim 128.
_LAUNCH = {"forward": (64, 64, 4, 2), "query": (64, 32, 4, 3), "key": (128, 128, 8, 1)}
# FP32 inputs (reference checks) need twice the shared memory per tile.
_LAUNCH_FP32 = (32, 32, 4, 1)
# Query heads per program of the kernel that prepares the output gradient.
_TERM_ROWS = 64
# Query positions per block of the relative-profile layout: a multiple of the query block of the
# forward and query kernels, which read and write whole tiles.
_ALIGN = 64
# Profile columns kept free on each side of a row, and the unit of the layout's column shift: a
# multiple of the key block of those two kernels, so that their tiles stay inside the rows.
_MARGIN = 64
# A kernel reads a global only as a constexpr.
_ALIGN_K = tl.constexpr(_ALIGN)
_MARGIN_K = tl.constexpr(_MARGIN)


def _launch(name: str, q: torch.Tensor) -> tuple[int, int, int, int]:
    """Return (query block, key block, warps, stages) of one kernel for q's dtype."""
    return _LAUNCH_FP32 if q.dtype == torch.float32 else _LAUNCH[name]


@triton.jit
def _load_rows(
    base, rows, offs_d, row_ok, EVEN: tl.constexpr
):  # pragma: no cover - Triton JIT executes on the GPU, outside Python tracing.
    """Load a [rows, width] tile; rows past the end read as zero unless the length divides the block."""
    if EVEN:
        return tl.load(base + rows[:, None] + offs_d[None, :])
    return tl.load(base + rows[:, None] + offs_d[None, :], mask=row_ok[:, None], other=0.0)


@triton.jit
def _profile_rows(
    offs_m, b, h, G, ROWS, SHIFT, H: tl.constexpr, PW: tl.constexpr, WIDE: tl.constexpr
):  # pragma: no cover - Triton JIT executes on the GPU, outside Python tracing.
    """Return, per query, the profile offset that plus a key's index addresses the pair's logit.

    The rows are ordered by the query position's residue in its block of _ALIGN positions, then
    by sequence, block and head. In a row, key n is column n - block start + SHIFT * _MARGIN, the
    same for every query of the block.
    """
    block = offs_m // _ALIGN_K
    row = (offs_m % _ALIGN_K) * ROWS + (b * G + block) * H + h
    if WIDE:
        row = row.to(tl.int64)
    return row * PW - block * _ALIGN_K + SHIFT * _MARGIN_K


@triton.jit
def _key_blocks(
    lo, pid_m, T, S, OFF, E, BM: tl.constexpr, BN: tl.constexpr
):  # pragma: no cover - Triton JIT executes on the GPU, outside Python tracing.
    """Return the key blocks of one query block: [first, near) without and [near, end) with relative logits."""
    kb_lo = tl.min(lo, axis=0) // BN
    q_last = tl.minimum(pid_m * BM + BM - 1, T - 1)
    kb_end = tl.minimum(q_last + OFF, S - 1) // BN + 1
    # Key blocks before kb_near lie beyond the relative extent of every query of the block.
    kb_near = tl.minimum(tl.maximum(kb_lo, tl.maximum(pid_m * BM + OFF - E + 1, 0) // BN), kb_end)
    return kb_lo, kb_near, kb_end


@triton.jit
def _scores(
    q, k, PROF, prof_row, offs_n, lo, hi, E, scale, PRECISION: tl.constexpr, REL: tl.constexpr, DENSE: tl.constexpr
):  # pragma: no cover - Triton JIT executes on the GPU, outside Python tracing.
    """Return the scores [queries, keys] of one tile and which pairs are allowed.

    A query sees the keys from its first key to its own position. With REL the tile holds
    keys within the relative extent of some query: their relative logits are added. The
    queries of a tile share the column of each key, so with DENSE the whole tile is read
    as aligned vectors and the pairs outside the extent are dropped afterwards; otherwise
    only the pairs within the extent are read, one element at a time.
    """
    s = tl.dot(q, tl.trans(k), input_precision=PRECISION) * scale
    allowed = (offs_n[None, :] >= lo[:, None]) & (offs_n[None, :] <= hi[:, None])
    if REL:
        in_rel = allowed & ((hi[:, None] - offs_n[None, :]) < E)
        if DENSE:
            r = tl.load(PROF + prof_row[:, None] + offs_n[None, :])
            s = s + tl.where(in_rel, r.to(tl.float32), 0.0)
        else:
            r = tl.load(PROF + prof_row[:, None] + offs_n[None, :], mask=in_rel, other=0.0)
            s = s + r.to(tl.float32)
    return s, allowed


@triton.jit
def _forward_tile(
    q,
    K,
    V,
    PROF,
    prof_row,
    lo,
    hi,
    m_i,
    l_i,
    acc,
    kb,
    b,
    kvh,
    S,
    E,
    scale,
    KVH: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    PRECISION: tl.constexpr,
    EVEN_N: tl.constexpr,
    WIDE: tl.constexpr,
    REL: tl.constexpr,
):  # pragma: no cover - Triton JIT executes on the GPU, outside Python tracing.
    """Fold one key block into the running softmax (base 2) and the value sum."""
    offs_n = kb * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    col_ok = offs_n < S
    col = b * S + offs_n
    if WIDE:
        col = col.to(tl.int64)
    col = (col * KVH + kvh) * D
    k = _load_rows(K, col, offs_d, col_ok, EVEN_N)
    s, allowed = _scores(q, k, PROF, prof_row, offs_n, lo, hi, E, scale, PRECISION, REL, True)
    s = tl.where(allowed, s * 1.4426950408889634, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, axis=1))
    alpha = tl.math.exp2(m_i - m_new)
    p = tl.math.exp2(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, axis=1)
    v = _load_rows(V, col, offs_d, col_ok, EVEN_N)
    acc = tl.dot(p.to(v.dtype), v, acc * alpha[:, None], input_precision=PRECISION)
    return m_new, l_i, acc


@triton.jit
def _forward_kernel(
    Q,
    K,
    V,
    PROF,
    SINK,
    LO,
    OUT,
    LSE,
    T,
    S,
    OFF,
    E,
    scale,
    G,
    ROWS,
    SHIFT,
    H: tl.constexpr,
    KVH: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    PRECISION: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    WIDE: tl.constexpr,
    PW: tl.constexpr,
):  # pragma: no cover - Triton JIT executes on the GPU, outside Python tracing.
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    kvh = h // (H // KVH)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    row_ok = offs_m < T
    row = b * T + offs_m
    if WIDE:
        row = row.to(tl.int64)
    row = row * H + h
    q = _load_rows(Q, row * D, offs_d, row_ok, EVEN_M)
    lo = tl.load(LO + b * T + offs_m, mask=row_ok, other=2147483647)
    hi = offs_m + OFF
    # The running softmax works in base 2. The sink is a key of value zero that every query
    # sees: it starts the softmax with weight one at its own logit.
    m_i = tl.load(SINK + row, mask=row_ok, other=0.0) * 1.4426950408889634
    l_i = tl.full([BM], 1.0, tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    prof_row = _profile_rows(offs_m, b, h, G, ROWS, SHIFT, H, PW, WIDE)
    kb_lo, kb_near, kb_end = _key_blocks(lo, pid_m, T, S, OFF, E, BM, BN)
    for kb in range(kb_lo, kb_near):
        m_i, l_i, acc = _forward_tile(
            q,
            K,
            V,
            PROF,
            prof_row,
            lo,
            hi,
            m_i,
            l_i,
            acc,
            kb,
            b,
            kvh,
            S,
            E,
            scale,
            KVH,
            D,
            BN,
            PRECISION,
            EVEN_N,
            WIDE,
            False,
        )
    for kb in range(kb_near, kb_end):
        m_i, l_i, acc = _forward_tile(
            q,
            K,
            V,
            PROF,
            prof_row,
            lo,
            hi,
            m_i,
            l_i,
            acc,
            kb,
            b,
            kvh,
            S,
            E,
            scale,
            KVH,
            D,
            BN,
            PRECISION,
            EVEN_N,
            WIDE,
            True,
        )
    out = acc / l_i[:, None]
    lse = (m_i + tl.math.log2(l_i)) * 0.6931471805599453
    if EVEN_M:
        tl.store(OUT + (row * D)[:, None] + offs_d[None, :], out)
        tl.store(LSE + row, lse)
    else:
        tl.store(OUT + (row * D)[:, None] + offs_d[None, :], out, mask=row_ok[:, None])
        tl.store(LSE + row, lse, mask=row_ok)


@triton.jit
def _query_tile(
    q,
    do,
    K,
    V,
    PROF,
    DPROF,
    prof_row,
    lo,
    hi,
    lse,
    delta,
    dq,
    kb,
    b,
    kvh,
    S,
    E,
    scale,
    KVH: tl.constexpr,
    D: tl.constexpr,
    BN: tl.constexpr,
    PRECISION: tl.constexpr,
    EVEN_N: tl.constexpr,
    WIDE: tl.constexpr,
    REL: tl.constexpr,
):  # pragma: no cover - Triton JIT executes on the GPU, outside Python tracing.
    """Add one key block's share of the query gradient and write its relative-logit gradient."""
    offs_n = kb * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    col_ok = offs_n < S
    col = b * S + offs_n
    if WIDE:
        col = col.to(tl.int64)
    col = (col * KVH + kvh) * D
    k = _load_rows(K, col, offs_d, col_ok, EVEN_N)
    v = _load_rows(V, col, offs_d, col_ok, EVEN_N)
    s, allowed = _scores(q, k, PROF, prof_row, offs_n, lo, hi, E, scale, PRECISION, REL, True)
    p = tl.where(allowed, tl.math.exp2((s - lse[:, None]) * 1.4426950408889634), 0.0)
    dp = tl.dot(do, tl.trans(v), input_precision=PRECISION)
    ds = p * (dp - delta[:, None])
    dq = tl.dot(ds.to(k.dtype), k, dq, input_precision=PRECISION)
    if REL:
        # Each query/key pair owns one profile element and this program owns its query rows. The whole
        # tile is written as aligned vectors: its other elements belong to no pair and are zero.
        in_rel = allowed & ((hi[:, None] - offs_n[None, :]) < E)
        tl.store(DPROF + prof_row[:, None] + offs_n[None, :], tl.where(in_rel, ds, 0.0).to(DPROF.dtype.element_ty))
    return dq


@triton.jit
def _backward_query_kernel(
    Q,
    K,
    V,
    PROF,
    LO,
    LSE,
    DO,
    DELTA,
    DQ,
    DPROF,
    T,
    S,
    OFF,
    E,
    scale,
    G,
    ROWS,
    SHIFT,
    H: tl.constexpr,
    KVH: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    PRECISION: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    WIDE: tl.constexpr,
    PW: tl.constexpr,
):  # pragma: no cover - Triton JIT executes on the GPU, outside Python tracing.
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    kvh = h // (H // KVH)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    row_ok = offs_m < T
    row = b * T + offs_m
    if WIDE:
        row = row.to(tl.int64)
    row = row * H + h
    q = _load_rows(Q, row * D, offs_d, row_ok, EVEN_M)
    do = _load_rows(DO, row * D, offs_d, row_ok, EVEN_M)
    lo = tl.load(LO + b * T + offs_m, mask=row_ok, other=2147483647)
    lse = tl.load(LSE + row, mask=row_ok, other=0.0)
    delta = tl.load(DELTA + row, mask=row_ok, other=0.0)
    hi = offs_m + OFF
    dq = tl.zeros([BM, D], tl.float32)
    prof_row = _profile_rows(offs_m, b, h, G, ROWS, SHIFT, H, PW, WIDE)
    kb_lo, kb_near, kb_end = _key_blocks(lo, pid_m, T, S, OFF, E, BM, BN)
    for kb in range(kb_lo, kb_near):
        dq = _query_tile(
            q,
            do,
            K,
            V,
            PROF,
            DPROF,
            prof_row,
            lo,
            hi,
            lse,
            delta,
            dq,
            kb,
            b,
            kvh,
            S,
            E,
            scale,
            KVH,
            D,
            BN,
            PRECISION,
            EVEN_N,
            WIDE,
            False,
        )
    for kb in range(kb_near, kb_end):
        dq = _query_tile(
            q,
            do,
            K,
            V,
            PROF,
            DPROF,
            prof_row,
            lo,
            hi,
            lse,
            delta,
            dq,
            kb,
            b,
            kvh,
            S,
            E,
            scale,
            KVH,
            D,
            BN,
            PRECISION,
            EVEN_N,
            WIDE,
            True,
        )
    if EVEN_M:
        tl.store(DQ + (row * D)[:, None] + offs_d[None, :], dq * scale)
    else:
        tl.store(DQ + (row * D)[:, None] + offs_d[None, :], dq * scale, mask=row_ok[:, None])


@triton.jit
def _clear_uncovered_kernel(
    DPROF,
    LO,
    T,
    S,
    OFF,
    E,
    G,
    ROWS,
    SHIFT,
    H: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    PW: tl.constexpr,
    WIDE: tl.constexpr,
):  # pragma: no cover - Triton JIT executes on the GPU, outside Python tracing.
    """Zero the profile-gradient columns that the query kernel leaves unwritten.

    With the same query blocks, the query kernel writes the columns of the key blocks that it reads
    with relative logits. The columns before and after them belong to no pair.
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    b = pid_bh // H
    h = pid_bh % H
    offs_m = pid_m * BM + tl.arange(0, BM)
    lo = tl.load(LO + b * T + offs_m, mask=offs_m < T, other=2147483647)
    kb_lo, kb_near, kb_end = _key_blocks(lo, pid_m, T, S, OFF, E, BM, BN)
    prof_row = _profile_rows(offs_m, b, h, G, ROWS, SHIFT, H, PW, WIDE)
    # Column 0 of these rows holds key first * BN, and a row is PW // BN key blocks wide.
    first = (pid_m * BM // _ALIGN_K * _ALIGN_K - SHIFT * _MARGIN_K) // BN
    last = first + PW // BN
    zeros = tl.zeros([BM, BN], DPROF.dtype.element_ty)
    for block in range(first, tl.minimum(tl.maximum(kb_near, first), last)):
        tl.store(DPROF + prof_row[:, None] + (block * BN + tl.arange(0, BN))[None, :], zeros)
    for block in range(tl.minimum(tl.maximum(kb_end, first), last), last):
        tl.store(DPROF + prof_row[:, None] + (block * BN + tl.arange(0, BN))[None, :], zeros)


@triton.jit
def _softmax_term_kernel(
    GRAD, OUT, DO, DELTA, heads, D: tl.constexpr, R: tl.constexpr
):  # pragma: no cover - Triton JIT executes on the GPU, outside Python tracing.
    """Per query head: sum(dO * O) in FP32, and dO rounded to the dtype of the products."""
    rows = tl.program_id(0) * R + tl.arange(0, R)
    row_ok = rows < heads
    offsets = (rows.to(tl.int64) * D)[:, None] + tl.arange(0, D)[None, :]
    grad = tl.load(GRAD + offsets, mask=row_ok[:, None], other=0.0)
    out = tl.load(OUT + offsets, mask=row_ok[:, None], other=0.0)
    tl.store(DELTA + rows, tl.sum(grad * out, axis=1), mask=row_ok)
    tl.store(DO + offsets, grad.to(DO.dtype.element_ty), mask=row_ok[:, None])


@triton.jit
def _key_tile(
    k,
    v,
    Q,
    DO,
    PROF,
    LO,
    LSE,
    DELTA,
    offs_n,
    dk,
    dv,
    qb,
    b,
    h,
    T,
    OFF,
    E,
    scale,
    G,
    ROWS,
    SHIFT,
    H: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    PRECISION: tl.constexpr,
    EVEN_M: tl.constexpr,
    WIDE: tl.constexpr,
    PW: tl.constexpr,
    REL: tl.constexpr,
):  # pragma: no cover - Triton JIT executes on the GPU, outside Python tracing.
    """Add one query block's share of the key and value gradients."""
    offs_m = qb * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    row_ok = offs_m < T
    row = b * T + offs_m
    if WIDE:
        row = row.to(tl.int64)
    row = row * H + h
    q = _load_rows(Q, row * D, offs_d, row_ok, EVEN_M)
    do = _load_rows(DO, row * D, offs_d, row_ok, EVEN_M)
    lo = tl.load(LO + b * T + offs_m, mask=row_ok, other=2147483647)
    lse = tl.load(LSE + row, mask=row_ok, other=0.0)
    delta = tl.load(DELTA + row, mask=row_ok, other=0.0)
    hi = offs_m + OFF
    prof_row = _profile_rows(offs_m, b, h, G, ROWS, SHIFT, H, PW, WIDE)
    s, allowed = _scores(q, k, PROF, prof_row, offs_n, lo, hi, E, scale, PRECISION, REL, False)
    p = tl.where(allowed, tl.math.exp2((s - lse[:, None]) * 1.4426950408889634), 0.0)
    dv = tl.dot(tl.trans(p).to(do.dtype), do, dv, input_precision=PRECISION)
    dp = tl.dot(do, tl.trans(v), input_precision=PRECISION)
    ds = p * (dp - delta[:, None])
    dk = tl.dot(tl.trans(ds).to(q.dtype), q, dk, input_precision=PRECISION)
    return dk, dv


@triton.jit
def _backward_key_kernel(
    Q,
    K,
    V,
    PROF,
    LO,
    LAST,
    LSE,
    DO,
    DELTA,
    DK,
    DV,
    T,
    S,
    OFF,
    E,
    scale,
    G,
    ROWS,
    SHIFT,
    H: tl.constexpr,
    KVH: tl.constexpr,
    D: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    PRECISION: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    WIDE: tl.constexpr,
    PW: tl.constexpr,
):  # pragma: no cover - Triton JIT executes on the GPU, outside Python tracing.
    pid_n = tl.program_id(0)
    pid_bk = tl.program_id(1)
    b = pid_bk // KVH
    kvh = pid_bk % KVH
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    col_ok = offs_n < S
    col = b * S + offs_n
    if WIDE:
        col = col.to(tl.int64)
    col = (col * KVH + kvh) * D
    k = _load_rows(K, col, offs_d, col_ok, EVEN_N)
    v = _load_rows(V, col, offs_d, col_ok, EVEN_N)
    last = tl.load(LAST + b * S + offs_n, mask=col_ok, other=-1)
    # Queries that can see a key of this block lie between the block's first key and the last such query.
    qb_lo = tl.maximum(pid_n * BN - OFF, 0) // BM
    qb_end = (tl.minimum(tl.max(last, axis=0), T - 1) + BM) // BM
    # Query blocks from this one on lie beyond the relative extent of every key of the block.
    qb_far = tl.maximum(pid_n * BN + BN - 1 + E - OFF + BM - 1, 0) // BM
    qb_far = tl.minimum(tl.maximum(qb_far, qb_lo), qb_end)
    dk = tl.zeros([BN, D], tl.float32)
    dv = tl.zeros([BN, D], tl.float32)
    for g in range(H // KVH):
        h = kvh * (H // KVH) + g
        for qb in range(qb_lo, qb_far):
            dk, dv = _key_tile(
                k,
                v,
                Q,
                DO,
                PROF,
                LO,
                LSE,
                DELTA,
                offs_n,
                dk,
                dv,
                qb,
                b,
                h,
                T,
                OFF,
                E,
                scale,
                G,
                ROWS,
                SHIFT,
                H,
                D,
                BM,
                PRECISION,
                EVEN_M,
                WIDE,
                PW,
                True,
            )
        for qb in range(qb_far, qb_end):
            dk, dv = _key_tile(
                k,
                v,
                Q,
                DO,
                PROF,
                LO,
                LSE,
                DELTA,
                offs_n,
                dk,
                dv,
                qb,
                b,
                h,
                T,
                OFF,
                E,
                scale,
                G,
                ROWS,
                SHIFT,
                H,
                D,
                BM,
                PRECISION,
                EVEN_M,
                WIDE,
                PW,
                False,
            )
    if EVEN_N:
        tl.store(DK + col[:, None] + offs_d[None, :], dk * scale)
        tl.store(DV + col[:, None] + offs_d[None, :], dv)
    else:
        tl.store(DK + col[:, None] + offs_d[None, :], dk * scale, mask=col_ok[:, None])
        tl.store(DV + col[:, None] + offs_d[None, :], dv, mask=col_ok[:, None])


def key_ranges(
    valid: torch.Tensor,
    positions: torch.Tensor,
    key_valid: torch.Tensor,
    key_positions: torch.Tensor,
    query_offset: int,
    window: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Describe which keys each query sees as one contiguous range of the global token stream.

    Args:
        valid: Boolean query validity [batch, sequence].
        positions: Logical query positions [batch, sequence], restarting at each document.
        key_valid: Boolean key validity [batch, key_sequence], global under CP.
        key_positions: Logical key positions [batch, key_sequence].
        query_offset: Global column of the first local query.
        window: Largest number of keys a query sees, or None for all keys of its document.

    Returns:
        ``first`` int32 [batch, sequence]: the first key of every query; its last key is its
        own global column, and an invalid query gets a first key after that, an empty range.
        ``last`` int32 [batch, key_sequence]: the last local query that sees each key, or -1.
        A document is a run of valid tokens whose positions increase by one, so the keys
        between a query's document start and the query itself are exactly its own document.
        A row's first document may start at a nonzero position without its earlier tokens
        (after left padding, or after the empty halo of the first CP rank); its range starts
        at the first key of the query's run of valid keys.
    """
    sequence, keys = valid.shape[1], key_valid.shape[1]
    column = torch.arange(sequence, device=valid.device) + query_offset
    key_column = torch.arange(keys, device=valid.device)
    run_start = torch.where(key_valid, 0, key_column + 1).cummax(1).values
    first = torch.maximum((column - positions).clamp_min(0), run_start[:, query_offset : query_offset + sequence])
    if window is not None:
        first = torch.maximum(first, column - (window - 1))
    first = torch.where(valid, first, column + 1)
    origin = key_column - key_positions
    ends_run = torch.ones_like(key_valid)
    ends_run[:, :-1] = ~key_valid[:, 1:] | (origin[:, 1:] != origin[:, :-1])
    run_end = torch.where(ends_run, key_column, keys).flip(1).cummin(1).values.flip(1)
    last = run_end if window is None else torch.minimum(run_end, key_column + (window - 1))
    last = torch.where(key_valid, last - query_offset, -1)
    return first.to(torch.int32).contiguous(), last.to(torch.int32).contiguous()


# Every layer of one forward receives the same layout tensors, so the ranges of the last
# layout are kept per window (local layers share one, full layers the other).
_RANGES: dict[int | None, tuple[tuple[torch.Tensor, ...], int, tuple[torch.Tensor, torch.Tensor]]] = {}


def _layout_ranges(
    valid: torch.Tensor,
    positions: torch.Tensor,
    key_valid: torch.Tensor,
    key_positions: torch.Tensor,
    query_offset: int,
    window: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return :func:`key_ranges`, reusing the result while the same layout tensors come in."""
    layout = (valid, positions, key_valid, key_positions)
    entry = _RANGES.get(window)
    if entry is not None and entry[1] == query_offset and all(a is b for a, b in zip(entry[0], layout)):
        return entry[2]
    # Activation checkpointing replays a layer's operations by their count. With several pipeline
    # microbatches in flight a layer can find the entry in its forward and miss it in its
    # recomputation, so the operations of a miss run outside the dispatch modes in both passes.
    with _disable_current_modes():
        ranges = key_ranges(valid, positions, key_valid, key_positions, query_offset, window)
    _RANGES[window] = (layout, query_offset, ranges)
    return ranges


attention_replay: RecomputeReplay[tuple[torch.Tensor, torch.Tensor]] = RecomputeReplay("Inkling2 attention")
"""Record/replay channel for a layer's attention output across the activation-checkpoint recomputation.

``Inkling2DecoderLayer.nemo_checkpoint_context_fn`` binds it around each checkpointed layer call when
``Inkling2TextConfig.reuse_attention_on_recompute`` is set. The checkpoint forward records the output
[batch, sequence, heads, head_dim] and the log-sum-exp [batch, sequence, heads] of each attention call,
both FP32; the recomputation takes them back instead of running the forward kernel. The backward reads
exactly these two forward results, so the gradients are unchanged.
"""


class _RelativeAttention(torch.autograd.Function):
    """Attention over document-local causal keys with relative logits and a zero-valued sink."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        profiles: torch.Tensor,
        sink: torch.Tensor,
        first: torch.Tensor,
        last: torch.Tensor,
        query_offset: int,
        scale: float,
        extent: int,
        shift: int,
    ) -> torch.Tensor:
        bound = attention_replay.current()
        kept = bound[0].take() if bound is not None and bound[1] == "replay" else None
        if kept is not None:
            out, lse = kept
            ctx.save_for_backward(q, k, v, profiles, sink, first, last, out, lse)
            ctx.query_offset = query_offset
            ctx.scale = scale
            ctx.layout = (extent, shift)
            return out
        batch, sequence, heads, width = q.shape
        keys, kv_heads = k.shape[1], k.shape[2]
        out = torch.empty(batch, sequence, heads, width, device=q.device, dtype=torch.float32)
        lse = torch.empty(batch, sequence, heads, device=q.device, dtype=torch.float32)
        block_m, block_n, warps, stages = _launch("forward", q)
        _forward_kernel[(triton.cdiv(sequence, block_m), batch * heads)](
            q,
            k,
            v,
            profiles,
            sink,
            first,
            out,
            lse,
            sequence,
            keys,
            query_offset,
            extent,
            scale,
            triton.cdiv(sequence, _ALIGN),
            profiles.shape[1],
            shift,
            H=heads,
            KVH=kv_heads,
            D=width,
            BM=block_m,
            BN=block_n,
            PRECISION=_precision(q),
            EVEN_M=sequence % block_m == 0,
            EVEN_N=keys % block_n == 0,
            WIDE=_wide(q, k, profiles),
            PW=profiles.shape[2],
            num_warps=warps,
            num_stages=stages,
        )
        if bound is not None and bound[1] == "record":
            # An alias without history: the returned tensor will own this call's graph node.
            bound[0].record((out.detach(), lse))
        ctx.save_for_backward(q, k, v, profiles, sink, first, last, out, lse)
        ctx.query_offset = query_offset
        ctx.scale = scale
        ctx.layout = (extent, shift)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_out: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        q, k, v, profiles, sink, first, last, out, lse = ctx.saved_tensors
        batch, sequence, heads, width = q.shape
        keys, kv_heads = k.shape[1], k.shape[2]
        extent, shift = ctx.layout
        blocks = triton.cdiv(sequence, _ALIGN)
        # sum(dO * O) is the softmax's own term; it is taken in FP32 before dO is rounded for the products.
        delta = torch.empty_like(lse)
        do = torch.empty_like(q)
        _softmax_term_kernel[(triton.cdiv(lse.numel(), _TERM_ROWS),)](
            grad_out.contiguous(), out, do, delta, lse.numel(), D=width, R=_TERM_ROWS
        )
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        # The buffer starts uninitialized. The query kernel writes the key blocks that it reads with
        # relative logits; the columns around those blocks are cleared first.
        dprof = torch.empty_like(profiles)
        block_m, block_n, warps, stages = _launch("query", q)
        _clear_uncovered_kernel[(blocks * _ALIGN // block_m, batch * heads)](
            dprof,
            first,
            sequence,
            keys,
            ctx.query_offset,
            extent,
            blocks,
            profiles.shape[1],
            shift,
            H=heads,
            BM=block_m,
            BN=block_n,
            PW=profiles.shape[2],
            WIDE=_wide(q, k, profiles),
        )
        _backward_query_kernel[(triton.cdiv(sequence, block_m), batch * heads)](
            q,
            k,
            v,
            profiles,
            first,
            lse,
            do,
            delta,
            dq,
            dprof,
            sequence,
            keys,
            ctx.query_offset,
            extent,
            ctx.scale,
            blocks,
            profiles.shape[1],
            shift,
            H=heads,
            KVH=kv_heads,
            D=width,
            BM=block_m,
            BN=block_n,
            PRECISION=_precision(q),
            EVEN_M=sequence % block_m == 0,
            EVEN_N=keys % block_n == 0,
            WIDE=_wide(q, k, profiles),
            PW=profiles.shape[2],
            num_warps=warps,
            num_stages=stages,
        )
        block_m, block_n, warps, stages = _launch("key", q)
        _backward_key_kernel[(triton.cdiv(keys, block_n), batch * kv_heads)](
            q,
            k,
            v,
            profiles,
            first,
            last,
            lse,
            do,
            delta,
            dk,
            dv,
            sequence,
            keys,
            ctx.query_offset,
            extent,
            ctx.scale,
            blocks,
            profiles.shape[1],
            shift,
            H=heads,
            KVH=kv_heads,
            D=width,
            BM=block_m,
            BN=block_n,
            PRECISION=_precision(q),
            EVEN_M=sequence % block_m == 0,
            EVEN_N=keys % block_n == 0,
            WIDE=_wide(q, k, profiles),
            PW=profiles.shape[2],
            num_warps=warps,
            num_stages=stages,
        )
        # The sink is a key of value zero: its gradient is minus its probability times sum(dO * O).
        dsink = -torch.exp(sink - lse) * delta
        return dq, dk, dv, dprof, dsink, None, None, None, None, None, None


def _wide(*tensors: torch.Tensor) -> bool:
    """Whether an element offset of one of the tensors needs 64 bits."""
    return any(tensor.numel() >= 2**31 for tensor in tensors)


def _precision(q: torch.Tensor) -> str:
    """Exact FP32 products for FP32 inputs (reference checks); low-precision inputs use their own."""
    return "ieee" if q.dtype == torch.float32 else "tf32"


def _aligned_profiles(relative: torch.Tensor, table: torch.Tensor, query_offset: int) -> tuple[torch.Tensor, int]:
    """Return every query's relative logits, laid out so that a tile of queries by keys is a rectangle.

    Stored by distance, the logits of neighbouring queries for one key sit one column apart, and a
    kernel has to read a tile element by element. Here the queries of a block of _ALIGN positions
    share their columns: key n is column n - block start + shift * _MARGIN. A query's row is then the
    table shifted by its position's residue in the block, so the rows are grouped by residue and one
    batched product serves all residues.

    Args:
        relative: Relative features [batch, sequence, heads, d_rel].
        table: Relative table [d_rel, extent]; column d holds distance d.
        query_offset: Global key column of the first local query.

    Returns:
        Logits [_ALIGN, batch * blocks * heads, width] in relative's dtype, ordered by residue, then
        sequence, block and head; and the shift.
    """
    batch, sequence, heads, features = relative.shape
    extent = table.shape[-1]
    blocks = -(-sequence // _ALIGN)
    padded = -(-extent // _MARGIN) * _MARGIN
    # The shift is a whole number of margins; the slack moves every row right to make it so.
    shift = -(-(padded + _MARGIN - query_offset) // _MARGIN)
    slack = shift * _MARGIN - (padded + _MARGIN - query_offset)
    width = padded + _ALIGN + 2 * _MARGIN + (_MARGIN if slack else 0)
    # A query's own key is the last column of the flipped table, which starts at column residue + start:
    # the rows of the residues are the windows of the padded table, taken from the last one back.
    start = padded - extent + _MARGIN + slack + 1
    flipped = F.pad(table.flip(-1), (_ALIGN - 1 + start, width - extent - start))
    shifted = flipped.unfold(-1, width, 1).flip(1).transpose(0, 1)
    if blocks * _ALIGN != sequence:
        relative = F.pad(relative, (0, 0, 0, 0, 0, blocks * _ALIGN - sequence))
    by_residue = relative.view(batch, blocks, _ALIGN, heads, features).permute(2, 0, 1, 3, 4)
    return torch.bmm(by_residue.reshape(_ALIGN, batch * blocks * heads, features), shifted), shift


def relative_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    relative: torch.Tensor,
    table: torch.Tensor,
    sink: torch.Tensor,
    valid: torch.Tensor,
    positions: torch.Tensor,
    key_valid: torch.Tensor,
    key_positions: torch.Tensor,
    query_offset: int,
    window: int | None,
) -> torch.Tensor:
    """Attend to the keys of each query's document, with relative logits and a zero-valued sink.

    Args:
        q: Queries [batch, sequence, heads, head_dim].
        k: Keys [batch, key_sequence, kv_heads, head_dim] in q's dtype.
        v: Values [batch, key_sequence, kv_heads, head_dim] in q's dtype.
        relative: Relative features [batch, sequence, heads, d_rel].
        table: Relative table [d_rel, extent]; column d holds distance d, and the relative
            logit of a query and a key at distance d is their product's column d, zero from
            the extent on.
        sink: Sink logits [batch, sequence, heads].
        valid: Boolean query validity [batch, sequence].
        positions: Logical query positions [batch, sequence], restarting at each document.
        key_valid: Boolean key validity [batch, key_sequence], global under CP.
        key_positions: Logical key positions [batch, key_sequence].
        query_offset: Global column of the first local query, zero without CP.
        window: Largest number of keys a local-attention query sees; None in full layers.

    Returns:
        FP32 context [batch, sequence, heads, head_dim]. Scores, softmax and sums accumulate
        in FP32; probabilities are rounded to the value dtype for the value product. Invalid
        queries return zero. Gradients reach q, k, v, relative, table and sink.
    """
    if not HAVE_TRITON:
        raise ImportError("The Triton attention kernel requires the triton package")
    width = q.shape[-1]
    # Triton tiles need a power-of-two width of at least 16; zero channels change no product.
    width_pad = max(16, 1 << (width - 1).bit_length()) - width
    if width_pad:
        q, k, v = F.pad(q, (0, width_pad)), F.pad(k, (0, width_pad)), F.pad(v, (0, width_pad))
    first, last = _layout_ranges(valid, positions, key_valid, key_positions, query_offset, window)
    profiles, shift = _aligned_profiles(relative, table, query_offset)
    out = _RelativeAttention.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        profiles,
        sink.float().contiguous(),
        first,
        last,
        query_offset,
        width**-0.5,
        table.shape[-1],
        shift,
    )
    return out[..., :width] if width_pad else out

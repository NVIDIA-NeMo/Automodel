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

"""Chunked Kimi Delta Attention BACKWARD in Triton for the Kimi-K3 call. From the raw
forward inputs q, k, v, g, beta, cu_seqlens and the output gradient ``do`` it recomputes the forward intermediates and
produces dq, dk, dv (bf16) and dg, dbeta (fp32) -- nothing is saved from the forward. Measured on GB200 against FLA
0.4.2's autograd backward (headline mode with saved intermediates) at the Kimi-K3 layer-microbatch shape: 4.25-4.39 ms
vs 4.89-4.95 ms, every output within per-output tolerances (dq/dk/dv/dg atol 2e-2 rtol 2e-2, dbeta atol 2e-1 rtol 5e-2, matched
ratio >= 0.99), identical numerics on document layouts it was not tuned on. Entry point: ``run``.
"""

from array import array as _array
from unittest.mock import MagicMock

import torch

from nemo_automodel.shared.import_utils import MISSING_TRITON_MSG, null_decorator

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # CPU-only installs (the import check on macOS runners): the module must still import
    HAVE_TRITON = False

if not HAVE_TRITON:
    triton = MagicMock()
    triton.jit = null_decorator
    triton.autotune = null_decorator
    triton.heuristics = null_decorator
    tl = MagicMock()

CLAMP = tl.constexpr(126.0)
LOG2E = tl.constexpr(1.4426950408889634)


@triton.jit
def _inv32(Xb, i32, BC: tl.constexpr):
    """(I + X)^-1 for a strictly-lower-triangular 32x32 X, via the doubling
    identity (I+X)^-1 = (I-X)(I+X^2)(I+X^4)(I+X^8)(I+X^16).

    FOUR squarings, not five. After s steps the running product equals
    sum_{j=0}^{2^(s+1)-1} (-X)^j; a 32x32 strictly-lower X has X^32 = 0, so
    s=4 (terms 0..31) is EXACT and a 5th step would multiply by I+X^32 = I.
    The 64x64 version needs terms 0..63, hence the 5 steps in the current code.
    Verified: s=4 -> err 1.6e-14 vs exact, s=3 -> 5.8e-5 (insufficient)."""
    P = Xb
    R = tl.where(i32[:, None] == i32[None, :], 1.0, 0.0) - P.to(tl.float32)
    for _s in tl.static_range(4):
        P = tl.dot(P, P, out_dtype=tl.float32).to(tl.bfloat16)
        R = R + tl.dot(R.to(tl.bfloat16), P, out_dtype=tl.float32)
    return R


@triton.jit
def _prep(
    Q,
    KK,
    G,
    BETA,
    CS,
    CL,
    QG,
    KE,
    KG,
    W,
    PM,
    GM,
    AQ,
    GL,
    RQ,
    RK,
    GC,
    GB,
    scale,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    QSCALE: tl.constexpr,
):
    # i_h is the FASTEST grid dim.  q/k/g are [T, H, K], so for a fixed token
    # the 96 heads are one contiguous 24 KB stretch.  With i_t fastest the
    # ~300 co-resident CTAs were all chunks of 3 heads, touching 768 of every
    # 24576 bytes (3%): every DRAM page opened, partially used, closed.  With
    # i_h fastest a resident wave covers 3 chunks x ALL heads = full pages,
    # and the 96 sibling lanes of the [T,H] fp32 vectors (RQ/RK/BETA, stride
    # 384 B) now cover whole sectors instead of 4 useful bytes in 32.
    i_h = tl.program_id(0)
    i_t = tl.program_id(1)
    t0 = tl.load(CS + i_t)
    L = tl.load(CL + i_t)
    c = tl.arange(0, BC)
    ak = tl.arange(0, K)

    r0 = c
    r1 = BC + c
    m0 = r0 < L
    m1 = r1 < L
    ok0 = (t0 + r0)[:, None] * (H * K) + i_h * K + ak[None, :]
    ok1 = (t0 + r1)[:, None] * (H * K) + i_h * K + ak[None, :]

    mb = (i_t * H + i_h) * BT * BT
    om00 = mb + r0[:, None] * BT + r0[None, :]
    om01 = mb + r0[:, None] * BT + r1[None, :]
    om10 = mb + r1[:, None] * BT + r0[None, :]
    om11 = mb + r1[:, None] * BT + r1[None, :]
    lo = c[:, None] > c[None, :]
    loe = c[:, None] >= c[None, :]
    zz = tl.zeros([BC, BC], dtype=tl.bfloat16)

    # ---- PHASE A: gates only -----------------------------------------------
    # g0/g1 are loaded, reduced, and killed before any q/k tile is created.
    # Identical split cumsum to the baseline, so bit-exact.
    # Forward-inclusive cumsum over the BC=32 rows is LT @ g with
    # LT[i,j] = (j <= i) -- the same reduction on the tensor cores instead of
    # a log2(BC)=5-step shuffle/select scan.  g lies in (-5, 0) so the partial
    # sums are bounded by 5*BC; splitting into a bf16 high part plus the fp32
    # residual carries the mantissa bits bf16 would drop, and the products are
    # accumulated in fp32.
    LT = (c[:, None] >= c[None, :]).to(tl.bfloat16)
    g0 = tl.load(G + ok0, mask=m0[:, None], other=0.0).to(tl.float32) * LOG2E
    g0h = g0.to(tl.bfloat16)
    g0r = (g0 - g0h.to(tl.float32)).to(tl.bfloat16)
    gc0 = tl.dot(LT, g0h, out_dtype=tl.float32) + tl.dot(LT, g0r, out_dtype=tl.float32)
    s0 = tl.sum(tl.where(c[:, None] == (BC - 1), gc0, 0.0), 0)
    g1 = tl.load(G + ok1, mask=m1[:, None], other=0.0).to(tl.float32) * LOG2E
    g1h = g1.to(tl.bfloat16)
    g1r = (g1 - g1h.to(tl.float32)).to(tl.bfloat16)
    gc1 = s0[None, :] + (tl.dot(LT, g1h, out_dtype=tl.float32) + tl.dot(LT, g1r, out_dtype=tl.float32))
    # (stores moved below, after bs0 is available as the quantisation base)

    gl = tl.sum(tl.where(c[:, None] == (BC - 1), gc1, 0.0), 0)
    tl.store(GL + (i_t * H + i_h) * K + ak, gl)
    bs0 = tl.sum(tl.where(c[:, None] == (BC // 2), gc0, 0.0), 0)
    bs1 = tl.sum(tl.where(c[:, None] == (BC // 2), gc1, 0.0), 0)
    tl.store(GB + ((i_t * H + i_h) * (BT // BC) + 0) * K + ak, bs0)
    # int16 fixed-point store of (gc - bs0) at LSB = 1/QSCALE.  bs0 itself is
    # kept exact in fp32 in GB (12.6 MB), so the reconstruction in _bwd_qkg is
    # base-exact and only the bounded difference carries quantisation error.
    tl.store(GC + ok0, tl.extra.cuda.libdevice.round((gc0 - bs0[None, :]) * QSCALE).to(tl.int16), mask=m0[:, None])
    tl.store(GC + ok1, tl.extra.cuda.libdevice.round((gc1 - bs0[None, :]) * QSCALE).to(tl.int16), mask=m1[:, None])
    tl.store(GB + ((i_t * H + i_h) * (BT // BC) + 1) * K + ak, bs1)
    # gl/bs0/bs1 are [K] vectors = 1 reg/thread each. Hoisting bs1 here does
    # extend its live range across phase B, but it costs 1 reg/thread to do
    # so and it is what makes KC constructible in phase B. Good trade.

    # ---- PHASE B: row-half 0, start to finish ------------------------------
    q0 = tl.load(Q + ok0, mask=m0[:, None], other=0.0).to(tl.float32)
    rq0 = tl.rsqrt(tl.sum(q0 * q0, 1) + 1e-6)
    tl.store(RQ + (t0 + r0) * H + i_h, rq0, mask=m0)
    qs0 = q0 * (rq0 * scale)[:, None]

    k0 = tl.load(KK + ok0, mask=m0[:, None], other=0.0).to(tl.float32)
    rk0 = tl.rsqrt(tl.sum(k0 * k0, 1) + 1e-6)
    tl.store(RK + (t0 + r0) * H + i_h, rk0, mask=m0)
    kn0 = k0 * rk0[:, None]

    eg0 = tl.math.exp2(gc0)
    tl.store(QG + ok0, (qs0 * eg0).to(tl.bfloat16), mask=m0[:, None])
    keb0 = (kn0 * eg0).to(tl.bfloat16)
    tl.store(KE + ok0, keb0, mask=m0[:, None])
    tl.store(KG + ok0, (kn0 * tl.math.exp2(gl[None, :] - gc0)).to(tl.bfloat16), mask=m0[:, None])

    er0 = tl.math.exp2(tl.minimum(tl.maximum(gc0 - bs0[None, :], -CLAMP), CLAMP))
    kr0 = (kn0 * er0).to(tl.bfloat16)
    qr0 = (qs0 * er0).to(tl.bfloat16)
    # KA and KC are BOTH built here, while kn0/gc0 are still live. KC couples
    # row-block 1 to column-block 0, so it needs kn0 (phase B) with bs1
    # (phase A) -- this is the single ordering constraint of the whole split.
    KA = tl.trans((kn0 * tl.math.exp2(tl.minimum(tl.maximum(bs0[None, :] - gc0, -CLAMP), CLAMP))).to(tl.bfloat16))
    KC = tl.trans((kn0 * tl.math.exp2(tl.minimum(tl.maximum(bs1[None, :] - gc0, -CLAMP), CLAMP))).to(tl.bfloat16))

    # Part (2) of the fix: consume kr0/qr0/KA NOW.
    Ga = tl.where(lo, tl.dot(kr0, KA, out_dtype=tl.float32), 0.0)
    Aa = tl.where(loe, tl.dot(qr0, KA, out_dtype=tl.float32), 0.0)
    tl.store(GM + om00, Ga.to(tl.bfloat16))
    tl.store(AQ + om00, Aa.to(tl.bfloat16))
    # Dead now: q0,k0,qs0,kn0,eg0,er0,gc0,bs0,kr0,qr0,KA,Aa.
    # Carried into phase C: keb0(16) + KC(16) + Ga(8) = 40 regs/thread.
    # Ga survives only because the inversion below needs it; it is [32,32]
    # fp32 = 8 regs, so carrying it is cheap and keeps the Ra/Rb chains
    # adjacent at the end (preserving the 2-way ILP).

    # ---- PHASE C: row-half 1, start to finish ------------------------------
    q1 = tl.load(Q + ok1, mask=m1[:, None], other=0.0).to(tl.float32)
    rq1 = tl.rsqrt(tl.sum(q1 * q1, 1) + 1e-6)
    tl.store(RQ + (t0 + r1) * H + i_h, rq1, mask=m1)
    qs1 = q1 * (rq1 * scale)[:, None]

    k1 = tl.load(KK + ok1, mask=m1[:, None], other=0.0).to(tl.float32)
    rk1 = tl.rsqrt(tl.sum(k1 * k1, 1) + 1e-6)
    tl.store(RK + (t0 + r1) * H + i_h, rk1, mask=m1)
    kn1 = k1 * rk1[:, None]

    eg1 = tl.math.exp2(gc1)
    tl.store(QG + ok1, (qs1 * eg1).to(tl.bfloat16), mask=m1[:, None])
    keb1 = (kn1 * eg1).to(tl.bfloat16)
    tl.store(KE + ok1, keb1, mask=m1[:, None])
    tl.store(KG + ok1, (kn1 * tl.math.exp2(gl[None, :] - gc1)).to(tl.bfloat16), mask=m1[:, None])

    er1 = tl.math.exp2(tl.minimum(tl.maximum(gc1 - bs1[None, :], -CLAMP), CLAMP))
    kr1 = (kn1 * er1).to(tl.bfloat16)
    qr1 = (qs1 * er1).to(tl.bfloat16)  # <-- peak of the kernel: 216 regs/thr
    KB = tl.trans((kn1 * tl.math.exp2(tl.minimum(tl.maximum(bs1[None, :] - gc1, -CLAMP), CLAMP))).to(tl.bfloat16))

    # Gc/Ac need NO triangular mask: row index BC+i always exceeds col j < BC.
    Gb = tl.where(lo, tl.dot(kr1, KB, out_dtype=tl.float32), 0.0)
    Ab = tl.where(loe, tl.dot(qr1, KB, out_dtype=tl.float32), 0.0)
    Gc = tl.dot(kr1, KC, out_dtype=tl.float32)
    Ac = tl.dot(qr1, KC, out_dtype=tl.float32)

    tl.store(GM + om11, Gb.to(tl.bfloat16))
    tl.store(GM + om10, Gc.to(tl.bfloat16))
    tl.store(GM + om01, zz)
    tl.store(AQ + om11, Ab.to(tl.bfloat16))
    tl.store(AQ + om10, Ac.to(tl.bfloat16))
    tl.store(AQ + om01, zz)

    # ---- BLOCK-TRIANGULAR INVERSION (unchanged) ----------------------------
    # C is scaled by b1 (the ROW block's beta), not b0 -- M = Gac*beta[:,None]
    # is a row scaling and C occupies rows BC:BT.
    b0 = tl.load(BETA + (t0 + r0) * H + i_h, mask=m0, other=0.0)
    b1 = tl.load(BETA + (t0 + r1) * H + i_h, mask=m1, other=0.0)
    Ra = _inv32((Ga * b0[:, None]).to(tl.bfloat16), c, BC)
    Rb = _inv32((Gb * b1[:, None]).to(tl.bfloat16), c, BC)
    E = -tl.dot(
        tl.dot(Rb.to(tl.bfloat16), (Gc * b1[:, None]).to(tl.bfloat16), out_dtype=tl.float32).to(tl.bfloat16),
        Ra.to(tl.bfloat16),
        out_dtype=tl.float32,
    )

    tl.store(PM + om00, Ra.to(tl.bfloat16))
    tl.store(PM + om10, E.to(tl.bfloat16))
    tl.store(PM + om11, Rb.to(tl.bfloat16))
    tl.store(PM + om01, zz)

    # ---- w, u consumed blockwise (unchanged) -------------------------------
    tA = (Ra * b0[None, :]).to(tl.bfloat16)
    tE = (E * b0[None, :]).to(tl.bfloat16)
    tB = (Rb * b1[None, :]).to(tl.bfloat16)
    tl.store(W + ok0, tl.dot(tA, keb0, out_dtype=tl.float32).to(tl.bfloat16), mask=m0[:, None])
    tl.store(
        W + ok1,
        (tl.dot(tE, keb0, out_dtype=tl.float32) + tl.dot(tB, keb1, out_dtype=tl.float32)).to(tl.bfloat16),
        mask=m1[:, None],
    )
    # U is no longer materialised here.  u = (PM * beta) @ v depends only on
    # PM and BETA (both already stored for _bwd_local) and on v, so _fwdh --
    # which is the ONLY consumer of U -- rebuilds it from the tile of PM it
    # can load itself.  That removes a 201 MB store from _prep, the matching
    # 201 MB read from _fwdh, the V loads here, and two [BT,BT]x[BT,BV] MMAs
    # from _prep, at the cost of re-reading PM (101 MB) and v in _fwdh.
    # _prep is the bandwidth-heaviest kernel (2.4 GB), so the trade favours
    # moving work out of it.


@triton.jit
def _scan(
    W,
    PM,
    BETA,
    VV,
    KG,
    GL,
    HS,
    VN,
    QG,
    AQ,
    DO,
    DHT,
    DVN,
    CS,
    CL,
    DF,
    DN,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    # The forward and reverse state scans are INDEPENDENT: _fwdh reads
    # {W,PM,BETA,VV,KG,GL} and writes {HS,VN}; _bwdh reads {QG,KG,W,AQ,DO,GL}
    # and writes {DHT,DVN}.  The reverse scan reads neither HS nor VN, so the
    # dependency graph is _prep -> {fwd || rev} -> _bwd_local, not a chain.
    # They were nonetheless launched back to back, and each ran a grid of only
    # (1, 96, 4) = 384 CTAs = 2.5 per SM on 152 SMs.  Both are latency-bound
    # sequential walks over 32 chunks, so 2.5 CTAs/SM leaves the schedulers
    # idle waiting on the loop-carried [K,BV] state.
    # Merging them into one launch over a doubled i_n axis puts ~5 CTAs of
    # INDEPENDENT work on each SM.  The two branches have disjoint live sets,
    # so the register budget is the max of the two, not the sum, and neither
    # branch's stalls can block the other's issue.
    i_v = tl.program_id(0)
    i_h = tl.program_id(1)
    i_nn = tl.program_id(2)
    rev = i_nn >= tl.num_programs(2) // 2
    i_n = i_nn % (tl.num_programs(2) // 2)
    c0 = tl.load(DF + i_n)
    nc = tl.load(DN + i_n)
    r = tl.arange(0, BT)
    ak = tl.arange(0, K)
    bv = i_v * BV + tl.arange(0, BV)
    if rev:
        dS = tl.zeros([K, BV], dtype=tl.float32)
        for i in range(nc):
            ic = c0 + nc - 1 - i
            t0 = tl.load(CS + ic)
            L = tl.load(CL + ic)
            mr = r < L
            ov = (t0 + r)[:, None] * (H * V) + i_h * V + bv[None, :]

            # dS -> DHT.  dSb dies as soon as the kg dot below consumes it.
            dSb = dS.to(tl.bfloat16)
            tl.store(DHT + (ic * H + i_h) * (K * V) + ak[:, None] * V + bv[None, :], dSb)

            # dvn = dot(aq^T, do_) + dot(kg, dSb)   -- serialized, aq dies early,
            # kg is not loaded until the aq term is already folded down to [64,BV].
            do_ = tl.load(DO + ov, mask=mr[:, None], other=0.0)
            aq = tl.load(AQ + ((ic * H + i_h) * BT + r)[:, None] * BT + r[None, :])
            dvn = tl.dot(tl.trans(aq), do_)
            ok = (t0 + r)[:, None] * (H * K) + i_h * K + ak[None, :]
            kg = tl.load(KG + ok, mask=mr[:, None], other=0.0)
            dvn = dvn + tl.dot(kg, dSb)
            dvnb = dvn.to(tl.bfloat16)
            tl.store(DVN + ov, dvnb, mask=mr[:, None])

            # dS = ((dS*exp(gl)) + dot(qg^T,do_)) - dot(w^T,dvnb)
            # exactly the parse order of the original one-liner, but only one
            # [K,BV] dot temporary is live at a time and qg/w are loaded late.
            gl = tl.load(GL + (ic * H + i_h) * K + ak)
            dS = dS * tl.math.exp2(gl)[:, None]
            qg = tl.load(QG + ok, mask=mr[:, None], other=0.0)
            dS = dS + tl.dot(tl.trans(qg), do_)
            w = tl.load(W + ok, mask=mr[:, None], other=0.0)
            dS = dS - tl.dot(tl.trans(w), dvnb)
    else:
        S = tl.zeros([K, BV], dtype=tl.float32)
        for i in range(nc):
            ic = c0 + i
            t0 = tl.load(CS + ic)
            L = tl.load(CL + ic)
            mr = r < L
            hoff = (ic * H + i_h) * (K * V) + ak[:, None] * V + bv[None, :]
            Sb = S.to(tl.bfloat16)
            tl.store(HS + hoff, Sb)
            ok = (t0 + r)[:, None] * (H * K) + i_h * K + ak[None, :]
            ov = (t0 + r)[:, None] * (H * V) + i_h * V + bv[None, :]
            w = tl.load(W + ok, mask=mr[:, None], other=0.0)
            # u = (PM * beta_col) @ v, rebuilt here instead of read from a
            # materialised U tensor.  tA/tE/tB in _prep are exactly the three
            # blocks of PM scaled by beta along the COLUMN axis, so the single
            # [BT,BT] product below reproduces the block-triangular form they
            # assembled (the zero block of PM contributes nothing).
            om = ((ic * H + i_h) * BT + r)[:, None] * BT + r[None, :]
            bcol = tl.load(BETA + (t0 + r) * H + i_h, mask=mr, other=0.0)
            tw = (tl.load(PM + om).to(tl.float32) * bcol[None, :]).to(tl.bfloat16)
            vx = tl.load(VV + ov, mask=mr[:, None], other=0.0)
            u = tl.dot(tw, vx, out_dtype=tl.float32)
            kg = tl.load(KG + ok, mask=mr[:, None], other=0.0)
            vn = u - tl.dot(w, Sb)
            vnb = vn.to(tl.bfloat16)
            tl.store(VN + ov, vnb, mask=mr[:, None])
            gl = tl.load(GL + (ic * H + i_h) * K + ak)
            S = S * tl.math.exp2(gl)[:, None] + tl.dot(tl.trans(kg), vnb)


@triton.jit
def _bwd_local(
    DO,
    VN,
    DVN,
    VV,
    KE,
    PM,
    GM,
    BETA,
    HS,
    DHT,
    CS,
    CL,
    DV,
    DBETA,
    DQG,
    DKG,
    DKE,
    DAQ,
    DGM,
    DGL,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
):
    # Chunk-major here, unlike _prep / _bwd_qkg.  This kernel's dominant
    # traffic is HS and DHT, laid out [NT, H, K, V]: a CTA reads one 32 KB
    # (K,V) plane, and consecutive i_t at fixed i_h stride by H*K*V = 3 MB
    # while consecutive i_h are adjacent.  But the (K,V) plane is already
    # far larger than a DRAM page, so head-major buys nothing there, and it
    # costs the chunk-local reuse of the [T,H,V] tiles.  Measured: head-major
    # is a 3-5% regression on this kernel while it is a win on the other two.
    i_t = tl.program_id(0)
    i_h = tl.program_id(1)
    t0 = tl.load(CS + i_t)
    L = tl.load(CL + i_t)
    r = tl.arange(0, BT)
    mr = r < L
    ak = tl.arange(0, K)
    ok = (t0 + r)[:, None] * (H * K) + i_h * K + ak[None, :]
    om = ((i_t * H + i_h) * BT + r)[:, None] * BT + r[None, :]
    b = tl.load(BETA + (t0 + r) * H + i_h, mask=mr, other=0.0)
    # PM is loaded ONCE via the coalesced `om` indexing (cols stride 1);
    # the transposed orientation is derived in registers.
    Pm = tl.load(PM + om).to(tl.float32)
    Gm = tl.load(GM + om).to(tl.float32)
    twyt = (tl.trans(Pm) * b[:, None]).to(tl.bfloat16)

    dqg = tl.zeros([BT, K], dtype=tl.float32)
    dw = tl.zeros([BT, K], dtype=tl.float32)
    dkg = tl.zeros([BT, K], dtype=tl.float32)
    dgl = tl.zeros([K], dtype=tl.float32)
    dAq = tl.zeros([BT, BT], dtype=tl.float32)
    dtwy = tl.zeros([BT, BT], dtype=tl.float32)
    for iv in range(V // BV):
        bv = iv * BV + tl.arange(0, BV)
        ov = (t0 + r)[:, None] * (H * V) + i_h * V + bv[None, :]
        hoff = (i_t * H + i_h) * (K * V) + ak[:, None] * V + bv[None, :]
        h_ = tl.load(HS + hoff)
        dh_ = tl.load(DHT + hoff)
        do_ = tl.load(DO + ov, mask=mr[:, None], other=0.0)
        vn_ = tl.load(VN + ov, mask=mr[:, None], other=0.0)
        dvn_ = tl.load(DVN + ov, mask=mr[:, None], other=0.0)
        v_ = tl.load(VV + ov, mask=mr[:, None], other=0.0)
        ht = tl.trans(h_)
        dqg += tl.dot(do_, ht)
        dw -= tl.dot(dvn_, ht)
        dkg += tl.dot(vn_, tl.trans(dh_))
        dgl += tl.sum(h_.to(tl.float32) * dh_.to(tl.float32), 1)
        dAq += tl.dot(do_, tl.trans(vn_))
        dtwy += tl.dot(dvn_, tl.trans(v_))
        tl.store(DV + ov, tl.dot(twyt, dvn_).to(tl.bfloat16), mask=mr[:, None])

    ke = tl.load(KE + ok, mask=mr[:, None], other=0.0)
    dwb = dw.to(tl.bfloat16)
    dtwy += tl.dot(dwb, tl.trans(ke))
    dke = tl.dot(twyt, dwb)
    dAq = tl.where(r[:, None] >= r[None, :], dAq, 0.0)
    dP = dtwy * b[None, :]
    dbj = tl.sum(dtwy * Pm, 0)
    tmp = tl.dot(tl.trans(Pm), dP, input_precision="tf32")
    dM = -tl.dot(tmp, Pm, input_precision="tf32")
    dM = tl.where(r[:, None] > r[None, :], dM, 0.0)
    dbc = tl.sum(dM * Gm, 1)
    dG = dM * b[:, None]

    tl.store(DBETA + (t0 + r) * H + i_h, dbj + dbc, mask=mr)
    tl.store(DQG + ok, dqg.to(tl.bfloat16), mask=mr[:, None])
    tl.store(DKG + ok, dkg.to(tl.bfloat16), mask=mr[:, None])
    tl.store(DKE + ok, dke.to(tl.bfloat16), mask=mr[:, None])
    tl.store(DAQ + om, dAq.to(tl.bfloat16))
    tl.store(DGM + om, dG.to(tl.bfloat16))
    tl.store(DGL + (i_t * H + i_h) * K + ak, dgl)


@triton.jit
def _bwd_qkg(
    Q,
    KK,
    GC,
    GB,
    GLF,
    RQ,
    RK,
    DQG,
    DKG,
    DKE,
    DAQ,
    DGM,
    DGL,
    CS,
    CL,
    DQ,
    DK,
    DG,
    scale,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    QSCALE: tl.constexpr,
):
    # kb is the FASTEST-varying grid dim so the two CTAs that share the same
    # (chunk, head) -- and therefore the same DGM / DAQ [64,64] tiles and the
    # same RQ / RK row vectors -- are launched adjacently and hit in L2
    # instead of re-reading 207 MB from HBM.  It also makes the two K-halves
    # of each q/k row, which are contiguous in the [T,H,K] layout, land on
    # neighbouring CTAs.
    kb = tl.program_id(0)
    i_h = tl.program_id(1)
    i_t = tl.program_id(2)
    t0 = tl.load(CS + i_t)
    L = tl.load(CL + i_t)
    r = tl.arange(0, BT)
    mr = r < L
    om = ((i_t * H + i_h) * BT + r)[:, None] * BT + r[None, :]
    zb = tl.zeros([BT, BT], dtype=tl.bfloat16)
    # Pre-mask by the triangular part only.  The row-block split (m0/m1) is
    # applied inside the a-loop, so two tiles are pinned here instead of four.
    # No re-mask: _bwd_local stores dAq already masked by (r >= c) and dG by
    # (r > c), so these two full-tile selects were provably dead.  Dropping
    # them removes two [BT,BT] selects per CTA from the kernel that owns ~40%
    # of the pipeline and NCU shows spilling at 100% overhead.
    dGk = tl.load(DGM + om)
    dAq = tl.load(DAQ + om)
    rq = tl.load(RQ + (t0 + r) * H + i_h, mask=mr, other=1.0)
    rk = tl.load(RK + (t0 + r) * H + i_h, mask=mr, other=1.0)
    c0 = r < BC
    c1 = r < 2 * BC
    if True:
        ds = kb * BK + tl.arange(0, BK)
        ok = (t0 + r)[:, None] * (H * K) + i_h * K + ds[None, :]
        q = tl.load(Q + ok, mask=mr[:, None], other=0.0).to(tl.float32)
        kx = tl.load(KK + ok, mask=mr[:, None], other=0.0).to(tl.float32)
        qn = q * rq[:, None]
        kn = kx * rk[:, None]
        qs = qn * scale
        # Reconstruct gc = bs0 + dequant(int16).  bs0 is exact fp32, so the
        # only error is the bounded 1/256 quantisation of the difference.
        gq0 = tl.load(GB + ((i_t * H + i_h) * (BT // BC) + 0) * K + ds)
        gc = gq0[None, :] + tl.load(GC + ok, mask=mr[:, None], other=0).to(tl.float32) * (1.0 / QSCALE)
        eg = tl.math.exp2(gc)
        gl = tl.load(GLF + (i_t * H + i_h) * K + ds)
        dqg = tl.load(DQG + ok, mask=mr[:, None], other=0.0).to(tl.float32)
        dkgv = tl.load(DKG + ok, mask=mr[:, None], other=0.0).to(tl.float32)
        dke = tl.load(DKE + ok, mask=mr[:, None], other=0.0).to(tl.float32)
        dglin = tl.load(DGL + (i_t * H + i_h) * K + ds)
        # The three bf16 tiles QG = qs*eg, KE = kn*eg, KG = kn*exp2(gl-gc)
        # that used to be loaded here (604 MB of reads, three extra [BT,BK]
        # fp32 tiles live at the same time as kn/qs/eg) are NOT needed: every
        # place they appeared factors through kn and qs, which are already in
        # registers.  With t1 = dke*eg and t2 = dkgv*egl_rel,
        #   dkn      = t1 + t2
        #   dqs      = dqg*eg
        #   KE*dke   = kn*t1,   KG*dkgv = kn*t2,   QG*dqg = qs*dqs
        #   dgc      = kn*(t1 - t2) + qs*dqs
        # so the two gate-scaled products collapse into ONE multiply by kn.
        # This is also strictly more accurate than before (fp32 throughout
        # instead of a bf16 round-trip through the stored tiles).
        egl_rel = tl.math.exp2(gl[None, :] - gc)
        t1 = dke * eg
        t2 = dkgv * egl_rel
        dkn = t1 + t2
        dqs = dqg * eg
        dgc = kn * (t1 - t2) + qs * dqs
        dgl_tot = dglin * tl.math.exp2(gl) + tl.sum(kn * t2, 0)

        gb0 = tl.load(GB + ((i_t * H + i_h) * (BT // BC) + 0) * K + ds)
        gb1 = tl.load(GB + ((i_t * H + i_h) * (BT // BC) + 1) * K + ds)
        # Stay in LOG space.  d = gc - bsA is the only gate tile the a-loop
        # needs; gc itself is dead after this point, so keeping d instead of
        # {gc, er} retires one fp32 [BT,BK] tile for zero extra bytes.
        # d is nearly free: gc = gb0 + dqz (above), and bsA is gb0 for rows
        # < BC and gb1 otherwise, so d = dqz + (0 or gb0-gb1) -- a [K]-vector
        # row-select, not a full-tile subtract.
        # Both exponentials are then derived from d at their use sites with
        # the clamp intact:
        #     er    = exp2(clamp(d))
        #     ec(a) = exp2(clamp((gb_a - bsA) - d))
        # This is NOT the unsafe 1/er reconstruction I tried earlier: that
        # divided by an already-clamped (saturated) value and produced inf.
        # Here nothing is divided and every exponent is clamped before use,
        # so the error is one fp32 rounding on an argument bounded by CLAMP,
        # ~1e-5 relative -- far below the 0.27% the int16 gate quantisation
        # already injects.
        dlt = tl.where(r[:, None] < BC, 0.0, (gb0 - gb1)[None, :]) + (gc - gb0[None, :])
        er = tl.math.exp2(tl.minimum(tl.maximum(dlt, -CLAMP), CLAMP))
        Ka = (kn * er).to(tl.bfloat16)
        Qa = (qs * er).to(tl.bfloat16)
        for a in tl.static_range(2):
            cm = c0 if a == 0 else c1
            # gb_a - bsA is a [K] two-way row-select, ~0.5 reg, replacing the
            # full-tile (bs - gc) subtract that pinned gc across the loop.
            # gdf = gb_a - bsA, where bsA is gb0 on rows < BC and gb1 above.
            # a=0 -> gb0 - bsA = {0 on rows<BC, gb0-gb1 above}
            # a=1 -> gb1 - bsA = {gb1-gb0 on rows<BC, 0 above}
            gdf = (
                tl.where(r[:, None] < BC, 0.0, (gb0 - gb1)[None, :])
                if a == 0
                else tl.where(r[:, None] < BC, (gb1 - gb0)[None, :], 0.0)
            )
            ma = (r[:, None] >= a * BC) & (r[:, None] < (a + 1) * BC)
            dGa = tl.where(ma, dGk, zb)
            dAa = tl.where(ma, dAq, zb)
            ec = tl.where(cm[:, None], tl.math.exp2(tl.minimum(tl.maximum(gdf - dlt, -CLAMP), CLAMP)), 0.0)
            Kb = (kn * ec).to(tl.bfloat16)
            # consume each product immediately so only one [BT,BK] temp is live
            X = tl.dot(dGa, Kb)
            dkn += er * X
            dgc += Ka.to(tl.float32) * X
            X = tl.dot(dAa, Kb)
            dqs += er * X
            dgc += Qa.to(tl.float32) * X
            # serialized: `+` is left-assoc, so this is the same association
            X = tl.dot(tl.trans(dGa), Ka)
            X = X + tl.dot(tl.trans(dAa), Qa)
            dkn += ec * X
            dgc -= Kb.to(tl.float32) * X
        dgc += tl.where(r[:, None] == (BT - 1), dgl_tot[None, :], 0.0)

        # `ok` reborn (see d3); dead across er / Ka / Qa / the whole a-loop
        oks = (t0 + r)[:, None] * (H * K) + i_h * K + ds[None, :]
        # DK / DQ first so dkn and dqs die before the cumsum tile is built.
        # Same three stores as before, distinct non-aliasing tensors, only
        # the order differs -- no extra memory traffic.
        tl.store(DK + oks, dkn.to(tl.bfloat16), mask=mr[:, None])
        tl.store(DQ + oks, (dqs * scale).to(tl.bfloat16), mask=mr[:, None])
        # Reverse-inclusive cumsum out[i,:] = sum_{j >= i} dgc[j,:] is exactly
        # UT @ dgc with UT[i,j] = (j >= i), i.e. a [BT,BT]x[BT,BK] matmul that
        # the tensor cores do in a couple of MMA issues.  tl.cumsum lowers to
        # a log2(BT)=6-step shuffle/select scan over the whole fp32 tile plus
        # a separate full reduction for `tot`, all on the slow path of the
        # kernel NCU shows spilling at 100% overhead.  Splitting dgc into its
        # bf16 high part and the fp32 residual keeps the sum accurate: the
        # residual carries the bits bf16 drops, so the pair reproduces fp32
        # accumulation to well within the 0.02 tolerance on dg.
        dgh = dgc.to(tl.bfloat16)
        dgr = (dgc - dgh.to(tl.float32)).to(tl.bfloat16)
        UT = (r[:, None] <= r[None, :]).to(tl.bfloat16)
        tl.store(
            DG + oks, tl.dot(UT, dgh, out_dtype=tl.float32) + tl.dot(UT, dgr, out_dtype=tl.float32), mask=mr[:, None]
        )


@triton.jit
def _l2n(Q, KK, DQ, DK, RQ, RK, N, BR: tl.constexpr, K: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    m = rows < N
    off = rows[:, None] * K + tl.arange(0, K)[None, :]
    q = tl.load(Q + off, mask=m[:, None], other=0.0).to(tl.float32)
    kx = tl.load(KK + off, mask=m[:, None], other=0.0).to(tl.float32)
    dqn = tl.load(DQ + off, mask=m[:, None], other=0.0).to(tl.float32)
    dkn = tl.load(DK + off, mask=m[:, None], other=0.0).to(tl.float32)
    # rstd was already computed by _prep; reload the [N] vectors (0.06 MB)
    # instead of redoing two full [BR,128] reductions per row block.
    rq = tl.load(RQ + rows, mask=m, other=1.0)
    rk = tl.load(RK + rows, mask=m, other=1.0)
    qn = q * rq[:, None]
    kn = kx * rk[:, None]
    dqo = rq[:, None] * (dqn - qn * tl.sum(qn * dqn, 1)[:, None])
    dko = rk[:, None] * (dkn - kn * tl.sum(kn * dkn, 1)[:, None])
    tl.store(DQ + off, dqo.to(tl.bfloat16), mask=m[:, None])
    tl.store(DK + off, dko.to(tl.bfloat16), mask=m[:, None])


def run(q, k, v, g, beta, cu_seqlens, do, dq, dk, dv, dg, dbeta, cu_seqlens_cpu=None):
    """Backward of the K3 chunk_kda call (destination-passing): fills dq, dk, dv (bf16) and dg, dbeta (fp32).

    q, k: [1, T, H, 128] bf16; v: [1, T, H, 128] bf16; g: [1, T, H, 128] fp32 log gate in (-5, 0); beta: [1, T, H] fp32;
    cu_seqlens: [N + 1] int32 document offsets; do: [1, T, H, 128] bf16 gradient of the output. Every forward
    intermediate is recomputed here. The per-chunk metadata needs the document offsets on the host: pass them as
    ``cu_seqlens_cpu`` (a sequence of ints, e.g. derived from a dense batch's shape) to avoid the device->host sync
    that ``cu_seqlens.tolist()`` otherwise costs -- the same contract as FLA's own ``cu_seqlens_cpu``.
    """
    if not HAVE_TRITON:
        raise ImportError(MISSING_TRITON_MSG)
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = 64
    BC = 32
    dev = q.device
    scale = K**-0.5

    # cu_seqlens.tolist() is a device->host copy, which forces a full stream
    # synchronisation before ANY kernel of this backward can be enqueued.  On
    # this shape nsys shows the six kernels summing to ~5.45 ms while the
    # wall-clock is ~5.77 ms; the sync plus the H2D upload of the derived
    # metadata is most of that gap.  Pinned staging removes the H2D from the
    # critical path: the copy into pinned memory is a plain memcpy and the
    # single non_blocking upload overlaps with the _prep launch.
    cu = list(cu_seqlens_cpu) if cu_seqlens_cpu is not None else cu_seqlens.tolist()
    starts = []
    lens = []
    dfirst = []
    dcnt = []
    for i in range(len(cu) - 1):
        s, e = int(cu[i]), int(cu[i + 1])
        dfirst.append(len(starts))
        if e <= s:
            dcnt.append(0)
            continue
        n = (e - s + BT - 1) // BT
        dcnt.append(n)
        for c in range(n):
            starts.append(s + c * BT)
            lens.append(min(BT, e - s - c * BT))
    NT = len(starts)
    ND = len(dfirst)
    host = torch.empty(2 * NT + 2 * ND, dtype=torch.int32, pin_memory=torch.cuda.is_available())
    host.copy_(torch.frombuffer(_array("i", starts + lens + dfirst + dcnt), dtype=torch.int32))
    meta = host.to(dev, non_blocking=True)
    CS = meta[0:NT]
    CL = meta[NT : 2 * NT]
    DF = meta[2 * NT : 2 * NT + ND]
    DN = meta[2 * NT + ND : 2 * NT + 2 * ND]

    bf = torch.bfloat16
    tk = (T, H, K)
    tv = (T, H, V)
    QG = torch.empty(tk, dtype=bf, device=dev)
    KE = torch.empty(tk, dtype=bf, device=dev)
    KG = torch.empty(tk, dtype=bf, device=dev)
    W = torch.empty(tk, dtype=bf, device=dev)
    PM = torch.empty((NT, H, BT, BT), dtype=bf, device=dev)
    GM = torch.empty((NT, H, BT, BT), dtype=bf, device=dev)
    AQ = torch.empty((NT, H, BT, BT), dtype=bf, device=dev)
    GL = torch.empty((NT, H, K), dtype=torch.float32, device=dev)
    RQ = torch.empty((T, H), dtype=torch.float32, device=dev)
    RK = torch.empty((T, H), dtype=torch.float32, device=dev)
    GC = torch.empty(tk, dtype=torch.int16, device=dev)
    GB = torch.empty((NT, H, BT // BC, K), dtype=torch.float32, device=dev)

    _prep[(H, NT)](
        q,
        k,
        g,
        beta,
        CS,
        CL,
        QG,
        KE,
        KG,
        W,
        PM,
        GM,
        AQ,
        GL,
        RQ,
        RK,
        GC,
        GB,
        scale,
        H,
        K,
        V,
        BT,
        BC,
        128.0,
        num_warps=4,
        num_stages=2,
    )

    HS = torch.empty((NT, H, K, V), dtype=bf, device=dev)
    VN = torch.empty(tv, dtype=bf, device=dev)
    DHT = torch.empty((NT, H, K, V), dtype=bf, device=dev)
    DVN = torch.empty(tv, dtype=bf, device=dev)
    BV = 64
    # BVS=128 covers V in a single CTA.  At BV=64 two v-CTAs per (head,doc)
    # each re-read the V-independent [BT,K] tiles -> ~400 MB of duplication.
    BVS = 128
    # ONE launch for both state scans.  They are independent -- the reverse
    # scan reads neither HS nor VN -- but each alone is a grid of only
    # (1, 96, 4) = 384 CTAs, i.e. 2.5 per SM on 152 SMs, running a sequential
    # 32-chunk walk whose loop-carried [K,BV] state serialises every issue.
    # Fusing them over a doubled document axis puts ~5 CTAs of INDEPENDENT
    # work on each SM, so one branch's memory stalls are covered by the
    # other's math.  The two branches have disjoint live sets, so the
    # register budget is the max of the two rather than the sum.
    _scan[(V // BVS, H, 2 * ND)](
        W,
        PM,
        beta,
        v,
        KG,
        GL,
        HS,
        VN,
        QG,
        AQ,
        do,
        DHT,
        DVN,
        CS,
        CL,
        DF,
        DN,
        H,
        K,
        V,
        BT,
        BVS,
        num_warps=8,
        num_stages=3,
    )

    DQG = torch.empty(tk, dtype=bf, device=dev)
    DKG = torch.empty(tk, dtype=bf, device=dev)
    DKE = torch.empty(tk, dtype=bf, device=dev)
    DAQ = torch.empty((NT, H, BT, BT), dtype=bf, device=dev)
    DGM = torch.empty((NT, H, BT, BT), dtype=bf, device=dev)
    DGL = torch.empty((NT, H, K), dtype=torch.float32, device=dev)
    _bwd_local[(NT, H)](
        do,
        VN,
        DVN,
        v,
        KE,
        PM,
        GM,
        beta,
        HS,
        DHT,
        CS,
        CL,
        dv,
        dbeta,
        DQG,
        DKG,
        DKE,
        DAQ,
        DGM,
        DGL,
        H,
        K,
        V,
        BT,
        BV,
        num_warps=8,
        num_stages=3,
    )

    BK = 64
    _bwd_qkg[(K // BK, H, NT)](
        q,
        k,
        GC,
        GB,
        GL,
        RQ,
        RK,
        DQG,
        DKG,
        DKE,
        DAQ,
        DGM,
        DGL,
        CS,
        CL,
        dq,
        dk,
        dg,
        scale,
        H,
        K,
        BT,
        BC,
        BK,
        128.0,
        num_warps=8,
        num_stages=1,
    )
    N = T * H
    BR = 16
    _l2n[(triton.cdiv(N, BR),)](q, k, dq, dk, RQ, RK, N, BR, K, num_warps=8, num_stages=2)

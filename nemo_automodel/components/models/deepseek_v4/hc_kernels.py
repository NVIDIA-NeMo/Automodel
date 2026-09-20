# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Fused RMSNorm + skinny projection for the DeepSeek-V4 mHC mixer.

The mixer computes ``rms_norm(streams) @ fn.T`` where ``streams`` is
``[tokens, hc_mult * hidden]`` and ``fn`` has only ``(2 + hc_mult) * hc_mult``
rows (24 for hc_mult=4).  That shape defeats cuBLAS: with a single 32-column
tile the grid is ``tokens / 32`` blocks, so a 12-layer step spent 59.5 ms in
``cutlass_80_wmma_tensorop_bf16_32x32_align2`` moving 6.4 GB, i.e. 108 GB/s or
about 3% of an H100's bandwidth, for 0.3% of the model's FLOPs.

These kernels split the reduction over K instead, so the grid is
``(tokens / BLOCK_M) * SPLIT_K`` and the work is bandwidth-bound as it should
be.  The normalisation statistic and the projection are computed in fp32; the
activations are read in their native dtype.  Enabled by
``BackendConfig.hc_proj_kernel``.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - exercised only without Triton
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _rstd_kernel(X, RSTD, M, K, eps, BLOCK_K: tl.constexpr):
        """Per-row reciprocal RMS: ``1 / sqrt(mean(x^2) + eps)``."""
        row = tl.program_id(0)
        acc = tl.zeros([BLOCK_K], tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(X + row * K + offs, mask=offs < K, other=0.0).to(tl.float32)
            acc += x * x
        mean = tl.sum(acc, axis=0) / K
        tl.store(RSTD + row, 1.0 / tl.sqrt(mean + eps))

    @triton.jit
    def _proj_fwd_kernel(
        X, W, RSTD, OUT, M, K, N: tl.constexpr, N_POW2: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        """``OUT[m, n] += sum_k x[m, k] * rstd[m] * W[n, k]`` over one K slice."""
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        split_k = tl.num_programs(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        rstd = tl.load(RSTD + rows, mask=row_mask, other=0.0)
        ns = tl.arange(0, N_POW2)
        n_mask = ns < N
        acc = tl.zeros([BLOCK_M, N_POW2], tl.float32)
        k_start = pid_k * BLOCK_K
        for k0 in range(k_start, K, split_k * BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs < K
            x = tl.load(X + rows[:, None] * K + offs[None, :], mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            w = tl.load(W + ns[:, None] * K + offs[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            acc += tl.dot(x.to(tl.float32), tl.trans(w.to(tl.float32)))
        acc = acc * rstd[:, None]
        tl.atomic_add(OUT + rows[:, None] * N + ns[None, :], acc, mask=row_mask[:, None] & n_mask[None, :])

    @triton.jit
    def _dw_kernel(
        X, DOUT, RSTD, DW, M, K, N: tl.constexpr, N_POW2: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        """``dW[n, k] += sum_m dout[m, n] * x[m, k] * rstd[m]`` over one M slice."""
        pid_k = tl.program_id(0)
        pid_m = tl.program_id(1)
        split_m = tl.num_programs(1)
        offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs < K
        ns = tl.arange(0, N_POW2)
        n_mask = ns < N
        acc = tl.zeros([N_POW2, BLOCK_K], tl.float32)
        for m0 in range(pid_m * BLOCK_M, M, split_m * BLOCK_M):
            rows = m0 + tl.arange(0, BLOCK_M)
            row_mask = rows < M
            rstd = tl.load(RSTD + rows, mask=row_mask, other=0.0)
            x = tl.load(X + rows[:, None] * K + offs[None, :], mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            d = tl.load(DOUT + rows[:, None] * N + ns[None, :], mask=row_mask[:, None] & n_mask[None, :], other=0.0)
            acc += tl.dot(tl.trans(d.to(tl.float32)), x.to(tl.float32) * rstd[:, None])
        tl.atomic_add(DW + ns[:, None] * K + offs[None, :], acc, mask=n_mask[:, None] & k_mask[None, :])

    @triton.jit
    def _s_kernel(
        X, DOUT, W, S, M, K, N: tl.constexpr, N_POW2: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr
    ):
        """Row-wise ``s[m] = sum_k (dout @ W)[m, k] * x[m, k]``, the RMSNorm backward statistic.

        Blocked over rows so ``W`` is read once per row block rather than once per row.
        """
        pid_m = tl.program_id(0)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        ns = tl.arange(0, N_POW2)
        n_mask = ns < N
        d = tl.load(DOUT + rows[:, None] * N + ns[None, :], mask=row_mask[:, None] & n_mask[None, :], other=0.0)
        acc = tl.zeros([BLOCK_M], tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs < K
            w = tl.load(W + ns[:, None] * K + offs[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            dxn = tl.dot(d.to(tl.float32), w.to(tl.float32))
            x = tl.load(X + rows[:, None] * K + offs[None, :], mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            acc += tl.sum(dxn * x.to(tl.float32), axis=1)
        tl.store(S + rows, acc, mask=row_mask)

    @triton.jit
    def _dx_kernel(
        X,
        DOUT,
        W,
        RSTD,
        S,
        DX,
        M,
        K,
        N: tl.constexpr,
        N_POW2: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """``dx = rstd * dxn - x * s * rstd^3 / K`` with ``dxn = dout @ W`` recomputed per block."""
        pid_m = tl.program_id(0)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        ns = tl.arange(0, N_POW2)
        n_mask = ns < N
        d = tl.load(DOUT + rows[:, None] * N + ns[None, :], mask=row_mask[:, None] & n_mask[None, :], other=0.0)
        rstd = tl.load(RSTD + rows, mask=row_mask, other=0.0)
        s = tl.load(S + rows, mask=row_mask, other=0.0)
        coeff = s * rstd * rstd * rstd / K
        for k0 in range(0, K, BLOCK_K):
            offs = k0 + tl.arange(0, BLOCK_K)
            k_mask = offs < K
            w = tl.load(W + ns[:, None] * K + offs[None, :], mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            dxn = tl.dot(d.to(tl.float32), w.to(tl.float32))
            x = tl.load(X + rows[:, None] * K + offs[None, :], mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            out = rstd[:, None] * dxn - x.to(tl.float32) * coeff[:, None]
            tl.store(DX + rows[:, None] * K + offs[None, :], out, mask=row_mask[:, None] & k_mask[None, :])


class _FusedHCProjection(torch.autograd.Function):
    """``rms_norm(x) @ w.T`` for a very skinny ``w``; see the module docstring."""

    @staticmethod
    def forward(ctx, x, w, eps):
        m, k = x.shape
        n = w.shape[0]
        rstd = torch.empty(m, dtype=torch.float32, device=x.device)
        out = torch.zeros(m, n, dtype=torch.float32, device=x.device)
        # Block sizes are capped so the fp32 tiles fit the 99 KB of shared memory an Ada SM
        # offers; Hopper has more but gains nothing from larger tiles on a bandwidth-bound op.
        red_k = min(1024, triton.next_power_of_2(k))
        dot_k = min(256, triton.next_power_of_2(k))
        _rstd_kernel[(m,)](x, rstd, m, k, eps, BLOCK_K=red_k, num_warps=8)
        block_m = 32
        split_k = max(1, min(8, k // dot_k))
        _proj_fwd_kernel[(triton.cdiv(m, block_m), split_k)](
            x,
            w,
            rstd,
            out,
            m,
            k,
            N=n,
            N_POW2=triton.next_power_of_2(n),
            BLOCK_M=block_m,
            BLOCK_K=dot_k,
            num_warps=8,
            num_stages=2,
        )
        ctx.save_for_backward(x, w, rstd)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, dout):
        x, w, rstd = ctx.saved_tensors
        m, k = x.shape
        n = w.shape[0]
        dout = dout.contiguous().float()
        n_pow2 = triton.next_power_of_2(n)
        row_k = min(128, triton.next_power_of_2(k))
        dot_k = min(128, triton.next_power_of_2(k))
        s = torch.empty(m, dtype=torch.float32, device=x.device)
        bwd_m = 32
        grid_m = (triton.cdiv(m, bwd_m),)
        _s_kernel[grid_m](
            x, dout, w, s, m, k, N=n, N_POW2=n_pow2, BLOCK_M=bwd_m, BLOCK_K=row_k, num_warps=8, num_stages=2
        )
        dx = torch.empty_like(x, dtype=torch.float32)
        _dx_kernel[grid_m](
            x, dout, w, rstd, s, dx, m, k, N=n, N_POW2=n_pow2, BLOCK_M=bwd_m, BLOCK_K=row_k, num_warps=8, num_stages=2
        )
        dw = torch.zeros_like(w, dtype=torch.float32)
        block_m = 64
        split_m = max(1, min(8, m // block_m))
        _dw_kernel[(triton.cdiv(k, dot_k), split_m)](
            x,
            dout,
            rstd,
            dw,
            m,
            k,
            N=n,
            N_POW2=n_pow2,
            BLOCK_M=block_m,
            BLOCK_K=dot_k,
            num_warps=8,
            num_stages=2,
        )
        return dx.to(x.dtype), dw.to(w.dtype), None


def fused_hc_projection(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Fused ``rms_norm(x, eps) @ w.T``. Falls back to eager ops without Triton or on CPU."""
    if not _HAS_TRITON or not x.is_cuda:
        normed = torch.nn.functional.rms_norm(x.float(), (x.shape[-1],), eps=eps)
        return torch.nn.functional.linear(normed, w.float())
    lead, k = x.shape[:-1], x.shape[-1]
    out = _FusedHCProjection.apply(x.reshape(-1, k).contiguous(), w.contiguous(), eps)
    return out.view(*lead, w.shape[0])

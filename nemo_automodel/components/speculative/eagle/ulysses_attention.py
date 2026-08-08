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
#
# Upstream attribution:
#   The sequence<->head all-to-all is the DeepSpeed-Ulysses collective
#   (microsoft/DeepSpeed, Apache-2.0). The EAGLE-3 mixed causal-block-0 +
#   TTT-diagonal joint softmax and its hand-written backward mirror
#   ``ring_attention._CachedRingAttention`` in this package; the FlashAttention
#   kernel dispatch reuses ``ring_attention``'s dense wrappers and the varlen
#   private entry points already used by ``zigzag_ring_attention``.

"""Ulysses (all-to-all) context-parallel attention for the EAGLE-3 draft.

Context parallelism shards the sequence: each rank holds a contiguous ``S/cp``
slice of the tokens. Ulysses turns that sequence shard into a *head* shard for
block-0 attention with an all-to-all, so a rank sees the **full** sequence for a
subset of the heads:

    ``[B, S_local, H, D]``  --all-to-all-->  ``[B, S_full, H/cp, D]``

With the full sequence local, block-0 (``Q @ K_0^T``, causal over the whole
sequence) runs a single dense or packed ``varlen`` FlashAttention, so
``cu_seqlens`` document boundaries carry through and CP composes with sequence
packing. The output is all-to-all'd back to the sequence shard, and the
per-position EAGLE-3 TTT diagonals (blocks ``i >= 1``) merge into it via the
online-softmax identity, shard-local and comm-free.

This is a hand-written ``autograd.Function`` (like the ring): the backward runs
the FlashAttention backward against the **merged** joint-softmax out/lse (not the
block-0-only softmax), which is required for the block-0 ``q``/``k`` gradients to
be correct whenever any diagonal step is present. The all-to-all is done with the
raw collective in forward and backward (grads are threaded by hand).
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from nemo_automodel.components.speculative.eagle.ring_attention import (
    _fa_backward,
    _fa_forward,
    _merge_diag,
)
from nemo_automodel.shared.import_utils import safe_import

HAVE_FLASH_ATTN, _fa = safe_import("flash_attn.flash_attn_interface")


# --------------------------------------------------------------------------- #
# Raw sequence<->head all-to-all (no autograd; the Function threads grads itself)
# --------------------------------------------------------------------------- #
def _all_to_all_single(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Blocking ``all_to_all_single`` splitting/joining along dim 0 (equal splits)."""
    x = x.contiguous()
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=group)
    return out


def _gather_seq_scatter_heads(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Sequence-sharded -> head-sharded: ``[B, S_local, H, D] -> [B, S_full, H/uly, D]``.

    Args:
        x: FlashAttention-layout tensor ``[batch, seq_local, heads, head_dim]`` --
            this rank's contiguous sequence shard with all heads; ``heads`` must be
            divisible by the Ulysses degree ``uly`` (the ``group`` world size).

    Returns:
        Tensor ``[batch, seq_full, heads // uly, head_dim]`` with the full sequence
        (ranks concatenated in rank order, preserving global token order) and only
        this rank's slice of the heads. Returned unchanged when ``uly == 1``.
    """
    uly = dist.get_world_size(group)
    if uly == 1:
        return x
    b, s_local, h, d = x.shape
    if h % uly != 0:
        raise ValueError(f"Ulysses degree ({uly}) must divide the head count ({h}).")
    h_local = h // uly
    send = x.reshape(b, s_local, uly, h_local, d).permute(2, 0, 1, 3, 4)  # [uly, B, S_loc, Hl, D]
    recv = _all_to_all_single(send, group)  # recv[j] == source rank j's tokens for my head slice
    return recv.permute(1, 0, 2, 3, 4).reshape(b, uly * s_local, h_local, d)  # [B, S_full, Hl, D]


def _scatter_seq_gather_heads(x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Head-sharded -> sequence-sharded: ``[B, S_full, H/uly, D] -> [B, S_local, H, D]``.

    Inverse of :func:`_gather_seq_scatter_heads`.

    Args:
        x: Tensor ``[batch, seq_full, heads_local, head_dim]`` -- the full sequence
            with this rank's slice of the heads; ``seq_full`` must be divisible by
            the Ulysses degree ``uly``.

    Returns:
        Tensor ``[batch, seq_full // uly, heads_local * uly, head_dim]`` -- this
        rank's contiguous sequence shard with all heads (heads concatenated in rank
        order). Returned unchanged when ``uly == 1``.
    """
    uly = dist.get_world_size(group)
    if uly == 1:
        return x
    b, s_full, h_local, d = x.shape
    if s_full % uly != 0:
        raise ValueError(f"Ulysses degree ({uly}) must divide the gathered sequence length ({s_full}).")
    s_local = s_full // uly
    send = x.reshape(b, uly, s_local, h_local, d).permute(1, 0, 2, 3, 4)  # [uly, B, S_loc, Hl, D]
    recv = _all_to_all_single(send, group)  # recv[j] == source rank j's head slice for my seq chunk
    return recv.permute(1, 2, 0, 3, 4).reshape(b, s_local, uly * h_local, d)  # [B, S_local, H, D]


def _gather_lse(lse: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Sequence-sharded -> head-sharded log-sum-exp: ``[B, H, S_local] -> [B, H/uly, S_full]``."""
    x = lse.transpose(1, 2).unsqueeze(-1)  # [B, S_local, H, 1]
    return _gather_seq_scatter_heads(x, group).squeeze(-1).transpose(1, 2)  # [B, H/uly, S_full]


def _scatter_lse(lse: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    """Head-sharded -> sequence-sharded log-sum-exp: ``[B, H/uly, S_full] -> [B, H, S_local]``."""
    x = lse.transpose(1, 2).unsqueeze(-1)  # [B, S_full, H/uly, 1]
    return _scatter_seq_gather_heads(x, group).squeeze(-1).transpose(1, 2)  # [B, H, S_local]


# --------------------------------------------------------------------------- #
# Block-0 FlashAttention over the gathered full sequence (dense or packed varlen)
# --------------------------------------------------------------------------- #
def _block0_forward(
    q_g: torch.Tensor,
    k_g: torch.Tensor,
    v_g: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal FlashAttention over the gathered sequence.

    Args:
        q_g, k_g, v_g: Gathered ``[batch, seq_full, heads_local, head_dim]``.
        scale: Softmax scale.
        cu_seqlens: GLOBAL packed-document boundaries ``[num_docs + 1]`` (int32) for
            the varlen path, or ``None`` for dense causal over the whole sequence.
        max_seqlen: Longest document length for the varlen path.

    Returns:
        ``(out, lse)`` with ``out`` ``[batch, seq_full, heads_local, head_dim]`` and
        ``lse`` ``[batch, heads_local, seq_full]`` (head-major).
    """
    if cu_seqlens is None:
        return _fa_forward(q_g, k_g, v_g, scale, causal=True)  # out [B,Sf,Hl,D], lse [B,Hl,Sf]
    b, s_full, hl, d = q_g.shape
    q_f = q_g.reshape(b * s_full, hl, d)
    k_f = k_g.reshape(b * s_full, hl, d)
    v_f = v_g.reshape(b * s_full, hl, d)
    out = _fa._flash_attn_varlen_forward(
        q_f, k_f, v_f, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, 0.0, scale, True, -1, -1, 0.0, None, False
    )
    out_flat, lse_flat = out[0], out[1]  # [b*Sf, Hl, D], [Hl, b*Sf]
    lse = lse_flat.reshape(hl, b, s_full).permute(1, 0, 2)  # [B, Hl, Sf]
    return out_flat.reshape(b, s_full, hl, d), lse


def _block0_backward(
    dout_g: torch.Tensor,
    q_g: torch.Tensor,
    k_g: torch.Tensor,
    v_g: torch.Tensor,
    out_g: torch.Tensor,
    lse_g: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    max_seqlen: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlashAttention backward over the gathered sequence, using the MERGED out/lse.

    Args:
        dout_g: Upstream grad, ``[batch, seq_full, heads_local, head_dim]``.
        q_g, k_g, v_g: Gathered q/k0/v0, same layout.
        out_g: The MERGED joint-softmax output gathered back, same layout / dtype as q.
        lse_g: The MERGED joint-softmax log-sum-exp, ``[batch, heads_local, seq_full]``.
        scale: Softmax scale.
        cu_seqlens, max_seqlen: Packed-document boundaries / longest doc, or ``None``.

    Returns:
        ``(dq, dk0, dv0)``, each ``[batch, seq_full, heads_local, head_dim]``.
    """
    if cu_seqlens is None:
        dq, dk, dv = torch.empty_like(q_g), torch.empty_like(k_g), torch.empty_like(v_g)
        _fa_backward(dout_g, q_g, k_g, v_g, out_g, lse_g, dq, dk, dv, scale, causal=True)
        return dq, dk, dv
    b, s_full, hl, d = q_g.shape
    dout_f = dout_g.reshape(b * s_full, hl, d)
    q_f = q_g.reshape(b * s_full, hl, d)
    k_f = k_g.reshape(b * s_full, hl, d)
    v_f = v_g.reshape(b * s_full, hl, d)
    out_f = out_g.reshape(b * s_full, hl, d)
    lse_f = lse_g.permute(1, 0, 2).reshape(hl, b * s_full)  # [Hl, b*Sf]
    dq_f, dk_f, dv_f = torch.empty_like(q_f), torch.empty_like(k_f), torch.empty_like(v_f)
    _fa._flash_attn_varlen_backward(
        dout_f,
        q_f,
        k_f,
        v_f,
        out_f,
        lse_f,
        dq_f,
        dk_f,
        dv_f,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        0.0,
        scale,
        True,
        -1,
        -1,
        0.0,
        None,
        False,
        None,
        False,
    )
    return (
        dq_f.reshape(b, s_full, hl, d),
        dk_f.reshape(b, s_full, hl, d),
        dv_f.reshape(b, s_full, hl, d),
    )


class _CachedUlyssesAttention(torch.autograd.Function):
    """EAGLE-3 mixed causal-ring-free attention under Ulysses context parallelism.

    Block 0 (``cache_k[:, 0]`` / ``cache_v[:, 0]``) is the causal sequence
    attention run over the all-to-all-gathered full sequence; blocks ``i >= 1`` are
    per-position TTT diagonals (same position, shard-local). Both are fused into one
    softmax via the online-softmax merge. The backward re-runs the block-0
    FlashAttention backward with the **merged** output/lse so it produces the
    correct joint-softmax gradient (the same identity ``_CachedRingAttention`` uses);
    the diagonal grads are added in closed form.

    Layout: ``q`` is ``[B, T_local, H, D]`` (FlashAttention layout); ``cache_k`` /
    ``cache_v`` carry a block axis ``[B, num_blocks, T_local, H, D]``.
    """

    @staticmethod
    def forward(ctx, q, cache_k, cache_v, group, scale, cu_seqlens, max_seqlen):
        q_g = _gather_seq_scatter_heads(q, group)
        k_g = _gather_seq_scatter_heads(cache_k[:, 0].contiguous(), group)
        v_g = _gather_seq_scatter_heads(cache_v[:, 0].contiguous(), group)
        out_g, lse_g = _block0_forward(q_g, k_g, v_g, scale, cu_seqlens, max_seqlen)
        acc_out = _scatter_seq_gather_heads(out_g, group).float()  # [B, T_local, H, D]
        acc_lse = _scatter_lse(lse_g, group).transpose(1, 2).float()  # [B, H, T] -> [B, T, H]
        num_blocks = cache_k.shape[1]
        q_f = q.float()
        for i in range(1, num_blocks):
            lse_i = (q_f * cache_k[:, i].float()).sum(-1) * scale  # [B, T_local, H]
            acc_out, acc_lse = _merge_diag(acc_out, acc_lse, cache_v[:, i].float(), lse_i)
        cu = cu_seqlens if cu_seqlens is not None else q.new_empty(0)
        ctx.save_for_backward(q, cache_k, cache_v, acc_out, acc_lse, cu)
        ctx.group, ctx.scale, ctx.max_seqlen, ctx.packed = group, scale, max_seqlen, cu_seqlens is not None
        return acc_out.to(q.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        q, cache_k, cache_v, merged_out, merged_lse, cu = ctx.saved_tensors
        group, scale = ctx.group, ctx.scale
        cu_seqlens = cu if ctx.packed else None
        num_blocks = cache_k.shape[1]
        dtype = q.dtype

        # Block 0: FlashAttention backward with the MERGED out/lse (gathered layout).
        q_g = _gather_seq_scatter_heads(q, group)
        k_g = _gather_seq_scatter_heads(cache_k[:, 0].contiguous(), group)
        v_g = _gather_seq_scatter_heads(cache_v[:, 0].contiguous(), group)
        dout_g = _gather_seq_scatter_heads(grad_out.contiguous().to(dtype), group)
        mout_g = _gather_seq_scatter_heads(merged_out.to(dtype), group)
        mlse_g = _gather_lse(merged_lse.transpose(1, 2).contiguous(), group)  # [B,T,H]->[B,H,T]->[B,Hl,Tf]
        dq_g, dk0_g, dv0_g = _block0_backward(dout_g, q_g, k_g, v_g, mout_g, mlse_g, scale, cu_seqlens, ctx.max_seqlen)
        dq = _scatter_seq_gather_heads(dq_g, group).float()
        dcache_k = torch.zeros_like(cache_k, dtype=torch.float32)
        dcache_v = torch.zeros_like(cache_v, dtype=torch.float32)
        dcache_k[:, 0] = _scatter_seq_gather_heads(dk0_g, group).float()
        dcache_v[:, 0] = _scatter_seq_gather_heads(dv0_g, group).float()

        # Diagonals: closed-form joint-softmax grads against the merged out/lse.
        grad_out_f = grad_out.float()
        q_f = q.float()
        for i in range(1, num_blocks):
            ki = cache_k[:, i].float()
            vi = cache_v[:, i].float()
            lse_i = (q_f * ki).sum(-1) * scale  # [B, T_local, H]
            wi = torch.exp(lse_i - merged_lse)  # joint-softmax weight of this diagonal
            d_out_i = grad_out_f * wi.unsqueeze(-1)
            d_lse_i = wi * (grad_out_f * (vi - merged_out)).sum(-1)
            dq = dq + d_lse_i.unsqueeze(-1) * scale * ki
            dcache_k[:, i] = d_lse_i.unsqueeze(-1) * scale * q_f
            dcache_v[:, i] = d_out_i

        return dq.to(dtype), dcache_k.to(cache_k.dtype), dcache_v.to(cache_v.dtype), None, None, None, None


def cached_ulysses_attention(
    q: torch.Tensor,
    cache_k: list[torch.Tensor],
    cache_v: list[torch.Tensor],
    group: dist.ProcessGroup,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
) -> torch.Tensor:
    """EAGLE-3 mixed causal-block-0 + TTT-diagonal attention under Ulysses CP.

    Args:
        q: This step's query, ``[batch, seq_local, heads, head_dim]`` (FlashAttention
            layout; the sequence is CP-sharded).
        cache_k: Per-TTT-step keys, each ``[batch, seq_local, heads, head_dim]``;
            index 0 is the step-0 sequence key, ``i >= 1`` the diagonal steps.
        cache_v: Per-TTT-step values, same layout as ``cache_k``.
        group: The Ulysses (context-parallel) process group.
        scale: Softmax scale (``head_dim ** -0.5``).
        cu_seqlens: GLOBAL (un-sharded) packed-document boundaries ``[num_docs + 1]``
            (int32), or ``None`` for a single causal stream.
        max_seqlen: Longest document length for the packed path.

    Returns:
        Attention output ``[batch, seq_local, heads, head_dim]`` in ``q``'s dtype.
    """
    ck = torch.stack(cache_k, dim=1)  # [B, num_blocks, T_local, H, D]
    cv = torch.stack(cache_v, dim=1)
    return _CachedUlyssesAttention.apply(q, ck, cv, group, scale, cu_seqlens, max_seqlen)

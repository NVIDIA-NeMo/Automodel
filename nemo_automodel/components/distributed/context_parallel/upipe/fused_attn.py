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
# Adapted from the Untied Ulysses reference implementation
# (https://github.com/togethercomputer/Untied-Ulysses), Apache-2.0.

"""Untied Ulysses ("UPipe") fused context-parallel attention.

Standard Ulysses projects every head at once, so the pre- and post-all-to-all
copies of Q, K and V are all live before attention starts. UPipe instead unties
the heads: it runs ``pipe_degree = n_heads / ulysses_degree`` sequential stages,
each projecting only as many query heads as there are Ulysses ranks, so that
after the all-to-all every rank owns exactly one head over the full sequence.
Peak QKV memory drops by ``pipe_degree``.

Two things make that work:

* The op consumes ``x`` and the raw ``wq``/``wk``/``wv`` weights rather than
  pre-projected Q/K/V. Projecting outside the op would materialise the full
  width and forfeit the saving, which is why this cannot be expressed as an
  ordinary attention backend.
* The backward saves no Q/K/V at all. It recomputes each stage's projection
  from ``x``, undoes RoPE with the inverse rotation (RoPE is unitary), and
  accumulates ``dx`` in place while writing ``dwq`` one row-block at a time.

Head ordering: to keep KV heads shardable one-per-rank under GQA, stage ``s``
pairs query head ``s * P + r`` with KV head ``(s // gqa_ratio) * P + r``, which
is a permutation of the standard ``i // gqa_ratio`` pairing. Callers must undo
it before ``o_proj``; see :func:`upipe_head_permutation`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

from nemo_automodel.components.distributed.context_parallel.upipe.all_to_all import all_to_all_4d
from nemo_automodel.components.distributed.context_parallel.upipe.ring_attn import attn_backward, attn_forward
from nemo_automodel.components.distributed.context_parallel.upipe.rotary import apply_rotary


def _resolve_group(group_name: str) -> Optional[dist.ProcessGroup]:
    """Resolve a process-group name, treating the empty string as absent.

    The custom-op boundary only accepts plain types, so groups cross it by name.
    """
    if not group_name:
        return None
    return dist.distributed_c10d._resolve_process_group(group_name)


@torch.library.custom_op("upipe::_upipe_attn_gqa_forward", mutates_args=(), device_types="cuda")
def upipe_attn_gqa_forward(
    ulysses_group_name: str,
    ring_group_name: str,
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_dim: int,
    dropout_p: float = 0.0,
    softmax_scale: float = 0.0,
    causal: bool = True,
    attn_type: str = "fa2",
    interleaved: bool = False,
) -> list[torch.Tensor]:
    """Staged forward pass for UPipe GQA attention.

    Args:
        ulysses_group_name: Name of the Ulysses process group.
        ring_group_name: Name of the ring process group, or ``""`` for none.
        x: Input hidden states ``[batch, local_seq, hidden]``.
        wq: Query projection weight ``[n_heads * head_dim, hidden]``.
        wk: Key projection weight ``[n_kv_heads * head_dim, hidden]``.
        wv: Value projection weight ``[n_kv_heads * head_dim, hidden]``.
        cos: Rotary cosine table ``[local_seq, head_dim / 2]``.
        sin: Rotary sine table ``[local_seq, head_dim / 2]``.
        head_dim: Attention head dimension.
        dropout_p: Dropout probability.
        softmax_scale: Softmax scale; 0 means ``head_dim ** -0.5``.
        causal: Whether to apply a causal mask.
        attn_type: ``"fa2"`` or ``"fa3"``.
        interleaved: Interleaved rather than half-split RoPE.

    Returns:
        ``[final_out, *lse_per_stage]``, where ``final_out`` is
        ``[batch, local_seq, n_heads, head_dim]`` in permuted head order.
    """
    ulysses_group = _resolve_group(ulysses_group_name)
    ring_group = _resolve_group(ring_group_name)

    bs, shard_seqlen, _hidden = x.shape
    # Derive head counts from the weights, not from the hidden size: head_dim
    # need not equal hidden // n_heads (e.g. Qwen3-32B).
    n_heads = wq.shape[0] // head_dim
    n_kv_heads_may_be_replicated = wk.shape[0] // head_dim
    gqa_ratio = n_heads // n_kv_heads_may_be_replicated

    ulysses_degree = 1 if ulysses_group is None else dist.get_world_size(ulysses_group)
    pipe_degree = n_heads // ulysses_degree

    if n_kv_heads_may_be_replicated % ulysses_degree != 0:
        raise ValueError(
            f"n_kv_heads ({n_kv_heads_may_be_replicated}) must be divisible by ulysses_degree ({ulysses_degree})"
        )

    if softmax_scale == 0.0:
        softmax_scale = head_dim ** (-0.5)

    wq_chunks = torch.chunk(wq, pipe_degree, dim=0)
    wk_chunks = torch.chunk(wk, pipe_degree // gqa_ratio, dim=0)
    wv_chunks = torch.chunk(wv, pipe_degree // gqa_ratio, dim=0)

    lse_list = []
    final_out = torch.empty([bs, shard_seqlen, n_heads, head_dim], device=x.device, dtype=x.dtype)

    k_out = None
    v_out = None

    for stage in range(pipe_degree):
        q_proj = F.linear(x, wq_chunks[stage]).view(bs, shard_seqlen, -1, head_dim)
        apply_rotary(q_proj, cos, sin, interleaved=interleaved, inplace=True)
        q_out = all_to_all_4d(q_proj, ulysses_group, scatter_idx=2, gather_idx=1)
        del q_proj

        if stage % gqa_ratio == 0:
            kv_idx = stage // gqa_ratio
            k_proj = F.linear(x, wk_chunks[kv_idx]).view(bs, shard_seqlen, -1, head_dim)
            v_proj = F.linear(x, wv_chunks[kv_idx]).view(bs, shard_seqlen, -1, head_dim)
            apply_rotary(k_proj, cos, sin, interleaved=interleaved, inplace=True)

            k_out = all_to_all_4d(k_proj, ulysses_group, scatter_idx=2, gather_idx=1)
            v_out = all_to_all_4d(v_proj, ulysses_group, scatter_idx=2, gather_idx=1)
            del k_proj, v_proj

        attn_out, lse = attn_forward(
            ring_group,
            q_out,
            k_out,
            v_out,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            attn_type=attn_type,
        )
        lse_list.append(lse)

        del q_out
        if (stage + 1) // gqa_ratio != stage // gqa_ratio:
            del k_out, v_out
            k_out = v_out = None

        out_local = all_to_all_4d(attn_out, ulysses_group, scatter_idx=1, gather_idx=2)
        del attn_out

        head_start = stage * ulysses_degree
        final_out[:, :, head_start : head_start + ulysses_degree, :] = out_local
        del out_local

    return [final_out] + lse_list


@torch.library.custom_op("upipe::_upipe_attn_gqa_backward", mutates_args=(), device_types="cuda")
def upipe_attn_gqa_backward(
    ulysses_group_name: str,
    ring_group_name: str,
    dout: torch.Tensor,
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    final_out: torch.Tensor,
    lse_list: list[torch.Tensor],
    head_dim: int,
    n_kv_heads: int,
    dropout_p: float = 0.0,
    softmax_scale: float = 0.0,
    causal: bool = True,
    attn_type: str = "fa2",
    deterministic: bool = False,
    interleaved: bool = False,
) -> list[torch.Tensor]:
    """Staged backward pass for UPipe GQA attention.

    Recomputes each stage's projections from ``x`` rather than reading saved
    activations, so no Q/K/V ever exists at full width.

    Args:
        ulysses_group_name: Name of the Ulysses process group.
        ring_group_name: Name of the ring process group, or ``""`` for none.
        dout: Gradient of the attention output, in permuted head order.
        x: Input hidden states from the forward pass.
        wq: Query projection weight.
        wk: Key projection weight.
        wv: Value projection weight.
        cos: Rotary cosine table.
        sin: Rotary sine table.
        final_out: Attention output from the forward pass.
        lse_list: Per-stage log-sum-exp tensors from the forward pass.
        head_dim: Attention head dimension.
        n_kv_heads: Pre-replication number of KV heads.
        dropout_p: Dropout probability.
        softmax_scale: Softmax scale; 0 means ``head_dim ** -0.5``.
        causal: Whether a causal mask was applied.
        attn_type: ``"fa2"`` or ``"fa3"``.
        deterministic: Use deterministic FlashAttention backward kernels.
        interleaved: Interleaved rather than half-split RoPE.

    Returns:
        ``[dx, dwq, dwk, dwv]``.
    """
    ulysses_group = _resolve_group(ulysses_group_name)
    ring_group = _resolve_group(ring_group_name)

    bs, shard_seqlen, hidden = x.shape
    n_heads = wq.shape[0] // head_dim
    n_kv_heads_may_be_replicated = wk.shape[0] // head_dim
    gqa_ratio = n_heads // n_kv_heads_may_be_replicated

    ulysses_degree = 1 if ulysses_group is None else dist.get_world_size(ulysses_group)
    pipe_degree = n_heads // ulysses_degree

    if softmax_scale == 0.0:
        softmax_scale = head_dim ** (-0.5)

    wq_chunks = torch.chunk(wq, pipe_degree, dim=0)
    wk_chunks = torch.chunk(wk, pipe_degree // gqa_ratio, dim=0)
    wv_chunks = torch.chunk(wv, pipe_degree // gqa_ratio, dim=0)

    final_out_chunks = list(torch.chunk(final_out, pipe_degree, dim=2))
    dout_chunks = list(torch.chunk(dout, pipe_degree, dim=2))

    dx = None
    dwq = torch.zeros_like(wq)
    dwk = torch.zeros_like(wk)
    dwv = torch.zeros_like(wv)

    dk_accum = [None for _ in range(pipe_degree // gqa_ratio)]
    dv_accum = [None for _ in range(pipe_degree // gqa_ratio)]

    k_out = None
    v_out = None

    x_flat = x.view(bs * shard_seqlen, -1)

    for stage in range(pipe_degree):
        q_proj = F.linear(x, wq_chunks[stage]).view(bs, shard_seqlen, -1, head_dim)
        apply_rotary(q_proj, cos, sin, interleaved=interleaved, inplace=True)
        q_out = all_to_all_4d(q_proj, ulysses_group, scatter_idx=2, gather_idx=1)
        del q_proj

        if stage % gqa_ratio == 0:
            kv_idx = stage // gqa_ratio
            k_proj = F.linear(x, wk_chunks[kv_idx]).view(bs, shard_seqlen, -1, head_dim)
            v_proj = F.linear(x, wv_chunks[kv_idx]).view(bs, shard_seqlen, -1, head_dim)
            apply_rotary(k_proj, cos, sin, interleaved=interleaved, inplace=True)

            k_out = all_to_all_4d(k_proj, ulysses_group, scatter_idx=2, gather_idx=1)
            v_out = all_to_all_4d(v_proj, ulysses_group, scatter_idx=2, gather_idx=1)
            del k_proj, v_proj

        out_a2a = all_to_all_4d(final_out_chunks[stage], ulysses_group, scatter_idx=2, gather_idx=1)
        dout_a2a = all_to_all_4d(dout_chunks[stage], ulysses_group, scatter_idx=2, gather_idx=1)
        final_out_chunks[stage] = None
        dout_chunks[stage] = None

        attn_dq, attn_dk, attn_dv = attn_backward(
            ring_group,
            dout_a2a,
            q_out,
            k_out,
            v_out,
            out_a2a,
            lse_list[stage],
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            deterministic=deterministic,
            attn_type=attn_type,
        )

        lse_list[stage] = None
        del dout_a2a, q_out, out_a2a
        if (stage + 1) // gqa_ratio != stage // gqa_ratio:
            del k_out, v_out
            k_out = v_out = None

        kv_idx = stage // gqa_ratio
        if dk_accum[kv_idx] is None:
            dk_accum[kv_idx] = attn_dk
            dv_accum[kv_idx] = attn_dv
        else:
            dk_accum[kv_idx].add_(attn_dk)
            dv_accum[kv_idx].add_(attn_dv)

        dq_local = all_to_all_4d(attn_dq, ulysses_group, scatter_idx=1, gather_idx=2)
        # RoPE is unitary, so the inverse rotation recovers the pre-rotary gradient.
        apply_rotary(dq_local, cos, sin, interleaved=interleaved, inplace=True, conjugate=True)
        dq_flat = dq_local.view(bs * shard_seqlen, -1)

        del attn_dq, attn_dk, attn_dv

        if dx is None:
            dx = dq_flat @ wq_chunks[stage]
        else:
            dx.addmm_(dq_flat, wq_chunks[stage])

        head_start = stage * (head_dim * ulysses_degree)
        dwq[head_start : head_start + head_dim * ulysses_degree, :] = dq_flat.T @ x_flat

        del dq_local, dq_flat

        if (stage + 1) % gqa_ratio == 0 or stage == pipe_degree - 1:
            dk_local = all_to_all_4d(dk_accum[kv_idx], ulysses_group, scatter_idx=1, gather_idx=2)
            dv_local = all_to_all_4d(dv_accum[kv_idx], ulysses_group, scatter_idx=1, gather_idx=2)

            dk_accum[kv_idx] = None
            dv_accum[kv_idx] = None

            apply_rotary(dk_local, cos, sin, interleaved=interleaved, inplace=True, conjugate=True)
            dk_flat = dk_local.view(bs * shard_seqlen, -1)
            dv_flat = dv_local.view(bs * shard_seqlen, -1)

            dx.addmm_(dk_flat, wk_chunks[kv_idx])
            dx.addmm_(dv_flat, wv_chunks[kv_idx])

            kv_head_start = kv_idx * (head_dim * ulysses_degree)
            kv_head_end = kv_head_start + head_dim * ulysses_degree
            dwk[kv_head_start:kv_head_end, :] = dk_flat.T @ x_flat
            dwv[kv_head_start:kv_head_end, :] = dv_flat.T @ x_flat

            del dk_local, dv_local, dk_flat, dv_flat

    dwk = _reduce_gqa_gradients(dwk, n_kv_heads, head_dim, hidden)
    dwv = _reduce_gqa_gradients(dwv, n_kv_heads, head_dim, hidden)

    return [dx.view(bs, shard_seqlen, -1).to(x.dtype), dwq, dwk, dwv]


def _reduce_gqa_gradients(
    dw: torch.Tensor,
    n_kv_heads: int,
    head_dim: int,
    hidden_dim: int,
) -> torch.Tensor:
    """Sum gradients across replicated KV weight slots and broadcast them back.

    A no-op unless ``wk``/``wv`` were replicated to reach ``ulysses_degree``
    KV heads, which the AutoModel integration currently forbids.
    """
    if dw.shape[0] // head_dim > n_kv_heads:
        n_rep = (dw.shape[0] // head_dim) // n_kv_heads
        dw = dw.view(n_kv_heads, n_rep, head_dim, hidden_dim)
        dw_sum = dw.sum(dim=1, keepdim=True)
        dw = dw_sum.expand(-1, n_rep, -1, -1).reshape(-1, hidden_dim)
    return dw


class UpipeAttnGQAFunc(torch.autograd.Function):
    """Autograd binding for the staged UPipe attention."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        head_dim: int,
        n_kv_heads: int,
        dropout_p: float,
        softmax_scale: Optional[float],
        causal: bool,
        ulysses_group_name: str,
        ring_group_name: str,
        attn_type: str,
        deterministic: bool,
        interleaved: bool,
    ) -> torch.Tensor:
        """Run the staged forward and stash only x, the weights, the output and the LSEs."""
        bs, seqlen, _hidden = x.shape

        if softmax_scale is None:
            softmax_scale = head_dim ** (-0.5)

        with torch.no_grad():
            outputs = torch.ops.upipe._upipe_attn_gqa_forward(
                ulysses_group_name,
                ring_group_name,
                x,
                wq,
                wk,
                wv,
                cos,
                sin,
                head_dim,
                dropout_p,
                softmax_scale,
                causal,
                attn_type,
                interleaved,
            )
            final_out = outputs[0]
            lse_list = outputs[1:]

        ctx.save_for_backward(x, wq, wk, wv, cos, sin, final_out, *lse_list)
        ctx.ulysses_group_name = ulysses_group_name
        ctx.ring_group_name = ring_group_name
        ctx.head_dim = head_dim
        ctx.n_kv_heads = n_kv_heads
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.attn_type = attn_type
        ctx.deterministic = deterministic
        ctx.interleaved = interleaved

        # -1 rather than the hidden size: n_heads * head_dim may differ from it.
        return final_out.view(bs, seqlen, -1)

    @staticmethod
    def backward(ctx, dout: torch.Tensor) -> tuple:
        """Recompute the stages and return gradients for x and the three weights."""
        saved = ctx.saved_tensors
        x, wq, wk, wv, cos, sin, final_out = saved[:7]
        lse_list = list(saved[7:])

        bs, seqlen, _hidden = x.shape
        n_heads = wq.shape[0] // ctx.head_dim

        dout = dout.view(bs, seqlen, n_heads, ctx.head_dim)

        with torch.no_grad():
            dx, dwq, dwk, dwv = torch.ops.upipe._upipe_attn_gqa_backward(
                ctx.ulysses_group_name,
                ctx.ring_group_name,
                dout,
                x,
                wq,
                wk,
                wv,
                cos,
                sin,
                final_out,
                lse_list,
                ctx.head_dim,
                ctx.n_kv_heads,
                ctx.dropout_p,
                ctx.softmax_scale,
                ctx.causal,
                ctx.attn_type,
                ctx.deterministic,
                ctx.interleaved,
            )

        return (dx, dwq, dwk, dwv) + (None,) * 12


def upipe_attn_gqa(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_dim: int,
    n_kv_heads: int,
    ulysses_group: Optional[dist.ProcessGroup],
    ring_group: Optional[dist.ProcessGroup] = None,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    attn_type: str = "fa2",
    deterministic: bool = False,
    interleaved: bool = False,
) -> torch.Tensor:
    """Fused projection + RoPE + context-parallel attention, staged over heads.

    Args:
        x: Input hidden states ``[batch, local_seq, hidden]``.
        wq: Query projection weight ``[n_heads * head_dim, hidden]``.
        wk: Key projection weight ``[n_kv_heads * head_dim, hidden]``.
        wv: Value projection weight ``[n_kv_heads * head_dim, hidden]``.
        cos: Rotary cosine table ``[local_seq, head_dim / 2]``.
        sin: Rotary sine table ``[local_seq, head_dim / 2]``.
        head_dim: Attention head dimension.
        n_kv_heads: Pre-replication number of KV heads.
        ulysses_group: Ulysses process group.
        ring_group: Ring process group, or None for pure Ulysses.
        dropout_p: Dropout probability.
        softmax_scale: Softmax scale; None means ``head_dim ** -0.5``.
        causal: Whether to apply a causal mask.
        attn_type: ``"fa2"`` or ``"fa3"``.
        deterministic: Use deterministic FlashAttention backward kernels.
        interleaved: Interleaved rather than half-split RoPE.

    Returns:
        ``[batch, local_seq, n_heads * head_dim]`` in permuted head order; undo
        the permutation with :func:`upipe_head_permutation` before ``o_proj``.
    """
    return UpipeAttnGQAFunc.apply(
        x,
        wq,
        wk,
        wv,
        cos,
        sin,
        head_dim,
        n_kv_heads,
        dropout_p,
        softmax_scale,
        causal,
        ulysses_group.group_name if ulysses_group is not None else "",
        ring_group.group_name if ring_group is not None else "",
        attn_type,
        deterministic,
        interleaved,
    )


def upipe_staged_attention(
    hidden_states: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    head_dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_perm: Optional[torch.Tensor],
    head_perm_inverse: Optional[torch.Tensor],
    ulysses_group: Optional[dist.ProcessGroup],
    ring_group: Optional[dist.ProcessGroup] = None,
    causal: bool = True,
    attn_type: str = "fa2",
    deterministic: bool = False,
    interleaved: bool = False,
) -> torch.Tensor:
    """Run UPipe attention and hand back heads in checkpoint order.

    Wraps :func:`upipe_attn_gqa` with the two corrections its staged head
    walk requires under GQA: ``wq``'s head blocks are gathered into the order
    the stages expect on the way in, and the output is scattered back on the
    way out. Both index operations are differentiable, so ``wq``'s gradient is
    un-permuted automatically.

    Args:
        hidden_states: Local hidden states ``[batch, local_seq, hidden]``.
        q_weight: ``q_proj.weight``, ``[n_heads * head_dim, hidden]``.
        k_weight: ``k_proj.weight``, ``[n_kv_heads * head_dim, hidden]``.
        v_weight: ``v_proj.weight``, ``[n_kv_heads * head_dim, hidden]``.
        cos: Rotary cosine table for the local positions, ``[batch, local_seq, head_dim]``.
        sin: Rotary sine table for the local positions, same shape as ``cos``.
        head_dim: Attention head dimension.
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads.
        head_perm: Slot-to-head permutation, or None when it is the identity.
        head_perm_inverse: Inverse of ``head_perm``, or None.
        ulysses_group: Ulysses process group.
        ring_group: Ring process group, or None for pure Ulysses.
        causal: Whether to apply a causal mask.
        attn_type: ``"fa2"`` or ``"fa3"``.
        deterministic: Use deterministic FlashAttention backward kernels.
        interleaved: Interleaved rather than half-split RoPE.

    Returns:
        ``[batch, local_seq, n_heads * head_dim]`` ready for ``o_proj``.
    """
    from nemo_automodel.components.distributed.context_parallel.upipe.rotary import (
        rope_tables_from_position_embeddings,
    )

    batch, local_seq, _ = hidden_states.shape
    rope_cos, rope_sin = rope_tables_from_position_embeddings(cos, sin, head_dim)

    if head_perm is not None:
        q_weight = q_weight.view(n_heads, head_dim, -1).index_select(0, head_perm).reshape(n_heads * head_dim, -1)

    out = upipe_attn_gqa(
        hidden_states,
        q_weight,
        k_weight,
        v_weight,
        rope_cos,
        rope_sin,
        head_dim=head_dim,
        n_kv_heads=n_kv_heads,
        ulysses_group=ulysses_group,
        ring_group=ring_group,
        causal=causal,
        attn_type=attn_type,
        deterministic=deterministic,
        interleaved=interleaved,
    )

    if head_perm_inverse is not None:
        out = out.view(batch, local_seq, n_heads, head_dim).index_select(2, head_perm_inverse)

    return out.reshape(batch, local_seq, n_heads * head_dim)

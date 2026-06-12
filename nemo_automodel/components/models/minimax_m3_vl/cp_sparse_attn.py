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

"""Context-parallel support for MiniMax M3 block-sparse DSA attention.

Under context parallelism the sequence is sharded across CP ranks with a
*load-balanced* layout (PyTorch's causal CP splits the sequence into
``2 * cp_size`` chunks and assigns rank ``r`` the pair ``{r, 2*cp_size-1-r}``),
so a rank's local positions are **not** a contiguous global span. The M3
lightning indexer builds its block-sparse mask from index q/k over the *global*
causal sequence, so a CP-aware sparse layer must gather the indexer inputs from
every rank and reorder them into global token order before selecting blocks.

This module holds the reorder primitives shared by the CP-aware attention. The
reorder math (``order_by_positions`` / ``restore_by_positions``) is factored out
as pure tensor functions so the load-balance inverse -- a silent-failure trap: a
wrong inverse trains without shape errors but never converges -- is unit-testable
on CPU without a process group.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch.autograd import Function

from nemo_automodel.components.models.gpt_oss.rope_utils import apply_rotary_emb_qk
from nemo_automodel.components.models.minimax_m3_vl.layers import MiniMaxM3Attention, select_sparse_blocks

# Compiled FlexAttention is built lazily (and cached) on first CP use so that
# importing this module / instantiating the attention on CPU does not require a
# FlexAttention-capable build. ``dynamic=True`` because the gathered global key
# length varies across batches / CP configs.
_COMPILED_FLEX_ATTENTION = None


def _get_compiled_flex_attention():
    global _COMPILED_FLEX_ATTENTION
    if _COMPILED_FLEX_ATTENTION is None:
        from torch.nn.attention.flex_attention import flex_attention

        _COMPILED_FLEX_ATTENTION = torch.compile(flex_attention, dynamic=True)
    return _COMPILED_FLEX_ATTENTION


class _AllGatherConcatFn(Function):
    """All-gather + concat with an autograd-safe backward.

    Forward concatenates equal-sized local shards from all CP ranks along
    ``dim``. Backward all-reduces the concatenated gradient and slices out this
    rank's shard. Mirrors ``qwen3_5_moe/cp_linear_attn.py``'s helper.
    """

    @staticmethod
    def forward(ctx, local_tensor: torch.Tensor, group: "dist.ProcessGroup", dim: int):
        dim = dim if dim >= 0 else local_tensor.ndim + dim
        world_size = dist.get_world_size(group)
        gathered = [torch.empty_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, local_tensor.contiguous(), group=group)

        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.dim = dim
        ctx.local_dim_size = local_tensor.size(dim)
        return torch.cat(gathered, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_full = grad_output.contiguous()
        dist.all_reduce(grad_full, op=dist.ReduceOp.SUM, group=ctx.group)
        start = ctx.rank * ctx.local_dim_size
        grad_local = grad_full.narrow(ctx.dim, start, ctx.local_dim_size).contiguous()
        return grad_local, None, None


def order_by_positions(
    gathered: torch.Tensor,
    gathered_positions: torch.Tensor,
    *,
    seq_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder a CP-gathered tensor from load-balanced order into global token order.

    Args:
        gathered: tensor whose ``seq_dim`` concatenates every CP rank's local shard
            in rank order (the output of an all-gather+concat).
        gathered_positions: 1-D global token positions aligned with ``gathered``
            along ``seq_dim`` (gathered the same way). Must be a permutation of
            ``0..S-1`` where ``S = gathered.size(seq_dim)``.
        seq_dim: the sequence dimension of ``gathered``.

    Returns:
        ``(global_tensor, sort_order)`` where ``global_tensor`` is ``gathered``
        reindexed so position ``p`` sits at index ``p`` along ``seq_dim``, and
        ``sort_order`` is the permutation applied (``argsort`` of the positions).

    Raises:
        ValueError: if ``gathered_positions`` is not a dense permutation of 0..S-1.
    """
    if gathered_positions.ndim != 1:
        raise ValueError(f"gathered_positions must be 1-D, got shape {tuple(gathered_positions.shape)}")
    sort_order = torch.argsort(gathered_positions)
    sorted_positions = gathered_positions.index_select(0, sort_order)
    expected = torch.arange(sorted_positions.numel(), device=sorted_positions.device, dtype=sorted_positions.dtype)
    if not torch.equal(sorted_positions, expected):
        raise ValueError(
            "MiniMax M3 CP sparse attention requires dense global token positions covering 0..S-1 "
            "after gathering CP shards (load-balanced causal sharding produces this); got a "
            "non-contiguous position set."
        )
    return gathered.index_select(seq_dim, sort_order), sort_order


def restore_by_positions(
    global_tensor: torch.Tensor,
    target_positions: torch.Tensor,
    *,
    seq_dim: int,
) -> torch.Tensor:
    """Select rows of a global-ordered tensor back into an arbitrary (local) position order.

    Inverse companion to :func:`order_by_positions`. Given a tensor indexed by
    global position along ``seq_dim`` (position ``p`` at index ``p``), return the
    slice in ``target_positions`` order -- e.g. this rank's load-balanced local
    positions, recovering the CP-sharded layout.

    Args:
        global_tensor: tensor indexed by global position along ``seq_dim``.
        target_positions: 1-D positions to select, in the desired output order.
        seq_dim: the sequence dimension of ``global_tensor``.

    Returns:
        ``global_tensor`` gathered along ``seq_dim`` at ``target_positions``.
    """
    if target_positions.ndim != 1:
        raise ValueError(f"target_positions must be 1-D, got shape {tuple(target_positions.shape)}")
    return global_tensor.index_select(seq_dim, target_positions.to(torch.long))


def _all_gather_concat_nograd(tensor: torch.Tensor, group: "dist.ProcessGroup", dim: int) -> torch.Tensor:
    """Plain (non-differentiable) all-gather + concat along ``dim``."""
    world_size = dist.get_world_size(group)
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    return torch.cat(gathered, dim=dim)


def cp_document_ids(positions: torch.Tensor) -> torch.Tensor:
    """Per-token document id from packed position ids (reset to 0 per document).

    ``doc_id = cumsum(positions == 0) - 1`` along the sequence dim: a 0-based id
    that increments at every position-0 (document start). A single sequence -> all
    zeros (so a same-document mask is all-True, a no-op). A trailing cp-pad (also
    position 0) opens a spurious extra document, but pad keys/queries are excluded
    by causality / the padding mask, so it is harmless.

    Args:
        positions: ``[B, T]`` long global-ordered position ids.

    Returns:
        ``[B, T]`` long document ids.
    """
    return (positions == 0).cumsum(dim=-1) - 1


def cp_load_balanced_global_slots(
    cp_size: int,
    t_local: int,
    device: torch.device,
    *,
    rank: int | None = None,
) -> torch.Tensor:
    """Global token-slot indices for PyTorch's causal context-parallel load balancing.

    Causal CP splits the (cp-padded) sequence into ``2 * cp_size`` equal chunks and
    assigns rank ``r`` the pair ``{r, 2*cp_size-1-r}`` (concatenated in that order),
    so the local length is ``2 * chunk``. This reconstructs each local slot's global
    index *structurally* -- independent of ``position_ids`` values -- which is robust
    to cp-padding (pad slots land at the global tail, where causality excludes them)
    and to the indexer's pad ``position_id`` fill.

    Args:
        cp_size: context-parallel size.
        t_local: local (per-rank) sequence length; must be even.
        rank: if given, return the ``[t_local]`` slots for that CP rank; otherwise
            return the ``[cp_size * t_local]`` slots for the rank-major all-gathered
            concatenation (rank 0's tokens, then rank 1's, ...).

    Returns:
        1-D long tensor of global slot indices (a permutation of ``0..T_global-1``).
    """
    if t_local % 2 != 0:
        raise ValueError(f"causal CP local length must be even (2*chunk), got {t_local}")
    chunk = t_local // 2

    def _slots_for(r: int) -> torch.Tensor:
        first = torch.arange(r * chunk, (r + 1) * chunk, device=device)
        second = torch.arange((2 * cp_size - 1 - r) * chunk, (2 * cp_size - r) * chunk, device=device)
        return torch.cat([first, second])

    if rank is not None:
        return _slots_for(rank)
    return torch.cat([_slots_for(r) for r in range(cp_size)])


class MiniMaxM3CPSparseAttention(MiniMaxM3Attention):
    """Context-parallel-aware drop-in for a MiniMax M3 sparse-attention layer.

    Inherits every parameter and the eager forward from ``MiniMaxM3Attention``.
    The only addition is ``_cp_mesh`` (set post-FSDP by the MoE parallelizer's
    ``apply_cp``). When CP is off (``_cp_mesh`` is None / size 1) it delegates to
    the parent's eager sparse forward, so non-CP runs are unaffected.

    Under CP (``cp_size > 1``) the sequence is sharded across ranks, so the DSA
    block selection -- which is causal over the *global* sequence -- cannot be
    built from a rank's local shard. This forward instead:

      1. projects q/k/v + indexer q/k locally and applies QK-norm + RoPE locally
         (``freqs_cis`` already encodes each token's global position, so phases
         stay correct after gathering);
      2. all-gathers k/v (autograd-safe) and the indexer key + token positions
         across the CP group, then reorders them into global token order
         (load-balanced CP sharding is non-contiguous -- see
         :func:`order_by_positions`);
      3. selects the top-k key blocks for the *local* queries against the global
         key sequence (:func:`select_sparse_blocks`);
      4. attends with FlexAttention over a ``BlockMask`` that encodes the block
         selection + token-level causal, with the local queries against the full
         gathered K/V (``enable_gqa=True``). FlexAttention has a real backward,
         so the gathered K/V gradients flow back to the local shards.

    Dense layers (0-2) are untouched; they use the standard DTensor-SDPA CP path.
    """

    _cp_mesh: Any

    def __init__(self, config: Any, backend: Any, *, is_sparse_attention_layer: bool = True):
        super().__init__(config, backend, is_sparse_attention_layer=is_sparse_attention_layer)
        self._cp_mesh = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        cp_mesh = self._cp_mesh
        if cp_mesh is None or cp_mesh.size() <= 1 or self.indexer is None:
            return super().forward(x, freqs_cis=freqs_cis, attention_mask=attention_mask, **attn_kwargs)
        return self._cp_forward(x, freqs_cis=freqs_cis, **attn_kwargs)

    def _cp_forward(self, x: torch.Tensor, *, freqs_cis: torch.Tensor, **attn_kwargs: Any) -> torch.Tensor:
        if x.dim() != 3:
            raise NotImplementedError("MiniMax M3 CP sparse attention supports bshd (3-D) input only.")
        bsz, t_local, _ = x.shape
        cp_group = self._cp_mesh.get_group()
        cp_size = self._cp_mesh.size()
        cp_rank = dist.get_rank(cp_group)

        # Structural global token-slot indices for the causal-CP load-balanced layout.
        # Used (instead of position_ids values) for both global reordering and causal
        # masking, so the trailing cp-pad slots are excluded by causality and the
        # indexer's pad position_id fill cannot corrupt ordering. For a single
        # (non-packed) sequence the global slot equals the token's position id.
        q_slots = cp_load_balanced_global_slots(cp_size, t_local, x.device, rank=cp_rank)  # [t_local]
        gathered_slots = cp_load_balanced_global_slots(cp_size, t_local, x.device)  # [T_global], rank-major
        sort_order = torch.argsort(gathered_slots)  # gathered-index -> global order

        # 1. main q/k/v (local) + QK-norm + RoPE. RoPE uses the local freqs_cis,
        #    which already encodes each token's global position.
        q = self.q_proj(x).view(bsz, t_local, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, t_local, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, t_local, self.num_kv_heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = apply_rotary_emb_qk(q, k, freqs_cis, format="bshd", rope_fusion=self.backend.rope_fusion)

        # 2. all-gather k/v (autograd-safe), reorder rank-major -> global slot order.
        k_g = _AllGatherConcatFn.apply(k, cp_group, 1).index_select(1, sort_order)  # [B, T_global, n_kv, D]
        v_g = _AllGatherConcatFn.apply(v, cp_group, 1).index_select(1, sort_order)

        # 2b. gather the padding mask so interior pad keys (real ragged-length SFT
        # batches) are masked in attention. padding_mask convention: True == pad.
        # cp-padding (trailing slots) is already excluded by causality; this covers
        # pad slots in the interior of the global sequence. None -> no padding.
        padding_mask = attn_kwargs.get("padding_mask", None)
        key_valid = None  # [B, T_global] bool, True == real (attendable) key
        if padding_mask is not None:
            local_valid = ~padding_mask.bool()
            if local_valid.dim() == 1:
                local_valid = local_valid.unsqueeze(0).expand(bsz, -1)
            key_valid = _all_gather_concat_nograd(local_valid, cp_group, dim=1).index_select(1, sort_order)

        # 2c. per-document boundaries (packed multi-document sequences). position_ids
        # reset to 0 at each document start, so doc_id = cumsum(pos == 0) - 1 over the
        # global slot order gives a 0-based per-token document id. A single sequence
        # yields all-zeros (same_doc is then all-True -> no effect on attention); the
        # trailing cp-pad (pos 0) starts a spurious doc but those keys are dropped by
        # causality / key_valid anyway.
        position_ids = attn_kwargs.get("position_ids", None)
        doc_global = None  # [B, T_global] long: document id per global key slot
        q_doc = None  # [B, t_local] long: document id per local query
        if position_ids is not None:
            # M3 text uses 1-D [B, S] position ids. The 3-D branch is defensive for an
            # mRoPE [3, B, S] layout (Qwen2-VL convention: axis 0 is the temporal/text
            # axis, which still resets to 0 per document), not expected for M3 today.
            if position_ids.dim() == 3:
                assert position_ids.shape[0] == 3, (
                    f"3-D position_ids expected mRoPE layout [3, B, S], got {tuple(position_ids.shape)}"
                )
                local_pos = position_ids[0]
            else:
                local_pos = position_ids
            if local_pos.dim() == 1:
                local_pos = local_pos.unsqueeze(0).expand(bsz, -1)
            pos_global = _all_gather_concat_nograd(local_pos.long(), cp_group, dim=1).index_select(1, sort_order)
            doc_global = cp_document_ids(pos_global)  # [B, T_global]
            q_doc = doc_global.index_select(1, q_slots)  # [B, t_local]

        # 3. block selection: local queries vs the global key sequence (no grad).
        with torch.no_grad():
            idxer = self.indexer
            idx_q = idxer.index_q_norm(
                idxer.index_q_proj(x).view(bsz, t_local, idxer.num_index_heads, idxer.index_head_dim)
            )
            idx_k = idxer.index_k_norm(idxer.index_k_proj(x).view(bsz, t_local, 1, idxer.index_head_dim))
            idx_q, idx_k = apply_rotary_emb_qk(
                idx_q, idx_k, freqs_cis, format="bshd", rope_fusion=idxer.backend.rope_fusion
            )
            idx_k_g = _all_gather_concat_nograd(idx_k, cp_group, dim=1).index_select(1, sort_order)  # [B,T_global,1,D]
            block_sel = select_sparse_blocks(
                idx_q,
                idx_k_g,
                block_size=idxer.block_size,
                topk_blocks=idxer.topk_blocks,
                init_blocks=idxer.init_blocks,
                local_blocks=idxer.local_blocks,
                score_type=idxer.score_type,
                q_positions=q_slots,
            )  # [B, H_idx, T_local, num_blocks]

        # 4. FlexAttention over the block-sparse mask (local queries x global K/V).
        out = self._flex_sparse_attention(
            q,
            k_g,
            v_g,
            block_sel=block_sel,
            q_positions=q_slots,
            key_valid=key_valid,
            doc_global=doc_global,
            q_doc=q_doc,
        )
        return self.o_proj(out.flatten(2))

    def _flex_sparse_attention(
        self,
        q: torch.Tensor,
        k_global: torch.Tensor,
        v_global: torch.Tensor,
        *,
        block_sel: torch.Tensor,
        q_positions: torch.Tensor,
        key_valid: torch.Tensor | None = None,
        doc_global: torch.Tensor | None = None,
        q_doc: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from torch.nn.attention.flex_attention import create_block_mask

        bsz, t_local = q.shape[0], q.shape[1]
        t_global = k_global.shape[1]
        block_size = self.indexer.block_size
        # GQA: each idx-head governs ``rep`` main heads (mirrors the eager builder).
        rep = self.num_heads // self.indexer.num_index_heads

        qh = q.transpose(1, 2).contiguous()  # [B, num_heads, T_local, D]
        kh = k_global.transpose(1, 2).contiguous()  # [B, n_kv, T_global, D]
        vh = v_global.transpose(1, 2).contiguous()

        # Composable predicate: causal (by global slot) + block selection, plus the
        # optional pad-key and per-document (block-diagonal) terms when supplied. The
        # ``is not None`` checks are resolved at trace time, so the compiled mask only
        # includes the active terms.
        def cp_sparse_mask(b, h, q_idx, kv_idx):
            h_idx = h // rep
            keep = (kv_idx <= q_positions[q_idx]) & block_sel[b, h_idx, q_idx, kv_idx // block_size]
            if key_valid is not None:
                keep = keep & key_valid[b, kv_idx]
            if doc_global is not None:
                keep = keep & (doc_global[b, kv_idx] == q_doc[b, q_idx])
            # Guarantee every query row attends at least its own key. Real queries
            # already include self (causal + same-doc + the forced local block), so
            # this is a no-op for them; it only rescues *pad* queries whose every
            # key is otherwise masked (key_valid + same-doc exclude the pad's own
            # padded keys) -> an all--inf softmax row -> NaN, which would corrupt the
            # residual stream / MoE router / EP all-to-all even though the pad's loss
            # is discarded. Self-attention to a pad's own (discarded) V is harmless.
            return keep | (kv_idx == q_positions[q_idx])

        block_mask = create_block_mask(
            cp_sparse_mask,
            B=bsz,
            H=self.num_heads,
            Q_LEN=t_local,
            KV_LEN=t_global,
            device=q.device,
            BLOCK_SIZE=block_size,
            _compile=True,
        )
        flex = _get_compiled_flex_attention()
        out = flex(qh, kh, vh, block_mask=block_mask, scale=self.head_dim**-0.5, enable_gqa=True)
        return out.transpose(1, 2)  # [B, T_local, num_heads, D]

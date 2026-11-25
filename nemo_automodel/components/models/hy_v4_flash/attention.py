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

"""Gated MLA with learnable sinks for Hy4-flash: sliding-window and compressed-indexer sparse layers.

Both layer types run the same *absorbed* latent attention. The per-head key up-projection is folded into the
query (``q_nope @ W_k``), every head then attends over one shared ``[kv_lora_rank + rope]`` latent per token, and
the value up-projection is applied to the latent output. This is mathematically the reference's expanded
attention, and it is what the shared FlashMLA-forward / cuDNN-backward sparse kernel
(``models/common/cudnn_sparse_attention``) consumes, so the kernel handles the learnable sink and per-document
index lists for both layer types:

* ``sliding_attention``: the indices of the last ``sliding_window`` tokens of the query's document.
* ``indexed_attention``: dense causal attention while a document is not longer than ``index_topk``;
  otherwise the tokens of the ``index_topk // index_kpool - 1`` best-scoring complete groups plus the query's own
  group, chosen by :class:`HY4FlashCompressedIndexer`.

Packed documents are isolated: indices only ever point into the query's own document and groups are
document-local.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.common.cudnn_sparse_attention import cudnn_sparse_attention
from nemo_automodel.components.models.deepseek_v41.cp import gather_sequence
from nemo_automodel.components.models.hy_v4_flash.config import AttentionType, HY4FlashTextConfig
from nemo_automodel.components.models.hy_v4_flash.layers import HY4FlashRMSNorm, apply_interleaved_rope


@dataclass(frozen=True)
class HY4FlashSparsePlan:
    """Kernel-ready key lists of this rank's queries.

    Attributes:
        indices: Int32 tensor of shape [local_tokens, width] of global key rows: an ascending valid prefix followed
            by ``-1``.
        lengths: Int32 tensor of shape [local_tokens], the valid-prefix length of each query (0 for padding).
    """

    indices: torch.Tensor
    lengths: torch.Tensor


def causal_prefix_plan(position: torch.Tensor, doc_start: torch.Tensor, width: int) -> HY4FlashSparsePlan:
    """Keys of the last ``width`` tokens (including the query) of each query, inside its document.

    This is the whole key list of a sliding-window layer (``width = sliding_window``) and of a sparse layer for
    documents not longer than ``index_topk`` (``width = index_topk``, i.e. dense causal attention).

    Args:
        position: Integer tensor of shape [queries], global token index of each query.
        doc_start: Integer tensor of shape [queries], global index of each query's document start
            (``position + 1`` for padding, which leaves the query without keys).
        width: Maximum number of keys per query.

    Returns:
        The plan, with ``indices`` of shape [queries, width] and ``lengths`` of shape [queries].
    """
    start = torch.maximum(position - (width - 1), doc_start)
    lengths = (position - start + 1).clamp_min(0)
    slots = torch.arange(width, device=position.device)
    indices = torch.where(slots < lengths[:, None], start[:, None] + slots, -1)
    return HY4FlashSparsePlan(indices.to(torch.int32), lengths.to(torch.int32))


@dataclass(frozen=True)
class HY4FlashDocLayout:
    """Document boundaries and layer-independent key lists of one packed row, built once per forward pass.

    Attributes:
        doc_ranges: ``(start, end)`` token intervals of the real documents in the *global* sequence
            (``end`` exclusive); padding is not listed.
        local_start: First global token held by this rank (0 without context parallelism).
        local_end: One past the last global token held by this rank.
        window_plan: Key lists of every sliding-window layer (width ``sliding_window``).
        dense_plan: Key lists of every sparse layer before selection (width ``index_topk``): exact for documents
            not longer than ``index_topk``; rows of longer documents are replaced by the indexer's selection.
        all_rows_nonempty: Whether this rank holds no padding token (every query has at least its own key).
        valid_rows: Int64 tensor of shape [valid_local_tokens] of the non-padding local queries, or ``None`` when
            ``all_rows_nonempty``.
        cp_group: Context-parallel group whose ranks hold the consecutive shards of the global sequence, or ``None``
            when this rank holds all of it. Every layer gathers its keys over this group.
    """

    doc_ranges: tuple[tuple[int, int], ...]
    local_start: int
    local_end: int
    window_plan: HY4FlashSparsePlan
    dense_plan: HY4FlashSparsePlan
    all_rows_nonempty: bool
    valid_rows: torch.Tensor | None
    cp_group: dist.ProcessGroup | None = None


@dataclass(frozen=True)
class HY4FlashRope:
    """Interleaved M-RoPE tables of one attention type over the *global* sequence.

    Attributes:
        cos: Tensor of shape [global_tokens, rope_dim // 2].
        sin: Tensor of shape [global_tokens, rope_dim // 2].
    """

    cos: torch.Tensor
    sin: torch.Tensor


class HY4FlashFP32Parameter(nn.Module):
    """FP32 parameter holder whose forward returns its weight and ignores the input.

    A bare parameter on a bf16 module would be cast by the FSDP mixed-precision policy. Holding it in its own module lets
    the parallelizer give it a dedicated fp32 unit (and one cross-block unit for the sinks); the ignored input argument
    ties the call to the activation graph so FSDP's pre/post hooks fire.
    """

    def __init__(self, value: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(value.to(torch.float32))

    def forward(self, reference: torch.Tensor) -> torch.Tensor:
        """Return a copy of the weight (``reference`` only anchors the FSDP hooks).

        The copy is what makes the result safe to use after this call: an FSDP unit that reshards after forward
        frees the unsharded parameter storage, so the parameter itself must not outlive the forward.
        """
        del reference
        return self.weight.clone()


class HY4FlashCompressedIndexer(nn.Module):
    """Group-pooled lightning indexer. Selection is discrete, so the indexer receives no loss gradient.

    Token keys are pooled ``index_kpool`` at a time through a softmax over fp32 value / gate projections
    (``key_value_projection`` / ``key_gate_projection`` biased by ``intra_group_ape``), normalised, and rotated
    with the sparse layer's RoPE at the first token of each group. Queries come from the attention's
    ``q_a_layernorm`` output.
    """

    def __init__(self, config: HY4FlashTextConfig) -> None:
        super().__init__()
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.kpool = config.index_kpool
        self.group_topk = config.index_topk // config.index_kpool
        self.rope_dim = config.qk_rope_head_dim
        self.query_projection = nn.Linear(
            config.q_lora_rank, self.n_heads * self.head_dim, bias=False, dtype=torch.bfloat16
        )
        self.head_weight_projection = nn.Linear(config.hidden_size, self.n_heads, bias=False, dtype=torch.bfloat16)
        self.token_key_projection = nn.Linear(config.hidden_size, self.head_dim, bias=False, dtype=torch.bfloat16)
        self.token_key_norm = HY4FlashRMSNorm(self.head_dim, config.rms_norm_eps, torch.bfloat16)
        self.key_value_projection = nn.Linear(self.head_dim, self.head_dim, bias=False, dtype=torch.float32)
        self.key_gate_projection = nn.Linear(self.head_dim, self.head_dim, bias=False, dtype=torch.float32)
        # The released key_norm weight is fp32; keep it exact instead of rounding it to bf16 (the output is cast back below).
        # One call per document, each with that document's group count: keep it off the shape-specialised core.
        self.key_norm = HY4FlashRMSNorm(self.head_dim, config.rms_norm_eps, torch.float32, static_shape=False)
        self.intra_group_ape = HY4FlashFP32Parameter(torch.empty(self.kpool, self.head_dim))
        # Queries are scored in chunks so the fp32 [queries, heads, groups] score block stays within this many bytes:
        # one chunk for a 4K document, 256 queries at a time for a 128K one. ``query_chunk`` pins the chunk instead.
        self.score_block_bytes = 2**29
        self.query_chunk: int | None = None
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def token_keys(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalised per-token index keys.

        Args:
            hidden_states: Tensor of shape [tokens, hidden].

        Returns:
            Tensor of shape [tokens, index_head_dim].
        """
        return self.token_key_norm(self.token_key_projection(hidden_states))

    @torch.no_grad()
    def enter_pooling_units(self, reference: torch.Tensor) -> None:
        """Enter the fp32 pooling modules once with an empty input.

        Pooling runs only for documents longer than ``index_topk``, so how often (and whether) a rank calls these
        modules depends on its data. Under FSDP each call of a sharded unit is a collective (all-gather), and ranks
        that issue different collective sequences deadlock. Entering the units here, unconditionally and in the same
        order on every rank, makes that sequence data independent; later calls find the parameters unsharded.

        Args:
            reference: Any tensor on the compute device (its values are unused).
        """
        empty = reference.new_zeros((0, self.kpool, self.head_dim), dtype=self.key_value_projection.weight.dtype)
        self.key_value_projection(empty)
        self.key_gate_projection(empty)
        self.intra_group_ape(empty)
        self.key_norm(empty[:, 0])

    @torch.no_grad()
    def pool_groups(self, token_keys: torch.Tensor) -> torch.Tensor:
        """Pool one document's token keys into complete-group keys.

        Args:
            token_keys: Tensor of shape [document_tokens, index_head_dim].

        Returns:
            Tensor of shape [document_tokens // index_kpool, index_head_dim] (a trailing partial group has no key).
        """
        groups = token_keys.shape[0] // self.kpool
        grouped = token_keys[: groups * self.kpool].reshape(groups, self.kpool, self.head_dim)
        with torch.autocast(device_type=token_keys.device.type, enabled=False):
            grouped = grouped.to(self.key_value_projection.weight.dtype)
            values = self.key_value_projection(grouped)
            gates = self.key_gate_projection(grouped) + self.intra_group_ape(grouped)
            pooled = (values * gates.softmax(dim=1)).sum(dim=1)
        return self.key_norm(pooled).to(token_keys.dtype)

    @torch.no_grad()
    def select(
        self,
        q_resid: torch.Tensor,
        hidden_states: torch.Tensor,
        query_rope: HY4FlashRope,
        group_keys: torch.Tensor,
        group_rope: HY4FlashRope,
        first_query_slot: int,
    ) -> HY4FlashSparsePlan:
        """Select the document-local tokens every query of one contiguous query interval attends to.

        A query recalls the ``index_topk // index_kpool - 1`` best-scoring groups that closed before its own group
        (ties broken towards the earlier group), plus the tokens of its own group up to itself. Scores are fp32 ``[queries, heads, groups]`` in the
        reference; they are materialised here a chunk of queries at a time and only over the groups the
        chunk can see, with the chunk sized so the block stays within ``score_block_bytes`` whatever the document length.

        Args:
            q_resid: Tensor of shape [queries, q_lora_rank]; consecutive tokens of one document.
            hidden_states: Tensor of shape [queries, hidden].
            query_rope: Tables of shape [queries, rope_dim // 2] at the queries' positions.
            group_keys: Tensor of shape [groups, index_head_dim], unrotated pooled keys of the document's leading
                complete groups; must cover every group that closes before the last query's group.
            group_rope: Tables of shape [groups, rope_dim // 2] at the first token of each group.
            first_query_slot: Document-local token index of the first query.

        Returns:
            Plan with ``indices`` [queries, index_topk] of document-local token indices and ``lengths`` [queries].
        """
        queries = q_resid.shape[0]
        device = q_resid.device
        nope = self.head_dim - self.rope_dim
        keys = torch.cat(
            (
                group_keys[:, :nope],
                apply_interleaved_rope(group_keys[None, :, None, nope:], group_rope.cos[None], group_rope.sin[None])[
                    0, :, 0
                ],
            ),
            dim=-1,
        ).float()
        query = self.query_projection(q_resid).view(queries, self.n_heads, self.head_dim)
        query = torch.cat(
            (
                query[..., :nope],
                apply_interleaved_rope(query[..., nope:].unsqueeze(0), query_rope.cos[None], query_rope.sin[None])[0],
            ),
            dim=-1,
        )
        head_weights = self.head_weight_projection(hidden_states).float() * (self.n_heads**-0.5 * self.head_dim**-0.5)
        slots = first_query_slot + torch.arange(queries, device=device)
        own_group = slots // self.kpool
        recall = self.group_topk - 1
        sentinel = torch.iinfo(torch.int32).max // self.kpool
        # One spare column absorbs the writes of the groups that are not chosen.
        recalled = torch.full((queries, recall + 1), sentinel, dtype=torch.long, device=device)
        chunk = self.query_chunk or max(1, self.score_block_bytes // (4 * self.n_heads * max(1, group_keys.shape[0])))
        for start in range(0, queries, chunk):
            end = min(start + chunk, queries)
            # Groups that closed before the last query of the chunk; later groups are invisible to all of it.
            visible = (first_query_slot + end - 1) // self.kpool
            if visible == 0:
                continue
            # Out of place on purpose: selective activation checkpointing caches matmul / bmm results, and a cached
            # tensor must not be mutated.
            scores = torch.matmul(query[start:end].float().flatten(0, 1), keys[:visible].T).relu()
            logits = torch.bmm(head_weights[start:end, None], scores.view(end - start, self.n_heads, visible))[:, 0]
            group = torch.arange(visible, device=device)
            closed = group < own_group[start:end, None]
            logits = logits.masked_fill(~closed, float("-inf"))
            # Top-k with a total order (score, then lowest group first), so the chosen set does not depend on how
            # the queries are chunked or split over context-parallel ranks: everything above the k-th score, then
            # the lowest-index groups tied with it. ``torch.topk`` alone leaves the order of ties unspecified, and
            # exact ties do occur (a group whose every head score is clipped by the ReLU scores exactly zero).
            budget = min(recall, visible)
            kth = logits.topk(budget, dim=-1).values[:, -1:]
            above = logits > kth
            tied = (logits == kth) & closed
            chosen = above | (tied & (tied.cumsum(-1) <= budget - above.sum(-1, keepdim=True)))
            # Compact the chosen groups to the front in ascending order (the kernel wants ascending key rows).
            slot = torch.where(chosen, chosen.cumsum(-1) - 1, recall)
            recalled[start:end].scatter_(1, slot, group.expand_as(slot))
        recalled = recalled[:, :recall]
        groups_recalled = (recalled < sentinel).sum(dim=-1)
        offsets = torch.arange(self.kpool, device=device)
        tokens = (recalled[:, :, None] * self.kpool + offsets).masked_fill(recalled[:, :, None] == sentinel, -1)
        indices = F.pad(tokens.flatten(1), (0, self.kpool), value=-1)
        own_tokens = own_group[:, None] * self.kpool + offsets
        own_tokens = own_tokens.masked_fill(own_tokens > slots[:, None], -1)
        indices.scatter_(1, groups_recalled[:, None] * self.kpool + offsets, own_tokens)
        lengths = groups_recalled * self.kpool + (slots - own_group * self.kpool + 1)
        return HY4FlashSparsePlan(indices.to(torch.int32), lengths.to(torch.int32))


def reference_sparse_latent_attention(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sink: torch.Tensor,
    scale: float,
    value_dim: int,
    chunk: int = 64,
) -> torch.Tensor:
    """PyTorch implementation of the shared sparse latent attention kernel (CPU/CUDA, any dtype).

    Args:
        q: Tensor of shape [queries, heads, latent_dim + rope_dim].
        kv: Tensor of shape [keys, latent_dim + rope_dim], shared by all heads.
        indices: Integer tensor of shape [queries, width] into ``kv`` rows, ``-1`` marking unused slots.
        sink: fp32 tensor of shape [heads], learned sink logits (scaled-logit space).
        scale: Softmax scale applied to ``q . k``.
        value_dim: Leading channels of ``kv`` used as the value (the latent width).
        chunk: Queries processed at a time.

    Returns:
        Tensor of shape [queries, heads, value_dim] in the dtype of ``q``.
    """
    outputs = []
    for start in range(0, q.shape[0], chunk):
        idx = indices[start : start + chunk].long()
        valid = idx >= 0
        gathered = kv[idx.clamp_min(0)].float()  # [c, width, dim]
        logits = torch.einsum("chd,cwd->chw", q[start : start + chunk].float(), gathered) * scale
        logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
        sink_col = sink.float()[None, :, None].expand(logits.shape[0], -1, 1)
        # A query without keys still has the sink column, so its probabilities are finite and its output is zero.
        probs = torch.cat((logits, sink_col), dim=-1).softmax(dim=-1)[..., :-1]
        outputs.append(torch.einsum("chw,cwd->chd", probs, gathered[..., :value_dim]).to(q.dtype))
    return torch.cat(outputs, dim=0)


class HY4FlashAttention(nn.Module):
    """Gated MLA of one decoder layer (``attention_type`` selects heads, indices and rotary theta)."""

    def __init__(self, config: HY4FlashTextConfig, layer_idx: int, backend: BackendConfig) -> None:
        super().__init__()
        if backend.attn not in ("cudnn", "sdpa"):
            raise ValueError(f"Hy4-flash attention supports backend.attn='cudnn' or 'sdpa', got {backend.attn!r}")
        if backend.rms_norm != "torch":
            raise ValueError(
                f"Hy4-flash builds its own RMSNorm; backend.rms_norm must be 'torch', got {backend.rms_norm!r}"
            )
        self.attention_type: AttentionType = config.attention_types[layer_idx]
        self.backend = backend
        self.num_heads = config.attention_heads(self.attention_type)
        self.nope_dim, self.rope_dim, self.v_dim = config.attention_dims(self.attention_type)
        self.kv_lora_rank = config.kv_lora_rank
        self.index_topk = config.index_topk
        self.scale = (self.nope_dim + self.rope_dim) ** -0.5
        hidden, dtype = config.hidden_size, torch.bfloat16
        linear = backend.linear
        self.q_a_proj = initialize_linear_module(linear, hidden, config.q_lora_rank, config.attention_bias, dtype=dtype)
        self.q_a_layernorm = HY4FlashRMSNorm(config.q_lora_rank, config.rms_norm_eps, dtype)
        self.q_b_proj = initialize_linear_module(
            linear, config.q_lora_rank, self.num_heads * (self.nope_dim + self.rope_dim), False, dtype=dtype
        )
        self.kv_a_proj_with_mqa = initialize_linear_module(
            linear, hidden, config.kv_lora_rank + self.rope_dim, config.attention_bias, dtype=dtype
        )
        self.kv_a_layernorm = HY4FlashRMSNorm(config.kv_lora_rank, config.rms_norm_eps, dtype)
        self.kv_b_proj = initialize_linear_module(
            linear, config.kv_lora_rank, self.num_heads * (self.nope_dim + self.v_dim), False, dtype=dtype
        )
        self.o_proj = initialize_linear_module(
            linear, self.num_heads * self.v_dim, hidden, config.attention_bias, dtype=dtype
        )
        self.gate_proj = initialize_linear_module(linear, hidden, self.num_heads * self.v_dim, False, dtype=dtype)
        self.sinks_param = HY4FlashFP32Parameter(torch.full((self.num_heads,), config.learnable_sink_init))
        self.compressed_indexer = (
            HY4FlashCompressedIndexer(config) if self.attention_type == "indexed_attention" else None
        )

    def setup_cp_attention(self, cp_mesh: DeviceMesh) -> None:
        """Accept the context-parallel mesh from the shared parallelizer without storing it.

        The parallelizer offers the mesh to sliding-window blocks only, so a group stored here would be missing on
        the sparse layers. Every layer instead reads the group from the forward's ``HY4FlashDocLayout.cp_group``
        (keys are gathered over it, queries stay local).
        """
        del cp_mesh

    def _absorbed_weights(self, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-head key / value up-projections of ``kv_b_proj`` as ``[heads, dim, latent]`` tensors.

        Evaluating the module on an identity keeps the result differentiable through LoRA adapters and valid
        when FSDP shards ``kv_b_proj`` (a direct ``.weight`` read would see the sharded DTensor).
        """
        eye = torch.eye(self.kv_lora_rank, dtype=dtype, device=device)
        weight = self.kv_b_proj(eye).T.reshape(self.num_heads, self.nope_dim + self.v_dim, self.kv_lora_rank)
        return weight[:, : self.nope_dim], weight[:, self.nope_dim :]

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        layout: HY4FlashDocLayout,
        rope: HY4FlashRope,
    ) -> torch.Tensor:
        """Attend within documents.

        Args:
            hidden_states: Tensor of shape [1, local_tokens, hidden] (the local interval of the packed row).
            layout: Document boundaries of the packed row.
            rope: Rotary tables of this layer's attention type over the global sequence (the indexer of a sparse
                layer uses the same tables).

        Returns:
            Tensor of shape [1, local_tokens, hidden]; positions outside every document are zero.
        """
        x = hidden_states[0]
        tokens = x.shape[0]
        lo, hi = layout.local_start, layout.local_end
        q_resid = self.q_a_layernorm(self.q_a_proj(x))
        q = self.q_b_proj(q_resid).view(tokens, self.num_heads, self.nope_dim + self.rope_dim)
        q_nope, q_rot = q[..., : self.nope_dim], q[..., self.nope_dim :]
        latent_rot = self.kv_a_proj_with_mqa(x)
        latent = self.kv_a_layernorm(latent_rot[..., : self.kv_lora_rank])
        local_rope = HY4FlashRope(rope.cos[lo:hi], rope.sin[lo:hi])
        q_rot = apply_interleaved_rope(q_rot[None], local_rope.cos[None], local_rope.sin[None])[0]
        k_rot = apply_interleaved_rope(
            latent_rot[None, :, None, self.kv_lora_rank :], local_rope.cos[None], local_rope.sin[None]
        )[0, :, 0]
        gate = torch.sigmoid(self.gate_proj(x)).view(tokens, self.num_heads, self.v_dim)

        w_k, w_v = self._absorbed_weights(x.dtype, x.device)
        q_abs = torch.einsum("qhd,hdc->qhc", q_nope, w_k)
        query = torch.cat((q_abs, q_rot), dim=-1)
        kv_local = torch.cat((latent, k_rot), dim=-1)
        token_keys = self.compressed_indexer.token_keys(x) if self.compressed_indexer is not None else None
        kv, keys = self._gather_keys(kv_local, token_keys, layout.cp_group)

        if self.compressed_indexer is None:
            plan = layout.window_plan
        else:
            plan = self._select(layout, rope, q_resid, x, keys)
        out_latent = self._attend(query, kv, plan, self.sinks_param(query), layout)
        out = torch.einsum("qhc,hvc->qhv", out_latent, w_v) * gate
        return self.o_proj(out.reshape(tokens, -1))[None]

    @torch.no_grad()
    def _select(
        self,
        layout: HY4FlashDocLayout,
        rope: HY4FlashRope,
        q_resid: torch.Tensor,
        hidden_states: torch.Tensor,
        token_keys: torch.Tensor,
    ) -> HY4FlashSparsePlan:
        """Key lists of a sparse layer: dense for short documents, the indexer's selection for long ones.

        Args:
            layout: Document boundaries of the packed row.
            rope: Rotary tables of the sparse layers over the global sequence (the indexer shares them).
            q_resid: Tensor of shape [local_tokens, q_lora_rank].
            hidden_states: Tensor of shape [local_tokens, hidden].
            token_keys: Tensor of shape [global_tokens, index_head_dim], normalised index keys.

        Returns:
            Plan over this rank's queries with global key rows.
        """
        indexer = self.compressed_indexer
        indexer.enter_pooling_units(hidden_states)
        lo, hi = layout.local_start, layout.local_end
        plan = layout.dense_plan
        indices = lengths = None
        for start, end in layout.doc_ranges:
            q_start, q_end = max(start, lo), min(end, hi)
            if end - start <= self.index_topk or q_end <= q_start:
                continue
            if indices is None:
                indices, lengths = plan.indices.clone(), plan.lengths.clone()
            # Only groups that close before the last local query's group can be recalled.
            groups = (q_end - 1 - start) // indexer.kpool
            group_tokens = slice(start, start + groups * indexer.kpool)
            first_tokens = slice(start, start + groups * indexer.kpool, indexer.kpool)
            local = slice(q_start - lo, q_end - lo)
            selected = indexer.select(
                q_resid[local],
                hidden_states[local],
                HY4FlashRope(rope.cos[q_start:q_end], rope.sin[q_start:q_end]),
                indexer.pool_groups(token_keys[group_tokens]),
                HY4FlashRope(rope.cos[first_tokens], rope.sin[first_tokens]),
                q_start - start,
            )
            indices[local] = torch.where(selected.indices >= 0, selected.indices + start, -1)
            lengths[local] = selected.lengths
        return plan if indices is None else HY4FlashSparsePlan(indices, lengths)

    def _gather_keys(
        self, kv_local: torch.Tensor, token_keys: torch.Tensor | None, cp_group: dist.ProcessGroup | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return the latent keys (and index token keys) over the global sequence.

        Without context parallelism the local tensors already are the global ones. With it, only the 576-wide latent
        keys (differentiable: backward reduce-scatters their gradients to the owning rank) and the 128-wide index token
        keys (no gradient) travel, instead of the 3072-wide hidden states.

        Args:
            kv_local: Tensor of shape [local_tokens, kv_lora_rank + rope_dim].
            token_keys: Optional tensor of shape [local_tokens, index_head_dim].
            cp_group: Context-parallel group of the forward, or ``None``.

        Returns:
            The same tensors over the global sequence.
        """
        if cp_group is None:
            return kv_local, token_keys
        kv = gather_sequence(kv_local[None], cp_group)[0]
        if token_keys is None:
            return kv, None
        with torch.no_grad():
            keys = gather_sequence(token_keys[None], cp_group)[0]
        return kv, keys

    def _attend(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        plan: HY4FlashSparsePlan,
        sinks: torch.Tensor,
        layout: HY4FlashDocLayout,
    ) -> torch.Tensor:
        """Run the sparse latent attention kernel (or its PyTorch twin).

        Args:
            query: Tensor of shape [queries, heads, kv_lora_rank + rope_dim].
            kv: Tensor of shape [keys, kv_lora_rank + rope_dim].
            plan: Key lists of the queries (``indices`` [queries, width] into ``kv`` rows, ``lengths`` [queries]).
            sinks: fp32 tensor of shape [heads], the learned sink logits.
            layout: Supplies the padding-row metadata the kernel's backward needs.

        Returns:
            Tensor of shape [queries, heads, kv_lora_rank]; zero for queries without keys.
        """
        if self.backend.attn == "cudnn":
            # The plan is already a compact ascending prefix, so the kernel skips its per-call sort.
            return cudnn_sparse_attention(
                query.contiguous(),
                kv.unsqueeze(1).contiguous(),
                plan.indices.unsqueeze(1),
                self.scale,
                topk_length=plan.lengths,
                all_rows_nonempty=layout.all_rows_nonempty,
                valid_row_indices=layout.valid_rows,
                attn_sink=sinks,
            )
        return reference_sparse_latent_attention(query, kv, plan.indices, sinks, self.scale, self.kv_lora_rank)

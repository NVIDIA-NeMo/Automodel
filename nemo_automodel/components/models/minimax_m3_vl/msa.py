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

"""Packed-document MSA sparse attention for MiniMax M3 on SM100.

One ``MSAMicrobatch`` per microbatch holds everything MSA shares across the attention layers and the
pipeline virtual stages: the canonical document map, the packed document layout, the lazily planned
block scorer and the padding mask (ADR 0010). ``sparse_attention`` runs the official flat forward
and the model-private backward on compact tokens; ``require_msa_support`` is the construction-time
gate. Everything here is BSHD in and BSHD out; the compact ``[tokens, ...]`` layout lives between
``pack`` and ``unpack``.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch.utils.weak import WeakIdKeyDictionary

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m3_vl import msa_bindings
from nemo_automodel.components.models.minimax_m3_vl.kernels import require_sm100
from nemo_automodel.components.models.minimax_m3_vl.kernels.msa_schedule import (
    BLOCK_SIZE,
    HEAD_DIM,
    INDEX_DIM,
    NUM_INDEX_HEADS,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    SOFTMAX_SCALE,
    TOPK_BLOCKS,
    MSABackwardSchedule,
)

_CACHE_ARGUMENTS = ("past_key_values", "cache_position", "page_table", "seqused_k", "prefix_cache")
_CROSS_ATTENTION_ARGUMENTS = ("encoder_hidden_states", "key_value_states")
# The compiled scorer variant is keyed on (dtype, qo_tile, single_wg, sparse_mode, page_size, split_kv,
# pack_factor); with everything else pinned it depends only on the longest document, and these five
# values are the measured boundaries between the reachable variants.
_WARMUP_MAX_DOCS = (16, 32, 64, 128, 256)


def require_msa_support(attention: Any, backend: BackendConfig) -> None:
    """Reject, at construction, an attention layer or backend the MSA kernels are not built for.

    Args:
        attention: The sparse attention layer: ``num_heads``, ``num_kv_heads``, ``head_dim`` and an
            ``indexer`` with ``num_index_heads``, ``block_size``, ``topk_blocks``, ``index_head_dim``
            and ``score_type``.
        backend: The model's backend selection.

    Raises:
        ValueError: If the topology is not the 64-query/4-KV-head, 128-channel, top-16 one, or the
            block score is not the ``max`` reduction.
        NotImplementedError: If the backend asks for FP8 projections or fused RoPE.
    """
    indexer = attention.indexer
    actual = (
        attention.num_heads,
        attention.num_kv_heads,
        attention.head_dim,
        indexer.num_index_heads,
        indexer.block_size,
        indexer.topk_blocks,
        indexer.index_head_dim,
    )
    expected = (NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, NUM_INDEX_HEADS, BLOCK_SIZE, TOPK_BLOCKS, INDEX_DIM)
    if actual != expected:
        # index_head_dim is load-bearing: the fused scorer's QK tile fixes the channel extent at 128 and
        # neither it nor its wrapper checks the argument, so a wider index would be silently truncated.
        raise ValueError(
            "MSA requires (num_heads, num_kv_heads, head_dim, num_index_heads, block_size, topk_blocks, "
            f"index_head_dim) = {expected}; got {actual}."
        )
    if indexer.score_type != "max":
        # The fused scorer reports unscaled QK maxima, so only a reduction that is invariant under a
        # positive rescaling of the logits keeps the same ranking; logsumexp is not.
        raise ValueError(f"MSA requires sparse_score_type='max'; got {indexer.score_type!r}.")
    if backend.te_fp8 is not None:
        raise NotImplementedError(
            "MiniMax M3 MSA first supports BF16 projection only; set backend.te_fp8=None or use sparse_attn='generic'."
        )
    if backend.rope_fusion:
        raise NotImplementedError(
            "MiniMax M3 MSA first supports rope_fusion=False only: the fused BSHD rotary path uses batch row 0's "
            "positions for every row (position_ids_to_freqs_cis), which corrupts packed per-document positions. "
            "Set backend.rope_fusion=False."
        )


def _reject_unsupported_runtime(attn_kwargs: Mapping[str, Any]) -> None:
    """Reject cache/THD/window/cross-attention/capture; tensor kwargs are checked only for presence."""
    qkv_format = attn_kwargs.get("qkv_format", "bshd")
    if qkv_format != "bshd":
        raise NotImplementedError(
            "MiniMax M3 MSA sparse attention supports BSHD (qkv_format='bshd') only; "
            f"got {qkv_format!r}. Set backend.sparse_attn='generic' for THD."
        )
    if attn_kwargs.get("use_cache", False):
        raise NotImplementedError("MiniMax M3 MSA supports cache-free prefill training only; set use_cache=False.")
    for cache_argument in _CACHE_ARGUMENTS:
        if attn_kwargs.get(cache_argument) is not None:
            raise NotImplementedError(
                "MiniMax M3 MSA supports cache-free flat prefill only; "
                f"got non-None {cache_argument}. Remove cache metadata or use sparse_attn='generic'."
            )
    if attn_kwargs.get("is_causal", True) is not True:
        raise NotImplementedError("MiniMax M3 MSA first supports causal self-attention only; set is_causal=True.")
    window_size = attn_kwargs.get("window_size", (-1, 0))
    full_causal_window = window_size is None or (isinstance(window_size, int) and window_size == -1)
    if isinstance(window_size, (tuple, list)):
        full_causal_window = tuple(window_size) == (-1, 0)
    if not full_causal_window:
        raise NotImplementedError(
            "MiniMax M3 MSA first supports full causal attention only; "
            f"got window_size={window_size!r}. Disable the sliding window."
        )
    for cross_attention_argument in _CROSS_ATTENTION_ARGUMENTS:
        if attn_kwargs.get(cross_attention_argument) is not None:
            raise NotImplementedError(
                "MiniMax M3 MSA first supports causal self-attention only; "
                f"got non-None {cross_attention_argument}. Use sparse_attn='generic' for cross-attention."
            )
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise NotImplementedError(
            "MiniMax M3 MSA does not support CUDA graph capture in the first delivery boundary; "
            "run outside capture or use sparse_attn='generic'."
        )


def _document_map(
    hidden: torch.Tensor,
    *,
    packed_seq_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Recover the int64 canonical document map ``[batch, sequence]`` (0 = padding) of one microbatch.

    Only the shape and device of ``hidden`` ``[batch, sequence, hidden]`` are read. Sources, in priority
    order: the packed loader's ``_packed_seq_ids``, a 2-D ``attention_mask`` holding document ids or a bool
    4-D block-causal ``attention_mask`` (decoded by ``_block_causal_documents``), a bool ``padding_mask``
    (one document per row), else one document per row.
    """
    if hidden.dim() != 3:
        raise NotImplementedError(f"MSA requires BSHD hidden states [batch, sequence, hidden], got {hidden.shape}")
    shape = tuple(hidden.shape[:2])
    for name, source in (("_packed_seq_ids", packed_seq_ids), ("attention_mask", attention_mask)):
        if source is None:
            continue
        if name == "attention_mask" and source.dim() == 4:
            return _block_causal_documents(source, shape).to(hidden.device)
        if source.dim() != 2:
            raise ValueError(
                f"MSA requires the compact document map: {name} must have shape [batch, sequence]={shape}, "
                f"got {tuple(source.shape)}. Packed loaders emit it for models that declare consumes_packed_seq_ids."
            )
        if tuple(source.shape) != shape or source.dtype.is_floating_point or source.dtype.is_complex:
            raise ValueError(
                f"{name} must be an integer or bool tensor of shape {shape}, got {tuple(source.shape)} {source.dtype}"
            )
        return source.to(device=hidden.device, dtype=torch.int64).contiguous()
    if padding_mask is not None:
        if tuple(padding_mask.shape) != shape:
            raise ValueError(f"padding_mask must have shape {shape}, got {tuple(padding_mask.shape)}")
        return (~padding_mask.to(device=hidden.device).bool()).to(torch.int64).contiguous()
    return torch.ones(shape, dtype=torch.int64, device=hidden.device)


def _block_causal_documents(mask: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Decode the document map of the dense mask a packed loader builds for a single-document pack.

    Kept until PR #3831 lets ``consumes_packed_seq_ids`` request the compact map for every pack; delete
    this decoder once that merges.

    Args:
        mask: bool ``[batch, 1, sequence, sequence]``; query row ``i`` keeps key ``j`` where true.
        shape: The expected ``(batch, sequence)``.

    Returns:
        int64 ``[batch, sequence]`` document ids, 0 for padding.

    Raises:
        ValueError: If the mask is not standard block-causal (every real query keeps exactly the causal
            keys of its own contiguous document, padding rows all false), checked with one host synchronization.
    """
    batch, sequence = shape
    if tuple(mask.shape) != (batch, 1, sequence, sequence) or mask.dtype != torch.bool:
        raise ValueError(
            f"MSA requires a bool block-causal attention_mask {(batch, 1, sequence, sequence)}, "
            f"got {tuple(mask.shape)} {mask.dtype}"
        )
    keep = mask[:, 0]
    real = torch.diagonal(keep, dim1=-2, dim2=-1)
    previous_visible = torch.diagonal(keep, offset=-1, dim1=-2, dim2=-1)
    starts = torch.cat((real[:, :1], real[:, 1:] & ~previous_visible), dim=-1)
    documents = starts.cumsum(dim=-1, dtype=torch.int64) * real
    positions = torch.arange(sequence, device=mask.device)
    standard = torch.ones((), dtype=torch.bool, device=mask.device)
    for start in range(0, sequence, 256):
        rows = documents[:, start : start + 256].unsqueeze(-1)
        expected = (rows > 0) & (rows == documents.unsqueeze(1)) & (positions <= positions[start : start + 256, None])
        standard &= (keep[:, start : start + 256] == expected).all()
    if not bool(standard.item()):
        raise ValueError(
            "MSA requires a standard bool block-causal attention_mask: each real query must keep exactly the "
            "causal keys of its own contiguous document, with padding rows false."
        )
    return documents.contiguous()


def _contiguous_runs(
    ids: torch.Tensor, batch_rows: torch.Tensor, is_real: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Check that every document is one contiguous run of tokens.

    Args:
        ids: int64 ``[batch * sequence]`` flat document map.
        batch_rows: int64 ``[batch * sequence]`` batch row of each flat token.
        is_real: bool ``[batch * sequence]``; True where ``ids > 0``.

    Returns:
        ``(valid, first_bad_row)`` 0-d tensors: whether no run is interrupted, and the first flat row
        that resumes a document after an interruption (-1 when none).
    """
    external_rows = torch.arange(ids.numel(), device=ids.device)
    if ids.numel() < 2:
        return torch.ones((), dtype=torch.bool, device=ids.device), torch.full((), -1, device=ids.device)
    # Distinct negative padding keys cannot form documents or collide with positive ids.
    key = torch.where(is_real, ids, -(external_rows + 1))
    order = torch.argsort(key, stable=True)
    order = order[torch.argsort(batch_rows[order], stable=True)]
    same_document = (batch_rows[order][1:] == batch_rows[order][:-1]) & (key[order][1:] == key[order][:-1])
    rows = external_rows[order]
    interrupted = same_document & (rows[1:] != rows[:-1] + 1)
    first_bad = torch.where(interrupted, rows[1:], torch.full_like(rows[1:], ids.numel())).min()
    return ~interrupted.any(), torch.where(first_bad == ids.numel(), torch.full_like(first_bad, -1), first_bad)


_MEMO: WeakIdKeyDictionary = WeakIdKeyDictionary()  # _packed_seq_ids -> (version, microbatch)


@dataclass(frozen=True)  # no slots: the scorer plan is a cached_property
class MSAMicrobatch:
    """One packed microbatch's MSA state, built once and shared by every attention layer and stage.

    Attributes:
        padding_mask: bool ``[batch, sequence]``; True where the token is padding.
        token_rows: int64 ``[tokens]`` flat ``batch * sequence`` row of each real token, in document order.
        workspace_positions: int64 ``[tokens]`` row of each compact token in the 128-aligned workspace.
        document_positions: int64 ``[tokens]`` document-local position of each compact token.
        document_workspace_starts: int32 ``[documents]`` workspace row where each document starts.
        cu_seqlens: int32 ``[documents + 1]`` compact document offsets, in ``pack`` row order.
        workspace_size: Rows of the aligned workspace; a positive multiple of 128.
        max_seqlen: Longest document, in tokens.
        forced_blocks: ``(init_blocks, local_blocks)`` of the model's indexer.
    """

    padding_mask: torch.Tensor
    token_rows: torch.Tensor
    workspace_positions: torch.Tensor
    document_positions: torch.Tensor
    document_workspace_starts: torch.Tensor
    cu_seqlens: torch.Tensor
    workspace_size: int
    max_seqlen: int
    forced_blocks: tuple[int, int]

    @classmethod
    def build(
        cls,
        hidden: torch.Tensor,
        *,
        packed_seq_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        padding_mask: torch.Tensor | None,
        attn_kwargs: Mapping[str, Any],
        forced_blocks: tuple[int, int],
    ) -> "MSAMicrobatch":
        """Return this microbatch's state, building it once per batch tensor.

        Every virtual pipeline stage is a deep-copied model that receives the same ``_packed_seq_ids``
        tensor, so its identity plus ``_version`` keys the memo; the entry lives as long as the batch,
        which is why the state keeps no reference to the tensor. Without ``_packed_seq_ids`` the state is
        rebuilt on every call.

        Args:
            hidden: ``[batch, sequence, hidden]`` hidden states; only the shape and device are read.
            packed_seq_ids: The loader's ``[batch, sequence]`` document map, or None.
            attention_mask: A 2-D document map or bool mask, a bool 4-D block-causal mask, or None.
            padding_mask: bool ``[batch, sequence]`` padding mask, or None.
            attn_kwargs: The forward's backend keyword arguments, checked for unsupported runtime features.
            forced_blocks: ``(init_blocks, local_blocks)`` of the model's indexer.

        Raises:
            NotImplementedError: For non-BSHD input, caches, non-causal or windowed attention, or CUDA graph capture.
            ValueError: If no source yields a well-formed document map.
        """
        _reject_unsupported_runtime(attn_kwargs)
        if packed_seq_ids is not None:
            entry = _MEMO.get(packed_seq_ids)
            if entry is not None and entry[0] == packed_seq_ids._version and entry[1].forced_blocks == forced_blocks:
                return entry[1]
        doc_ids = _document_map(
            hidden, packed_seq_ids=packed_seq_ids, attention_mask=attention_mask, padding_mask=padding_mask
        )
        microbatch = cls.from_document_map(doc_ids, forced_blocks=forced_blocks)
        if packed_seq_ids is not None:
            _MEMO[packed_seq_ids] = (packed_seq_ids._version, microbatch)
        return microbatch

    @classmethod
    def from_document_map(cls, doc_ids: torch.Tensor, *, forced_blocks: tuple[int, int]) -> "MSAMicrobatch":
        """Derive the packed layout of ``doc_ids`` with exactly one device-to-host synchronization.

        Args:
            doc_ids: Integer ``[batch, sequence]`` document map: 0 marks padding, each positive id one
                contiguous run of tokens within its row.
            forced_blocks: ``(init_blocks, local_blocks)`` of the model's indexer.

        Raises:
            ValueError: If the map is empty, holds negative ids, no real token, an interrupted document,
                or coordinates past int32.
        """
        if (
            doc_ids.dim() != 2
            or doc_ids.numel() == 0
            or doc_ids.dtype == torch.bool
            or doc_ids.dtype.is_floating_point
            or doc_ids.dtype.is_complex
        ):
            raise ValueError(
                "doc_ids must be a non-empty integer tensor of shape [batch, sequence], "
                f"got {tuple(doc_ids.shape)} {doc_ids.dtype}"
            )
        batch_size, sequence_length = doc_ids.shape
        device = doc_ids.device
        ids = doc_ids.reshape(-1).to(torch.int64)
        num_external = ids.numel()
        external_rows = torch.arange(num_external, device=device)
        batch_rows = torch.div(external_rows, sequence_length, rounding_mode="floor")
        is_real = ids > 0
        previous = (
            torch.cat((ids.new_full((1,), -1), ids[:-1])),
            torch.cat((batch_rows.new_full((1,), -1), batch_rows[:-1])),
        )
        following = (
            torch.cat((ids[1:], ids.new_full((1,), -1))),
            torch.cat((batch_rows[1:], batch_rows.new_full((1,), -1))),
        )
        is_run_start = is_real & ((ids != previous[0]) | (batch_rows != previous[1]))
        is_run_end = is_real & ((ids != following[0]) | (batch_rows != following[1]))
        run_start = torch.where(is_run_start, external_rows, torch.full_like(external_rows, -1)).cummax(0).values
        run_end = (
            torch.where(is_run_end, external_rows, torch.full_like(external_rows, num_external))
            .flip(0)
            .cummin(0)
            .values.flip(0)
        )
        run_length = run_end - run_start + 1
        aligned_run_length = torch.where(
            is_real, -(-run_length // BLOCK_SIZE) * BLOCK_SIZE, torch.zeros_like(run_length)
        )
        aligned_prefix = torch.where(is_run_start, aligned_run_length, torch.zeros_like(run_length)).cumsum(0)
        runs_are_valid, first_bad = _contiguous_runs(ids, batch_rows, is_real)
        probe = torch.stack(
            (
                is_real.sum(),
                is_run_start.sum(),
                aligned_prefix[-1],
                torch.where(is_run_start, run_length, torch.zeros_like(run_length)).max(),
                (ids >= 0).all().to(torch.int64),
                runs_are_valid.to(torch.int64),
                first_bad,
            )
        )
        num_tokens, num_documents, workspace_size, max_seqlen, ids_valid, structure_valid, bad_row = (
            probe.tolist()
        )  # the one sync
        if not ids_valid:
            raise ValueError("doc_ids must be non-negative (0 = padding, positive = document id)")
        if num_tokens == 0:
            raise ValueError("doc_ids must contain at least one real token (a positive document id)")
        if not structure_valid:
            raise ValueError(
                "doc_ids must give each document one contiguous run of tokens; the document at flat token "
                f"index {bad_row} resumes after an interruption"
            )
        if max(num_tokens, workspace_size, max_seqlen) > torch.iinfo(torch.int32).max:
            raise ValueError(
                f"MSA document coordinates must fit int32, got tokens={num_tokens}, workspace_size={workspace_size}, "
                f"max_seqlen={max_seqlen}"
            )
        aligned_start = aligned_prefix - aligned_run_length
        # Prefix ranks permit a fixed-shape scatter; clone token rows before reusing the partition buffer.
        partitioned = torch.empty_like(external_rows)
        real_rank = is_real.cumsum(0) - 1
        partitioned.scatter_(0, torch.where(is_real, real_rank, (~is_real).cumsum(0) + num_tokens - 1), external_rows)
        token_rows = partitioned[:num_tokens].clone()
        run_rank = is_run_start.cumsum(0) - 1
        partitioned.scatter_(
            0, torch.where(is_run_start, run_rank, (~is_run_start).cumsum(0) + num_documents - 1), external_rows
        )
        run_rows = partitioned[:num_documents]
        document_lengths = run_length[run_rows].to(torch.int32)
        return cls(
            padding_mask=~is_real.view(batch_size, sequence_length),
            token_rows=token_rows,
            workspace_positions=(aligned_start + external_rows - run_start)[token_rows].contiguous(),
            document_positions=(external_rows - run_start)[token_rows].contiguous(),
            document_workspace_starts=aligned_start[run_rows].to(torch.int32).contiguous(),
            cu_seqlens=torch.cat(
                (torch.zeros(1, dtype=torch.int32, device=device), document_lengths.cumsum(0, dtype=torch.int32))
            ),
            workspace_size=workspace_size,
            max_seqlen=max_seqlen,
            forced_blocks=forced_blocks,
        )

    def pack(self, external: torch.Tensor) -> torch.Tensor:
        """Gather ``external[batch, sequence, ...]`` to ``[tokens, ...]`` in document order; may alias the input."""
        rows = self.padding_mask.numel()
        flat = external.reshape(rows, *external.shape[2:])
        return flat if self.token_rows.numel() == rows else flat.index_select(0, self.token_rows)

    def unpack(self, packed: torch.Tensor) -> torch.Tensor:
        """Scatter ``packed[tokens, ...]`` back to ``[batch, sequence, ...]`` with zero padding; may alias the input."""
        shape = (*self.padding_mask.shape, *packed.shape[1:])
        if self.token_rows.numel() == self.padding_mask.numel():
            return packed.reshape(shape)
        rows = packed.new_zeros((self.padding_mask.numel(), *packed.shape[1:]))
        return rows.index_copy_(0, self.token_rows, packed).reshape(shape)

    @cached_property
    def _plan(self) -> "_SelectionPlan":
        """The scorer plan of this microbatch, built on the first block selection.

        The first plan of a process also compiles every scorer variant production can reach: the
        ``from_pretrained`` path skips ``initialize_weights``, so this is the one model-owned point both
        load paths pass through before the first scoring pass (ADR 0010).
        """
        device = self.cu_seqlens.device
        require_sm100(device)
        _warm_scorer(device, self.forced_blocks)
        return _SelectionPlan.build(self)

    def select_blocks(self, index_q: torch.Tensor, index_k: torch.Tensor) -> torch.Tensor:
        """Choose each query's key blocks within its own document for one layer.

        Selection is a hard top-k over unnormalized QK maxima, so it is not differentiable: call it
        under ``torch.no_grad``.

        Args:
            index_q: bf16 ``[tokens, 4, 128]`` index queries, post norm and RoPE.
            index_k: bf16 ``[tokens, 1, 128]`` shared index key, post norm and RoPE.

        Returns:
            int32 ``[4, tokens, 16]`` document-local block ids, padded with -1: the canonical support.
        """
        return self._plan.select(index_q, index_k)


@dataclass(frozen=True, slots=True)
class _SelectionPlan:
    """The layer-invariant half of block selection: the FMHA plan, the score shape and each query's geometry.

    Pinned to ``split_prefill_decode=False`` and ``num_kv_splits=1``: the first splits a batch whose
    first document is short into two sub-plans (2.9x on the score pass plus two host syncs per call),
    the second lets the planner pick a variant from an SM-count estimate.
    """

    plan: Any
    score_shape: tuple[int, int, int]
    num_blocks: int
    candidate: torch.Tensor  # bool [tokens, blocks]: blocks of the query's own document at or before its own
    forced: torch.Tensor  # bool [tokens, blocks]: the candidates kept whatever they score

    @classmethod
    def build(cls, msa: MSAMicrobatch) -> "_SelectionPlan":
        """Plan the scorer for ``msa`` and derive its ``[tokens, blocks]`` candidate and forced masks."""
        # fmha_sm100_plan needs host lengths; this is the microbatch's second and last sync.
        lengths = msa.cu_seqlens.diff().cpu()
        plan = msa_bindings.kernels().fmha_sm100_plan(
            lengths,
            lengths,
            NUM_INDEX_HEADS,
            num_kv_heads=1,
            causal=True,
            output_maxscore=True,
            num_kv_splits=1,
            split_prefill_decode=False,
        )
        split_batch, _, _, score_plan, _ = plan
        if split_batch:
            raise RuntimeError("the MSA planner split the batch into decode and prefill sub-plans")
        max_k_tiles = int(score_plan["max_k_tiles"])
        tokens = int(lengths.sum())
        if max_k_tiles <= 0:
            # api.py:616-619 prints and sets max_k_tiles = -1, then falls back to dense attention with
            # max_score=None, which would read as "no blocks selected" rather than as a failure.
            raise RuntimeError(f"the MSA scorer disabled maxscore: {NUM_INDEX_HEADS} * max_k_tiles * {tokens} > 2**31")
        own_block = msa.document_positions // BLOCK_SIZE
        num_blocks = -(-msa.max_seqlen // BLOCK_SIZE)
        blocks = torch.arange(num_blocks, device=own_block.device)
        # A query's document-local position lies inside its own document, so "at or before my own
        # block" is the whole permission.
        candidate = blocks <= own_block[:, None]
        init_blocks, local_blocks = msa.forced_blocks
        forced = blocks < init_blocks
        if local_blocks > 0:
            forced = forced | (blocks == own_block[:, None])
        return cls(plan, (NUM_INDEX_HEADS, max_k_tiles, tokens), num_blocks, candidate, forced & candidate)

    def select(self, index_q: torch.Tensor, index_k: torch.Tensor) -> torch.Tensor:
        """Score bf16 index_q[tokens, 4, 128] against index_k[tokens, 1, 128] -> int32 [4, tokens, 16] block ids."""
        # The score buffer is reused across layers, so tiles the kernel does not write hold the previous
        # layer's values; the selection rule rejects exactly those. v is never read with output_o=False.
        _, max_score = msa_bindings.kernels().fmha_sm100(
            index_q,
            index_k,
            index_k,
            self.plan,
            max_score=_score_scratch(index_q.device, self.score_shape),
            output_o=False,
            output_maxscore=True,
        )
        score = max_score[:, : self.num_blocks].permute(0, 2, 1)
        score = score.masked_fill(~self.candidate, float("-inf")).masked_fill_(self.forced, float("inf"))
        if score.shape[-1] < TOPK_BLOCKS:
            score = F.pad(score, (0, TOPK_BLOCKS - score.shape[-1]), value=float("-inf"))
        values, indices = score.topk(TOPK_BLOCKS, dim=-1)
        return torch.where(values == float("-inf"), -1, indices).to(torch.int32).contiguous()


_SCORE_SCRATCH: dict[torch.device, torch.Tensor] = {}


def _score_scratch(device: torch.device, shape: tuple[int, int, int]) -> torch.Tensor:
    """Return the process-wide score buffer of ``device``, grown to ``shape``.

    The scorer stores rather than accumulates and every tile the selection rule reads is written by the
    same call, so one buffer serves every layer and microbatch; scoring passes never overlap because MSA
    is single-stream and rejects CUDA-graph capture. ``max_k_tiles`` is rounded up to 128 tiles whatever
    the documents are, so per-plan buffers would cost 224 MiB where this one costs 11.28 MiB.
    """
    elements = shape[0] * shape[1] * shape[2]
    buffer = _SCORE_SCRATCH.get(device)
    if buffer is None or buffer.numel() < elements:
        buffer = _SCORE_SCRATCH[device] = torch.empty(elements, dtype=torch.float32, device=device)
    return buffer[:elements].view(shape)


@lru_cache(maxsize=None)
def _warm_scorer(device: torch.device, forced_blocks: tuple[int, int]) -> None:
    """Compile every reachable scorer variant once per process and device, ~44 s each on a cold cache.

    Warming runs the production path over synthetic one-document microbatches, so the variant compiled
    here is the variant production reaches by construction.
    """
    for max_doc in _WARMUP_MAX_DOCS:
        microbatch = MSAMicrobatch.from_document_map(
            torch.ones((1, max_doc), dtype=torch.int64, device=device), forced_blocks=forced_blocks
        )
        _SelectionPlan.build(microbatch).select(
            torch.zeros((max_doc, NUM_INDEX_HEADS, INDEX_DIM), dtype=torch.bfloat16, device=device),
            torch.zeros((max_doc, 1, INDEX_DIM), dtype=torch.bfloat16, device=device),
        )


def sparse_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q2k: torch.Tensor, msa: MSAMicrobatch
) -> torch.Tensor:
    """Run MSA sparse attention on compact tokens.

    Args:
        q: bf16 ``[tokens, 64, 128]`` queries after RoPE.
        k: bf16 ``[tokens, 4, 128]`` keys after RoPE.
        v: bf16 ``[tokens, 4, 128]`` values.
        q2k: int32 ``[4, tokens, 16]`` canonical support from ``msa.select_blocks``.
        msa: The microbatch the tokens were packed by.

    Returns:
        bf16 ``[tokens, 64, 128]`` attention output; ``q``, ``k`` and ``v`` receive gradients.

    Raises:
        NotImplementedError: Under ``torch.use_deterministic_algorithms``: the backward accumulates dK/dV
            with FP32 atomics and dQ with packed bf16 atomics, so it is not bitwise deterministic.
        ValueError: If ``q``, ``k`` or ``v`` is not bf16.
    """
    if torch.are_deterministic_algorithms_enabled():
        raise NotImplementedError(
            "MiniMax M3 MSA backward accumulates dK/dV with FP32 atomics and dQ with packed 16-bit atomics, "
            "so it is not bitwise deterministic; disable torch deterministic algorithms or use sparse_attn='generic'."
        )
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError(f"MiniMax M3 MSA first supports BF16 q/k/v only; got q={q.dtype}, k={k.dtype}, v={v.dtype}.")
    return _SparseAttention.apply(q, k, v, q2k, msa)


def _aligned(compact: torch.Tensor, positions: torch.Tensor, workspace_size: int) -> torch.Tensor:
    """Scatter ``compact[tokens, H, D]`` to rows ``positions[tokens]`` of a zero-filled ``[workspace_size, H, D]``."""
    return compact.new_zeros((workspace_size, *compact.shape[1:])).index_copy_(0, positions, compact)


class _SparseAttention(torch.autograd.Function):
    """The official flat forward, its saved schedule, and the backward-only aligned K/V workspace."""

    @staticmethod
    def forward(
        ctx: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q2k: torch.Tensor, msa: MSAMicrobatch
    ) -> torch.Tensor:
        """Run bf16 q[tokens, 64, 128], k/v[tokens, 4, 128], int32 q2k[4, tokens, 16] -> bf16 out[tokens, 64, 128]."""
        ctx.set_materialize_grads(False)
        kernels = msa_bindings.kernels()
        cu_seqlens, max_seqlen = msa.cu_seqlens, msa.max_seqlen
        # The CSR extension lacks a CUDAGuard; bind both launches to the tensor's device.
        with torch.cuda.device(q.device):
            row_ptr, q_indices, schedule = kernels.build_k2q_csr(
                q2k,
                cu_seqlens,
                cu_seqlens,
                BLOCK_SIZE,
                total_k=q.shape[0],
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                total_rows=msa.workspace_size // BLOCK_SIZE,
                qhead_per_kv=NUM_Q_HEADS // NUM_KV_HEADS,
                return_schedule=True,
            )
            out, lse = kernels.sparse_atten_func(
                q,
                k,
                v,
                row_ptr,
                q_indices,
                TOPK_BLOCKS,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                blk_kv=BLOCK_SIZE,
                causal=True,
                softmax_scale=SOFTMAX_SCALE,
                partial_dtype=torch.bfloat16,
                return_softmax_lse=True,
                schedule=schedule,
            )
        ctx.save_for_backward(
            q,
            k,
            v,
            out,
            lse,
            row_ptr,
            q_indices,
            schedule.scheduler_metadata,
            schedule.work_count,
            msa.workspace_positions,
            msa.document_workspace_starts,
            cu_seqlens,
        )
        ctx.workspace_size = msa.workspace_size
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_out: torch.Tensor | None) -> tuple[Any, ...]:
        """Map bf16 grad_out[tokens, 64, 128] to dq[tokens, 64, 128], dk/dv[tokens, 4, 128] and two None slots."""
        q, k, v, out, lse, row_ptr, q_indices, scheduler_metadata, work_count, positions, starts, cu_seqlens = (
            ctx.saved_tensors
        )
        if grad_out is None:
            grad_out = torch.zeros_like(out)
        schedule = MSABackwardSchedule(row_ptr, q_indices, scheduler_metadata, work_count, cu_seqlens, starts)
        with torch.cuda.device(q.device):
            dq, dk_aligned, dv_aligned = msa_bindings.kernels().run_backward(
                q=q,
                k_aligned=_aligned(k, positions, ctx.workspace_size),
                v_aligned=_aligned(v, positions, ctx.workspace_size),
                grad_out=grad_out,
                lse=lse,
                out=out,
                schedule=schedule,
            )
        return dq, dk_aligned.index_select(0, positions), dv_aligned.index_select(0, positions), None, None

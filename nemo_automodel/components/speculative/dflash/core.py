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

"""DFlash online training wrapper.

Ported from SpecForge's ``specforge/core/dflash.py``. ``DFlashTrainerModule``
samples a set of anchor positions per sequence, builds one parallel draft block
per anchor (the block's first token is the real anchor token, the rest are
``MASK``), runs the draft model under a bespoke block attention mask, and
computes a block-wise cross-entropy loss against the ground-truth continuation
of each anchor.

Three training objectives are supported via ``loss_type``:

* ``"dflash"`` (default): the DFlash paper's fixed-anchor objective. Only block
  position 0 is a real token; positions ``1..block_size-1`` are supervised with
  the decay-weighted CE of Eq. 4 (``w_k = exp(-(k-1)/gamma)``).
* ``"dpace*"``: the D-PACE objective (arXiv:2605.18810). Same fixed-anchor block
  as ``"dflash"``, but the CE is reweighted by detached Dynamic Position-Aware
  weights built from the draft's own confidence (see ``DFlashDecayLoss``);
  ``dpace_alpha`` smooths those confidence products.
* ``"variable_prefix"``: the D2SD VP-Drafter objective (arXiv:2606.04446). Each
  block draws a visible-prefix length ``l`` from a truncated geometric prior
  (``Pr(l) ~ prefix_weight_base ** l``), positions ``< l`` are filled with the
  real tokens, and only the masked positions ``>= l`` are supervised, with the
  decay re-anchored at the prefix boundary (``w_k = exp(-(k-l)/gamma)``). This
  trains the draft to re-draft block suffixes behind a variable accepted
  prefix, the regime a variable-prefix drafter sees at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask

from nemo_automodel.components.attention.dflash_mask import (
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from nemo_automodel.components.loss.dllm_loss import _DFLASH_LOSS_TYPES as _DECAY_LOSS_TYPES
from nemo_automodel.components.loss.dllm_loss import DFlashDecayLoss
from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel


def _context_doc_ids(seq_lens: torch.Tensor, seq_len: int, device: torch.device) -> torch.Tensor:
    """Per-context-token document id ``[B, S]`` from packed ``seq_lens`` ``[B, max_docs]``.

    Mirrors the ``doc_id`` construction in ``build_block_causal_additive_mask``: a
    token's id is the number of document boundaries at or before its position, so
    0-length padding entries never split a real document.
    """
    boundaries = seq_lens.to(device).cumsum(dim=1)  # [B, max_docs]
    positions = torch.arange(seq_len, device=device)
    return (boundaries.unsqueeze(1) <= positions.view(1, -1, 1)).sum(dim=2)


def _to_full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Materialise a (possibly tensor-parallel) tensor as a plain local tensor.

    Under tensor parallelism the target's column-parallel ``lm_head`` and
    vocab-parallel ``embed_tokens`` return ``DTensor`` outputs. The draft and the
    block-wise loss consume plain tensors, so gather the full tensor. A no-op for
    an already-plain (unsharded / replicated) tensor.
    """
    return tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor


class NoValidAnchorsError(ValueError):
    """Raised when a batch has no sample long enough to form a DFlash block.

    A DFlash anchor needs at least ``block_size + 1`` supervised tokens (the
    anchor plus its block). Datasets always contain some short conversations;
    the training loop catches this and skips the offending micro-batch rather
    than aborting the run.
    """


@dataclass
class DFlashStepMetrics:
    """Per-step training outputs for the DFlash draft."""

    loss: torch.Tensor
    accuracy: torch.Tensor
    valid_tokens: torch.Tensor


# DFlashDecayLoss owns the fixed-anchor objectives (``dflash`` plus the D-PACE
# variants); ``variable_prefix`` computes its loss inline here. Deriving from its
# set keeps the two in sync when a new decay-loss variant is added.
_DFLASH_LOSS_TYPES = tuple(sorted(_DECAY_LOSS_TYPES)) + ("variable_prefix",)


class DFlashTrainerModule(nn.Module):
    """DFlash online training wrapper with block-wise CE loss."""

    def __init__(
        self,
        draft_model: Qwen3DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
        loss_type: str = "dflash",
        dpace_alpha: float = 0.5,
        prefix_weight_base: float = 0.9,
    ):
        super().__init__()
        if loss_type not in _DFLASH_LOSS_TYPES:
            raise ValueError(f"loss_type must be one of {_DFLASH_LOSS_TYPES}, got {loss_type!r}")
        if prefix_weight_base <= 0:
            raise ValueError(f"prefix_weight_base must be > 0, got {prefix_weight_base}")
        self.draft_model = draft_model
        # Keep the frozen target lm_head / embed_tokens as NON-registered
        # references. Under tensor parallelism their weights are DTensors; a
        # registered (DDP-wrapped) sharded param would break DDP's parameter
        # broadcast/bucketing. Non-registering keeps the DDP-wrapped trainer to
        # the plain draft params only, while ``self.lm_head`` / ``self.embed_tokens``
        # attribute access still works. (Mirrors EAGLE core_v12's _target_lm_head.)
        object.__setattr__(self, "lm_head", target_lm_head)
        object.__setattr__(self, "embed_tokens", target_embed_tokens)
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma
        self.loss_type = loss_type
        self.dpace_alpha = float(dpace_alpha)
        self.prefix_weight_base = float(prefix_weight_base)
        # Smallest visible-prefix length variable-prefix training samples (and the
        # slice point of its loss); single source of truth for both methods.
        self._min_prefix = min(2, block_size - 1)

        # Block-wise decay-weighted CE for the fixed-anchor objectives (``dflash``
        # and the D-PACE variants). ``normalize="mean"`` gives a local per-micro-batch
        # decay-weighted mean; ``loss_decay_gamma=None`` disables decay (uniform
        # weights). The variable-prefix objective needs per-block data-dependent
        # weights and computes its loss inline instead (see _variable_prefix_loss).
        self.loss_fn = (
            None
            if loss_type == "variable_prefix"
            else DFlashDecayLoss(
                loss_gamma=loss_decay_gamma,
                normalize="mean",
                loss_type=loss_type,
                dpace_alpha=dpace_alpha,
            )
        )

        # Per-block offset constant (block_size,) for label gathering / position ids.
        self.register_buffer("_block_offsets", torch.arange(block_size).view(1, 1, -1), persistent=False)

    def _sample_anchor_positions(
        self,
        seq_len: int,
        loss_mask: torch.Tensor,
        device: torch.device,
        doc_remaining: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns ``(anchors, keep_mask)``.

        ``doc_remaining`` ``[B, S]`` (sequence packing) restricts anchors so the
        whole block stays inside one document (``doc_remaining >= block_size - 1``),
        the per-document analogue of the ``anchor <= seq_len - block_size`` bound.
        This is required for correctness -- ``_build_block_targets`` gathers labels
        by absolute offset and does not encode document boundaries, so a block that
        crossed one would be supervised on the next document's tokens. A side effect
        is that a packed document shorter than ``block_size`` yields no anchors (the
        unpacked path still supervises such a short sequence's partial block); pack
        with documents at least ``block_size`` long to avoid dropping their signal.
        """
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        if doc_remaining is not None:
            # Keep the block within the anchor's document: its last predicted
            # token (anchor + block_size - 1) must still be a real token of that
            # document, i.e. at least block_size - 1 real tokens follow the anchor.
            valid = valid & (doc_remaining[:, : max_anchor + 1] >= bs - 1)
        valid_counts = valid.sum(dim=1)
        # ``valid`` already restricts positions to ``[0, seq_len - block_size]``, so
        # every valid position has room for a full block and is a legitimate anchor.
        # Draw up to the richest sample's valid count (per-sample padding is handled
        # by ``keep_mask`` below); no -1, which would spuriously raise when the
        # richest sample has exactly one valid anchor and always drop one otherwise.
        max_n = min(self.num_anchors, int(valid_counts.max().item()))
        if max_n <= 0:
            raise NoValidAnchorsError(
                "No valid anchor positions in this batch; every sample has fewer than "
                f"block_size+1 ({bs + 1}) supervised tokens."
            )

        indices = torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        masked_indices = torch.where(valid, indices, torch.tensor(seq_len + 1, device=device))

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device))
        return anchors, keep_mask

    def _sample_prefix_lengths(self, bsz: int, n_blocks: int, device: torch.device) -> torch.Tensor:
        """Sample a visible-prefix length per block for variable-prefix training.

        A prefix length ``l`` means block positions ``[0, l)`` are visible real
        tokens and positions ``[l, block_size)`` are masked prediction targets.
        Lengths follow D2SD's truncated geometric prior ``Pr(l) ~ base ** l`` over
        ``[min(2, block_size - 1), block_size - 1]``; a base below 1 biases toward
        short prefixes. The lower bound skips the degenerate fixed-anchor DFlash
        case, and the upper bound keeps at least one masked target per block.

        Returns:
            prefix_lengths: Long tensor of shape ``[batch, blocks]``.
        """
        min_prefix = self._min_prefix
        max_prefix = self.block_size - 1
        if max_prefix <= min_prefix:
            return torch.full((bsz, n_blocks), min_prefix, dtype=torch.long, device=device)
        prefix_ids = torch.arange(min_prefix, max_prefix + 1, device=device, dtype=torch.float32)
        weights = self.prefix_weight_base**prefix_ids
        samples = torch.multinomial(weights, num_samples=bsz * n_blocks, replacement=True)
        return samples.view(bsz, n_blocks) + min_prefix

    def _create_position_ids(
        self, anchor_positions: torch.Tensor, context_position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Position ids for the parallel draft blocks (anchor position + offset).

        Without packing the anchor's position equals its row index, so the block
        positions are ``anchor + offset``. Under packing ``context_position_ids``
        ``[B, S]`` holds per-document reset positions, so the block's base position
        is gathered from it at the anchor (``context_position_ids[anchor] + offset``)
        to keep the draft's RoPE phase document-local.
        """
        bsz = anchor_positions.shape[0]
        offsets = torch.arange(self.block_size, device=anchor_positions.device).view(1, 1, -1)
        if context_position_ids is None:
            base = anchor_positions.unsqueeze(-1)
        else:
            base = torch.gather(context_position_ids, 1, anchor_positions.clamp(min=0)).unsqueeze(-1)
        pos_ids = base + offsets
        return pos_ids.view(bsz, -1)

    def _create_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
        """Embed each block as ``[anchor_token, MASK, MASK, ...]`` (invalid blocks all MASK)."""
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        noise_ids = torch.full((bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device)
        block_starts = (torch.arange(n, device=device) * bs).unsqueeze(0).expand(bsz, -1)

        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)

        flat_batch_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        noise_ids[flat_batch_idx, block_starts] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )
        # A tensor-parallel target's embed_tokens is vocab-parallel and returns a
        # DTensor; gather it so the (plain) draft can consume the noise embedding.
        return _to_full_tensor(self.embed_tokens(noise_ids))

    def _create_vp_noise_embed(self, input_ids, anchor_positions, block_keep_mask, prefix_lengths):
        """Embed the draft blocks with a visible prefix of real tokens, then ``MASK``.

        The variable-prefix analogue of :meth:`_create_noise_embed`: block
        positions ``< prefix_lengths`` hold the real sequence tokens, the rest
        (and every position of an invalid block) hold ``MASK``.

        Args:
            input_ids: Long tensor of shape ``[batch, sequence]``.
            anchor_positions: Long tensor of shape ``[batch, blocks]``; each block's
                start position in the sequence.
            block_keep_mask: Bool tensor of shape ``[batch, blocks]``; invalid
                (padding) blocks are embedded as all ``MASK``.
            prefix_lengths: Long tensor of shape ``[batch, blocks]``; this block's
                visible-prefix length.

        Returns:
            noise_embedding: Tensor of shape ``[batch, blocks * block_size, hidden]``.
        """
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]

        token_positions = anchor_positions.unsqueeze(-1) + self._block_offsets  # [B, N, bs]
        safe_positions = token_positions.clamp(0, seq_len - 1)
        real_tokens = torch.gather(input_ids.unsqueeze(1).expand(-1, n, -1), 2, safe_positions)
        visible_prefix = self._block_offsets < prefix_lengths.unsqueeze(-1)
        fill_mask = visible_prefix & block_keep_mask.unsqueeze(-1) & (token_positions < seq_len)
        mask_tokens = torch.full_like(real_tokens, self.mask_token_id)
        noise_ids = torch.where(fill_mask, real_tokens, mask_tokens).view(bsz, n * self.block_size)
        # A tensor-parallel target's embed_tokens is vocab-parallel and returns a
        # DTensor; gather it so the (plain) draft can consume the noise embedding.
        return _to_full_tensor(self.embed_tokens(noise_ids))

    def _build_block_targets(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        seq_len: int,
        label_start: int = 0,
        doc_remaining: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-block ground-truth tokens and the supervised-position mask.

        Returns ``(label_indices, target_ids, block_mask)`` each of shape
        ``[B, N, block_size]``. ``label_indices[..., k]`` is the sequence position
        block position ``k`` predicts (``anchor + label_start + k``); ``block_mask``
        is the product of block validity, in-bounds, and the gathered loss mask.
        Shared by the block-wise trainers (DFlash, JetSpec, and Domino, which passes
        ``label_start=1`` for ``shift_label``) so the label/mask gathering lives in
        one place. Under packing, ``doc_remaining`` ``[B, S]`` truncates labels at
        the anchor's document boundary: anchor sampling only keeps offsets up to
        ``block_size - 1`` inside the anchor's document, and a shifted label window
        (``label_start > 0``) reaches one past that guarantee.
        """
        n = anchor_positions.size(1)
        label_offsets = self._block_offsets + label_start  # [1, 1, bs]
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets  # [B, N, bs]
        valid_label_mask = label_indices < seq_len
        if doc_remaining is not None:
            doc_rem_at_anchor = torch.gather(doc_remaining, 1, anchor_positions.clamp(min=0)).unsqueeze(-1)
            valid_label_mask = valid_label_mask & (label_offsets <= doc_rem_at_anchor)
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        target_ids = torch.gather(input_ids.unsqueeze(1).expand(-1, n, -1), 2, safe_label_indices)
        gathered_loss_mask = torch.gather(loss_mask.unsqueeze(1).expand(-1, n, -1), 2, safe_label_indices)
        block_mask = block_keep_mask.unsqueeze(-1).float() * valid_label_mask.float() * gathered_loss_mask
        return label_indices, target_ids, block_mask

    def _prepare_block_inputs(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        doc_remaining: torch.Tensor | None = None,
        causal: bool = False,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, "torch.Tensor | BlockMask", Optional[torch.Tensor]
    ]:
        """Shared block-drafting prologue: anchors, noise embedding, positions, mask.

        Centralises the sequence-packing handling for every block-wise trainer
        (DFlash and its Domino / JetSpec subclasses): anchors are sampled so each
        block stays inside one document, the block's context prefix is restricted
        to its anchor's document, and the draft RoPE uses per-document positions.

        Under ``loss_type="variable_prefix"`` the block embedding shows a sampled
        real-token prefix (``_create_vp_noise_embed``) instead of the single anchor
        token, and the sampled ``prefix_lengths`` are returned for the loss; every
        other ``loss_type`` returns ``prefix_lengths=None``.

        Args:
            input_ids:     ``[B, S]`` context token ids (long).
            loss_mask:     ``[B, S]`` supervised-token mask.
            position_ids:  ``[B, S]`` per-document reset positions (packing), or ``None``.
            seq_lens:      ``[B, max_docs]`` packed document lengths, or ``None`` (unpacked).
            doc_remaining: ``[B, S]`` remaining real tokens of each position's document.
            causal:        When True, build the in-block-causal (JetSpec) mask instead
                of the bidirectional (DFlash / Domino) one.

        Returns:
            ``(anchor_positions [B, N], block_keep_mask [B, N], noise_embedding
            [B, N*block_size, H], full_position_ids [B, S + N*block_size],
            attention mask, prefix_lengths [B, N] or None)``; the mask is a flex
            ``BlockMask`` or a dense additive ``[B, 1, N*block_size, S + N*block_size]``
            tensor, per backend.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        packed = seq_lens is not None
        if packed and (position_ids is None or doc_remaining is None):
            # A partial set would silently drop the in-document anchor constraint
            # (cross-document supervision) or the per-document RoPE positions.
            raise ValueError(
                "Sequence packing requires position_ids, seq_lens, and doc_remaining together; "
                "got seq_lens without the other packing metadata."
            )

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device, doc_remaining=doc_remaining if packed else None
        )
        if self.loss_type == "variable_prefix":
            prefix_lengths = self._sample_prefix_lengths(bsz, anchor_positions.shape[1], device)
            noise_embedding = self._create_vp_noise_embed(input_ids, anchor_positions, block_keep_mask, prefix_lengths)
        else:
            prefix_lengths = None
            noise_embedding = self._create_noise_embed(input_ids, anchor_positions, block_keep_mask)

        if packed:
            context_position_ids = position_ids
            ctx_doc_id = _context_doc_ids(seq_lens, seq_len, device)
            anchor_doc_id = torch.gather(ctx_doc_id, 1, anchor_positions.clamp(min=0))
        else:
            context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
            ctx_doc_id = None
            anchor_doc_id = None
        draft_position_ids = self._create_position_ids(anchor_positions, context_position_ids if packed else None)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        if self.attention_backend == "flex_attention":
            attn_mask = create_dflash_block_mask(
                anchor_positions,
                block_keep_mask,
                seq_len,
                self.block_size,
                device,
                causal=causal,
                ctx_doc_id=ctx_doc_id,
                anchor_doc_id=anchor_doc_id,
            )
        else:
            attn_mask = create_dflash_sdpa_mask(
                anchor_positions,
                block_keep_mask,
                seq_len,
                self.block_size,
                device,
                dtype=noise_embedding.dtype,
                causal=causal,
                ctx_doc_id=ctx_doc_id,
                anchor_doc_id=anchor_doc_id,
            )
        return anchor_positions, block_keep_mask, noise_embedding, full_position_ids, attn_mask, prefix_lengths

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        doc_remaining: torch.Tensor | None = None,
    ) -> DFlashStepMetrics:
        """Parallel block-wise training forward pass.

        Sequence packing (``position_ids`` ``[B, S]`` per-document reset positions,
        ``seq_lens`` ``[B, max_docs]`` document lengths, ``doc_remaining`` ``[B, S]``)
        keeps every block inside one document: anchors are constrained so the block
        does not cross a boundary, the block's context prefix attends only within the
        anchor's document, and the draft's RoPE uses the per-document positions.
        """
        bsz, seq_len = input_ids.shape

        anchor_positions, block_keep_mask, noise_embedding, full_position_ids, dflash_attn_mask, prefix_lengths = (
            self._prepare_block_inputs(
                input_ids, loss_mask, position_ids=position_ids, seq_lens=seq_lens, doc_remaining=doc_remaining
            )
        )

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
        )
        # A tensor-parallel target's lm_head is column-parallel and returns
        # vocab-sharded (DTensor) logits; gather to a full tensor for the loss.
        logits = _to_full_tensor(self.lm_head(output_hidden))

        n = anchor_positions.size(1)
        bs = self.block_size

        # Block position k predicts the token at anchor + k.
        _, target_ids, block_mask = self._build_block_targets(
            input_ids, loss_mask, anchor_positions, block_keep_mask, seq_len
        )

        if self.loss_type == "variable_prefix":
            return self._variable_prefix_loss(logits.view(bsz, n, bs, -1), target_ids, block_mask, prefix_lengths)

        # Drop block position 0 (the clean anchor token, never a target); the
        # remaining bs-1 predicted positions are what the loss supervises.
        pred_logits = logits.view(bsz, n, bs, -1)[:, :, 1:, :].reshape(bsz, n * (bs - 1), -1)
        pred_targets = target_ids[:, :, 1:].reshape(bsz, n * (bs - 1))
        pred_mask = block_mask[:, :, 1:].reshape(bsz, n * (bs - 1))

        loss_fn = self.loss_fn
        assert loss_fn is not None, "loss_fn is constructed for every loss_type except 'variable_prefix'"
        loss_out = loss_fn(pred_logits, pred_targets, pred_mask, num_tokens=None, block_size=bs)

        count_per_pos = loss_out.draft_count_per_pos
        valid_tokens = count_per_pos.sum()
        accuracy = loss_out.draft_correct_per_pos.sum() / (valid_tokens + 1e-6)

        return DFlashStepMetrics(
            loss=loss_out.total_loss, accuracy=accuracy.detach(), valid_tokens=valid_tokens.detach()
        )

    def _variable_prefix_loss(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        block_mask: torch.Tensor,
        prefix_lengths: torch.Tensor,
    ) -> DFlashStepMetrics:
        """Decay-weighted CE over each block's masked suffix (D2SD Eq. for L_VP).

        Only positions at or past the sampled visible prefix are supervised, and
        the exponential decay restarts at the prefix boundary:
        ``w_k = exp(-(k - l) / gamma)`` for block position ``k >= l``
        (``loss_decay_gamma=None`` disables decay). The loss is the weighted mean
        ``sum(nll * w) / sum(w)``, mirroring the fixed-anchor path's
        ``normalize="mean"``. Assumes every ``prefix_lengths`` entry is at least
        ``min(2, block_size - 1)`` (what :meth:`_sample_prefix_lengths` produces),
        so the leading always-visible positions can be sliced off before the CE.

        Args:
            logits: Tensor of shape ``[batch, blocks, block_size, vocab]``.
            target_ids: Long tensor of shape ``[batch, blocks, block_size]``; the
                ground-truth token at ``anchor + k`` for block position ``k``.
            block_mask: Tensor of shape ``[batch, blocks, block_size]``; 0/1
                product of block validity, in-bounds, and the loss mask.
            prefix_lengths: Long tensor of shape ``[batch, blocks]``; this block's
                visible-prefix length.

        Returns:
            DFlashStepMetrics with the weighted-mean loss, the argmax accuracy
            over supervised positions, and the supervised-position count.
        """
        # Positions below the minimum prefix are visible in every block; drop them
        # before the CE like the fixed-anchor path drops position 0.
        min_prefix = self._min_prefix
        offsets = self._block_offsets[..., min_prefix:]
        logits = logits[:, :, min_prefix:, :]
        target_ids = target_ids[:, :, min_prefix:]
        block_mask = block_mask[:, :, min_prefix:]

        supervised = block_mask * (offsets >= prefix_lengths.unsqueeze(-1)).float()
        weights = supervised
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            effective_pos = (offsets - prefix_lengths.unsqueeze(-1)).clamp(min=0).float()
            weights = supervised * torch.exp(-effective_pos / self.loss_decay_gamma)

        vocab = logits.shape[-1]
        token_nll = F.cross_entropy(logits.reshape(-1, vocab), target_ids.reshape(-1), reduction="none").view_as(
            supervised
        )
        loss = (token_nll * weights).sum() / (weights.sum() + 1e-6)

        valid_tokens = supervised.sum()
        correct = ((logits.argmax(dim=-1) == target_ids).float() * supervised).sum()
        accuracy = correct / (valid_tokens + 1e-6)
        return DFlashStepMetrics(loss=loss, accuracy=accuracy.detach(), valid_tokens=valid_tokens.detach())

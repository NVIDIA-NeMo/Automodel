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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import BlockMask

from nemo_automodel.components.attention.dflash_mask import (
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
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
    ):
        super().__init__()
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

        # Block-wise decay-weighted CE. ``normalize="mean"`` gives a local
        # per-micro-batch decay-weighted mean; ``loss_decay_gamma=None`` disables
        # decay (uniform weights).
        self.loss_fn = DFlashDecayLoss(loss_gamma=loss_decay_gamma, normalize="mean")

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

    def _build_block_targets(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-block ground-truth tokens and the supervised-position mask.

        Returns ``(label_indices, target_ids, block_mask)`` each of shape
        ``[B, N, block_size]``. ``label_indices[..., k]`` is the sequence position
        block position ``k`` predicts (``anchor + k``); ``block_mask`` is the product
        of block validity, in-bounds, and the gathered loss mask. Shared by the
        block-wise trainers (DFlash here, JetSpec) so the label/mask gathering lives
        in one place.
        """
        n = anchor_positions.size(1)
        label_indices = anchor_positions.unsqueeze(-1) + self._block_offsets  # [B, N, bs]
        valid_label_mask = label_indices < seq_len
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, "torch.Tensor | BlockMask"]:
        """Shared block-drafting prologue: anchors, noise embedding, positions, mask.

        Centralises the sequence-packing handling for every block-wise trainer
        (DFlash and its Domino / JetSpec subclasses): anchors are sampled so each
        block stays inside one document, the block's context prefix is restricted
        to its anchor's document, and the draft RoPE uses per-document positions.

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
            attention mask)``; the mask is a flex ``BlockMask`` or a dense additive
            ``[B, 1, N*block_size, S + N*block_size]`` tensor, per backend.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        packed = seq_lens is not None

        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device, doc_remaining=doc_remaining if packed else None
        )
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
        return anchor_positions, block_keep_mask, noise_embedding, full_position_ids, attn_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        doc_remaining: Optional[torch.Tensor] = None,
    ) -> DFlashStepMetrics:
        """Parallel block-wise training forward pass.

        Sequence packing (``position_ids`` ``[B, S]`` per-document reset positions,
        ``seq_lens`` ``[B, max_docs]`` document lengths, ``doc_remaining`` ``[B, S]``)
        keeps every block inside one document: anchors are constrained so the block
        does not cross a boundary, the block's context prefix attends only within the
        anchor's document, and the draft's RoPE uses the per-document positions.
        """
        bsz, seq_len = input_ids.shape

        anchor_positions, block_keep_mask, noise_embedding, full_position_ids, dflash_attn_mask = (
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

        # Drop block position 0 (the clean anchor token, never a target); the
        # remaining bs-1 predicted positions are what the loss supervises.
        pred_logits = logits.view(bsz, n, bs, -1)[:, :, 1:, :].reshape(bsz, n * (bs - 1), -1)
        pred_targets = target_ids[:, :, 1:].reshape(bsz, n * (bs - 1))
        pred_mask = block_mask[:, :, 1:].reshape(bsz, n * (bs - 1))

        loss_out = self.loss_fn(pred_logits, pred_targets, pred_mask, num_tokens=None, block_size=bs)

        count_per_pos = loss_out.draft_count_per_pos
        valid_tokens = count_per_pos.sum()
        accuracy = loss_out.draft_correct_per_pos.sum() / (valid_tokens + 1e-6)

        return DFlashStepMetrics(
            loss=loss_out.total_loss, accuracy=accuracy.detach(), valid_tokens=valid_tokens.detach()
        )

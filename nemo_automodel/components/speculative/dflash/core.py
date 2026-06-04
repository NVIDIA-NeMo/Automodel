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
import torch.nn.functional as F

from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on torch build
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None


@dataclass
class DFlashStepMetrics:
    """Per-step training outputs for the DFlash draft."""

    loss: torch.Tensor
    accuracy: torch.Tensor
    valid_tokens: torch.Tensor


def create_dflash_sdpa_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Dense boolean attention mask for the SDPA backend.

    See :func:`create_dflash_block_mask` for the rules; this materializes the
    same predicate as a ``(B, 1, Q_LEN, KV_LEN)`` boolean tensor.
    """
    B, N = anchor_positions.shape
    Q_LEN = N * block_size

    q_indices = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)
    kv_indices = torch.arange(S + N * block_size, device=device).view(1, 1, 1, -1)

    q_block_ids = q_indices // block_size
    anchor_expanded = anchor_positions.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)

    mask_context = (kv_indices < S) & (kv_indices < anchor_expanded)

    is_draft = kv_indices >= S
    kv_block_ids = (kv_indices - S) // block_size
    mask_draft = is_draft & (q_block_ids == kv_block_ids)

    valid_block = block_keep_mask.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)
    return (mask_context | mask_draft) & valid_block


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
):
    """Construct a Flex Attention ``BlockMask`` for DFlash training.

    KV layout: ``[Context (S tokens) | Block_0 | Block_1 | ... | Block_{n-1}]``
    Q  layout: ``[Block_0 | Block_1 | ... | Block_{n-1}]``

    Rules:
      1. Each block sees context strictly before its anchor (``kv_idx < anchor``).
      2. Intra-block attention is bidirectional.
      3. Different blocks are invisible to each other.
      4. Invalid blocks (``block_keep_mask=False``) see nothing.
    """
    B, N = anchor_positions.shape

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        safe_q_block_id = q_block_id.clamp(max=N - 1)
        anchor_pos = anchor_positions[b, safe_q_block_id]

        is_context = kv_idx < S
        # Strictly less than: matches inference, where the target hidden at the
        # anchor position is not yet available as context.
        mask_context = is_context & (kv_idx < anchor_pos)

        is_draft = kv_idx >= S
        kv_block_id = (kv_idx - S) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)

        is_valid_block = block_keep_mask[b, safe_q_block_id]
        in_bounds = q_block_id < N
        return (mask_context | mask_draft) & is_valid_block & in_bounds

    Q_LEN = N * block_size
    KV_LEN = S + N * block_size
    return create_block_mask(dflash_mask_mod, B=B, H=None, Q_LEN=Q_LEN, KV_LEN=KV_LEN, device=device)


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
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns ``(anchors, keep_mask)``."""
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()) - 1)
        if max_n <= 0:
            raise ValueError(
                "No valid anchor positions; ensure each sample has at least "
                f"block_size+1 ({bs + 1}) supervised tokens before batching."
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

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        """Absolute position ids for the parallel draft blocks (anchor + offset)."""
        bsz = anchor_positions.shape[0]
        offsets = torch.arange(self.block_size, device=anchor_positions.device).view(1, 1, -1)
        pos_ids = anchor_positions.unsqueeze(-1) + offsets
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
        return self.embed_tokens(noise_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashStepMetrics:
        """Parallel block-wise training forward pass."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        anchor_positions, block_keep_mask = self._sample_anchor_positions(seq_len, loss_mask, device)
        noise_embedding = self._create_noise_embed(input_ids, anchor_positions, block_keep_mask)

        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        draft_position_ids = self._create_position_ids(anchor_positions)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        if self.attention_backend == "flex_attention":
            dflash_attn_mask = create_dflash_block_mask(
                anchor_positions, block_keep_mask, seq_len, self.block_size, device
            )
        else:
            dflash_attn_mask = create_dflash_sdpa_mask(
                anchor_positions, block_keep_mask, seq_len, self.block_size, device
            )

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
        )
        logits = self.lm_head(output_hidden)

        # Labels: position k of a block predicts token at anchor + k.
        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1), 2, safe_label_indices
        )

        # Weight mask: valid block * in-bounds label * exclude anchor (pos 0) * loss_mask.
        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        weight_mask = weight_mask * valid_label_mask.float()
        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()
        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1), 2, safe_label_indices
        )
        weight_mask = weight_mask * original_loss_mask_gathered

        binary_eval_mask = weight_mask.view(-1).clone()

        # Loss decay: exp(-(k-1)/gamma) so the first prediction (k=1) keeps weight 1.0.
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            decay_weights = torch.exp(-(k - 1).clamp(min=0).float() / self.loss_decay_gamma)
            weight_mask = weight_mask * decay_weights

        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)
        flat_weights = weight_mask.view(-1)

        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
        valid_token_count = flat_weights.sum() + 1e-6
        loss = (loss_per_token * flat_weights).sum() / valid_token_count

        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum() + 1e-6
            accuracy = correct.sum().float() / actual_token_count

        return DFlashStepMetrics(loss=loss, accuracy=accuracy, valid_tokens=binary_eval_mask.sum().detach())

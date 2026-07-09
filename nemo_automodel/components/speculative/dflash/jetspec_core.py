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

"""JetSpec online training wrapper.

JetSpec (arXiv:2606.18394, "Breaking the Scaling Ceiling of Speculative Decoding
with Parallel Tree Drafting") reuses the DFlash parallel draft backbone but makes
two changes that turn the block-parallel draft into a *causal* parallel tree
drafter:

1. **Block-causal attention** (paper §2.2). DFlash drafts a block bidirectionally,
   so each node's distribution is branch-agnostic and the constructed tree can be
   internally inconsistent. JetSpec masks the in-block attention causally -- a
   query at within-block offset ``i`` attends only to offsets ``j <= i`` -- so each
   branch is conditioned on its own ancestor prefix and the draft factorization
   mirrors the target's autoregressive order. This is the ``causal=True`` path of
   :func:`~nemo_automodel.components.attention.dflash_mask.create_dflash_block_mask`.
2. **Forward-KL distillation** (paper §2.3, Eq. 8-9). Instead of DFlash's
   hard-label decay-weighted CE, JetSpec matches the target model's per-position
   soft distribution with a temperature-scaled forward-KL objective
   (:class:`~nemo_automodel.components.loss.kd_loss.KDLoss`), so the draft preserves
   the teacher's relative preferences across plausible continuations.

``JetSpecTrainerModule`` subclasses :class:`DFlashTrainerModule` and reuses its
anchor sampling, ``[anchor, MASK, ...]`` noise-block construction, and absolute
position ids; only the attention mask (causal) and the loss (forward-KL against
teacher logits) are JetSpec-specific. The draft model itself is the unmodified
``Qwen3DFlashDraftModel`` -- the mask is supplied by this wrapper, so no new draft
architecture is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from nemo_automodel.components.attention.dflash_mask import (
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from nemo_automodel.components.loss.kd_loss import KDLoss
from nemo_automodel.components.speculative.dflash.core import (
    DFlashTrainerModule,
    NoValidAnchorsError,
    _to_full_tensor,
)
from nemo_automodel.components.speculative.dflash.domino_core import compute_accept_len
from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel

# Ignore index marking non-supervised draft positions in the KD labels tensor.
_IGNORE_INDEX = -100


@dataclass
class JetSpecStepMetrics:
    """Per-step training outputs for the JetSpec draft.

    ``loss``/``accuracy``/``valid_tokens`` mirror ``DFlashStepMetrics`` so the
    shared DFlash training loop consumes them unchanged. ``accept_len`` is the
    expected accepted-prefix length per block (a greedy acceptance-length / tau
    proxy), which is the headline quantity for speculative decoding -- far more
    informative than the depth-averaged token accuracy.
    """

    loss: torch.Tensor
    accuracy: torch.Tensor
    valid_tokens: torch.Tensor
    accept_len: torch.Tensor


class JetSpecTrainerModule(DFlashTrainerModule):
    """JetSpec online training wrapper: causal parallel drafting + forward-KL distillation."""

    def __init__(
        self,
        draft_model: Qwen3DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        kd_temperature: float = 1.0,
        kd_chunk_size: int = 0,
    ):
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend=attention_backend,
            num_anchors=num_anchors,
            loss_decay_gamma=None,
        )
        # Forward KL(P_target || Q_draft) with Hinton T^2 scaling and uniform
        # weighting over active draft positions (paper Eq. 9, no depth weighting in
        # the main results). The parent's DFlashDecayLoss (``self.loss_fn``) is left
        # unused -- JetSpec is a distillation objective, not hard-label CE.
        self.kd_temperature = float(kd_temperature)
        self.kd_chunk_size = int(kd_chunk_size)
        self.kd_loss_fn = KDLoss(
            ignore_index=_IGNORE_INDEX, temperature=self.kd_temperature, chunk_size=self.kd_chunk_size
        )

    def _gather_teacher_logits(
        self,
        target_logits: torch.Tensor,
        label_indices: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Teacher distribution for each predicted position, gathered from the target logits.

        Block offset ``k`` (``k = 1..bs-1``) predicts the token at sequence
        position ``anchor + k``; the target model's autoregressive distribution for
        that token, on the ground-truth prefix, is its logits at position
        ``anchor + k - 1``. Those source positions are ``label_indices[..., :-1]``
        (``anchor + 0 .. anchor + bs-2``). Returns ``[B, N*(bs-1), V]``.
        """
        bsz = label_indices.shape[0]
        vocab = target_logits.size(-1)
        teacher_idx = label_indices[:, :, :-1].clamp(max=seq_len - 1)  # [B, N, bs-1]
        teacher_idx = teacher_idx.reshape(bsz, -1, 1).expand(-1, -1, vocab)
        return torch.gather(target_logits, 1, teacher_idx)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> JetSpecStepMetrics:
        """Causal parallel block-wise forward with a forward-KL distillation loss.

        ``target_logits`` is the frozen target's full-vocab logits ``[B, S, V]``
        (captured by ``HFDFlashTargetModel(capture_logits=True)``); it supplies the
        teacher distribution for every supervised draft position.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        bs = self.block_size

        anchor_positions, block_keep_mask = self._sample_anchor_positions(seq_len, loss_mask, device)
        n = anchor_positions.size(1)
        label_indices, target_ids, block_mask = self._build_block_targets(
            input_ids, loss_mask, anchor_positions, block_keep_mask, seq_len
        )
        # Drop block offset 0 (the clean anchor token, never predicted); the
        # remaining bs-1 positions are supervised.
        pred_targets = target_ids[:, :, 1:].reshape(bsz, -1)
        valid = block_mask[:, :, 1:].reshape(bsz, -1) > 0.5
        if not valid.any():
            # Anchors exist but none has a supervised continuation (every predicted
            # position is loss-masked); nothing to distill. Skip like a short batch.
            raise NoValidAnchorsError("No supervised draft positions in this batch.")

        noise_embedding = self._create_noise_embed(input_ids, anchor_positions, block_keep_mask)
        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        draft_position_ids = self._create_position_ids(anchor_positions)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        if self.attention_backend == "flex_attention":
            attn_mask = create_dflash_block_mask(anchor_positions, block_keep_mask, seq_len, bs, device, causal=True)
        else:
            attn_mask = create_dflash_sdpa_mask(
                anchor_positions, block_keep_mask, seq_len, bs, device, dtype=noise_embedding.dtype, causal=True
            )

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=attn_mask,
        )
        logits = _to_full_tensor(self.lm_head(output_hidden))  # [B, N*bs, V]
        student_logits = logits.view(bsz, n, bs, -1)[:, :, 1:, :].reshape(bsz, n * (bs - 1), -1)

        teacher_logits = self._gather_teacher_logits(_to_full_tensor(target_logits), label_indices, seq_len)

        # KD labels mark validity (ignored positions are excluded from the mean);
        # the token ids themselves are never used as targets by the forward KL.
        kd_labels = torch.where(valid, pred_targets, torch.full_like(pred_targets, _IGNORE_INDEX))
        loss = self.kd_loss_fn(student_logits, teacher_logits, kd_labels)

        with torch.no_grad():
            valid_tokens = valid.sum()
            # Per-token acceptance: does the draft's greedy token match the TARGET's
            # greedy token? This is the greedy speculative-decoding accept condition
            # (paper Eq. 12). JetSpec distills toward the target, so the target's
            # argmax is the right reference; comparing against the raw ground-truth
            # token (DFlash's hard-label metric) understates agreement, especially
            # when the training data was regenerated by a different model.
            draft_ids = student_logits.argmax(dim=-1)
            target_ids = teacher_logits.argmax(dim=-1)
            accuracy = ((draft_ids == target_ids) & valid).sum().float() / valid_tokens.clamp_min(1).float()

            # Expected accepted-prefix length per block (a tau proxy): consecutive
            # draft==target matches from the first drafted depth, +1 for the always
            # accepted anchor token, averaged over blocks that have any supervised
            # position. This is the headline speculative-decoding quantity and is
            # far more meaningful than the depth-averaged ``accuracy`` above.
            d4 = draft_ids.view(bsz, n, bs - 1)
            t4 = target_ids.view(bsz, n, bs - 1)
            v4 = valid.view(bsz, n, bs - 1)
            block_accept = compute_accept_len(d4, t4, v4)  # [B, N]
            valid_block = v4.any(dim=2)
            accept_len = ((block_accept + 1.0) * valid_block).sum() / valid_block.sum().clamp_min(1).float()

        return JetSpecStepMetrics(
            loss=loss,
            accuracy=accuracy.detach(),
            valid_tokens=valid_tokens.detach(),
            accept_len=accept_len.detach(),
        )

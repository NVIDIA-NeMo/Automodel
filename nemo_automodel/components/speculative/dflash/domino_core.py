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

"""Domino online training wrapper.

Ported from SpecForge's ``specforge/core/domino.py`` (sgl-project/SpecForge#571).

Domino extends the parallel DFlash draft backbone with a lightweight *causal*
correction head. DFlash drafts a whole block in a single non-causal forward, so
each predicted position is blind to the (drafted) tokens earlier in its own
block. The Domino head fixes that: a GRU encodes a causal state from the block's
previous tokens, and a low-rank projection of ``[backbone hidden | GRU state]``
produces a logit delta that is *added* to the parallel base logits.

Training jointly supervises two logits with a base-anchor curriculum::

    loss = (1 - lambda_base) * final_loss + lambda_base * base_loss

``final_loss`` is the CE on the Domino-refined logits and ``base_loss`` the CE on
the backbone-only base logits. ``lambda_base`` decays from its start value to 0
over the first ``decay_ratio`` fraction of training, so early steps keep the
parallel backbone strong and later steps let the correction head take over.

``DominoTrainerModule`` reuses the DFlash anchor sampling, noise-block
construction, and block attention mask (it subclasses ``DFlashTrainerModule``);
only the head, the dual-logit loss, and the metrics are Domino-specific. The
Domino head parameters (``prefix_gru``, ``embed_proj``) live on the DFlash draft
model and are enabled via ``dflash_config.projector_type='domino'``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.attention.dflash_mask import (
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from nemo_automodel.components.speculative.dflash.core import DFlashTrainerModule, _to_full_tensor
from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel


def get_lambda_base(
    global_step: int,
    total_steps: int,
    lambda_start: float = 1.0,
    decay_ratio: float = 0.5,
) -> float:
    """Base-anchor curriculum weight at ``global_step``.

    ``lambda_base`` starts at ``lambda_start`` and decays linearly to 0 over the
    first ``decay_ratio`` fraction of ``total_steps``, then stays at 0. The result
    is clamped to ``[0, 1]``.
    """
    decay_steps = max(1, int(total_steps * decay_ratio))
    progress = min(global_step / decay_steps, 1.0)
    lambda_base = lambda_start * (1.0 - progress)
    return max(0.0, min(1.0, lambda_base))


def compute_accept_len(
    pred_ids_4d: torch.Tensor,
    target_ids_4d: torch.Tensor,
    valid_mask_4d: torch.Tensor,
) -> torch.Tensor:
    """Per-block acceptance length: consecutive correct predictions from position 0.

    Invalid (masked-out) positions count as correct so they never truncate a block
    prematurely; the trailing valid mask then zeros their contribution.
    """
    correct = (pred_ids_4d == target_ids_4d) | (~valid_mask_4d)
    accept_prefix = correct.long().cumprod(dim=2) * valid_mask_4d.long()
    return accept_prefix.sum(dim=2).float()


@dataclass
class DominoStepMetrics:
    """Per-step training outputs for the Domino draft.

    ``loss``/``accuracy``/``valid_tokens`` mirror ``DFlashStepMetrics`` so the
    DFlash training loop consumes them unchanged. The remaining fields are
    diagnostics for the two supervised logits and the curriculum weight.
    """

    loss: torch.Tensor
    accuracy: torch.Tensor
    valid_tokens: torch.Tensor
    final_loss: torch.Tensor
    base_loss: torch.Tensor
    base_accuracy: torch.Tensor
    accept_len: torch.Tensor
    base_accept_len: torch.Tensor
    lambda_base: torch.Tensor


class DominoTrainerModule(DFlashTrainerModule):
    """Domino online training wrapper: DFlash backbone + causal correction head."""

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
        shift_label: bool = False,
    ):
        super().__init__(
            draft_model=draft_model,
            target_lm_head=target_lm_head,
            target_embed_tokens=target_embed_tokens,
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend=attention_backend,
            num_anchors=num_anchors,
            loss_decay_gamma=loss_decay_gamma,
        )
        if getattr(draft_model, "projector_type", None) != "domino":
            raise ValueError(
                "DominoTrainerModule requires a draft model built with "
                "dflash_config.projector_type='domino' (so prefix_gru / embed_proj exist)."
            )
        self.shift_label = shift_label

    @property
    def _suffix_start(self) -> int:
        """First block position that receives the Domino correction.

        With ``shift_label`` the block predicts ``anchor+1+k``, so position 0 is
        already a real next-token prediction; otherwise position 0 is the clean
        anchor and is excluded. ``pure_draft_prefix_len`` reserves additional
        leading positions for the backbone-only (uncorrected) base logits.
        """
        pure_prefix = getattr(self.draft_model, "pure_draft_prefix_len", 0)
        return pure_prefix if self.shift_label else (1 + pure_prefix)

    def _build_domino_head_inputs(
        self,
        input_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        target_ids: torch.Tensor,
        output_hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reshape backbone hidden states and gather the block's previous tokens.

        ``prev_ids[..., k]`` is the token that precedes block-position ``k``'s
        target -- the input token at ``anchor+k`` when ``shift_label`` (the GRU
        consumes ground-truth context), else the target sequence itself.
        """
        bsz, n, bs = target_ids.shape
        hidden4d = output_hidden.reshape(bsz, n, bs, output_hidden.shape[-1])

        prev_ids = target_ids
        if self.shift_label:
            prev_offsets = torch.arange(0, self.block_size, device=input_ids.device).view(1, 1, -1)
            prev_indices = (anchor_positions.unsqueeze(-1) + prev_offsets).clamp(max=input_ids.size(1) - 1)
            prev_ids = torch.gather(
                input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
                2,
                prev_indices,
            )
        return hidden4d, prev_ids

    def _apply_domino_head(
        self,
        base_logits4d: torch.Tensor,
        hidden4d: torch.Tensor,
        prev_ids: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Add the GRU-conditioned low-rank correction to the suffix base logits."""
        bsz, n, bs = target_ids.shape
        # A tensor-parallel target's embed_tokens is vocab-parallel and returns a
        # DTensor; gather it so the (plain) GRU consumes a plain tensor. ``prev_ids``
        # already equals ``target_ids`` when not ``shift_label`` (set by
        # ``_build_domino_head_inputs``), so one embed call covers both branches.
        block_emb = _to_full_tensor(self.embed_tokens(prev_ids))
        if self.shift_label:
            gru_inputs = block_emb.reshape(bsz * n, bs, -1)
            gru_out, _ = self.draft_model.prefix_gru(gru_inputs)
            gru_out = gru_out.reshape(bsz, n, bs, -1)
            prefix_states = gru_out[:, :, self._suffix_start :, :]
        else:
            gru_inputs = block_emb[:, :, : bs - 1, :].reshape(bsz * n, bs - 1, -1)
            gru_out, _ = self.draft_model.prefix_gru(gru_inputs)
            gru_out = gru_out.reshape(bsz, n, bs - 1, -1)
            prefix_states = gru_out[:, :, self._suffix_start - 1 :, :]
        z_n = hidden4d[:, :, self._suffix_start :, :]
        concat_features = torch.cat([z_n, prefix_states], dim=-1)
        logits_e = self.draft_model.embed_proj(concat_features)

        prefix_logits = base_logits4d[:, :, : self._suffix_start, :]
        suffix_logits = base_logits4d[:, :, self._suffix_start :, :] + logits_e
        return torch.cat([prefix_logits, suffix_logits], dim=2)

    def _compute_weighted_losses(
        self,
        final_logits: torch.Tensor,
        base_logits: torch.Tensor,
        target_ids: torch.Tensor,
        weight_mask: torch.Tensor,
        lambda_base: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decay-weighted CE on both logits, mixed by the curriculum weight."""
        flat_targets = target_ids.reshape(-1)
        flat_weights = weight_mask.reshape(-1)
        valid_token_count = flat_weights.sum() + 1e-6

        final_loss_per_token = F.cross_entropy(
            final_logits.reshape(-1, final_logits.size(-1)), flat_targets, reduction="none"
        )
        final_loss = (final_loss_per_token * flat_weights).sum() / valid_token_count

        base_loss_per_token = F.cross_entropy(
            base_logits.reshape(-1, base_logits.size(-1)), flat_targets, reduction="none"
        )
        base_loss = (base_loss_per_token * flat_weights).sum() / valid_token_count

        loss = (1.0 - lambda_base) * final_loss + lambda_base * base_loss
        return loss, final_loss, base_loss

    def _compute_extra_metrics(
        self,
        pred_ids: torch.Tensor,
        flat_base_logits: torch.Tensor,
        flat_targets: torch.Tensor,
        binary_eval_mask: torch.Tensor,
        actual_token_count: torch.Tensor,
        target_ids: torch.Tensor,
        eval_weight_mask: torch.Tensor,
        final_loss: torch.Tensor,
        base_loss: torch.Tensor,
        lambda_base: float,
    ) -> Dict[str, torch.Tensor]:
        """Diagnostics for both heads (acceptance length, base accuracy). No gradient."""
        bsz, n, bs = target_ids.shape

        base_pred_ids = torch.argmax(flat_base_logits, dim=-1)
        base_correct = (base_pred_ids == flat_targets) & (binary_eval_mask > 0.5)
        base_accuracy = base_correct.sum().float() / actual_token_count

        valid_mask_4d = (eval_weight_mask > 0).bool()
        pred_accept_len = compute_accept_len(pred_ids.view(bsz, n, bs), target_ids, valid_mask_4d)
        base_accept_len = compute_accept_len(base_pred_ids.view(bsz, n, bs), target_ids, valid_mask_4d)

        valid_block_mask = valid_mask_4d.any(dim=2)
        num_valid_blocks = valid_block_mask.sum().float() + 1e-6
        avg_accept_len = ((pred_accept_len + 1.0) * valid_block_mask.float()).sum() / num_valid_blocks
        base_avg_accept_len = ((base_accept_len + 1.0) * valid_block_mask.float()).sum() / num_valid_blocks

        return {
            "final_loss": final_loss.detach(),
            "base_loss": base_loss.detach(),
            "base_accuracy": base_accuracy.detach(),
            "accept_len": avg_accept_len.detach(),
            "base_accept_len": base_avg_accept_len.detach(),
            "lambda_base": torch.tensor(float(lambda_base), device=final_loss.device),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        lambda_base: float = 0.0,
    ) -> DominoStepMetrics:
        """Parallel block-wise training forward with the Domino correction head."""
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
                anchor_positions, block_keep_mask, seq_len, self.block_size, device, dtype=noise_embedding.dtype
            )

        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
        )

        # --- Labels: block position k predicts anchor + label_start + k ---
        label_start = 1 if self.shift_label else 0
        label_offsets = torch.arange(label_start, label_start + self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_target_indices = label_indices.clamp(max=seq_len - 1)
        n = anchor_positions.size(1)
        target_ids = torch.gather(input_ids.unsqueeze(1).expand(-1, n, -1), 2, safe_target_indices)

        bsz, n, bs = target_ids.shape
        # A tensor-parallel target's lm_head is column-parallel and returns
        # vocab-sharded (DTensor) logits; gather to a full tensor for the loss.
        base_logits = _to_full_tensor(self.lm_head(output_hidden))
        hidden4d, prev_ids = self._build_domino_head_inputs(
            input_ids=input_ids,
            anchor_positions=anchor_positions,
            target_ids=target_ids,
            output_hidden=output_hidden,
        )
        base_logits4d = base_logits.reshape(bsz, n, bs, -1)
        final_logits = self._apply_domino_head(
            base_logits4d=base_logits4d,
            hidden4d=hidden4d,
            prev_ids=prev_ids,
            target_ids=target_ids,
        ).reshape(bsz, n * bs, -1)

        # --- Weight mask: block validity * bounds * (exclude anchor) * loss_mask ---
        weight_mask = block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        weight_mask = weight_mask * valid_label_mask.float()
        if not self.shift_label:
            pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
            weight_mask = weight_mask * (pos_in_block > 0).float()
        gathered_loss_mask = torch.gather(loss_mask.unsqueeze(1).expand(-1, n, -1), 2, safe_target_indices)
        weight_mask = weight_mask * gathered_loss_mask

        # Binary eval mask (pre-decay) for accuracy / acceptance-length stats. The
        # decay below rebinds weight_mask to a fresh tensor (out-of-place), so these
        # keep referencing the pre-decay mask without a clone.
        eval_weight_mask = weight_mask
        binary_eval_mask = weight_mask.view(-1)

        # --- Block-position loss decay: first valid position keeps weight 1.0 ---
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(1, 1, -1)
            offset = 0 if self.shift_label else 1
            decay_weights = torch.exp(-(k - offset).clamp(min=0).float() / self.loss_decay_gamma)
            weight_mask = weight_mask * decay_weights

        loss, final_loss, base_loss = self._compute_weighted_losses(
            final_logits=final_logits,
            base_logits=base_logits,
            target_ids=target_ids,
            weight_mask=weight_mask,
            lambda_base=lambda_base,
        )

        with torch.no_grad():
            flat_logits = final_logits.reshape(-1, final_logits.size(-1))
            flat_base_logits = base_logits.reshape(-1, base_logits.size(-1))
            flat_targets = target_ids.reshape(-1)
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            actual_token_count = binary_eval_mask.sum() + 1e-6
            accuracy = correct.sum().float() / actual_token_count
            metrics = self._compute_extra_metrics(
                pred_ids=pred_ids,
                flat_base_logits=flat_base_logits,
                flat_targets=flat_targets,
                binary_eval_mask=binary_eval_mask,
                actual_token_count=actual_token_count,
                target_ids=target_ids,
                eval_weight_mask=eval_weight_mask,
                final_loss=final_loss,
                base_loss=base_loss,
                lambda_base=lambda_base,
            )

        return DominoStepMetrics(
            loss=loss,
            accuracy=accuracy.detach(),
            valid_tokens=binary_eval_mask.sum().detach(),
            final_loss=metrics["final_loss"],
            base_loss=metrics["base_loss"],
            base_accuracy=metrics["base_accuracy"],
            accept_len=metrics["accept_len"],
            base_accept_len=metrics["base_accept_len"],
            lambda_base=metrics["lambda_base"],
        )

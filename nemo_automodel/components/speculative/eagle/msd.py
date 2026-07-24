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

"""Multimodal speculative decoding draft model and training objective.

MSD extends EAGLE-1/2 feature drafting to VLM targets.  Text positions use the
usual concatenation of target features and next-token embeddings.  Image
positions instead pass the target VLM's already-computed image embeddings
directly into the draft transformer, preserving the non-causal relationship
between visual patches.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from nemo_automodel.components.loss.soft_ce import masked_soft_cross_entropy
from nemo_automodel.components.speculative.eagle.draft_llama_v12 import (
    LlamaEagleDraftModel,
    _build_causal_mask,
)


class MultimodalEagleDraftModel(LlamaEagleDraftModel):
    """EAGLE-1/2 draft model with modality-aware input feature construction."""

    config_class = PretrainedConfig

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        target_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next-position target hidden states from multimodal features.

        Args:
            inputs_embeds: Tensor of shape [batch, sequence, hidden], containing
                the target VLM input embeddings shifted one position to the left.
                Image positions already contain projected vision features.
            target_hidden_states: Tensor of shape [batch, sequence, hidden], the
                frozen target language model's unshifted final hidden states.
            attention_mask: Tensor of shape [batch, sequence], where one denotes
                a real token and zero denotes padding.
            image_mask: Bool tensor of shape [batch, sequence], aligned with
                ``inputs_embeds`` and true for projected image-token positions.

        Returns:
            Tensor of shape [batch, sequence, hidden], the draft predictions for
            the target's next-position hidden states.
        """
        if inputs_embeds.shape != target_hidden_states.shape:
            raise ValueError(
                "inputs_embeds and target_hidden_states must have identical [batch, sequence, hidden] shapes, "
                f"got {tuple(inputs_embeds.shape)} and {tuple(target_hidden_states.shape)}."
            )
        if attention_mask.shape != inputs_embeds.shape[:2] or image_mask.shape != inputs_embeds.shape[:2]:
            raise ValueError(
                "attention_mask and image_mask must both have shape [batch, sequence] matching inputs_embeds, "
                f"got {tuple(attention_mask.shape)} and {tuple(image_mask.shape)} for {tuple(inputs_embeds.shape)}."
            )

        inputs_embeds = inputs_embeds.to(target_hidden_states.dtype)
        text_features = self.fc(torch.cat((inputs_embeds, target_hidden_states), dim=-1))
        hidden_states = torch.where(image_mask.bool().unsqueeze(-1), inputs_embeds, text_features)

        batch_size, sequence_length, _ = hidden_states.shape
        position_ids = (
            torch.arange(sequence_length, device=hidden_states.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        causal_mask = _build_causal_mask(attention_mask, hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=causal_mask, position_ids=position_ids)
        return self.norm(hidden_states)


@dataclass
class MSDStepMetrics:
    """Aggregated metrics from one multimodal speculative-decoding training step."""

    loss: torch.Tensor
    hidden_loss: torch.Tensor
    token_loss: torch.Tensor
    accuracy: torch.Tensor
    valid_tokens: torch.Tensor


class MSDTrainerModule(nn.Module):
    """Draft-side training module for multimodal speculative decoding."""

    def __init__(
        self,
        draft_model: nn.Module,
        *,
        target_lm_head: nn.Module,
        hidden_loss_weight: float = 1.0,
        token_loss_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.draft_model = draft_model
        object.__setattr__(self, "_target_lm_head", target_lm_head)
        self.hidden_loss_weight = hidden_loss_weight
        self.token_loss_weight = token_loss_weight
        self.hidden_loss_fn = nn.SmoothL1Loss(reduction="none")

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project predicted hidden states through the frozen target language head.

        Args:
            hidden_states: Tensor of shape [..., hidden], with arbitrary leading
                dimensions.

        Returns:
            Tensor of shape [..., vocab], containing target-vocabulary logits.
        """
        weight = self._target_lm_head.weight
        if hasattr(weight, "full_tensor"):
            weight = weight.full_tensor()
        return F.linear(hidden_states, weight.detach())

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        input_hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        target_logits: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> MSDStepMetrics:
        """Run one multimodal speculative-decoding training step.

        Args:
            inputs_embeds: Tensor of shape [batch, sequence, hidden], the target
                VLM input embeddings shifted left by one position.
            attention_mask: Tensor of shape [batch, sequence], where one denotes
                a real token and zero denotes padding.
            loss_mask: Bool tensor of shape [batch, sequence], true where the
                shifted next-token target is supervised.
            input_hidden_states: Tensor of shape [batch, sequence, hidden], the
                target's unshifted final hidden states used as draft features.
            target_hidden_states: Tensor of shape [batch, sequence, hidden], the
                target's final hidden states shifted left by one position.
            target_logits: Tensor of shape [batch, sequence, vocab], the frozen
                target logits shifted left by one position.
            image_mask: Bool tensor of shape [batch, sequence], true where
                ``inputs_embeds`` contains a projected image token.

        Returns:
            MSDStepMetrics containing scalar loss terms, first-token accuracy, and
            the number of supervised token positions.
        """
        predicted_hidden_states = self.draft_model(
            inputs_embeds=inputs_embeds,
            target_hidden_states=input_hidden_states,
            attention_mask=attention_mask,
            image_mask=image_mask,
        )
        predicted_logits = self.compute_logits(predicted_hidden_states)
        valid_mask = loss_mask.bool()
        valid_tokens = valid_mask.sum()
        hidden_loss_per_token = self.hidden_loss_fn(predicted_hidden_states, target_hidden_states).mean(dim=-1)
        hidden_loss = (
            hidden_loss_per_token[valid_mask].mean() if valid_mask.any() else hidden_loss_per_token.sum() * 0.0
        )

        target_probs = torch.softmax(target_logits.float(), dim=-1).detach()
        token_loss = masked_soft_cross_entropy(
            logits=predicted_logits,
            target_probs=target_probs,
            position_mask=valid_mask.unsqueeze(-1),
        )
        loss = self.hidden_loss_weight * hidden_loss + self.token_loss_weight * token_loss

        correct = (predicted_logits.argmax(dim=-1) == target_logits.argmax(dim=-1)) & valid_mask
        accuracy = correct.sum() / valid_tokens.clamp_min(1)
        return MSDStepMetrics(
            loss=loss,
            hidden_loss=hidden_loss,
            token_loss=token_loss,
            accuracy=accuracy,
            valid_tokens=valid_tokens,
        )

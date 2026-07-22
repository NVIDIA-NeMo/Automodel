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

"""Unit tests for multimodal speculative decoding components."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import LlamaConfig

from nemo_automodel.components.speculative.eagle.msd import (
    MSDTrainerModule,
    MultimodalEagleDraftModel,
)
from nemo_automodel.components.speculative.eagle.msd_target import HFMSDTargetModel


def _draft_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=11,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )


def test_msd_draft_directly_uses_projected_image_embeddings() -> None:
    """Image positions bypass EAGLE's text-feature projection."""
    draft = MultimodalEagleDraftModel(_draft_config())
    with torch.no_grad():
        draft.fc.weight.zero_()
        for parameter in draft.layers.parameters():
            parameter.zero_()
        draft.norm.weight.fill_(1.0)

    inputs_embeds = torch.zeros(1, 3, 8)
    inputs_embeds[0, 1] = torch.arange(1, 9)
    target_hidden_states = torch.randn(1, 3, 8)
    output = draft(
        inputs_embeds=inputs_embeds,
        target_hidden_states=target_hidden_states,
        attention_mask=torch.ones(1, 3),
        image_mask=torch.tensor([[False, True, False]]),
    )

    assert torch.allclose(output[0, 0], torch.zeros(8))
    assert torch.allclose(output[0, 2], torch.zeros(8))
    assert torch.count_nonzero(output[0, 1]) == 8


def test_msd_draft_rejects_misaligned_masks() -> None:
    """The draft fails early when per-token multimodal tensors disagree."""
    draft = MultimodalEagleDraftModel(_draft_config())
    with pytest.raises(ValueError, match="attention_mask and image_mask"):
        draft(
            inputs_embeds=torch.randn(1, 3, 8),
            target_hidden_states=torch.randn(1, 3, 8),
            attention_mask=torch.ones(1, 2),
            image_mask=torch.zeros(1, 3, dtype=torch.bool),
        )


class _PerfectDraft(nn.Module):
    """Return the target features supplied by the test batch."""

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        target_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return target features while retaining each tensor input in the graph.

        Args:
            inputs_embeds: Tensor of shape [batch, sequence, hidden].
            target_hidden_states: Tensor of shape [batch, sequence, hidden].
            attention_mask: Tensor of shape [batch, sequence].
            image_mask: Bool tensor of shape [batch, sequence].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        return target_hidden_states + 0.0 * (
            inputs_embeds.sum() + attention_mask.sum() + image_mask.to(inputs_embeds.dtype).sum()
        )


def test_msd_trainer_masks_supervision_and_backpropagates() -> None:
    """MSD loss ignores masked positions and propagates a finite gradient."""
    lm_head = nn.Linear(4, 7, bias=False)
    trainer = MSDTrainerModule(_PerfectDraft(), target_lm_head=lm_head)
    target_hidden = torch.randn(1, 3, 4, requires_grad=True)
    target_logits = lm_head(target_hidden.detach())
    metrics = trainer(
        inputs_embeds=torch.randn(1, 3, 4),
        attention_mask=torch.ones(1, 3),
        loss_mask=torch.tensor([[True, False, True]]),
        input_hidden_states=target_hidden,
        target_hidden_states=target_hidden,
        target_logits=target_logits,
        image_mask=torch.tensor([[False, True, False]]),
    )

    metrics.loss.backward()
    assert metrics.valid_tokens.item() == 2
    assert torch.isfinite(metrics.loss)
    assert target_hidden.grad is not None
    assert torch.isfinite(target_hidden.grad).all()


class _TinyLanguageBackbone(nn.Module):
    """Language backbone used to verify the MSD target embedding hook."""

    def forward(self, *, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Return the fused language inputs unchanged.

        Args:
            inputs_embeds: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        return inputs_embeds


class _TinyVLM(nn.Module):
    """Minimal VLM whose image placeholder is replaced before the language model."""

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(image_token_id=3)
        self.embeddings = nn.Embedding(8, 4)
        self.model = _TinyLanguageBackbone()
        self.lm_head = nn.Linear(4, 8, bias=False)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
        return_dict: bool,
        use_cache: bool,
        pixel_values: torch.Tensor,
    ) -> SimpleNamespace:
        """Fuse an image feature into the token embedding sequence.

        Args:
            input_ids: Long tensor of shape [batch, sequence].
            attention_mask: Tensor of shape [batch, sequence].
            output_hidden_states: Whether hidden states are requested.
            return_dict: Whether a structured output is requested.
            use_cache: Whether key-value caching is requested.
            pixel_values: Tensor of shape [batch, channels, height, width].

        Returns:
            SimpleNamespace whose ``hidden_states`` contains a Tensor of shape
            [batch, sequence, hidden] and whose ``logits`` has shape [batch,
            sequence, vocab].
        """
        del attention_mask, output_hidden_states, return_dict, use_cache
        embeds = self.embeddings(input_ids)
        image_features = pixel_values.mean(dim=(1, 2, 3), keepdim=False).view(-1, 1, 1).expand(-1, 1, 4)
        embeds = torch.where(input_ids.eq(self.config.image_token_id).unsqueeze(-1), image_features, embeds)
        hidden_states = self.model(inputs_embeds=embeds)
        return SimpleNamespace(hidden_states=(hidden_states,), logits=self.lm_head(hidden_states))


def test_msd_target_captures_fused_image_embeddings_and_alignment() -> None:
    """The target wrapper preserves vision features and VLM label alignment."""
    model = _TinyVLM()
    wrapper = HFMSDTargetModel(model)
    input_ids = torch.tensor([[1, 3, 2]])
    pixel_values = torch.full((1, 3, 2, 2), 5.0)
    batch = wrapper.generate_batch(
        input_ids=input_ids,
        attention_mask=torch.ones(1, 3),
        loss_mask=torch.tensor([[True, False, True]]),
        model_inputs={"input_ids": input_ids, "attention_mask": torch.ones(1, 3), "pixel_values": pixel_values},
    )

    assert torch.equal(batch.image_mask, torch.tensor([[True, False, False]]))
    assert torch.equal(batch.loss_mask, torch.tensor([[True, False, True]]))
    assert torch.allclose(batch.inputs_embeds[0, 0], torch.full((4,), 5.0))
    assert torch.equal(batch.target_hidden_states[:, -1], torch.zeros(1, 4))

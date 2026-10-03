# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for Qwen3.5 dense VLM last-stage hidden states under pipeline parallelism."""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers.models.qwen3_5")

from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForConditionalGeneration

HIDDEN, VOCAB = 8, 32


class _LanguageModel(nn.Module):
    """Non-first PP stage backbone: no embed_tokens, echoes the incoming hidden states."""

    embed_tokens = None

    def forward(self, inputs_embeds=None, **kwargs):
        return types.SimpleNamespace(last_hidden_state=inputs_embeds)


def _build_last_stage() -> Qwen3_5ForConditionalGeneration:
    model = Qwen3_5ForConditionalGeneration.__new__(Qwen3_5ForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = types.SimpleNamespace(image_token_id=None, video_token_id=None, vision_start_token_id=None)
    model.cp_mesh = None
    model.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
    model.model = types.SimpleNamespace(language_model=_LanguageModel())
    model.train()
    return model


def test_declares_hidden_state_support():
    assert Qwen3_5ForConditionalGeneration._pp_return_hidden_states_supported is True


def test_last_stage_returns_hidden_states_when_requested():
    model = _build_last_stage()
    model._pp_return_hidden_states = True
    hidden = torch.randn(2, 5, HIDDEN)

    output = model(input_ids=hidden)

    assert output is hidden


def test_last_stage_returns_logits_by_default():
    model = _build_last_stage()
    hidden = torch.randn(2, 5, HIDDEN)

    output = model(input_ids=hidden)

    assert output.shape == (2, 5, VOCAB)

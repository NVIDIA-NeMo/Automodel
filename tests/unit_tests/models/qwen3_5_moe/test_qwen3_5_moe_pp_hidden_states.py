# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for Qwen3.5-MoE last-stage hidden states under pipeline parallelism.

FusedLinearCrossEntropy applies ``lm_head`` inside the loss, so the recipe sets
``_pp_return_hidden_states`` on the last stage and the forward must return hidden
states. MTP stages return a ``(logits, *mtp_per_depth_h)`` tuple and keep a logit loss.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers.models.qwen3_5_moe")

from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForConditionalGeneration

HIDDEN, VOCAB = 8, 32


def _build_last_stage(*, mtp_enabled: bool = False) -> Qwen3_5MoeForConditionalGeneration:
    """Barebones last pipeline stage whose backbone echoes the incoming hidden states."""
    model = Qwen3_5MoeForConditionalGeneration.__new__(Qwen3_5MoeForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = types.SimpleNamespace(
        image_token_id=None,
        video_token_id=None,
        vision_start_token_id=None,
        text_config=types.SimpleNamespace(hidden_size=HIDDEN, vocab_size=VOCAB),
    )
    model.mtp_config = types.SimpleNamespace(enabled=mtp_enabled)
    # PP stage splitting drops the MTP module from non-final stages; the flag must not depend on it.
    model.mtp = None
    model.cp_mesh = None
    model.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    class _Backbone(nn.Module):
        language_model = types.SimpleNamespace(embed_tokens=None)

        def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
            hidden = inputs_embeds if inputs_embeds is not None else input_ids
            return types.SimpleNamespace(last_hidden_state=hidden)

    model.model = _Backbone()
    model.train()
    return model


@pytest.mark.parametrize(("mtp_enabled", "supported"), [(False, True), (True, False)])
def test_hidden_state_support_follows_mtp_config(mtp_enabled, supported):
    model = _build_last_stage(mtp_enabled=mtp_enabled)
    assert model._pp_return_hidden_states_supported is supported


def test_last_stage_returns_hidden_states_when_requested():
    model = _build_last_stage()
    hidden = torch.randn(2, 5, HIDDEN)
    model._pp_return_hidden_states = True

    output = model(inputs_embeds=hidden)

    assert output is hidden


def test_last_stage_returns_logits_by_default():
    model = _build_last_stage()
    hidden = torch.randn(2, 5, HIDDEN)

    output = model(inputs_embeds=hidden)

    logits = output.logits if hasattr(output, "logits") else output
    assert logits.shape == (2, 5, VOCAB)


def test_stage_metas_emit_hidden_size_when_requested():
    model = _build_last_stage()
    model._pp_return_hidden_states = True
    _, outs = model.get_pipeline_stage_metas(is_first=False, microbatch_size=1, seq_len=6, dtype=torch.float32)
    assert outs[0].shape == (1, 6, HIDDEN)

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

"""Tests for the DFlash trainer module (anchor sampling, noise, block-wise loss)."""

from __future__ import annotations

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.speculative.dflash.core import DFlashStepMetrics, DFlashTrainerModule
from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel

VOCAB = 64
HIDDEN = 32
NUM_TARGET_LAYERS = 8
TARGET_LAYER_IDS = [1, 3, 5]
BLOCK_SIZE = 4
MASK_ID = VOCAB - 1


def _build_trainer(num_anchors=8, loss_decay_gamma=None, loss_type="dflash", dpace_alpha=0.5):
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    cfg.num_target_layers = NUM_TARGET_LAYERS
    cfg.block_size = BLOCK_SIZE
    cfg.dflash_config = {"mask_token_id": MASK_ID, "target_layer_ids": TARGET_LAYER_IDS}
    cfg._attn_implementation = "sdpa"
    draft = Qwen3DFlashDraftModel(cfg)
    lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
    embed = torch.nn.Embedding(VOCAB, HIDDEN)
    return DFlashTrainerModule(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=MASK_ID,
        block_size=BLOCK_SIZE,
        attention_backend="sdpa",
        num_anchors=num_anchors,
        loss_decay_gamma=loss_decay_gamma,
        loss_type=loss_type,
        dpace_alpha=dpace_alpha,
    )


def _inputs(bsz=2, seq_len=24):
    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB - 1, (bsz, seq_len))
    loss_mask = torch.ones(bsz, seq_len)
    hidden = torch.randn(bsz, seq_len, len(TARGET_LAYER_IDS) * HIDDEN)
    return input_ids, hidden, loss_mask


def test_forward_returns_finite_loss_and_grads_flow_to_draft():
    trainer = _build_trainer(loss_decay_gamma=7.0)
    input_ids, hidden, loss_mask = _inputs()
    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    assert isinstance(out, DFlashStepMetrics)
    assert torch.isfinite(out.loss) and out.loss.item() > 0
    assert 0.0 <= out.accuracy.item() <= 1.0
    assert out.valid_tokens.item() > 0
    out.loss.backward()
    grad = sum(p.grad.abs().sum().item() for p in trainer.draft_model.parameters() if p.grad is not None)
    assert grad > 0


def test_forward_supports_dpace_loss_and_grads_flow_to_draft():
    trainer = _build_trainer(loss_type="dpace", dpace_alpha=0.4)
    input_ids, hidden, loss_mask = _inputs()
    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    assert isinstance(out, DFlashStepMetrics)
    assert torch.isfinite(out.loss) and out.loss.item() > 0
    out.loss.backward()
    grad = sum(p.grad.abs().sum().item() for p in trainer.draft_model.parameters() if p.grad is not None)
    assert grad > 0


def test_noise_embed_keeps_anchor_token_and_masks_rest():
    trainer = _build_trainer()
    bsz, seq_len = 1, 20
    input_ids = torch.arange(1, seq_len + 1).view(1, seq_len)  # distinct, non-mask tokens
    anchors = torch.tensor([[2, 8]])
    keep = torch.tensor([[True, False]])
    # Patch embed_tokens to identity-ish so we can read back token ids.
    trainer.embed_tokens = torch.nn.Embedding(VOCAB + seq_len + 1, 1)
    with torch.no_grad():
        trainer.embed_tokens.weight.copy_(torch.arange(VOCAB + seq_len + 1).view(-1, 1).float())
    emb = trainer._create_noise_embed(input_ids, anchors, keep)
    ids = emb.view(bsz, -1).round().long()
    # block 0 valid: position 0 holds the anchor token (input_ids[0,2]==3), rest MASK
    assert ids[0, 0].item() == input_ids[0, 2].item()
    assert (ids[0, 1:BLOCK_SIZE] == MASK_ID).all()
    # block 1 invalid: every position is MASK
    assert (ids[0, BLOCK_SIZE : 2 * BLOCK_SIZE] == MASK_ID).all()


def test_position_ids_are_anchor_plus_offset():
    trainer = _build_trainer()
    anchors = torch.tensor([[2, 10]])
    pos = trainer._create_position_ids(anchors)
    assert pos.tolist() == [[2, 3, 4, 5, 10, 11, 12, 13]]


def test_sample_anchor_positions_respect_loss_mask():
    trainer = _build_trainer(num_anchors=4)
    seq_len = 20
    loss_mask = torch.ones(1, seq_len)
    loss_mask[0, 10:] = 0  # only first 10 positions supervised
    anchors, keep = trainer._sample_anchor_positions(seq_len, loss_mask, torch.device("cpu"))
    valid_anchors = anchors[keep]
    assert (valid_anchors < 10).all()


def test_no_valid_anchors_raises():
    trainer = _build_trainer()
    seq_len = 20
    loss_mask = torch.zeros(1, seq_len)  # nothing supervised
    with pytest.raises(ValueError):
        trainer._sample_anchor_positions(seq_len, loss_mask, torch.device("cpu"))

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

"""Tests for the Domino trainer module (causal head, dual-logit loss, curriculum)."""

from __future__ import annotations

import pytest
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.speculative.dflash.domino_core import (
    DominoStepMetrics,
    DominoTrainerModule,
    compute_accept_len,
    get_lambda_base,
)
from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel

VOCAB = 64
HIDDEN = 32
NUM_TARGET_LAYERS = 8
TARGET_LAYER_IDS = [1, 3, 5]
BLOCK_SIZE = 4
MASK_ID = VOCAB - 1
EMB_DIM = 16
GRU_HIDDEN = 24


def _draft_config(projector_type="domino", pure_prefix=1, shift_label=True):
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
    dflash_config = {"mask_token_id": MASK_ID, "target_layer_ids": TARGET_LAYER_IDS}
    if projector_type is not None:
        dflash_config.update(
            {
                "projector_type": projector_type,
                "emb_dim": EMB_DIM,
                "gru_hidden_dim": GRU_HIDDEN,
                "pure_draft_prefix_len": pure_prefix,
                "shift_label": shift_label,
            }
        )
    cfg.dflash_config = dflash_config
    cfg._attn_implementation = "sdpa"
    return cfg


def _build_trainer(num_anchors=8, loss_decay_gamma=None, pure_prefix=1, shift_label=True):
    draft = Qwen3DFlashDraftModel(_draft_config(pure_prefix=pure_prefix, shift_label=shift_label))
    lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
    embed = torch.nn.Embedding(VOCAB, HIDDEN)
    return DominoTrainerModule(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=MASK_ID,
        block_size=BLOCK_SIZE,
        attention_backend="sdpa",
        num_anchors=num_anchors,
        loss_decay_gamma=loss_decay_gamma,
        shift_label=shift_label,
    )


def _inputs(bsz=2, seq_len=24):
    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB - 1, (bsz, seq_len))
    loss_mask = torch.ones(bsz, seq_len)
    hidden = torch.randn(bsz, seq_len, len(TARGET_LAYER_IDS) * HIDDEN)
    return input_ids, hidden, loss_mask


# --------------------------------------------------------------------------- #
# Draft head construction
# --------------------------------------------------------------------------- #


def test_draft_builds_domino_head():
    draft = Qwen3DFlashDraftModel(_draft_config())
    assert isinstance(draft.prefix_gru, torch.nn.GRU)
    assert draft.prefix_gru.input_size == HIDDEN
    assert draft.prefix_gru.hidden_size == GRU_HIDDEN
    # embed_proj projects [hidden | gru] -> emb_dim -> vocab.
    assert draft.embed_proj[0].in_features == HIDDEN + GRU_HIDDEN
    assert draft.embed_proj[-1].out_features == VOCAB
    assert draft.shift_label is True
    assert draft.pure_draft_prefix_len == 1


def test_draft_without_projector_has_no_head():
    draft = Qwen3DFlashDraftModel(_draft_config(projector_type=None))
    assert draft.projector_type is None
    assert not hasattr(draft, "prefix_gru")


def test_draft_unknown_projector_raises():
    with pytest.raises(ValueError, match="Unknown draft projector_type"):
        Qwen3DFlashDraftModel(_draft_config(projector_type="mystery"))


def test_trainer_requires_domino_draft():
    draft = Qwen3DFlashDraftModel(_draft_config(projector_type=None))
    with pytest.raises(ValueError, match="projector_type='domino'"):
        DominoTrainerModule(
            draft_model=draft,
            target_lm_head=torch.nn.Linear(HIDDEN, VOCAB, bias=False),
            target_embed_tokens=torch.nn.Embedding(VOCAB, HIDDEN),
            mask_token_id=MASK_ID,
            block_size=BLOCK_SIZE,
            attention_backend="sdpa",
        )


# --------------------------------------------------------------------------- #
# Forward pass
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("shift_label", [True, False])
def test_forward_returns_metrics_and_grads_flow_to_head(shift_label):
    trainer = _build_trainer(loss_decay_gamma=7.0, shift_label=shift_label)
    input_ids, hidden, loss_mask = _inputs()
    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, lambda_base=0.3)
    assert isinstance(out, DominoStepMetrics)
    assert torch.isfinite(out.loss) and out.loss.item() > 0
    assert 0.0 <= out.accuracy.item() <= 1.0
    assert 0.0 <= out.base_accuracy.item() <= 1.0
    assert out.valid_tokens.item() > 0
    assert torch.isfinite(out.final_loss) and torch.isfinite(out.base_loss)
    # Accept length is at least the always-accepted anchor (>= ~1.0; the 1e-6
    # denominator term keeps it a hair under 1.0 for an all-miss untrained model).
    assert out.accept_len.item() > 0.99 and out.base_accept_len.item() > 0.99
    assert out.lambda_base.item() == pytest.approx(0.3)

    out.loss.backward()
    gru_grad = sum(p.grad.abs().sum().item() for p in trainer.draft_model.prefix_gru.parameters() if p.grad is not None)
    proj_grad = sum(
        p.grad.abs().sum().item() for p in trainer.draft_model.embed_proj.parameters() if p.grad is not None
    )
    assert gru_grad > 0
    assert proj_grad > 0


def test_lambda_base_one_equals_base_loss():
    trainer = _build_trainer()
    input_ids, hidden, loss_mask = _inputs()
    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, lambda_base=1.0)
    assert out.loss.item() == pytest.approx(out.base_loss.item(), rel=1e-5)


def test_lambda_base_zero_equals_final_loss():
    trainer = _build_trainer()
    input_ids, hidden, loss_mask = _inputs()
    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, lambda_base=0.0)
    assert out.loss.item() == pytest.approx(out.final_loss.item(), rel=1e-5)


def test_suffix_start_depends_on_shift_label():
    # shift_label=True -> pure_prefix; shift_label=False -> 1 + pure_prefix.
    assert _build_trainer(pure_prefix=1, shift_label=True)._suffix_start == 1
    assert _build_trainer(pure_prefix=1, shift_label=False)._suffix_start == 2
    assert _build_trainer(pure_prefix=2, shift_label=True)._suffix_start == 2


# --------------------------------------------------------------------------- #
# Curriculum schedule + acceptance length
# --------------------------------------------------------------------------- #


def test_get_lambda_base_schedule():
    # Linear decay from start to 0 over decay_ratio * total steps, then clamps at 0.
    assert get_lambda_base(0, 100, lambda_start=1.0, decay_ratio=1.0) == pytest.approx(1.0)
    assert get_lambda_base(50, 100, lambda_start=1.0, decay_ratio=1.0) == pytest.approx(0.5)
    assert get_lambda_base(100, 100, lambda_start=1.0, decay_ratio=1.0) == pytest.approx(0.0)
    assert get_lambda_base(200, 100, lambda_start=1.0, decay_ratio=1.0) == pytest.approx(0.0)
    # Half-decay: lambda_base hits 0 at the midpoint.
    assert get_lambda_base(50, 100, lambda_start=1.0, decay_ratio=0.5) == pytest.approx(0.0)
    assert get_lambda_base(0, 0, lambda_start=1.0, decay_ratio=0.5) == pytest.approx(1.0)


def test_compute_accept_len():
    # block 0: first two predictions correct then a miss -> accept_len 2.
    # block 1: first prediction wrong -> accept_len 0.
    pred = torch.tensor([[[1, 2, 9, 4], [9, 2, 3, 4]]])
    target = torch.tensor([[[1, 2, 3, 4], [1, 2, 3, 4]]])
    valid = torch.ones(1, 2, 4, dtype=torch.bool)
    accept = compute_accept_len(pred, target, valid)
    assert accept.tolist() == [[2.0, 0.0]]
    # An invalid trailing position never truncates the accepted prefix.
    valid2 = torch.tensor([[[True, True, False, True], [True, True, True, True]]])
    pred2 = torch.tensor([[[1, 2, 0, 4], [1, 2, 3, 4]]])
    accept2 = compute_accept_len(pred2, target, valid2)
    assert accept2.tolist() == [[3.0, 4.0]]

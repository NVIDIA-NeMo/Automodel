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

"""Tests for the DFlash 2 trainer module (block CE + candidate-selection CE)."""

from __future__ import annotations

from unittest import mock

import pytest
import torch
import torch.nn.functional as F
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel.components.speculative.dflash.core import DFlashTrainerModule
from nemo_automodel.components.speculative.dflash.dflash2_core import (
    DFlash2StepMetrics,
    DFlash2TrainerModule,
)
from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel
from nemo_automodel.components.speculative.dflash.draft_qwen3_dflash2 import Qwen3DFlash2DraftModel

VOCAB = 64
HIDDEN = 32
NUM_TARGET_LAYERS = 8
TARGET_LAYER_IDS = [1, 3, 5]
BLOCK_SIZE = 4
MASK_ID = VOCAB - 1
TOP_K = 4


def _draft_cfg(attention_backend="sdpa", selector_top_k=TOP_K):
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
    cfg.dflash_config = {
        "mask_token_id": MASK_ID,
        "target_layer_ids": TARGET_LAYER_IDS,
        "conv_group_size": 8,
        "selector_rank": 16,
        "selector_top_k": selector_top_k,
    }
    cfg._attn_implementation = attention_backend
    return cfg


def _build_trainer(
    num_anchors=8,
    loss_decay_gamma=None,
    loss_type="dflash",
    dpace_alpha=0.5,
    attention_backend="sdpa",
    selector_loss_weight=1.0,
    selector_top_k=TOP_K,
):
    torch.manual_seed(0)
    draft = Qwen3DFlash2DraftModel(_draft_cfg(attention_backend, selector_top_k))
    return DFlash2TrainerModule(
        draft_model=draft,
        target_lm_head=torch.nn.Linear(HIDDEN, VOCAB, bias=False),
        target_embed_tokens=torch.nn.Embedding(VOCAB, HIDDEN),
        mask_token_id=MASK_ID,
        block_size=BLOCK_SIZE,
        attention_backend=attention_backend,
        num_anchors=num_anchors,
        loss_decay_gamma=loss_decay_gamma,
        loss_type=loss_type,
        dpace_alpha=dpace_alpha,
        selector_loss_weight=selector_loss_weight,
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
    assert isinstance(out, DFlash2StepMetrics)
    assert torch.isfinite(out.loss) and out.loss.item() > 0
    assert torch.isfinite(out.base_loss) and torch.isfinite(out.selector_loss)
    assert 0.0 <= out.accuracy.item() <= 1.0
    assert 0.0 <= out.base_accuracy.item() <= 1.0
    assert 0.0 <= out.candidate_recall.item() <= 1.0
    assert out.valid_tokens.item() > 0
    assert out.loss_weight.item() > 0
    torch.testing.assert_close(out.accuracy, out.correct_tokens / out.valid_tokens)
    torch.testing.assert_close(out.base_accuracy, out.base_correct_tokens / out.valid_tokens)
    torch.testing.assert_close(out.accept_len, out.accept_len_sum / out.valid_blocks)
    torch.testing.assert_close(out.base_accept_len, out.base_accept_len_sum / out.valid_blocks)
    out.loss.backward()
    grad = sum(p.grad.abs().sum().item() for p in trainer.draft_model.parameters() if p.grad is not None)
    assert grad > 0


@pytest.mark.parametrize("loss_type", ["dflash", "dpace"])
def test_selector_term_is_the_only_difference_from_dflash(loss_type):
    """With ``selector_loss_weight=0`` the objective must be DFlash's, exactly.

    The convolutions and the selector start at the identity, so a zero-weighted
    selector term leaves a loss that has to match ``DFlashTrainerModule`` on the
    same weights, the same sampled anchors, and the same ``loss_type``. If it does
    not, the DFlash 2 wrapper has changed the backbone objective rather than only
    adding to it -- covers both the fixed-decay and the D-PACE schedule.
    """
    trainer2 = _build_trainer(loss_decay_gamma=7.0, loss_type=loss_type, selector_loss_weight=0.0)
    torch.manual_seed(0)
    dflash_draft = Qwen3DFlashDraftModel(_draft_cfg())
    dflash_draft.load_state_dict(
        {k: v for k, v in trainer2.draft_model.state_dict().items() if k in dflash_draft.state_dict()}
    )
    trainer1 = DFlashTrainerModule(
        draft_model=dflash_draft,
        target_lm_head=trainer2.lm_head,
        target_embed_tokens=trainer2.embed_tokens,
        mask_token_id=MASK_ID,
        block_size=BLOCK_SIZE,
        attention_backend="sdpa",
        num_anchors=8,
        loss_decay_gamma=7.0,
        loss_type=loss_type,
    )

    input_ids, hidden, loss_mask = _inputs()
    torch.manual_seed(1234)
    out2 = trainer2(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    torch.manual_seed(1234)
    out1 = trainer1(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)

    torch.testing.assert_close(out2.loss, out1.loss)
    torch.testing.assert_close(out2.base_loss, out1.loss)
    torch.testing.assert_close(out2.loss_weight, out1.loss_weight)
    torch.testing.assert_close(out2.valid_tokens, out1.valid_tokens)
    torch.testing.assert_close(out2.base_accuracy, out1.accuracy)


def test_selector_loss_is_added_on_top_of_the_block_ce():
    trainer = _build_trainer(loss_decay_gamma=7.0, selector_loss_weight=0.5)
    input_ids, hidden, loss_mask = _inputs()
    torch.manual_seed(1234)
    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    torch.testing.assert_close(out.loss.detach(), out.base_loss + 0.5 * out.selector_loss)


def test_selector_gradient_reaches_the_codebooks():
    """The selector term must actually train the selector, not just the backbone.

    Uses a top-k spanning the vocabulary so every supervised position carries
    selector signal; a narrower top-k on an untrained tiny model would leave the
    term empty and pass the assertion vacuously.
    """
    trainer = _build_trainer(loss_decay_gamma=7.0, selector_top_k=VOCAB)
    input_ids, hidden, loss_mask = _inputs()
    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    assert out.selector_loss.item() > 0
    out.loss.backward()
    selector = trainer.draft_model.candidate_selector
    assert selector.successor_codebook.grad.abs().sum() > 0
    for name, param in trainer.draft_model.named_parameters():
        if "conv" in name:
            assert param.grad is not None, name


def test_a_batch_with_no_candidate_hits_leaves_a_finite_zero_selector_loss():
    """An untrained draft can miss the true token at every position.

    The selector then has nothing to learn from, so its term must be exactly zero
    rather than a NaN out of an empty weighted mean, and training must still make
    progress on the backbone term.
    """
    trainer = _build_trainer(loss_decay_gamma=7.0, selector_top_k=1)
    input_ids, hidden, loss_mask = _inputs()
    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    if out.candidate_recall.item() == 0.0:
        torch.testing.assert_close(out.selector_loss, torch.tensor(0.0))
        torch.testing.assert_close(out.loss.detach(), out.base_loss)
    out.loss.backward()
    for name, param in trainer.draft_model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


def test_candidate_recall_is_one_when_top_k_covers_the_vocabulary():
    """Every true token is a candidate, so the selector supervises every position.

    ``candidate_recall`` is the ceiling the selector can reach; a top-k spanning
    the whole vocabulary must report 1.0, which pins the masking of positions
    whose true token missed the candidate list.
    """
    trainer = _build_trainer(loss_decay_gamma=7.0, selector_top_k=VOCAB)
    input_ids, hidden, loss_mask = _inputs()
    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    torch.testing.assert_close(out.candidate_recall, torch.tensor(1.0))


def test_candidate_recall_bounds_the_selector_accuracy():
    trainer = _build_trainer(loss_decay_gamma=7.0, selector_top_k=8)
    input_ids, hidden, loss_mask = _inputs()
    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    assert out.accuracy.item() <= out.candidate_recall.item() + 1e-6


@pytest.mark.parametrize("attention_backend", ["eager", "sdpa"])
def test_padding_blocks_do_not_nan_loss_or_grads(attention_backend):
    """A batch mixing sequence lengths produces padding blocks (``block_keep_mask``
    has False entries). Those must not NaN either loss term or the gradients: a
    fully-masked attention row NaNs the softmax, and the selector's extra
    top-k / gather path is a second place that contamination could hide."""
    trainer = _build_trainer(loss_decay_gamma=7.0, attention_backend=attention_backend)
    input_ids, hidden, loss_mask = _inputs(bsz=2, seq_len=24)
    loss_mask[0, 5:] = 0.0

    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    assert torch.isfinite(out.loss) and out.loss.item() > 0
    assert torch.isfinite(out.selector_loss) and torch.isfinite(out.base_loss)
    out.loss.backward()
    for name, param in trainer.draft_model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


def test_requires_a_dflash2_draft_model():
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="candidate_selector"):
        DFlash2TrainerModule(
            draft_model=Qwen3DFlashDraftModel(_draft_cfg()),
            target_lm_head=torch.nn.Linear(HIDDEN, VOCAB, bias=False),
            target_embed_tokens=torch.nn.Embedding(VOCAB, HIDDEN),
            mask_token_id=MASK_ID,
            block_size=BLOCK_SIZE,
            attention_backend="sdpa",
        )


def test_rejects_a_negative_selector_loss_weight():
    with pytest.raises(ValueError, match="selector_loss_weight"):
        _build_trainer(selector_loss_weight=-1.0)


def test_rejects_variable_prefix_loss_type():
    """The selector teacher-forces the predecessor from the fixed-anchor block
    layout (see ``_selector_scores``), which a variable visible prefix breaks."""
    with pytest.raises(ValueError, match="variable_prefix"):
        _build_trainer(loss_type="variable_prefix")


def test_dpace_loss_type_runs_finite_and_trains_both_terms():
    """``loss_type="dpace"`` must run end to end: a finite loss for both terms,
    and gradients reaching the backbone, the convolutions, and the selector."""
    trainer = _build_trainer(loss_type="dpace", dpace_alpha=0.35, selector_top_k=VOCAB)
    input_ids, hidden, loss_mask = _inputs()
    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    assert torch.isfinite(out.loss) and out.loss.item() > 0
    assert torch.isfinite(out.base_loss) and torch.isfinite(out.selector_loss)
    assert out.selector_loss.item() > 0
    out.loss.backward()
    selector = trainer.draft_model.candidate_selector
    assert selector.successor_codebook.grad.abs().sum() > 0
    for name, param in trainer.draft_model.named_parameters():
        if "conv" in name:
            assert param.grad is not None, name


def test_dpace_alpha_changes_the_selector_loss():
    """``dpace_alpha`` must reach the selector term too, not only the base one --
    otherwise the two terms would silently disagree on the D-PACE schedule."""
    input_ids, hidden, loss_mask = _inputs()

    torch.manual_seed(1234)
    low_alpha = _build_trainer(loss_type="dpace", dpace_alpha=0.1, selector_top_k=VOCAB)(
        input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask
    )
    torch.manual_seed(1234)
    high_alpha = _build_trainer(loss_type="dpace", dpace_alpha=0.9, selector_top_k=VOCAB)(
        input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask
    )

    assert not torch.allclose(low_alpha.selector_loss, high_alpha.selector_loss, atol=1e-4)


def test_dpace_with_forced_candidate_misses_runs_finite_and_trains_the_selector():
    """Every other D-PACE test uses ``selector_top_k == vocab_size``, which makes
    ``has_target`` always true and hides how a missed candidate interacts with
    the D-PACE weight recursion. A narrow top-k forces real misses on this tiny,
    untrained model."""
    trainer = _build_trainer(loss_type="dpace", dpace_alpha=0.35, selector_top_k=TOP_K)
    input_ids, hidden, loss_mask = _inputs()

    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    assert out.candidate_recall.item() < 1.0

    assert torch.isfinite(out.loss) and torch.isfinite(out.selector_loss)
    out.loss.backward()
    for name, param in trainer.draft_model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name


def test_dpace_loss_weight_uses_num_anchors_not_the_achieved_block_count():
    """Mirrors the DFlash regression: the achieved block count varies per
    micro-batch and per DP rank, so the D-PACE "mean" denominator must stay
    ``bsz * num_anchors`` rather than following it."""
    trainer = _build_trainer(loss_type="dpace", dpace_alpha=0.4)
    input_ids, hidden, loss_mask = _inputs()
    bsz = input_ids.shape[0]
    loss_mask[:, 4:] = 0.0  # far fewer valid anchor positions than num_anchors=8

    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)

    assert out.loss_weight.item() == float(bsz * trainer.num_anchors)


@pytest.mark.parametrize("loss_type,expected_cross_entropy_calls", [("dflash", 2), ("dpace", 3)])
def test_dflash_loss_type_skips_the_redundant_backbone_confidence_pass(loss_type, expected_cross_entropy_calls):
    """``position_weights`` never reads the backbone confidence's values for
    ``loss_type="dflash"``, so recomputing a second full-vocabulary CE for it
    (on top of the one already inside ``loss_fn``, and the selector's own) would
    be a pure, and on a real model sized, waste."""
    trainer = _build_trainer(loss_type=loss_type, selector_top_k=VOCAB)
    input_ids, hidden, loss_mask = _inputs()

    with mock.patch(
        "nemo_automodel.components.speculative.dflash.dflash2_core.F.cross_entropy", wraps=F.cross_entropy
    ) as cross_entropy:
        trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)

    assert cross_entropy.call_count == expected_cross_entropy_calls

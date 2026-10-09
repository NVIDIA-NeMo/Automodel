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

import pytest
import torch
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
    attention_backend="sdpa",
    selector_loss_weight=1.0,
    selector_top_k=TOP_K,
    max_total_anchors=None,
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
        max_total_anchors=max_total_anchors,
        loss_decay_gamma=loss_decay_gamma,
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


def test_selector_term_is_the_only_difference_from_dflash():
    """With ``selector_loss_weight=0`` the objective must be DFlash's, exactly.

    The convolutions and the selector start at the identity, so a zero-weighted
    selector term leaves a loss that has to match ``DFlashTrainerModule`` on the
    same weights and the same sampled anchors. If it does not, the DFlash 2
    wrapper has changed the backbone objective rather than only adding to it.
    """
    trainer2 = _build_trainer(loss_decay_gamma=7.0, selector_loss_weight=0.0)
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
    )

    input_ids, hidden, loss_mask = _inputs()
    torch.manual_seed(1234)
    out2 = trainer2(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)
    torch.manual_seed(1234)
    out1 = trainer1(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask)

    torch.testing.assert_close(out2.loss, out1.loss)
    torch.testing.assert_close(out2.base_loss, out1.loss)
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


def test_label_ids_supervise_labels_but_selector_predecessor_is_real_anchor(monkeypatch):
    trainer = _build_trainer(loss_decay_gamma=7.0, selector_top_k=VOCAB)
    input_ids, hidden, loss_mask = _inputs(bsz=2, seq_len=12)
    label_ids = (input_ids + 7) % (VOCAB - 1)
    anchors = torch.tensor([[2], [3]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    monkeypatch.setattr(
        trainer,
        "_sample_anchor_positions",
        lambda seq_len, loss_mask, device, doc_remaining=None: (anchors.to(device), keep.to(device)),
    )
    captured = {}
    original_selector_scores = trainer._selector_scores

    def capture_selector_scores(hidden, logits, target_ids):
        captured["target_ids"] = target_ids.detach().clone()
        return original_selector_scores(hidden, logits, target_ids)

    monkeypatch.setattr(trainer, "_selector_scores", capture_selector_scores)
    out = trainer(input_ids=input_ids, hidden_states=hidden, loss_mask=loss_mask, label_ids=label_ids)
    assert torch.isfinite(out.loss)

    expected = torch.stack(
        [
            torch.cat((input_ids[row, anchor : anchor + 1], label_ids[row, anchor + 1 : anchor + BLOCK_SIZE]))
            for row, anchor in enumerate((2, 3))
        ]
    ).unsqueeze(1)
    torch.testing.assert_close(captured["target_ids"], expected)


def test_label_ids_must_match_input_shape_and_dtype():
    trainer = _build_trainer()
    input_ids, hidden, loss_mask = _inputs(bsz=1, seq_len=12)
    with pytest.raises(ValueError, match="match input_ids shape"):
        trainer(input_ids, hidden, loss_mask, label_ids=input_ids[:, :-1])
    with pytest.raises(ValueError, match="must have dtype"):
        trainer(input_ids, hidden, loss_mask, label_ids=input_ids.float())


def test_dflash2_exposes_separate_differentiable_loss_terms():
    trainer = _build_trainer(selector_loss_weight=0.7, loss_decay_gamma=4.0)
    input_ids, hidden, loss_mask = _inputs()
    metrics = trainer(input_ids, hidden, loss_mask)
    torch.testing.assert_close(metrics.loss, metrics.base_loss + 0.7 * metrics.selector_loss)
    assert metrics.base_loss.requires_grad and metrics.selector_loss.requires_grad
    assert not metrics.selector_loss_denominator.requires_grad
    # Candidate-dependent denominator can be smaller than the backbone weight.
    assert 0 <= metrics.selector_loss_denominator <= metrics.loss_weight
    params = tuple(trainer.draft_model.parameters())
    total = torch.autograd.grad(metrics.loss, params, retain_graph=True, allow_unused=True)
    separated = torch.autograd.grad(metrics.base_loss + 0.7 * metrics.selector_loss, params, allow_unused=True)
    for actual, expected in zip(separated, total):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("budget", [0, -1])
def test_total_anchor_budget_rejects_nonpositive(budget):
    with pytest.raises(ValueError, match="max_total_anchors"):
        _build_trainer(max_total_anchors=budget)


def test_dflash2_total_anchor_budget_reaches_shared_sampler():
    trainer = _build_trainer(max_total_anchors=4)
    input_ids, hidden, mask = _inputs(bsz=2)
    metrics = trainer(input_ids, hidden, mask)
    assert metrics.valid_blocks == 4


def _fix_anchors(monkeypatch, trainer, anchors):
    anchors = torch.tensor(anchors)
    keep = torch.ones_like(anchors, dtype=torch.bool)
    monkeypatch.setattr(
        trainer,
        "_sample_anchor_positions",
        lambda seq_len, loss_mask, device, doc_remaining=None: (anchors.to(device), keep.to(device)),
    )


def test_split_loss_terms_reconstruct_the_unsharded_weighted_mean(monkeypatch):
    """``term * denominator`` must be additive across data-parallel shards.

    A caller normalizing over its DP group computes ``sum(term_r * w_r) / sum(w_r)``.
    Splitting one batch into two shards with different supervision depths must
    reproduce the unsharded terms, which only holds if each denominator is the
    exact decay-weighted sum inside its term.
    """
    # top_k=32 leaves some true tokens outside the candidates, so the selector
    # denominator is a strict, non-trivial subset of the backbone one.
    trainer = _build_trainer(loss_decay_gamma=4.0, selector_top_k=32)
    input_ids, hidden, loss_mask = _inputs(bsz=2, seq_len=12)
    label_ids = (input_ids + 7) % (VOCAB - 1)
    loss_mask[1, 6:] = 0
    anchors = ([[1, 4]], [[2, 4]])

    shards = []
    for row, row_anchors in enumerate(anchors):
        _fix_anchors(monkeypatch, trainer, row_anchors)
        rows = slice(row, row + 1)
        shards.append(trainer(input_ids[rows], hidden[rows], loss_mask[rows], label_ids=label_ids[rows]))
    _fix_anchors(monkeypatch, trainer, [a[0] for a in anchors])
    full = trainer(input_ids, hidden, loss_mask, label_ids=label_ids)

    # Decayed, not counted: row 0 has two full blocks, row 1 one full block plus
    # one cut to a single position by its loss mask.
    decay = torch.exp(-torch.arange(BLOCK_SIZE - 1) / 4.0)
    torch.testing.assert_close(shards[0].loss_weight, 2 * decay.sum())
    torch.testing.assert_close(shards[1].loss_weight, decay.sum() + 1)
    assert 0 < full.selector_loss_denominator < full.loss_weight

    for term, denominator in (("base_loss", "loss_weight"), ("selector_loss", "selector_loss_denominator")):
        weights = [getattr(shard, denominator) for shard in shards]
        torch.testing.assert_close(sum(weights), getattr(full, denominator))
        merged = sum(getattr(shard, term) * w for shard, w in zip(shards, weights)) / sum(weights)
        torch.testing.assert_close(merged, getattr(full, term), rtol=1e-5, atol=1e-6)


def test_bf16_base_loss_reconstructs_its_numerator_within_rounding(monkeypatch):
    """BF16 logits keep ``base_loss * loss_weight`` within a few BF16 roundings.

    ``DFlashDecayLoss`` reduces in the logits dtype while ``loss_weight`` is FP32,
    so the product is not exact under BF16 training. The NLL, decay weights, both
    sums and the quotient each round once (unit roundoff ``2**-8``), so the
    relative error is bounded by about ``6 * 2**-8``.
    """
    trainer = _build_trainer(loss_decay_gamma=4.0)
    compute_logits = trainer.draft_model.compute_logits
    captured = {}

    def bf16_logits(*args, **kwargs):
        captured["logits"] = compute_logits(*args, **kwargs).to(torch.bfloat16)
        return captured["logits"]

    monkeypatch.setattr(trainer.draft_model, "compute_logits", bf16_logits)
    input_ids, hidden, loss_mask = _inputs(bsz=2, seq_len=24)
    loss_mask[:, 1::5] = 0
    anchors = torch.tensor([[1, 6, 11, 16], [3, 8, 13, 18]])
    _fix_anchors(monkeypatch, trainer, anchors.tolist())
    out = trainer(input_ids, hidden * 4, loss_mask)
    assert out.base_loss.dtype == torch.bfloat16 and out.loss_weight.dtype == torch.float32

    positions = anchors.unsqueeze(-1) + torch.arange(1, BLOCK_SIZE)
    targets = torch.gather(input_ids.unsqueeze(1).expand(-1, anchors.shape[1], -1), 2, positions)
    weights = torch.gather(loss_mask.unsqueeze(1).expand(-1, anchors.shape[1], -1), 2, positions)
    weights = weights * torch.exp(-torch.arange(BLOCK_SIZE - 1) / 4.0)
    logits = captured["logits"].view(*anchors.shape, BLOCK_SIZE, VOCAB)[:, :, 1:].float()
    nll = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1), reduction="none")
    numerator = (nll.view_as(weights) * weights).sum()

    torch.testing.assert_close(out.loss_weight, weights.sum())
    torch.testing.assert_close(out.base_loss.float() * out.loss_weight, numerator, rtol=6 * 2**-8, atol=0)


def test_packed_label_ids_match_per_document_rows(monkeypatch):
    """Packing two documents with ``label_ids`` must equal running them as rows.

    Each document's blocks take context and clean anchors from ``input_ids`` and
    targets from ``label_ids`` of that document only. The ``4`` anchor crosses the
    first document's end; packing must mask the next document's labels exactly
    as the unpacked row masks out-of-range positions.
    """
    trainer = _build_trainer(loss_decay_gamma=4.0, selector_top_k=32)
    doc_len = 6
    input_ids, hidden, loss_mask = _inputs(bsz=2, seq_len=doc_len)
    label_ids = (input_ids + 7) % (VOCAB - 1)

    _fix_anchors(monkeypatch, trainer, [[1, 4], [2, 1]])
    unpacked = trainer(input_ids, hidden, loss_mask, label_ids=label_ids)

    _fix_anchors(monkeypatch, trainer, [[1, 4, doc_len + 2, doc_len + 1]])

    def run_packed(packed_labels):
        return trainer(
            input_ids.reshape(1, -1),
            hidden.reshape(1, 2 * doc_len, -1),
            loss_mask.reshape(1, -1),
            position_ids=torch.arange(doc_len).repeat(2).unsqueeze(0),
            seq_lens=torch.tensor([[doc_len, doc_len]]),
            doc_remaining=torch.arange(doc_len - 1, -1, -1).repeat(2).unsqueeze(0),
            label_ids=packed_labels,
        )

    packed = run_packed(label_ids.reshape(1, -1))
    for field in ("valid_tokens", "loss_weight", "selector_loss_denominator", "base_loss", "selector_loss"):
        torch.testing.assert_close(getattr(packed, field), getattr(unpacked, field))
    assert not torch.allclose(packed.base_loss, run_packed(None).base_loss)

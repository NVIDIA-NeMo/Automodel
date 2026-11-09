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

"""Tests for the DSpine trainer module and its step schedule."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.speculative.dflash.core import NoValidAnchorsError
from nemo_automodel.components.speculative.dflash.draft_qwen3 import Qwen3DFlashDraftModel
from nemo_automodel.components.speculative.dflash.draft_qwen3_dspine import Qwen3DSpineDraftModel
from nemo_automodel.components.speculative.dflash.dspine_core import DSpineSchedule, DSpineTrainerModule
from tests.unit_tests.speculative.test_dspine_draft import (
    BLOCK_SIZE,
    HIDDEN,
    MASK_ID,
    TARGET_LAYER_IDS,
    TOP_K,
    VOCAB,
    build_draft,
    draft_cfg,
    randomize_writes,
)

BSZ = 2
SEQ_LEN = 20


def build_trainer(
    num_anchors=6,
    loss_decay_gamma=7.0,
    objective_chunk_blocks=128,
    attention_backend="sdpa",
    randomize=True,
    seed=0,
    **kwargs,
):
    draft = build_draft(seed=seed, attention_backend=attention_backend)
    if randomize:
        randomize_writes(draft, scale=0.2)
    torch.manual_seed(seed + 1)
    lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
    embed = torch.nn.Embedding(VOCAB, HIDDEN)
    lm_head.requires_grad_(False)
    embed.requires_grad_(False)
    return DSpineTrainerModule(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=MASK_ID,
        block_size=BLOCK_SIZE,
        attention_backend=attention_backend,
        num_anchors=num_anchors,
        loss_decay_gamma=loss_decay_gamma,
        objective_chunk_blocks=objective_chunk_blocks,
        **kwargs,
    )


def make_batch(seed=0, loss_mask=None):
    gen = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, MASK_ID, (BSZ, SEQ_LEN), generator=gen)
    hidden_states = torch.randn(BSZ, SEQ_LEN, len(TARGET_LAYER_IDS) * HIDDEN, generator=gen)
    target_logits = 3.0 * torch.randn(BSZ, SEQ_LEN, VOCAB, generator=gen)
    if loss_mask is None:
        loss_mask = torch.ones(BSZ, SEQ_LEN)
    return dict(input_ids=input_ids, hidden_states=hidden_states, loss_mask=loss_mask, target_logits=target_logits)


def run(trainer, batch, seed=0, **kwargs):
    torch.manual_seed(seed)  # same anchors / teacher-forcing draws across calls
    return trainer(**batch, **kwargs)


def bridge_grads(trainer):
    return {
        name: p.grad.clone()
        for name, p in trainer.draft_model.named_parameters()
        if p.grad is not None and name.startswith(("selected_prefix_bridge.", "layers.0.", "fc."))
    }


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def reference_loss(trainer, batch, alignment_beta, seed=0):
    """The corrected SpecForge DSpine objective (sgl-project/SpecForge#1038), unchunked.

    Re-derives the anchors, draft forward and every term from the trainer's public
    pieces with plain full-tensor math, as an independent check of the chunked,
    checkpointed implementation.
    """
    torch.manual_seed(seed)
    input_ids, loss_mask = batch["input_ids"], batch["loss_mask"]
    anchors, keep, noise, positions, mask, _ = trainer._prepare_block_inputs(input_ids, loss_mask, causal=True)
    label_idx, block_ids, block_mask = trainer._build_block_targets(input_ids, loss_mask, anchors, keep, SEQ_LEN)
    draft = trainer.draft_model
    out = draft(
        position_ids=positions,
        attention_mask=mask,
        noise_embedding=noise,
        target_hidden=batch["hidden_states"],
        block_token_ids=block_ids,
    )
    n = anchors.shape[1]
    head = trainer.lm_head.weight
    hidden = out.hidden_states.view(BSZ, n, BLOCK_SIZE, HIDDEN)[:, :, 1:]
    pre = out.pre_injection.view(BSZ, n, BLOCK_SIZE, HIDDEN)[:, :, 1:]
    labels, preds = block_ids[:, :, 1:], block_ids[:, :, :-1]
    supervised = block_mask[:, :, 1:].cumprod(-1)
    w = supervised * torch.exp(-torch.arange(BLOCK_SIZE - 1) / 7.0)
    teacher = torch.stack([batch["target_logits"][b][label_idx[b, :, :-1]] for b in range(BSZ)])

    logits = hidden @ head.T
    ce = F.cross_entropy(logits.flatten(0, 2), labels.flatten(), reduction="none").view_as(w)
    l1 = (logits.softmax(-1) - teacher.softmax(-1)).abs().sum(-1)

    cand = logits.topk(TOP_K, dim=-1).indices
    scores = (draft.refine(pre, preds).unsqueeze(-2) * head[cand]).sum(-1)
    hit = (cand == labels.unsqueeze(-1)).any(-1)
    slot = (cand == labels.unsqueeze(-1)).long().argmax(-1)
    r_ce = F.cross_entropy(scores.flatten(0, 2), slot.flatten(), reduction="none").view_as(w)
    eligible = teacher.softmax(-1).gather(-1, cand).sum(-1) > 1e-4
    r_l1 = (scores.softmax(-1) - teacher.gather(-1, cand).softmax(-1)).abs().sum(-1)
    refine_ce = (r_ce * hit * w).sum() / (hit * w).sum()
    refine_l1 = (r_l1 * eligible * w).sum() / (eligible * w).sum()

    codes = draft.selected_prefix_bridge.codebook[labels]
    feats = out.layer_features.view(-1, BSZ, n, BLOCK_SIZE, out.layer_features.shape[-1])[:, :, :, 1:]
    align = (1 - F.cosine_similarity(feats, codes.unsqueeze(0), dim=-1)) * w
    alignment = align.sum() / feats.shape[0]

    a = 0.1
    numerator = (
        a * (ce * w).sum()
        + (1 - a) * (l1 * w).sum()
        + (a * refine_ce + (1 - a) * refine_l1) * supervised.sum()
        + alignment_beta * ((ce * w).sum() / w.sum()).detach() * alignment
    )
    return numerator / BSZ


@pytest.mark.parametrize("alignment_beta", [0.0, 0.5])
def test_loss_matches_the_reference_objective(alignment_beta):
    trainer = build_trainer()
    batch = make_batch()
    metrics = run(trainer, batch, alignment_beta=alignment_beta)
    expected = reference_loss(trainer, batch, alignment_beta)
    torch.testing.assert_close(metrics.loss, expected, rtol=1e-4, atol=1e-4)
    assert metrics.loss_weight.item() == BSZ


def test_chunking_changes_neither_the_loss_nor_the_gradients():
    batch = make_batch()
    results = []
    for chunk in (1, 5, 128):
        trainer = build_trainer(objective_chunk_blocks=chunk)
        metrics = run(trainer, batch, alignment_beta=0.5)
        metrics.loss.backward()
        results.append((metrics, bridge_grads(trainer)))
    (ref_metrics, ref_grads), *others = results
    assert ref_grads
    for metrics, grads in others:
        torch.testing.assert_close(metrics.loss, ref_metrics.loss, rtol=1e-5, atol=1e-5)
        assert metrics.correct_tokens == ref_metrics.correct_tokens
        assert grads.keys() == ref_grads.keys()
        for name in grads:
            torch.testing.assert_close(grads[name], ref_grads[name], rtol=1e-4, atol=1e-5)


def test_gradients_reach_the_backbone_and_bridge_but_not_the_frozen_tensors():
    trainer = build_trainer()
    run(trainer, make_batch(), alignment_beta=0.5).loss.backward()
    bridge = trainer.draft_model.selected_prefix_bridge
    for param in (bridge.read_weight, bridge.early_read_weights, bridge.feature_write_weights, bridge.prev.weight):
        assert param.grad is not None and param.grad.abs().sum() > 0
    assert trainer.draft_model.fc.weight.grad.abs().sum() > 0
    assert trainer.lm_head.weight.grad is None
    assert not bridge.codebook.requires_grad


def test_alignment_term_adds_beta_times_detached_backbone_ce():
    trainer = build_trainer()
    batch = make_batch()
    off = run(trainer, batch, alignment_beta=0.0)
    on = run(trainer, batch, alignment_beta=0.25)
    backbone_ce = off.backbone_ce_sum / off.weight_sum
    expected = off.loss + 0.25 * backbone_ce * off.alignment_sum / BSZ
    torch.testing.assert_close(on.loss, expected, rtol=1e-5, atol=1e-5)


def test_loss_is_finite_and_zero_free_when_a_block_has_no_candidate_hits(monkeypatch):
    trainer = build_trainer()
    batch = make_batch()
    real_topk = torch.Tensor.topk

    def miss_topk(self, k, dim=-1, **kwargs):
        # Candidates from the tail of the vocabulary that no label uses.
        out = real_topk(self, k, dim=dim, **kwargs)
        return type(out)((out.values, torch.full_like(out.indices, MASK_ID)))

    monkeypatch.setattr(torch.Tensor, "topk", miss_topk)
    metrics = run(trainer, batch)
    assert torch.isfinite(metrics.loss)
    assert metrics.refine_ce_weight == 0 and metrics.refine_ce == 0
    assert metrics.candidate_hits == 0 and metrics.correct_tokens == 0
    metrics.loss.backward()


# ---------------------------------------------------------------------------
# Supervision and metrics
# ---------------------------------------------------------------------------


def test_supervision_stops_at_the_first_unsupervised_token():
    loss_mask = torch.ones(BSZ, SEQ_LEN)
    # Every position after index 2 is unsupervised except an isolated island at 5.
    loss_mask[:, 3:] = 0
    loss_mask[:, 5] = 1
    trainer = build_trainer(num_anchors=32)
    metrics = run(trainer, make_batch(loss_mask=loss_mask))
    # Valid anchors are 0, 1, 2 and 5 (anchor <= SEQ_LEN - BLOCK_SIZE with loss_mask=1);
    # their contiguous supervised prefixes after the anchor have 2, 1, 0 and 0 tokens.
    assert metrics.valid_tokens.item() == BSZ * 3


def test_refined_accept_len_equals_the_serial_greedy_walk():
    """The teacher-forced refinement metric must equal what decoding emits: walk each
    block left to right, rescoring the top-k candidates with the token just chosen."""
    trainer = build_trainer().eval()
    batch = make_batch()
    with torch.no_grad():
        metrics = run(trainer, batch)
        torch.manual_seed(0)
        anchors, keep, noise, positions, mask, _ = trainer._prepare_block_inputs(
            batch["input_ids"], batch["loss_mask"], causal=True
        )
        _, block_ids, _ = trainer._build_block_targets(batch["input_ids"], batch["loss_mask"], anchors, keep, SEQ_LEN)
        draft = trainer.draft_model
        out = draft(
            position_ids=positions,
            attention_mask=mask,
            noise_embedding=noise,
            target_hidden=batch["hidden_states"],
            block_token_ids=block_ids,
        )
        n = anchors.shape[1]
        head = trainer.lm_head.weight
        hidden = out.hidden_states.view(BSZ, n, BLOCK_SIZE, HIDDEN)
        pre = out.pre_injection.view(BSZ, n, BLOCK_SIZE, HIDDEN)
        accepted = []
        for b in range(BSZ):
            for blk in range(n):
                previous, length = block_ids[b, blk, 0], 0
                for t in range(1, BLOCK_SIZE):
                    cand = (hidden[b, blk, t] @ head.T).topk(TOP_K).indices
                    scores = head[cand] @ draft.refine(pre[b, blk, t], previous)
                    previous = cand[scores.argmax()]
                    if previous != block_ids[b, blk, t]:
                        break
                    length += 1
                accepted.append(length + 1)
    torch.testing.assert_close(metrics.accept_len, torch.tensor(sum(accepted) / len(accepted)))


def test_teacher_forcing_applies_only_in_training_mode():
    trainer = build_trainer()
    batch = make_batch()
    trainer.train()
    forced = run(trainer, batch, teacher_forcing=1.0)
    plain = run(trainer, batch, teacher_forcing=0.0)
    assert not torch.allclose(forced.loss, plain.loss)
    trainer.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            run(trainer, batch, teacher_forcing=1.0).loss, run(trainer, batch, teacher_forcing=0.0).loss
        )


@pytest.mark.parametrize("attention_backend", ["eager", "sdpa"])
def test_padding_blocks_and_packing_keep_the_loss_finite(attention_backend):
    trainer = build_trainer(num_anchors=12, attention_backend=attention_backend)
    batch = make_batch()
    batch["loss_mask"][1, 4:] = 0  # sample 1 has far fewer anchors -> padded blocks
    metrics = run(trainer, batch, alignment_beta=0.5, teacher_forcing=0.5)
    assert torch.isfinite(metrics.loss)
    metrics.loss.backward()
    assert all(torch.isfinite(g).all() for g in bridge_grads(trainer).values())

    seq_lens = torch.tensor([[12, 8], [20, 0]])
    position_ids = torch.cat([torch.arange(12), torch.arange(8)]).unsqueeze(0)
    position_ids = torch.cat([position_ids, torch.arange(SEQ_LEN).unsqueeze(0)])
    doc_remaining = torch.stack(
        [torch.cat([torch.arange(11, -1, -1), torch.arange(7, -1, -1)]), torch.arange(SEQ_LEN - 1, -1, -1)]
    )
    packed = run(trainer, make_batch(), position_ids=position_ids, seq_lens=seq_lens, doc_remaining=doc_remaining)
    assert torch.isfinite(packed.loss)


def test_a_batch_without_anchors_raises_no_valid_anchors():
    trainer = build_trainer()
    with pytest.raises(NoValidAnchorsError):
        run(trainer, make_batch(loss_mask=torch.zeros(BSZ, SEQ_LEN)))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_requires_a_dspine_draft_model():
    draft = Qwen3DFlashDraftModel(draft_cfg())
    with pytest.raises(TypeError, match="Qwen3DSpineDraftModel"):
        DSpineTrainerModule(
            draft, torch.nn.Linear(HIDDEN, VOCAB, bias=False), torch.nn.Embedding(VOCAB, HIDDEN), MASK_ID
        )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"block_size": BLOCK_SIZE + 1}, "block_size"),
        ({"distribution_ce_weight": 1.5}, "distribution_ce_weight"),
        ({"candidate_mass_threshold": 1.0}, "candidate_mass_threshold"),
        ({"objective_chunk_blocks": 0}, "objective_chunk_blocks"),
    ],
)
def test_rejects_invalid_arguments(kwargs, match):
    draft = build_draft()
    args = dict(block_size=BLOCK_SIZE, attention_backend="sdpa")
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        DSpineTrainerModule(
            draft, torch.nn.Linear(HIDDEN, VOCAB, bias=False), torch.nn.Embedding(VOCAB, HIDDEN), MASK_ID, **args
        )


def test_rejects_a_draft_whose_transfer_space_was_never_initialized():
    """Training on the placeholder codebook would feed zero anchor codes silently."""
    torch.manual_seed(0)
    draft = Qwen3DSpineDraftModel(draft_cfg())
    with pytest.raises(ValueError, match="initialize_transfer_space"):
        DSpineTrainerModule(
            draft,
            torch.nn.Linear(HIDDEN, VOCAB, bias=False),
            torch.nn.Embedding(VOCAB, HIDDEN),
            MASK_ID,
            block_size=BLOCK_SIZE,
        )


def test_rejects_a_head_with_bias():
    with pytest.raises(ValueError, match="bias"):
        DSpineTrainerModule(
            build_draft(),
            torch.nn.Linear(HIDDEN, VOCAB, bias=True),
            torch.nn.Embedding(VOCAB, HIDDEN),
            MASK_ID,
            block_size=BLOCK_SIZE,
        )


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


def make_schedule(**overrides):
    args = dict(
        alignment_weight=0.5,
        alignment_warmup_steps=10,
        teacher_forcing_probability=0.5,
        teacher_forcing_hold_fraction=1 / 6,
        teacher_forcing_zero_fraction=1 / 3,
        total_steps=60,
    )
    args.update(overrides)
    return DSpineSchedule(**args)


def test_alignment_beta_warms_up_linearly():
    schedule = make_schedule()
    assert [schedule.alignment_beta(s) for s in (0, 5, 10, 50)] == [0.0, 0.25, 0.5, 0.5]
    assert make_schedule(alignment_warmup_steps=0).alignment_beta(0) == 0.5


def test_teacher_forcing_holds_then_decays_to_zero():
    schedule = make_schedule()
    # Held for the first sixth (10 steps), zero from a third (20 steps) on.
    assert schedule.teacher_forcing(0) == 0.5
    assert schedule.teacher_forcing(10) == 0.5
    assert math.isclose(schedule.teacher_forcing(15), 0.25)
    assert schedule.teacher_forcing(20) == 0.0
    assert schedule.teacher_forcing(59) == 0.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"alignment_weight": -1.0},
        {"alignment_warmup_steps": -1},
        {"teacher_forcing_probability": 1.5},
        {"teacher_forcing_hold_fraction": 0.5, "teacher_forcing_zero_fraction": 0.4},
        {"total_steps": 0},
    ],
)
def test_schedule_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        make_schedule(**overrides)

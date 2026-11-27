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

"""Methodology invariants of the target-agnostic DFly2 pieces: key lists, objective, trainer metrics."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.speculative.dfly2.common import (
    KEY_LIST_ALIGNMENT,
    DFly2TargetFeatures,
    build_block_key_indices,
    context_key_indices,
)
from nemo_automodel.components.speculative.dfly2.core import DFly2TrainerModule
from nemo_automodel.components.speculative.dfly2.loss import compute_dfly2_loss
from nemo_automodel.components.speculative.dfly2.target import DFly2TargetModel
from nemo_automodel.components.speculative.dspark.common import DSparkForwardOutput

pytestmark = pytest.mark.cpu_only


def _context_indices() -> torch.Tensor:
    # Row r lists keys r and 10 + r (two segments of a 10-row context).
    rows = torch.arange(5)
    return torch.stack((rows, rows + 10), dim=-1).view(1, 5, 2)


def test_block_reads_the_row_before_its_anchor_then_its_own_block() -> None:
    indices = build_block_key_indices(
        _context_indices(),
        torch.tensor([[3, 1]]),
        torch.tensor([[True, True]]),
        num_context_keys=20,
        block_size=2,
        causal=False,
    )
    assert indices.shape == (1, 4, KEY_LIST_ALIGNMENT) and indices.dtype == torch.int32
    # Block 0 (anchor 3) reads row 2; block 1 (anchor 1) reads row 0. Draft keys follow the context.
    assert indices[0, :, :4].tolist() == [
        [2, 12, 20, 21],
        [2, 12, 20, 21],
        [0, 10, 22, 23],
        [0, 10, 22, 23],
    ]
    assert (indices[0, :, 4:] == -1).all()


def test_dropped_blocks_and_sequence_start_anchors_read_nothing_from_the_context() -> None:
    indices = build_block_key_indices(
        _context_indices(),
        torch.tensor([[0, 4]]),
        torch.tensor([[True, False]]),
        num_context_keys=20,
        block_size=2,
        causal=True,
    )
    # Anchor 0 has no verified token before it, but still sees its own block, causally.
    assert indices[0, 0, :4].tolist() == [-1, -1, 20, -1]
    assert indices[0, 1, :4].tolist() == [-1, -1, 20, 21]
    # A dropped block sees no key at all, so its output is zero.
    assert (indices[0, 2:] == -1).all()


def test_context_key_indices_masks_only_the_draft_keys() -> None:
    indices = torch.tensor([[[3, -1, 20, 21]]], dtype=torch.int32)
    assert context_key_indices(indices, 20).tolist() == [[[3, -1, -1, -1]]]


def test_target_features_check_their_shapes() -> None:
    with pytest.raises(ValueError, match="one key tensor per key-list"):
        DFly2TargetFeatures(
            context_keys=(torch.zeros(1, 4, 2),), context_indices=(), last_hidden_states=torch.zeros(1, 2, 3)
        )
    with pytest.raises(ValueError, match="must match"):
        DFly2TargetFeatures(
            context_keys=(torch.zeros(1, 4, 2),),
            context_indices=(torch.zeros(1, 3, 2, dtype=torch.long),),
            last_hidden_states=torch.zeros(1, 2, 3),
        )


def _outputs(draft_logits: torch.Tensor, target_logits: torch.Tensor, eval_mask: torch.Tensor, confidence=None):
    batch, blocks, drafted, _ = draft_logits.shape
    return DSparkForwardOutput(
        draft_logits=draft_logits,
        target_ids=target_logits.argmax(-1),
        eval_mask=eval_mask,
        block_keep_mask=eval_mask.any(-1),
        confidence_pred=confidence,
        aligned_target_logits=target_logits,
    )


def _reference_loss(draft_logits, target_logits, target_ids, mask, alpha, eta):
    """The objective written out slot by slot."""
    q = draft_logits.float().softmax(-1)
    p = target_logits.float().softmax(-1)
    tv = 0.5 * (p - q).abs().sum(-1)
    kl = (p * (p.log() - q.log())).sum(-1)
    lam = torch.exp(-eta * (1 - tv))
    lk = lam * kl + (1 - lam) * tv
    confidence = q.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    smooth = torch.where(mask, (1 - alpha) * confidence + alpha, torch.ones_like(confidence))
    prefix = smooth.cumprod(-1)
    weights = (prefix * mask).flip(-1).cumsum(-1).flip(-1) * mask
    return (lk * weights).sum() / weights.sum(), (tv * weights).sum() / weights.sum()


def test_objective_matches_the_written_out_formula() -> None:
    torch.manual_seed(0)
    draft_logits = torch.randn(2, 3, 4, 11, requires_grad=True)
    target_logits = torch.randn(2, 3, 4, 11)
    mask = torch.ones(2, 3, 4, dtype=torch.bool)
    mask[0, 1, 2:] = False
    mask[1, 2] = False
    outputs = _outputs(draft_logits, target_logits, mask)
    loss, terms = compute_dfly2_loss(outputs=outputs, dpace_alpha=0.5, lk_eta=3.0, confidence_head_alpha=1.0)
    expected, expected_tv = _reference_loss(draft_logits, target_logits, outputs.target_ids, mask, 0.5, 3.0)
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(terms["l1_loss"], expected_tv, rtol=1e-5, atol=1e-6)
    # The D-PACE weights and the LK mixing weight are detached: the gradient is that of the
    # weighted LK sum with both held fixed.
    loss.backward()
    draft_grad = draft_logits.grad.clone()
    draft_logits.grad = None
    expected.backward()
    assert not torch.allclose(draft_grad, draft_logits.grad)
    assert torch.isfinite(draft_grad).all() and draft_grad[1, 2].abs().max() == 0


def test_identical_distributions_give_zero_distillation_and_full_acceptance() -> None:
    logits = torch.randn(1, 2, 3, 7)
    mask = torch.ones(1, 2, 3, dtype=torch.bool)
    loss, terms = compute_dfly2_loss(
        outputs=_outputs(logits.clone(), logits, mask), dpace_alpha=0.5, lk_eta=3.0, confidence_head_alpha=1.0
    )
    torch.testing.assert_close(loss, torch.zeros(()), atol=1e-6, rtol=0)
    torch.testing.assert_close(terms["accept_rate_per_pos_num"], terms["accept_rate_per_pos_den"])
    # Every drafted token accepted: tau counts the anchor plus the three drafted tokens.
    torch.testing.assert_close(terms["tau_num"] / terms["tau_den"], torch.tensor(4.0))


def test_confidence_head_regresses_the_acceptance_probability() -> None:
    torch.manual_seed(1)
    draft_logits = torch.randn(1, 2, 3, 7)
    target_logits = torch.randn(1, 2, 3, 7)
    mask = torch.ones(1, 2, 3, dtype=torch.bool)
    confidence = torch.zeros(1, 2, 3, requires_grad=True)
    _, without = compute_dfly2_loss(
        outputs=_outputs(draft_logits, target_logits, mask), dpace_alpha=0.5, lk_eta=3.0, confidence_head_alpha=1.0
    )
    loss, terms = compute_dfly2_loss(
        outputs=_outputs(draft_logits, target_logits, mask, confidence),
        dpace_alpha=0.5,
        lk_eta=3.0,
        confidence_head_alpha=1.0,
    )
    # At logit 0 the BCE is log 2 for any target; the loss adds it on top of the LK term.
    torch.testing.assert_close(terms["confidence_loss"], torch.tensor(2.0).log())
    torch.testing.assert_close(terms["loss"], without["loss"] + terms["confidence_loss"])
    loss.backward()
    accept = 1 - 0.5 * (target_logits.softmax(-1) - draft_logits.softmax(-1)).abs().sum(-1)
    # d BCE / d logit = sigmoid(0) - target, so the gradient sign follows 0.5 - acceptance.
    assert torch.equal(confidence.grad.sign(), (0.5 - accept).sign())
    assert terms["confidence_diag_den"] > 0


def test_objective_requires_the_teacher() -> None:
    outputs = _outputs(torch.randn(1, 1, 2, 5), torch.randn(1, 1, 2, 5), torch.ones(1, 1, 2, dtype=torch.bool))
    outputs.aligned_target_logits = None
    with pytest.raises(ValueError, match="aligned_target_logits"):
        compute_dfly2_loss(outputs=outputs, dpace_alpha=0.5, lk_eta=3.0, confidence_head_alpha=1.0)


class _StubDraft(nn.Module):
    """Return fixed outputs over two drafted slots, recording the features it receives."""

    def __init__(self, outputs: DSparkForwardOutput):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.outputs = outputs
        self.seen = None

    def forward(self, **kwargs):
        self.seen = kwargs
        self.outputs.draft_logits = self.outputs.draft_logits + self.weight
        return self.outputs


def test_trainer_reports_per_drafted_token_acceptance_over_the_block() -> None:
    torch.manual_seed(2)
    outputs = _outputs(torch.randn(1, 2, 2, 5), torch.randn(1, 2, 2, 5), torch.ones(1, 2, 2, dtype=torch.bool))
    draft = _StubDraft(outputs)
    trainer = DFly2TrainerModule(draft)
    keys, indices = (torch.zeros(1, 4, 2),), (torch.zeros(1, 2, 3, dtype=torch.int32),)
    metrics = trainer(
        input_ids=torch.zeros(1, 2, dtype=torch.long),
        loss_mask=torch.ones(1, 2),
        context_keys=keys,
        context_indices=indices,
        target_last_hidden_states=torch.zeros(1, 2, 3),
    )
    # A block of 3 slots drafts 2 tokens: the per-position sums have block_size entries and
    # the last one, which no drafted token fills, stays empty so it is not logged.
    assert metrics.accept_rate_per_pos_den.tolist() == [2.0, 2.0, 0.0]
    assert metrics.accept_rate_per_pos_num[-1] == 0
    assert draft.seen["context_keys"] is keys and draft.seen["context_indices"] is indices
    metrics.loss.backward()
    assert draft.weight.grad is not None


def test_trainer_rejects_invalid_dpace_alpha() -> None:
    with pytest.raises(ValueError, match="dpace_alpha"):
        DFly2TrainerModule(nn.Linear(1, 1), dpace_alpha=1.5)


def test_target_wrapper_requires_a_feature_provider() -> None:
    with pytest.raises(TypeError, match="does not provide DFly2 target features"):
        DFly2TargetModel(nn.Linear(1, 1), [1])

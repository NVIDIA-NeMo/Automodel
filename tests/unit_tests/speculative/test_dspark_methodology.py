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

"""Correctness tests for the DSpark training methodology.

These pin the *method* itself -- the three-term objective, its position-decay
weighting, the total-variation/acceptance identity, eval-masking, and
anchor-validity -- against independently re-derived reference values. All run on
CPU: the objective and the sampler are plain tensor math (no FlexAttention).
"""

import torch
import torch.nn.functional as F

from nemo_automodel.components.speculative.dspark import loss as loss_module
from nemo_automodel.components.speculative.dspark.common import (
    DSparkForwardOutput,
    build_anchor_candidate_mask,
    sample_anchor_positions,
)
from nemo_automodel.components.speculative.dspark.loss import compute_dspark_loss


def _output(
    *,
    draft_logits,
    target_ids,
    eval_mask,
    block_keep_mask,
    confidence_pred=None,
    aligned_target_logits=None,
):
    return DSparkForwardOutput(
        draft_logits=draft_logits,
        target_ids=target_ids,
        eval_mask=eval_mask,
        block_keep_mask=block_keep_mask,
        confidence_pred=confidence_pred,
        aligned_target_logits=aligned_target_logits,
    )


def _ce_per_token(draft_logits, target_ids):
    b, a, k, v = draft_logits.shape
    return F.cross_entropy(draft_logits.reshape(-1, v), target_ids.reshape(-1), reduction="none").reshape(b, a, k)


def test_cross_entropy_term_matches_masked_mean():
    """With uniform weights, the CE term is the eval-masked mean cross-entropy."""
    torch.manual_seed(0)
    draft_logits = torch.randn(1, 1, 2, 4)
    target_ids = torch.tensor([[[1, 3]]])
    eval_mask = torch.ones(1, 1, 2, dtype=torch.bool)
    block_keep_mask = torch.ones(1, 1, dtype=torch.bool)

    loss = compute_dspark_loss(
        outputs=_output(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
        ),
        loss_decay_gamma=None,
        ce_loss_alpha=1.0,
        l1_loss_alpha=0.0,
        confidence_head_alpha=0.0,
    )

    ce = _ce_per_token(draft_logits, target_ids)
    expected = ce.sum() / (eval_mask.sum() + 1e-6)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_position_decay_weighting():
    """Position k carries weight exp(-k/gamma); the CE term reflects that exactly."""
    torch.manual_seed(1)
    draft_logits = torch.randn(1, 1, 3, 5)
    target_ids = torch.tensor([[[0, 2, 4]]])
    eval_mask = torch.ones(1, 1, 3, dtype=torch.bool)
    block_keep_mask = torch.ones(1, 1, dtype=torch.bool)
    gamma = 2.0

    loss = compute_dspark_loss(
        outputs=_output(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
        ),
        loss_decay_gamma=gamma,
        ce_loss_alpha=1.0,
        l1_loss_alpha=0.0,
        confidence_head_alpha=0.0,
    )

    ce = _ce_per_token(draft_logits, target_ids)[0, 0]
    w = torch.exp(-torch.arange(3, dtype=torch.float32) / gamma)
    expected = (ce * w).sum() / (w.sum() + 1e-6)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_eval_mask_excludes_disabled_positions():
    """A position with eval_mask=0 contributes neither to the numerator nor denominator."""
    torch.manual_seed(2)
    draft_logits = torch.randn(1, 1, 2, 4)
    target_ids = torch.tensor([[[1, 2]]])
    block_keep_mask = torch.ones(1, 1, dtype=torch.bool)
    eval_mask = torch.tensor([[[True, False]]])

    loss = compute_dspark_loss(
        outputs=_output(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
        ),
        loss_decay_gamma=None,
        ce_loss_alpha=1.0,
        l1_loss_alpha=0.0,
        confidence_head_alpha=0.0,
    )

    ce = _ce_per_token(draft_logits, target_ids)[0, 0]
    expected = ce[0] / (1.0 + 1e-6)  # only position 0 is supervised
    assert torch.allclose(loss, expected, atol=1e-6)


def test_total_variation_term_zero_when_draft_matches_target():
    """L_tv is the TV distance; it vanishes when draft logits equal target logits."""
    torch.manual_seed(3)
    draft_logits = torch.randn(1, 1, 2, 4)
    aligned = draft_logits.clone()
    target_ids = torch.tensor([[[1, 2]]])
    eval_mask = torch.ones(1, 1, 2, dtype=torch.bool)
    block_keep_mask = torch.ones(1, 1, dtype=torch.bool)

    loss, terms = compute_dspark_loss(
        outputs=_output(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            aligned_target_logits=aligned,
        ),
        loss_decay_gamma=None,
        ce_loss_alpha=0.0,
        l1_loss_alpha=1.0,
        confidence_head_alpha=0.0,
        return_terms=True,
    )
    # Draft == target -> TV distance 0 -> the l1 term and the total loss are 0.
    assert terms["l1_loss"].item() == 0.0
    assert loss.item() == 0.0


def test_acceptance_rate_and_tau_match_reference():
    """accept_rate is the eval-masked mean TV acceptance; tau is the expected
    accepted block length (running product over the block, plus the anchor)."""
    torch.manual_seed(5)
    draft_logits = torch.randn(1, 1, 3, 5)
    aligned = torch.randn(1, 1, 3, 5)
    target_ids = torch.zeros(1, 1, 3, dtype=torch.long)
    eval_mask = torch.ones(1, 1, 3, dtype=torch.bool)
    block_keep_mask = torch.ones(1, 1, dtype=torch.bool)

    _, terms = compute_dspark_loss(
        outputs=_output(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            aligned_target_logits=aligned,
        ),
        loss_decay_gamma=None,
        ce_loss_alpha=0.0,
        l1_loss_alpha=1.0,
        confidence_head_alpha=0.0,
        return_terms=True,
    )

    tv = (draft_logits.softmax(-1) - aligned.softmax(-1)).abs().sum(-1)
    accept = (1.0 - 0.5 * tv).clamp(0.0, 1.0)
    mask = eval_mask.float()
    expected_accept = (accept * mask).sum() / (mask.sum() + 1e-6)
    expected_tau = (accept[0, 0].cumprod(0).sum() + 1.0) / (1.0 + 1e-6)
    assert torch.allclose(terms["accept_rate"], expected_accept, atol=1e-6)
    assert torch.allclose(terms["tau"], expected_tau, atol=1e-6)


def test_confidence_calibration_matches_reference():
    """The confidence diagnostics are the eval-masked mean absolute error and
    signed bias of the confidence head against the measured acceptance rate."""
    torch.manual_seed(6)
    draft_logits = torch.randn(1, 1, 3, 5)
    aligned = torch.randn(1, 1, 3, 5)
    confidence_pred = torch.randn(1, 1, 3)
    target_ids = torch.zeros(1, 1, 3, dtype=torch.long)
    eval_mask = torch.ones(1, 1, 3, dtype=torch.bool)
    block_keep_mask = torch.ones(1, 1, dtype=torch.bool)

    _, terms = compute_dspark_loss(
        outputs=_output(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            confidence_pred=confidence_pred,
            aligned_target_logits=aligned,
        ),
        loss_decay_gamma=None,
        ce_loss_alpha=0.0,
        l1_loss_alpha=0.0,
        confidence_head_alpha=1.0,
        return_terms=True,
    )

    tv = (draft_logits.softmax(-1) - aligned.softmax(-1)).abs().sum(-1)
    accept = (1.0 - 0.5 * tv).clamp(0.0, 1.0)
    mask = eval_mask.float()
    error = confidence_pred.sigmoid() - accept
    den = mask.sum() + 1e-6
    assert torch.allclose(terms["confidence_abs_error"], (error.abs() * mask).sum() / den, atol=1e-6)
    assert torch.allclose(terms["confidence_bias"], (error * mask).sum() / den, atol=1e-6)


def test_chunked_probability_distance_matches_reference_and_gradient():
    torch.manual_seed(4)
    draft = torch.randn(2, 2, 3, 7, requires_grad=True)
    target = torch.randn_like(draft)
    chunked = loss_module._compute_l1_dist_per_token(
        draft_logits=draft,
        aligned_target_logits=target,
        chunk_size=2,
    )
    expected = (draft.softmax(dim=-1) - target.softmax(dim=-1)).abs().sum(dim=-1)

    torch.testing.assert_close(chunked, expected)
    chunked.sum().backward()
    chunked_grad = draft.grad.detach().clone()

    draft.grad = None
    expected.sum().backward()
    torch.testing.assert_close(draft.grad, chunked_grad)


def test_anchor_candidates_require_two_consecutive_supervised_tokens():
    """A position is a valid anchor only if it and its first draft target are supervised."""
    loss_mask = torch.tensor([[1, 1, 0, 1, 1, 1]], dtype=torch.uint8)
    valid = build_anchor_candidate_mask(seq_len=6, loss_mask=loss_mask)
    # candidate i valid iff loss_mask[i] and loss_mask[i+1]; over i in [0, seq_len-1).
    expected = torch.tensor([[True, False, False, True, True]])
    assert torch.equal(valid, expected)


def test_sampled_anchors_are_valid_and_counted():
    """Sampled anchors fall on valid candidates and block_keep_mask counts them."""
    torch.manual_seed(7)
    loss_mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.uint8)
    anchors, keep = sample_anchor_positions(seq_len=6, loss_mask=loss_mask, num_anchors=8, device=torch.device("cpu"))
    valid = build_anchor_candidate_mask(seq_len=6, loss_mask=loss_mask)
    num_valid = int(valid.sum().item())
    assert int(keep.sum().item()) == min(num_valid, 8)
    for pos, kept in zip(anchors[0].tolist(), keep[0].tolist()):
        if kept:
            assert valid[0, pos], f"anchor {pos} is not a valid candidate"

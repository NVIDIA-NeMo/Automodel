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

"""Tests for MTP acceptance-length evaluation (compute_mtp_prefix_hits)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.speculative.mtp.core import (
    MTPAcceptanceMetrics,
    compute_mtp_prefix_hits,
    simulated_accept_length,
)

IGNORE_INDEX = -100
VOCAB = 6


def _constant_class_logits(shape: tuple[int, ...], cls: int, vocab: int = VOCAB) -> torch.Tensor:
    """Logits whose argmax is ``cls`` at every position."""
    logits = torch.zeros(*shape, vocab)
    logits[..., cls] = 10.0
    return logits


@pytest.fixture
def labels() -> torch.Tensor:
    # labels[t] is already the next-token target at input position t (standard
    # causal-LM convention), so rolling it left once more derives the depth-1
    # MTP target, matching calculate_mtp_loss's own rolling.
    return torch.tensor([[1, 1, 1]], dtype=torch.long)


def test_draft_matching_verifier_is_accepted_even_against_a_wrong_label(labels):
    """Acceptance tracks the model's own greedy pick, not the dataset label.

    Verifier and draft both predict class 2 while the label is class 1: every
    draft should be accepted (tau == 2), since the label never enters the
    comparison -- only validity masking.
    """
    verifier_logits = _constant_class_logits(labels.shape, cls=2)
    draft_logits = [_constant_class_logits(labels.shape, cls=2)]

    metrics = compute_mtp_prefix_hits(
        model=nn.Identity(),
        mtp_per_depth_logits=draft_logits,
        verifier_logits=verifier_logits,
        labels=labels,
        ignore_index=IGNORE_INDEX,
    )

    torch.testing.assert_close(metrics.valid, torch.tensor([2]))
    torch.testing.assert_close(metrics.prefix_hits, torch.tensor([2]))
    torch.testing.assert_close(simulated_accept_length(metrics.prefix_hits, metrics.valid), torch.tensor(2.0))


def test_draft_matching_the_label_is_rejected_against_a_disagreeing_verifier(labels):
    """The inverse of the above: matching the dataset label is not acceptance.

    Verifier predicts class 2 while the draft predicts class 1 (the label):
    every draft should be rejected (tau == 1), even though the draft is
    "correct" against the dataset.
    """
    verifier_logits = _constant_class_logits(labels.shape, cls=2)
    draft_logits = [_constant_class_logits(labels.shape, cls=1)]

    metrics = compute_mtp_prefix_hits(
        model=nn.Identity(),
        mtp_per_depth_logits=draft_logits,
        verifier_logits=verifier_logits,
        labels=labels,
        ignore_index=IGNORE_INDEX,
    )

    torch.testing.assert_close(metrics.valid, torch.tensor([2]))
    torch.testing.assert_close(metrics.prefix_hits, torch.tensor([0]))
    torch.testing.assert_close(simulated_accept_length(metrics.prefix_hits, metrics.valid), torch.tensor(1.0))


def test_cu_seqlens_masks_cross_document_positions():
    """``cu_seqlens`` (THD layout) must exclude a document's last token from the
    acceptance count, the same packed-sequence boundary ``calculate_mtp_loss``
    enforces -- not just the global trailing position every depth already masks.
    """
    labels = torch.ones(1, 8, dtype=torch.long)
    verifier_logits = _constant_class_logits(labels.shape, cls=2)
    draft_logits = [_constant_class_logits(labels.shape, cls=2)]
    cu_seqlens = torch.tensor([0, 4, 8])  # two 4-token documents

    with_boundary = compute_mtp_prefix_hits(
        model=nn.Identity(),
        mtp_per_depth_logits=draft_logits,
        verifier_logits=verifier_logits,
        labels=labels,
        ignore_index=IGNORE_INDEX,
        cu_seqlens=cu_seqlens,
    )
    # Position 7 is the global trailing position (masked regardless of boundaries);
    # position 3 is the first document's last token, whose depth-1 target rolls
    # into the second document and must also be excluded.
    torch.testing.assert_close(with_boundary.valid, torch.tensor([6]))
    torch.testing.assert_close(with_boundary.prefix_hits, torch.tensor([6]))

    without_boundary = compute_mtp_prefix_hits(
        model=nn.Identity(),
        mtp_per_depth_logits=draft_logits,
        verifier_logits=verifier_logits,
        labels=labels,
        ignore_index=IGNORE_INDEX,
    )
    # Without cu_seqlens only the global trailing position is excluded, proving
    # the boundary argument is what fixes the cross-document contamination.
    torch.testing.assert_close(without_boundary.valid, torch.tensor([7]))


def test_second_depth_wrong_breaks_the_chain():
    """Depth 2 disagreeing with the verifier zeroes the joint prefix, not just depth 2's own count."""
    labels = torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long)
    verifier_logits = _constant_class_logits(labels.shape, cls=2)
    draft_logits = [_constant_class_logits(labels.shape, cls=2), _constant_class_logits(labels.shape, cls=3)]

    metrics = compute_mtp_prefix_hits(
        model=nn.Identity(),
        mtp_per_depth_logits=draft_logits,
        verifier_logits=verifier_logits,
        labels=labels,
        ignore_index=IGNORE_INDEX,
    )

    torch.testing.assert_close(metrics.valid, torch.tensor([4, 3]))
    torch.testing.assert_close(metrics.prefix_hits, torch.tensor([4, 0]))
    torch.testing.assert_close(simulated_accept_length(metrics.prefix_hits, metrics.valid), torch.tensor(2.0))


def test_hidden_states_mode_matches_logits_mode_through_lm_head(labels):
    verifier_logits = _constant_class_logits(labels.shape, cls=2)
    lm_head = nn.Linear(3, VOCAB, bias=False)
    model = nn.Module()
    model.lm_head = lm_head

    hidden = [torch.randn(1, 3, 3)]
    with torch.no_grad():
        logits = [lm_head(h) for h in hidden]

    from_hidden = compute_mtp_prefix_hits(
        model=model, mtp_per_depth_h=hidden, verifier_logits=verifier_logits, labels=labels, ignore_index=IGNORE_INDEX
    )
    from_logits = compute_mtp_prefix_hits(
        model=model,
        mtp_per_depth_logits=logits,
        verifier_logits=verifier_logits,
        labels=labels,
        ignore_index=IGNORE_INDEX,
    )

    torch.testing.assert_close(from_hidden.prefix_hits, from_logits.prefix_hits)
    torch.testing.assert_close(from_hidden.valid, from_logits.valid)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        pytest.param({"mtp_per_depth_h": [torch.randn(1, 3, 3)]}, "lm_head", id="hidden_states_without_lm_head"),
        pytest.param(
            {"mtp_per_depth_h": [torch.randn(1, 3, 3)], "mtp_per_depth_logits": [torch.randn(1, 3, VOCAB)]},
            "exactly one",
            id="both_h_and_logits",
        ),
        pytest.param({}, "exactly one", id="neither_h_nor_logits"),
    ],
)
def test_invalid_inputs_raise(kwargs, match, labels):
    verifier_logits = _constant_class_logits(labels.shape, cls=0)
    with pytest.raises(ValueError, match=match):
        compute_mtp_prefix_hits(
            model=nn.Identity(), verifier_logits=verifier_logits, labels=labels, ignore_index=IGNORE_INDEX, **kwargs
        )


def test_returns_mtp_acceptance_metrics_type(labels):
    verifier_logits = _constant_class_logits(labels.shape, cls=0)
    metrics = compute_mtp_prefix_hits(
        model=nn.Identity(),
        mtp_per_depth_logits=[_constant_class_logits(labels.shape, cls=0)],
        verifier_logits=verifier_logits,
        labels=labels,
        ignore_index=IGNORE_INDEX,
    )
    assert isinstance(metrics, MTPAcceptanceMetrics)

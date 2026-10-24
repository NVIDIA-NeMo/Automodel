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


def _onehot_logits(targets: torch.Tensor, vocab: int) -> torch.Tensor:
    """Logits whose argmax equals ``targets`` wherever it is a valid class id."""
    safe_targets = targets.clamp_min(0)
    return nn.functional.one_hot(safe_targets, num_classes=vocab).float() * 10.0


def _wrong_logits_like(targets: torch.Tensor, vocab: int) -> torch.Tensor:
    """Logits that always argmax to class 0, regardless of ``targets``."""
    logits = torch.zeros(*targets.shape, vocab)
    logits[..., 0] = 10.0
    return logits


@pytest.fixture
def labels() -> torch.Tensor:
    # labels[t] is already the next-token target at input position t (standard
    # causal-LM convention), so rolling it left once more derives the depth-1
    # MTP target, matching calculate_mtp_loss's own rolling.
    return torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)


def _depth_targets(labels: torch.Tensor) -> list[torch.Tensor]:
    # Hand-derived expected per-depth targets from calculate_mtp_loss's roll rule:
    # depth 0: roll left 1, mask last 1 position.
    # depth 1: roll left 1 again (cumulative), mask last 2 positions.
    return [
        torch.tensor([[2, 3, 4, 5, IGNORE_INDEX]]),
        torch.tensor([[3, 4, 5, IGNORE_INDEX, IGNORE_INDEX]]),
    ]


def test_all_depths_correct_gives_full_valid_and_hit_counts(labels):
    targets = _depth_targets(labels)
    logits = [_onehot_logits(t, VOCAB) for t in targets]

    metrics = compute_mtp_prefix_hits(
        model=nn.Identity(),
        mtp_per_depth_logits=logits,
        labels=labels,
        ignore_index=IGNORE_INDEX,
    )

    assert isinstance(metrics, MTPAcceptanceMetrics)
    torch.testing.assert_close(metrics.valid, torch.tensor([4, 3]))
    torch.testing.assert_close(metrics.prefix_hits, torch.tensor([4, 3]))
    torch.testing.assert_close(simulated_accept_length(metrics.prefix_hits, metrics.valid), torch.tensor(3.0))


def test_second_depth_wrong_breaks_the_chain(labels):
    targets = _depth_targets(labels)
    logits = [_onehot_logits(targets[0], VOCAB), _wrong_logits_like(targets[1], VOCAB)]

    metrics = compute_mtp_prefix_hits(
        model=nn.Identity(),
        mtp_per_depth_logits=logits,
        labels=labels,
        ignore_index=IGNORE_INDEX,
    )

    # valid counts are unaffected by correctness, only by the ignore_index mask.
    torch.testing.assert_close(metrics.valid, torch.tensor([4, 3]))
    # depth-1 hits unaffected; depth-2 hits drop to zero and the chain breaks.
    torch.testing.assert_close(metrics.prefix_hits, torch.tensor([4, 0]))
    torch.testing.assert_close(simulated_accept_length(metrics.prefix_hits, metrics.valid), torch.tensor(2.0))


def test_hidden_states_mode_matches_logits_mode_through_lm_head(labels):
    targets = _depth_targets(labels)
    lm_head = nn.Linear(3, VOCAB, bias=False)
    model = nn.Module()
    model.lm_head = lm_head

    hidden = [torch.randn(1, 5, 3) for _ in targets]
    with torch.no_grad():
        logits = [lm_head(h) for h in hidden]

    from_hidden = compute_mtp_prefix_hits(
        model=model,
        mtp_per_depth_h=hidden,
        labels=labels,
        ignore_index=IGNORE_INDEX,
    )
    from_logits = compute_mtp_prefix_hits(
        model=model,
        mtp_per_depth_logits=logits,
        labels=labels,
        ignore_index=IGNORE_INDEX,
    )

    torch.testing.assert_close(from_hidden.prefix_hits, from_logits.prefix_hits)
    torch.testing.assert_close(from_hidden.valid, from_logits.valid)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        pytest.param({"mtp_per_depth_h": [torch.randn(1, 5, 3)]}, "lm_head", id="hidden_states_without_lm_head"),
        pytest.param(
            {"mtp_per_depth_h": [torch.randn(1, 5, 3)], "mtp_per_depth_logits": [torch.randn(1, 5, VOCAB)]},
            "exactly one",
            id="both_h_and_logits",
        ),
        pytest.param({}, "exactly one", id="neither_h_nor_logits"),
        pytest.param(
            {
                "mtp_per_depth_logits": [torch.randn(1, 5, VOCAB), torch.randn(1, 5, VOCAB)],
                "mtp_per_depth_targets": [torch.zeros(1, 5, dtype=torch.long)],
            },
            "Expected 2 mtp_per_depth_targets",
            id="targets_length_mismatch",
        ),
    ],
)
def test_invalid_inputs_raise(kwargs, match, labels):
    with pytest.raises(ValueError, match=match):
        compute_mtp_prefix_hits(model=nn.Identity(), labels=labels, ignore_index=IGNORE_INDEX, **kwargs)


def test_precomputed_targets_bypass_label_rolling():
    precomputed = [
        torch.tensor([[2, 3, IGNORE_INDEX]]),
        torch.tensor([[3, IGNORE_INDEX, IGNORE_INDEX]]),
    ]
    logits = [_onehot_logits(t, VOCAB) for t in precomputed]

    metrics = compute_mtp_prefix_hits(
        model=nn.Identity(),
        mtp_per_depth_logits=logits,
        mtp_per_depth_targets=precomputed,
        labels=torch.zeros(1, 3, dtype=torch.long),  # unused when targets are precomputed
        ignore_index=IGNORE_INDEX,
    )

    torch.testing.assert_close(metrics.valid, torch.tensor([2, 1]))
    torch.testing.assert_close(metrics.prefix_hits, torch.tensor([2, 1]))


def test_precomputed_targets_length_mismatch_raises():
    with pytest.raises(ValueError, match="Expected 2 mtp_per_depth_targets"):
        compute_mtp_prefix_hits(
            model=nn.Identity(),
            mtp_per_depth_logits=[torch.randn(1, 3, VOCAB), torch.randn(1, 3, VOCAB)],
            mtp_per_depth_targets=[torch.zeros(1, 3, dtype=torch.long)],
            labels=torch.zeros(1, 3, dtype=torch.long),
            ignore_index=IGNORE_INDEX,
        )

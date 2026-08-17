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

import pytest
import torch
import torch.nn.functional as F

from tests.functional_tests.checkpoint_robustness.parity_metrics import (
    _compute_parity_metrics,
    _parity_failures,
    _resolve_parity_thresholds,
    _validate_logits,
)


def test_identical_logits_have_zero_divergence_and_unit_cosine():
    logits = torch.tensor([[[2.0, 0.0, -1.0], [0.5, 0.25, -0.5]]])

    metrics = _compute_parity_metrics(logits, logits.clone(), chunk_tokens=1)

    assert metrics.token_count == 2
    assert metrics.vocab_size == 3
    assert metrics.mean_kl == pytest.approx(0.0, abs=1e-8)
    assert metrics.p95_kl == pytest.approx(0.0, abs=1e-8)
    assert metrics.max_kl == pytest.approx(0.0, abs=1e-8)
    assert metrics.cosine_similarity == pytest.approx(1.0)
    assert metrics.mean_absolute_logit_difference == 0.0
    assert metrics.max_absolute_logit_difference == 0.0


def test_full_logit_metrics_match_direct_reference_computation():
    reference = torch.tensor([[[1.0, -0.5, 0.25], [0.0, 2.0, -1.0]]])
    candidate = torch.tensor([[[0.75, -0.25, 0.0], [0.5, 1.25, -0.5]]])
    reference_flat = reference.reshape(-1, 3).float()
    candidate_flat = candidate.reshape(-1, 3).float()
    reference_log_probs = F.log_softmax(reference_flat, dim=-1)
    expected_kl = (reference_log_probs.exp() * (reference_log_probs - F.log_softmax(candidate_flat, dim=-1))).sum(-1)

    metrics = _compute_parity_metrics(reference, candidate, chunk_tokens=1)

    assert metrics.mean_kl == pytest.approx(expected_kl.mean().item(), rel=1e-6, abs=1e-8)
    assert metrics.p95_kl == pytest.approx(torch.quantile(expected_kl, 0.95).item(), rel=1e-6, abs=1e-8)
    assert metrics.max_kl == pytest.approx(expected_kl.max().item(), rel=1e-6, abs=1e-8)
    assert metrics.cosine_similarity == pytest.approx(
        F.cosine_similarity(reference.flatten(), candidate.flatten(), dim=0).item(), rel=1e-6
    )
    absolute_difference = (reference - candidate).abs()
    assert metrics.mean_absolute_logit_difference == pytest.approx(absolute_difference.mean().item())
    assert metrics.max_absolute_logit_difference == pytest.approx(absolute_difference.max().item())


def test_p95_is_stable_against_a_single_token_outlier_while_max_remains_diagnostic():
    reference = torch.zeros(1, 100, 2)
    candidate = reference.clone()
    candidate[0, -1] = torch.tensor([20.0, -20.0])

    metrics = _compute_parity_metrics(reference, candidate)

    assert metrics.mean_kl > 0.0
    assert metrics.p95_kl == pytest.approx(0.0, abs=1e-8)
    assert metrics.max_kl > 1.0


def test_metric_results_do_not_depend_on_chunk_size():
    generator = torch.Generator().manual_seed(1234)
    reference = torch.randn(2, 7, 11, generator=generator)
    candidate = reference + 0.01 * torch.randn(2, 7, 11, generator=generator)

    single_token_chunks = _compute_parity_metrics(reference, candidate, chunk_tokens=1)
    all_token_chunk = _compute_parity_metrics(reference, candidate, chunk_tokens=14)

    assert single_token_chunks.to_dict() == pytest.approx(all_token_chunk.to_dict(), rel=1e-6, abs=1e-8)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_logits_are_rejected(bad_value):
    logits = torch.zeros(1, 2, 3)
    logits[0, 1, 2] = bad_value

    with pytest.raises(ValueError, match="non-finite"):
        _validate_logits(logits)
    with pytest.raises(ValueError, match="non-finite"):
        _compute_parity_metrics(torch.zeros_like(logits), logits)


def test_named_profiles_are_ordered_and_gate_mean_and_p95():
    strict = _resolve_parity_thresholds("strict", "cross_framework")
    standard = _resolve_parity_thresholds("standard", "cross_framework")
    relaxed = _resolve_parity_thresholds("relaxed", "cross_framework")
    reference = torch.zeros(1, 100, 2)
    candidate = reference.clone()
    candidate[:, :10, 0] = 0.25
    metrics = _compute_parity_metrics(reference, candidate)

    assert strict.mean_kl < standard.mean_kl < relaxed.mean_kl
    assert strict.p95_kl < standard.p95_kl < relaxed.p95_kl
    failures = _parity_failures(metrics, strict)
    assert any("mean KL" in failure for failure in failures)
    assert any("p95 KL" in failure for failure in failures)


def test_legacy_numeric_overrides_preserve_max_kl_semantics():
    reference = torch.zeros(1, 100, 2)
    candidate = reference.clone()
    candidate[0, -1] = torch.tensor([20.0, -20.0])
    metrics = _compute_parity_metrics(reference, candidate)
    relaxed = _resolve_parity_thresholds("relaxed", "cross_framework")

    failures = _parity_failures(metrics, relaxed, legacy_max_kl_threshold=1.0)

    assert len(failures) == 1
    assert failures[0].startswith("max KL")


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="Unknown parity tolerance profile"):
        _resolve_parity_thresholds("custom", "cross_framework")


@pytest.mark.parametrize("bad_threshold", [float("nan"), float("inf"), -1.0])
def test_invalid_legacy_kl_threshold_is_rejected(bad_threshold):
    logits = torch.zeros(1, 2, 3)
    metrics = _compute_parity_metrics(logits, logits)
    thresholds = _resolve_parity_thresholds("standard", "same_implementation")

    with pytest.raises(ValueError, match="finite and non-negative"):
        _parity_failures(metrics, thresholds, legacy_max_kl_threshold=bad_threshold)

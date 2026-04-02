# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for dLLM loss functions (MDLMCrossEntropyLoss, HybridDiffusionLLMLoss)."""

import pytest
import torch

import torch.nn.functional as F

from nemo_automodel.components.loss.dllm_loss import (
    DLLMLossOutput,
    MDLMCrossEntropyLoss,
    _compute_per_token_nll,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

B, L, V = 2, 8, 32  # batch, seq_len, vocab


@pytest.fixture
def dummy_inputs():
    """Create minimal inputs shared across tests."""
    torch.manual_seed(42)
    logits = torch.randn(B, L, V)
    target_ids = torch.randint(0, V, (B, L))
    # Supervised positions: first 6 of 8
    loss_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]] * B)
    # Corrupted positions: subset of supervised
    noise_mask = torch.tensor([[0, 1, 0, 1, 1, 0, 0, 0]] * B).bool()
    p_mask = torch.full((B, L), 0.5)
    return logits, target_ids, noise_mask, p_mask, loss_mask


# ---------------------------------------------------------------------------
# MDLMCrossEntropyLoss
# ---------------------------------------------------------------------------


class TestMDLMCrossEntropyLoss:
    def test_returns_dllm_loss_output(self, dummy_inputs):
        logits, target_ids, noise_mask, p_mask, loss_mask = dummy_inputs
        loss_fn = MDLMCrossEntropyLoss()
        result = loss_fn(logits, target_ids, noise_mask, p_mask, loss_mask)
        assert isinstance(result, DLLMLossOutput)

    def test_total_loss_equals_dllm_loss(self, dummy_inputs):
        """For MDLM, total_loss and dllm_loss should be equal (no AR component)."""
        logits, target_ids, noise_mask, p_mask, loss_mask = dummy_inputs
        loss_fn = MDLMCrossEntropyLoss()
        result = loss_fn(logits, target_ids, noise_mask, p_mask, loss_mask)
        assert torch.allclose(result.total_loss, result.dllm_loss, atol=1e-6)

    def test_loss_is_positive(self, dummy_inputs):
        logits, target_ids, noise_mask, p_mask, loss_mask = dummy_inputs
        loss_fn = MDLMCrossEntropyLoss()
        result = loss_fn(logits, target_ids, noise_mask, p_mask, loss_mask)
        assert result.total_loss.item() > 0

    def test_zero_loss_when_no_noise(self, dummy_inputs):
        """If nothing is corrupted, loss should be zero."""
        logits, target_ids, _, p_mask, loss_mask = dummy_inputs
        noise_mask = torch.zeros(B, L, dtype=torch.bool)
        loss_fn = MDLMCrossEntropyLoss()
        result = loss_fn(logits, target_ids, noise_mask, p_mask, loss_mask)
        assert result.total_loss.item() == 0.0

    def test_normalization_by_num_diffusion_tokens(self, dummy_inputs):
        logits, target_ids, noise_mask, p_mask, loss_mask = dummy_inputs
        loss_fn = MDLMCrossEntropyLoss()
        result_unnorm = loss_fn(logits, target_ids, noise_mask, p_mask, loss_mask)
        result_norm = loss_fn(
            logits, target_ids, noise_mask, p_mask, loss_mask, num_diffusion_tokens=10
        )
        # Normalized loss should be unnormalized / 10
        assert torch.allclose(result_norm.total_loss, result_unnorm.total_loss / 10, atol=1e-5)


# ---------------------------------------------------------------------------
# _compute_per_token_nll helper
# ---------------------------------------------------------------------------


class TestComputePerTokenNLL:
    def test_plain_tensor_matches_ce(self):
        """Plain tensor path should match F.cross_entropy(reduction='none')."""
        torch.manual_seed(42)
        logits = torch.randn(2, 8, 32)
        targets = torch.randint(0, 32, (2, 8))
        nll = _compute_per_token_nll(logits, targets)
        ref = F.cross_entropy(
            logits.reshape(-1, 32), targets.reshape(-1), reduction="none"
        ).reshape(2, 8)
        assert torch.allclose(nll, ref)

    def test_output_shape(self):
        """Output shape should be [B, L]."""
        logits = torch.randn(4, 16, 64)
        targets = torch.randint(0, 64, (4, 16))
        nll = _compute_per_token_nll(logits, targets)
        assert nll.shape == (4, 16)

    def test_positive_values(self):
        """NLL should be non-negative."""
        logits = torch.randn(2, 8, 32)
        targets = torch.randint(0, 32, (2, 8))
        nll = _compute_per_token_nll(logits, targets)
        assert (nll >= 0).all()

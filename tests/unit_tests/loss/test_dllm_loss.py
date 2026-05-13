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

"""Tests for dLLM loss functions (MDLMCrossEntropyLoss, DFlashDecayLoss)."""

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.loss.dllm_loss import (
    DFlashDecayLoss,
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
        result_norm = loss_fn(logits, target_ids, noise_mask, p_mask, loss_mask, num_diffusion_tokens=10)
        # Normalized loss should be unnormalized / 10
        assert torch.allclose(result_norm.total_loss, result_unnorm.total_loss / 10, atol=1e-5)

    def test_numerical_correctness_against_reference(self):
        """Verify loss matches hand-computed reference: sum(CE * mask * 1/p_mask) / N.

        Reference formula (from dllm/core/trainers/mdlm.py):
            loss = sum_{i in masked} CE_i * (1/t) / sum(maskable)
        where t = p_mask (the corruption probability).
        """
        torch.manual_seed(123)
        B_test, L_test, V_test = 2, 4, 8
        logits = torch.randn(B_test, L_test, V_test)
        target_ids = torch.randint(0, V_test, (B_test, L_test))
        loss_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        noise_mask = torch.tensor([[True, False, True, False], [False, True, False, False]])
        p_mask = torch.tensor([[0.4, 0.4, 0.4, 0.4], [0.6, 0.6, 0.6, 0.6]])

        # Hand-compute reference
        ce = F.cross_entropy(logits.reshape(-1, V_test), target_ids.reshape(-1), reduction="none").reshape(
            B_test, L_test
        )
        mask = noise_mask & loss_mask.bool()
        weighted = ce * mask.float() * (1.0 / p_mask)
        num_supervised = loss_mask.sum().item()
        expected = weighted.sum() / num_supervised

        loss_fn = MDLMCrossEntropyLoss()
        result = loss_fn(logits, target_ids, noise_mask, p_mask, loss_mask, num_diffusion_tokens=int(num_supervised))
        assert torch.allclose(result.total_loss, expected, atol=1e-5)

    def test_loss_only_at_corrupted_supervised_positions(self):
        """Loss should be zero for positions that are corrupted but NOT supervised,
        and for positions that are supervised but NOT corrupted."""
        torch.manual_seed(99)
        logits = torch.randn(1, 6, 16)
        target_ids = torch.randint(0, 16, (1, 6))
        # Only position 2 is both corrupted AND supervised
        loss_mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
        noise_mask = torch.tensor([[False, False, True, True, False, False]])
        p_mask = torch.full((1, 6), 0.5)

        loss_fn = MDLMCrossEntropyLoss()
        result = loss_fn(logits, target_ids, noise_mask, p_mask, loss_mask)

        # Compute expected: only position 2 contributes
        ce = F.cross_entropy(logits.reshape(-1, 16), target_ids.reshape(-1), reduction="none").reshape(1, 6)
        expected = ce[0, 2] * (1.0 / 0.5)
        assert torch.allclose(result.total_loss, expected, atol=1e-5)


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
        ref = F.cross_entropy(logits.reshape(-1, 32), targets.reshape(-1), reduction="none").reshape(2, 8)
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


# ---------------------------------------------------------------------------
# DFlashDecayLoss
# ---------------------------------------------------------------------------

B_D, T_D, V_D = 2, 15, 32  # batch, block_size-1 (15 predicted per block_size=16), vocab


@pytest.fixture
def dflash_inputs():
    torch.manual_seed(7)
    logits = torch.randn(B_D, T_D, V_D)
    target_ids = torch.randint(0, V_D, (B_D, T_D))
    block_mask = torch.ones(B_D, T_D)
    return logits, target_ids, block_mask


class TestDFlashDecayLoss:
    def test_returns_dllm_loss_output(self, dflash_inputs):
        logits, target_ids, block_mask = dflash_inputs
        loss_fn = DFlashDecayLoss(loss_gamma=7.0)
        result = loss_fn(logits, target_ids, block_mask)
        assert isinstance(result, DLLMLossOutput)

    def test_loss_is_positive(self, dflash_inputs):
        logits, target_ids, block_mask = dflash_inputs
        loss_fn = DFlashDecayLoss(loss_gamma=7.0)
        result = loss_fn(logits, target_ids, block_mask)
        assert result.total_loss.item() > 0

    def test_zero_loss_when_mask_all_zero(self, dflash_inputs):
        logits, target_ids, _ = dflash_inputs
        block_mask = torch.zeros(B_D, T_D)
        loss_fn = DFlashDecayLoss(loss_gamma=7.0)
        result = loss_fn(logits, target_ids, block_mask)
        assert result.total_loss.item() == 0.0

    def test_normalization_by_num_tokens(self, dflash_inputs):
        logits, target_ids, block_mask = dflash_inputs
        loss_fn = DFlashDecayLoss(loss_gamma=7.0)
        result_unnorm = loss_fn(logits, target_ids, block_mask)
        result_norm = loss_fn(logits, target_ids, block_mask, num_tokens=10)
        assert torch.allclose(result_norm.total_loss, result_unnorm.total_loss / 10, atol=1e-5)

    def test_decay_weights_decrease_monotonically(self):
        """First predicted position has higher weight than the last."""
        torch.manual_seed(0)
        B, T, V = 1, 8, 16
        logits = torch.zeros(B, T, V)  # uniform CE so only weights differ
        target_ids = torch.zeros(B, T, dtype=torch.long)
        loss_fn = DFlashDecayLoss(loss_gamma=2.0)

        mask_first = torch.zeros(B, T)
        mask_first[:, 0] = 1.0
        loss_first = loss_fn(logits, target_ids, mask_first).total_loss

        mask_last = torch.zeros(B, T)
        mask_last[:, -1] = 1.0
        loss_last = loss_fn(logits, target_ids, mask_last).total_loss

        assert loss_first > loss_last

    def test_block_size_resets_decay_per_block(self):
        """With block_size, each block starts fresh at weight=1; without it weights
        decay monotonically across the full concatenated sequence."""
        torch.manual_seed(1)
        block_size, n, gamma = 4, 2, 2.0
        T = n * (block_size - 1)
        B, V = 1, 8
        logits = torch.randn(B, T, V)
        target_ids = torch.randint(0, V, (B, T))
        block_mask = torch.ones(B, T)
        loss_fn = DFlashDecayLoss(loss_gamma=gamma)

        result_reset = loss_fn(logits, target_ids, block_mask, block_size=block_size)
        result_mono = loss_fn(logits, target_ids, block_mask)
        assert not torch.allclose(result_reset.total_loss, result_mono.total_loss, atol=1e-4)

        T_per = block_size - 1
        w_single = torch.exp(-torch.arange(T_per, dtype=torch.float) / gamma)
        w_mono = torch.exp(-torch.arange(T, dtype=torch.float) / gamma)
        assert torch.allclose(w_single.repeat(n)[:T_per], w_mono[:T_per])
        assert w_single.repeat(n)[T_per] > w_mono[T_per]  # second block resets to 1

    def test_multi_block_loss_is_scalar(self):
        block_size, n_blocks = 8, 3
        T = n_blocks * (block_size - 1)
        B, V = 2, 64
        logits = torch.randn(B, T, V)
        target_ids = torch.randint(0, V, (B, T))
        block_mask = torch.ones(B, T)
        result = DFlashDecayLoss(loss_gamma=4.0)(logits, target_ids, block_mask, block_size=block_size)
        assert result.total_loss.ndim == 0

    def test_gamma_controls_decay_rate(self):
        """Larger γ → slower decay → different total loss than small γ."""
        torch.manual_seed(2)
        T, V = 10, 16
        logits = torch.randn(1, T, V)
        target_ids = torch.randint(0, V, (1, T))
        block_mask = torch.ones(1, T)

        loss_fast = DFlashDecayLoss(loss_gamma=1.0)(logits, target_ids, block_mask).total_loss
        loss_slow = DFlashDecayLoss(loss_gamma=100.0)(logits, target_ids, block_mask).total_loss

        assert loss_fast.item() > 0
        assert loss_slow.item() > 0
        assert not torch.allclose(loss_fast, loss_slow, atol=1e-3)

    def test_paper_default_gammas(self):
        """Verify loss runs without error for all three paper-default block sizes."""
        for block_size, gamma in [(16, 7.0), (10, 5.0), (8, 4.0)]:
            T = block_size - 1
            logits = torch.randn(1, T, 32)
            target_ids = torch.randint(0, 32, (1, T))
            block_mask = torch.ones(1, T)
            loss_fn = DFlashDecayLoss(loss_gamma=gamma)
            result = loss_fn(logits, target_ids, block_mask, block_size=block_size)
            assert result.total_loss.item() > 0, f"zero loss for block_size={block_size}"

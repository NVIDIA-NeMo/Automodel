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
    BlockDiffusionCrossEntropyLoss,
    DFlashDecayLoss,
    DLLMLossOutput,
    HybridDiffusionLLMLoss,
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
# HybridDiffusionLLMLoss
# ---------------------------------------------------------------------------


class TestHybridDiffusionLLMLoss:
    def test_diffusion_only_when_no_causal_logits(self, dummy_inputs):
        """Without causal logits, total_loss == alpha * dllm_loss (no AR term)."""
        logits, target_ids, noise_mask, p_mask, loss_mask = dummy_inputs
        loss_fn = HybridDiffusionLLMLoss(alpha=0.3)
        result = loss_fn(logits, target_ids, noise_mask, p_mask, loss_mask)
        assert torch.allclose(result.total_loss, result.dllm_loss, atol=1e-6)

    def test_ar_component_increases_total_loss(self, dummy_inputs):
        """When causal logits are present, total_loss > alpha * dllm_loss."""
        logits, target_ids, noise_mask, p_mask, loss_mask = dummy_inputs
        causal_logits = torch.randn(B, L, V)
        combined_logits = torch.cat([logits, causal_logits], dim=1)  # [B, 2L, V]
        loss_fn = HybridDiffusionLLMLoss(alpha=0.3)
        result = loss_fn(
            combined_logits,
            target_ids,
            noise_mask,
            p_mask,
            loss_mask,
            loss_mask_ar=loss_mask,
        )
        assert result.total_loss.item() > result.dllm_loss.item()

    def test_alpha_scales_diffusion_loss(self, dummy_inputs):
        logits, target_ids, noise_mask, p_mask, loss_mask = dummy_inputs
        result_a03 = HybridDiffusionLLMLoss(alpha=0.3)(logits, target_ids, noise_mask, p_mask, loss_mask)
        result_a10 = HybridDiffusionLLMLoss(alpha=1.0)(logits, target_ids, noise_mask, p_mask, loss_mask)
        ratio = result_a03.total_loss.item() / result_a10.total_loss.item()
        assert abs(ratio - 0.3) < 1e-5

    def test_zero_dllm_loss_when_no_noise(self, dummy_inputs):
        logits, target_ids, _, p_mask, loss_mask = dummy_inputs
        noise_mask = torch.zeros(B, L, dtype=torch.bool)
        loss_fn = HybridDiffusionLLMLoss(alpha=0.3)
        result = loss_fn(logits, target_ids, noise_mask, p_mask, loss_mask)
        assert result.dllm_loss.item() == 0.0

    def test_normalization_by_num_diffusion_tokens(self, dummy_inputs):
        logits, target_ids, noise_mask, p_mask, loss_mask = dummy_inputs
        loss_fn = HybridDiffusionLLMLoss(alpha=1.0)
        result_unnorm = loss_fn(logits, target_ids, noise_mask, p_mask, loss_mask)
        result_norm = loss_fn(logits, target_ids, noise_mask, p_mask, loss_mask, num_diffusion_tokens=10)
        assert torch.allclose(result_norm.total_loss, result_unnorm.total_loss / 10, atol=1e-5)

    def test_ar_normalization(self, dummy_inputs):
        """AR loss should be normalized by num_ar_tokens."""
        logits, target_ids, noise_mask, p_mask, loss_mask = dummy_inputs
        causal_logits = torch.randn(B, L, V)
        combined_logits = torch.cat([logits, causal_logits], dim=1)
        loss_fn = HybridDiffusionLLMLoss(alpha=0.3)
        result_unnorm = loss_fn(
            combined_logits,
            target_ids,
            noise_mask,
            p_mask,
            loss_mask,
            loss_mask_ar=loss_mask,
        )
        result_norm = loss_fn(
            combined_logits,
            target_ids,
            noise_mask,
            p_mask,
            loss_mask,
            loss_mask_ar=loss_mask,
            num_diffusion_tokens=10,
            num_ar_tokens=10,
        )
        assert torch.allclose(result_norm.total_loss, result_unnorm.total_loss / 10, atol=1e-5)

    def test_separate_causal_logits_path_matches_concat(self, dummy_inputs):
        """Passing causal_logits separately should produce the same result as the concat layout."""
        logits, target_ids, noise_mask, p_mask, loss_mask = dummy_inputs
        causal_logits = torch.randn(B, L, V)
        combined_logits = torch.cat([logits, causal_logits], dim=1)
        loss_fn = HybridDiffusionLLMLoss(alpha=0.3)
        result_concat = loss_fn(
            combined_logits,
            target_ids,
            noise_mask,
            p_mask,
            loss_mask,
            loss_mask_ar=loss_mask,
        )
        result_separate = loss_fn(
            logits,
            target_ids,
            noise_mask,
            p_mask,
            loss_mask,
            loss_mask_ar=loss_mask,
            causal_logits=causal_logits,
        )
        assert torch.allclose(result_concat.total_loss, result_separate.total_loss, atol=1e-5)


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


# ---------------------------------------------------------------------------
# DFlashDecayLoss
# ---------------------------------------------------------------------------

B_D, N_D, K_D, V_D = 2, 5, 3, 32  # batch, blocks, block_size-1 (predicted per block), vocab


@pytest.fixture
def dflash_inputs():
    torch.manual_seed(7)
    logits = torch.randn(B_D, N_D, K_D, V_D)
    target_ids = torch.randint(0, V_D, (B_D, N_D, K_D))
    block_mask = torch.ones(B_D, N_D, K_D)
    return logits, target_ids, block_mask


class TestDFlashDecayLoss:
    def test_zero_loss_when_mask_all_zero(self, dflash_inputs):
        logits, target_ids, _ = dflash_inputs
        block_mask = torch.zeros(B_D, N_D, K_D)
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
        """Within a block, the first predicted position weighs more than the last."""
        torch.manual_seed(0)
        B, N, K, V = 1, 1, 8, 16
        logits = torch.zeros(B, N, K, V)  # uniform CE so only weights differ
        target_ids = torch.zeros(B, N, K, dtype=torch.long)
        loss_fn = DFlashDecayLoss(loss_gamma=2.0)

        mask_first = torch.zeros(B, N, K)
        mask_first[:, :, 0] = 1.0
        loss_first = loss_fn(logits, target_ids, mask_first).total_loss

        mask_last = torch.zeros(B, N, K)
        mask_last[:, :, -1] = 1.0
        loss_last = loss_fn(logits, target_ids, mask_last).total_loss

        assert loss_first > loss_last

    def test_decay_resets_per_block(self):
        """The [B, n, k] contract makes the reset structural: offset 0 of every
        block gets weight 1, and the same decay curve repeats in each block."""
        B, N, K, V, gamma = 1, 2, 3, 8, 2.0
        logits = torch.zeros(B, N, K, V)  # uniform CE -> only weights differ
        target_ids = torch.zeros(B, N, K, dtype=torch.long)
        loss_fn = DFlashDecayLoss(loss_gamma=gamma)

        def masked_loss(block_idx, offset):
            m = torch.zeros(B, N, K)
            m[0, block_idx, offset] = 1.0
            return loss_fn(logits, target_ids, m).total_loss

        # Offset 0 resets to weight 1 in every block -> identical loss across blocks.
        assert torch.isclose(masked_loss(0, 0), masked_loss(1, 0))
        # The decay curve is identical in each block (same offset -> same weight).
        assert torch.isclose(masked_loss(0, K - 1), masked_loss(1, K - 1))
        # Within a block, weight decays with offset.
        assert masked_loss(1, 0) > masked_loss(1, K - 1)

    def test_gamma_controls_decay_rate(self):
        """Larger gamma -> slower decay -> different total loss than small gamma."""
        torch.manual_seed(2)
        N, K, V = 1, 10, 16
        logits = torch.randn(1, N, K, V)
        target_ids = torch.randint(0, V, (1, N, K))
        block_mask = torch.ones(1, N, K)

        loss_fast = DFlashDecayLoss(loss_gamma=1.0)(logits, target_ids, block_mask).total_loss
        loss_slow = DFlashDecayLoss(loss_gamma=100.0)(logits, target_ids, block_mask).total_loss

        assert not torch.allclose(loss_fast, loss_slow, atol=1e-3)

    def test_invalid_loss_type_raises(self):
        with pytest.raises(ValueError, match="loss_type must be"):
            DFlashDecayLoss(loss_type="bogus")

    def test_invalid_dpace_alpha_raises(self):
        with pytest.raises(ValueError, match="dpace_alpha must be"):
            DFlashDecayLoss(loss_type="dpace", dpace_alpha=1.5)

    @pytest.mark.parametrize(
        "loss_type",
        ["dpace", "dpace-cumulative-confidence-only", "dpace-continuation-value-only"],
    )
    def test_dpace_matches_reference_with_block_reset(self, loss_type):
        torch.manual_seed(13)
        bsz, n, k, vocab = 2, 3, 4, 17
        logits = torch.randn(bsz, n, k, vocab)
        target_ids = torch.randint(0, vocab, (bsz, n, k))
        mask = (torch.rand(bsz, n, k) > 0.25).float()
        alpha = 0.35

        token_nll = F.cross_entropy(
            logits.reshape(-1, vocab),
            target_ids.reshape(-1),
            reduction="none",
        ).view(bsz, n, k)
        prob = torch.exp(-token_nll)
        smooth = torch.where(mask > 0, (1.0 - alpha) * prob + alpha, torch.ones_like(prob))
        prefix = torch.cumprod(smooth, dim=-1)  # resets per block: cumprod over k
        suffix = torch.flip(torch.cumsum(torch.flip(prefix * mask, dims=[-1]), dim=-1), dims=[-1])
        if loss_type == "dpace-cumulative-confidence-only":
            ref_weight = prefix
        elif loss_type == "dpace-continuation-value-only":
            ref_weight = suffix / prefix.clamp_min(torch.finfo(prefix.dtype).tiny)
        else:
            ref_weight = suffix
        # normalize="mean" => the reference's per-sequence normalization (divide by
        # the batch size, independent of the sampled anchor count). Dividing by the
        # D-PACE weight sum instead would cancel the weight magnitudes the objective
        # encodes (and bias the DP gradient average).
        expected = (token_nll * mask * ref_weight).sum() / float(bsz)

        out = DFlashDecayLoss(loss_type=loss_type, dpace_alpha=alpha, normalize="mean")(logits, target_ids, mask)

        assert torch.isclose(expected, out.total_loss, atol=1e-6)
        assert out.loss_denominator.item() == float(bsz)

    def test_dpace_mean_divides_by_batch_size_not_weight_sum(self):
        """``normalize="mean"`` normalizes D-PACE per sequence, as the reference does.

        The reference divides the D-PACE weighted sum by the batch size, independent
        of the sampled anchor count. The D-PACE weights carry the objective's signal,
        so a weight-sum denominator would cancel it; it is also data-dependent, hence
        different on every DP rank, which would bias the gradient average (that stays
        exact only when every rank divides by the same constant).
        """
        torch.manual_seed(21)
        bsz, n, k, vocab = 2, 3, 4, 17
        logits = torch.randn(bsz, n, k, vocab)
        target_ids = torch.randint(0, vocab, (bsz, n, k))
        mask = (torch.rand(bsz, n, k) > 0.25).float()
        alpha = 0.35

        token_nll = F.cross_entropy(logits.reshape(-1, vocab), target_ids.reshape(-1), reduction="none").view(
            bsz, n, k
        )
        prob = torch.exp(-token_nll)
        smooth = torch.where(mask > 0, (1.0 - alpha) * prob + alpha, torch.ones_like(prob))
        prefix = torch.cumprod(smooth, dim=-1)
        weight = torch.flip(torch.cumsum(torch.flip(prefix * mask, dims=[-1]), dim=-1), dims=[-1])

        weighted_sum = (token_nll * mask * weight).sum()
        batch_loss = weighted_sum / float(bsz)
        weight_sum_loss = weighted_sum / ((mask * weight).sum() + 1e-6)

        out = DFlashDecayLoss(loss_type="dpace", dpace_alpha=alpha, normalize="mean")(logits, target_ids, mask)

        assert torch.isclose(out.total_loss, batch_loss, atol=1e-6)
        assert not torch.isclose(out.total_loss, weight_sum_loss, atol=1e-6)
        assert out.loss_denominator.item() == float(bsz)

    def test_dpace_honors_num_tokens_not_batch_size(self):
        """D-PACE must normalize by the global ``num_tokens`` (the default
        ``normalize="tokens"``), like dflash — not the local batch size — so
        ``loss_type`` stays orthogonal to the normalization denominator."""
        torch.manual_seed(7)
        bsz, n, k, vocab = 2, 3, 4, 17
        logits = torch.randn(bsz, n, k, vocab)
        target_ids = torch.randint(0, vocab, (bsz, n, k))
        block_mask = torch.ones(bsz, n, k)
        loss_fn = DFlashDecayLoss(loss_type="dpace", dpace_alpha=0.5)  # normalize="tokens"

        num_tokens = 37
        scaled = loss_fn(logits, target_ids, block_mask, num_tokens=num_tokens)
        raw = loss_fn(logits, target_ids, block_mask, num_tokens=None).total_loss
        # Dividing by num_tokens (not bsz=2) scales the summed loss by 1/num_tokens.
        assert torch.isclose(scaled.total_loss * num_tokens, raw, atol=1e-5)
        assert scaled.loss_denominator.item() == float(num_tokens)

    def test_dpace_alpha_changes_loss(self, dflash_inputs):
        logits, target_ids, block_mask = dflash_inputs

        low_alpha = DFlashDecayLoss(loss_type="dpace", dpace_alpha=0.1)(logits, target_ids, block_mask).total_loss
        high_alpha = DFlashDecayLoss(loss_type="dpace", dpace_alpha=0.9)(logits, target_ids, block_mask).total_loss

        assert not torch.allclose(low_alpha, high_alpha, atol=1e-4)

    def test_loss_denominator_reports_dflash_denominators(self, dflash_inputs):
        """loss_denominator is the exact scalar the loss divided by: num_tokens in
        the ``"tokens"`` mode, the effective decay-weight sum in dflash ``"mean"``."""
        logits, target_ids, block_mask = dflash_inputs
        tokens = DFlashDecayLoss(loss_gamma=7.0)(logits, target_ids, block_mask, num_tokens=13)
        assert tokens.loss_denominator.item() == 13.0

        mean = DFlashDecayLoss(loss_gamma=7.0, normalize="mean")(logits, target_ids, block_mask)
        expected_denom = (torch.exp(-torch.arange(K_D, dtype=torch.float) / 7.0).view(1, 1, K_D) * block_mask).sum()
        assert torch.isclose(mean.loss_denominator, expected_denom + 1e-6, atol=1e-4)


class TestDFlashDraftAccuracy:
    """Per-position draft top-1 accuracy (correct, count) sums.

    The loss returns per-rank raw (correct, count) sums per block offset;
    the recipe SUM-allreduces both and divides post-reduction, so the
    reduction works for arbitrary per-rank token distributions without
    smuggling a per-rank denominator into the numerator.
    """

    def test_returns_per_offset_counts(self, dflash_inputs):
        """The [B, n, k] contract always yields per-offset counts of shape [k]."""
        logits, target_ids, block_mask = dflash_inputs
        result = DFlashDecayLoss(loss_gamma=7.0)(logits, target_ids, block_mask)
        assert result.draft_correct_per_pos.shape == (K_D,)
        assert result.draft_count_per_pos.shape == (K_D,)

    def test_perfect_predictions_give_full_counts(self):
        """argmax == target everywhere -> per-pos correct equals per-pos count."""
        B, N, K, V = 2, 3, 4, 8
        target_ids = torch.randint(0, V, (B, N, K))
        logits = torch.full((B, N, K, V), -10.0)
        logits.scatter_(3, target_ids.unsqueeze(-1), 10.0)  # peak at the target
        block_mask = torch.ones(B, N, K)
        result = DFlashDecayLoss(loss_gamma=7.0)(logits, target_ids, block_mask)
        assert result.draft_correct_per_pos.shape == (K,)
        assert torch.equal(result.draft_correct_per_pos, result.draft_count_per_pos)
        # Each of the K offsets has B * N valid positions.
        assert torch.all(result.draft_count_per_pos == B * N)

    def test_counts_exclude_masked_positions(self):
        """Positions with block_mask=0 must not contribute to correct OR count."""
        B, N, K, V = 1, 1, 4, 8
        target_ids = torch.zeros(B, N, K, dtype=torch.long)
        logits = torch.full((B, N, K, V), -10.0)
        logits[..., 0] = 10.0  # always predicts class 0 == target
        logits[0, 0, 3] = 0.0
        logits[0, 0, 3, 1] = 10.0  # last offset predicts wrong
        block_mask = torch.tensor([[[1.0, 1.0, 1.0, 0.0]]])
        result = DFlashDecayLoss(loss_gamma=7.0)(logits, target_ids, block_mask)
        # offsets 0,1,2 are correct + counted; offset 3 is masked -> zero count, zero correct
        assert result.draft_correct_per_pos.tolist() == [1.0, 1.0, 1.0, 0.0]
        assert result.draft_count_per_pos.tolist() == [1.0, 1.0, 1.0, 0.0]

    def test_draft_acc_per_pos_sums_over_batch_and_blocks(self):
        """(correct, count) sum over (batch, blocks) to per-offset [k] vectors."""
        correct = torch.tensor([[[True, False], [True, True]]])  # [B=1, n=2, k=2]
        block_mask = torch.ones(1, 2, 2)
        correct_per_pos, count_per_pos = DFlashDecayLoss._draft_acc_per_pos(correct, block_mask)
        assert correct_per_pos.tolist() == [2.0, 1.0]  # offset 0: both blocks; offset 1: one block
        assert count_per_pos.tolist() == [2.0, 2.0]

    @pytest.mark.parametrize(
        "loss_type",
        ["dflash", "dpace", "dpace-cumulative-confidence-only", "dpace-continuation-value-only"],
    )
    def test_fused_matches_nonfused(self, loss_type):
        """forward_fused and forward must agree on loss and per-position sums."""
        torch.manual_seed(3)
        B, N, K, D, V = 2, 2, 3, 16, 32
        hidden = torch.randn(B, N, K, D)
        weight = torch.randn(V, D)
        bias = torch.randn(V)
        target_ids = torch.randint(0, V, (B, N, K))
        block_mask = torch.ones(B, N, K)
        loss_fn = DFlashDecayLoss(
            loss_gamma=7.0,
            use_fused_linear_ce=True,
            chunk_size=4,
            loss_type=loss_type,
        )

        logits = torch.nn.functional.linear(hidden, weight, bias)
        ref = loss_fn(logits, target_ids, block_mask, num_tokens=B * N * K)
        fused = loss_fn.forward_fused(
            hidden,
            weight,
            target_ids,
            block_mask,
            num_tokens=B * N * K,
            lm_head_bias=bias,
        )

        assert torch.allclose(ref.total_loss, fused.total_loss, atol=1e-4)
        assert torch.equal(ref.draft_correct_per_pos, fused.draft_correct_per_pos)
        assert torch.equal(ref.draft_count_per_pos, fused.draft_count_per_pos)

    def test_paper_default_first_offset_weight_is_one(self):
        """The first predicted position of a block must have decay weight 1.0 for the
        paper's (block_size, gamma) defaults. This locks Eq. 4 and the published
        triples (16/7, 10/5, 8/4) — the per-block [k] curve is applied to every
        block by construction, so if anyone retunes _decay_weights and shifts the
        start point, every block's first supervision gets the wrong weight."""
        for block_size, gamma in [(16, 7.0), (10, 5.0), (8, 4.0)]:
            loss_fn = DFlashDecayLoss(loss_gamma=gamma)
            k = block_size - 1
            w = loss_fn._decay_weights(k, torch.device("cpu"), torch.float32)
            assert w.shape == (k,), f"block_size={block_size}: weights shape mismatch"
            assert torch.isclose(w[0], torch.tensor(1.0)), (
                f"block_size={block_size}: first weight {w[0].item()} != 1.0"
            )
            assert w[0] > w[-1], f"block_size={block_size}: weights do not decay within block"

    def test_recipe_per_pos_metrics_dict_construction(self):
        """Lock the recipe-side contract: given the loss's per-rank
        ``(draft_correct_per_pos, draft_count_per_pos)`` tensors and the
        post-reduction divide it performs, the metrics dict must contain
        ``draft_acc`` plus one ``draft_acc_k{k}`` key per offset with the
        correct value. Mirrors train_ft.py:_run_train_optim_step verbatim
        so it catches drift in the recipe's reduction shape."""
        B, N, bs, V = 2, 2, 5, 8
        K = bs - 1
        torch.manual_seed(11)
        target_ids = torch.randint(0, V, (B, N, K))
        logits = torch.randn(B, N, K, V)
        block_mask = torch.ones(B, N, K)
        loss_fn = DFlashDecayLoss(loss_gamma=7.0)
        result = loss_fn(logits, target_ids, block_mask)

        # Simulate the recipe's post-reduction divide + key construction.
        correct_per_pos = result.draft_correct_per_pos
        count_per_pos = result.draft_count_per_pos
        total_correct = correct_per_pos.sum().item()
        total_count = count_per_pos.sum().item()
        draft_acc = total_correct / total_count
        draft_acc_per_pos = (correct_per_pos / count_per_pos.clamp_min(1.0)).tolist()

        metrics = {"loss": 0.0, "draft_acc": draft_acc}
        for k, v in enumerate(draft_acc_per_pos, start=1):
            metrics[f"draft_acc_k{k}"] = v

        # One key per block offset; values match the per-pos quotient.
        assert set(metrics) == {"loss", "draft_acc"} | {f"draft_acc_k{k}" for k in range(1, bs)}
        for k in range(1, bs):
            expected = (correct_per_pos[k - 1] / count_per_pos[k - 1].clamp_min(1.0)).item()
            assert metrics[f"draft_acc_k{k}"] == pytest.approx(expected, abs=1e-6)
        # Overall acc derives consistently from the per-pos sums.
        assert metrics["draft_acc"] == pytest.approx(total_correct / total_count, abs=1e-6)

    def test_dp_sum_reduction_yields_global_accuracy(self):
        """SUM-allreduce of per-rank (correct, count) per position, then divide
        post-reduction, yields the correct global per-position accuracy and
        overall accuracy. This is the property the recipe relies on for
        distributed-correct logging under FSDP2.
        """
        torch.manual_seed(5)
        B, N, K, V = 1, 2, 3, 8  # 3 offsets x 2 blocks
        # Two uneven "shards" with the same shape but different content.
        t0 = torch.randint(0, V, (B, N, K))
        t1 = torch.randint(0, V, (B, N, K))
        l0 = torch.randn(B, N, K, V)
        l1 = torch.randn(B, N, K, V)
        m0 = torch.ones(B, N, K)
        m1 = torch.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]])  # one position masked

        loss_fn = DFlashDecayLoss(loss_gamma=7.0)
        r0 = loss_fn(l0, t0, m0)
        r1 = loss_fn(l1, t1, m1)

        # Recipe pattern: SUM-allreduce across shards, then divide.
        correct_global = r0.draft_correct_per_pos + r1.draft_correct_per_pos
        count_global = r0.draft_count_per_pos + r1.draft_count_per_pos
        per_pos_acc = correct_global / count_global.clamp_min(1.0)
        overall_acc = correct_global.sum() / count_global.sum()

        # Hand-computed reference
        c0 = ((l0.argmax(-1) == t0).float() * m0).sum(dim=(0, 1))
        c1 = ((l1.argmax(-1) == t1).float() * m1).sum(dim=(0, 1))
        n0 = m0.sum(dim=(0, 1))
        n1 = m1.sum(dim=(0, 1))
        expected_per_pos = (c0 + c1) / (n0 + n1).clamp_min(1.0)
        expected_overall = (c0 + c1).sum() / (n0 + n1).sum()
        assert torch.allclose(per_pos_acc, expected_per_pos, atol=1e-6)
        assert torch.isclose(overall_acc, expected_overall, atol=1e-6)


class TestIDLMLoss:
    """I-DLM block-diffusion loss: two CEs over the ``[x_t | x_0]`` halves.

    The reference is the official combined-loss math (``train/sft/trainer.py``):
    a decode CE on the noisy half and a verify CE on the clean half, both over
    the Dream-shifted response (answer) positions.
    """

    def _reference(self, logits, target, answer_mask, valid_mask, alpha, L):
        noisy, clean = logits[:, :L, :], logits[:, L : 2 * L, :]
        shift_target = target[:, 1:]
        supervise = (answer_mask[:, 1:].bool() & valid_mask[:, 1:].bool()).float()
        denom = supervise.sum().clamp_min(1)

        def ce(lg):
            pt = F.cross_entropy(
                lg[:, :-1, :].reshape(-1, lg.size(-1)).float(),
                shift_target.reshape(-1),
                reduction="none",
            ).view(shift_target.shape)
            return (pt * supervise).sum() / denom

        ce_noisy, ce_clean = ce(noisy), ce(clean)
        return ce_noisy + alpha * ce_clean, ce_noisy, ce_clean

    def _inputs(self):
        torch.manual_seed(0)
        B, L, V = 2, 12, 32
        logits = torch.randn(B, 2 * L, V)  # [x_t | x_0]
        target = torch.randint(0, V, (B, L))
        valid_mask = torch.ones(B, L, dtype=torch.long)
        valid_mask[1, 10:] = 0  # pad the tail of sample 1
        # All-masked: supervised region (after a 4-token prompt) is the response.
        answer_mask = torch.zeros(B, L, dtype=torch.bool)
        answer_mask[:, 4:] = True
        answer_mask = answer_mask & valid_mask.bool()
        return logits, target, answer_mask, valid_mask, L

    def test_matches_reference_fixed_alpha(self):
        from nemo_automodel.components.loss.dllm_loss import IDLMLoss

        logits, target, answer_mask, valid_mask, L = self._inputs()
        out = IDLMLoss(clean_loss_weight=0.2)(logits, target, answer_mask, valid_mask, seq_len=L)
        ref_total, ref_noisy, _ = self._reference(logits, target, answer_mask, valid_mask, 0.2, L)
        assert torch.isclose(out.total_loss, ref_total, atol=1e-5)
        assert torch.isclose(out.dllm_loss, ref_noisy, atol=1e-5)

    def test_auto_balance_scales_clean_to_noisy(self):
        from nemo_automodel.components.loss.dllm_loss import IDLMLoss

        logits, target, answer_mask, valid_mask, L = self._inputs()
        out = IDLMLoss(auto_balance=True)(logits, target, answer_mask, valid_mask, seq_len=L)
        _, ce_noisy, ce_clean = self._reference(logits, target, answer_mask, valid_mask, 0.0, L)
        alpha = (ce_noisy / ce_clean.clamp_min(1e-6)).detach()
        expected = ce_noisy + alpha * ce_clean
        assert torch.isclose(out.total_loss, expected, atol=1e-5)

    def test_excludes_padding_and_finite(self):
        """Both CEs ignore padded positions and never produce NaN/inf."""
        from nemo_automodel.components.loss.dllm_loss import IDLMLoss

        logits, target, answer_mask, valid_mask, L = self._inputs()
        out = IDLMLoss(clean_loss_weight=1.0)(logits, target, answer_mask, valid_mask, seq_len=L)
        assert torch.isfinite(out.total_loss)
        assert out.total_loss.item() > 0

    def test_normalization_by_num_diffusion_tokens(self):
        """A global denominator rescales the loss vs the local supervised count.

        The recipe passes an all-reduced ``num_diffusion_tokens`` so the loss is a
        DP/grad-accum-invariant token-mean; here we assert the denominator swap is
        exact (local sum-of-CE divided by the supplied global count).
        """
        from nemo_automodel.components.loss.dllm_loss import IDLMLoss

        logits, target, answer_mask, valid_mask, L = self._inputs()
        loss_fn = IDLMLoss(clean_loss_weight=0.2)
        local = loss_fn(logits, target, answer_mask, valid_mask, seq_len=L)

        # Recover the un-normalised (summed) loss from the local-denominator run.
        local_denom = (answer_mask[:, 1:].bool() & valid_mask[:, 1:].bool()).sum().clamp_min(1)
        global_denom = 97
        summed = local.total_loss * local_denom
        out = loss_fn(logits, target, answer_mask, valid_mask, seq_len=L, num_diffusion_tokens=global_denom)
        assert torch.isclose(out.total_loss, summed / global_denom, atol=1e-5)


def _init_single_process_group():
    """Trivial single-process gloo group for DTensor tests (mirrors test_kd_loss)."""
    if not torch.distributed.is_available():
        return None
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29507", rank=0, world_size=1)
    return torch.distributed.group.WORLD


@pytest.fixture(scope="module")
def trivial_pg():
    process_group_already_initialized = torch.distributed.is_initialized()
    pg = _init_single_process_group()
    if pg is None:
        pytest.skip("torch.distributed not available")
    try:
        yield pg
    finally:
        if not process_group_already_initialized and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


class TestIDLMLossDTensor:
    """IDLMLoss on a vocab-sharded DTensor must match the plain-tensor result.

    Single-process (world_size=1) gloo group: the shard is trivial, but this
    exercises the DTensor code path (the ``full_tensor()`` materialisation in
    ``_compute_per_token_nll`` plus the logit-shift slice on a DTensor) that the
    FSDP2/TP training path hits, deterministically and without a GPU.
    """

    def test_vocab_sharded_dtensor_matches_plain(self, trivial_pg):
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Shard, distribute_tensor

        from nemo_automodel.components.loss.dllm_loss import IDLMLoss

        torch.manual_seed(0)
        B, L, V = 2, 12, 32
        logits = torch.randn(B, 2 * L, V)  # [x_t | x_0]
        target = torch.randint(0, V, (B, L))
        valid_mask = torch.ones(B, L, dtype=torch.long)
        answer_mask = torch.zeros(B, L, dtype=torch.bool)
        answer_mask[:, 4:] = True

        loss_fn = IDLMLoss(clean_loss_weight=0.2)
        plain = loss_fn(logits, target, answer_mask, valid_mask, seq_len=L).total_loss

        mesh = init_device_mesh("cpu", (1,))
        dlogits = distribute_tensor(logits, mesh, [Shard(-1)])  # vocab-sharded
        sharded = loss_fn(dlogits, target, answer_mask, valid_mask, seq_len=L).total_loss

        assert torch.allclose(plain, sharded, atol=1e-5), f"plain {plain.item():.6f} != DTensor {sharded.item():.6f}"


class TestDFlashNormalizeMean:
    """``normalize="mean"`` divides by the effective weight sum (decay-weighted mean)."""

    def _weighted_mean(self, logits_full, target_full, block_mask_full, gamma):
        bsz, n, bs, _ = logits_full.shape
        offsets = torch.arange(bs).view(1, 1, -1)
        weight = block_mask_full * (offsets > 0).float()
        if gamma is not None:
            decay = torch.exp(-(torch.arange(bs).float() - 1).clamp(min=0) / gamma).view(1, 1, -1)
            weight = weight * decay
        nll = F.cross_entropy(logits_full.reshape(-1, logits_full.size(-1)), target_full.reshape(-1), reduction="none")
        flat_w = weight.reshape(-1)
        return (nll * flat_w).sum() / (flat_w.sum() + 1e-6)

    @pytest.mark.parametrize("gamma", [7.0, None])
    def test_mean_matches_reference(self, gamma):
        torch.manual_seed(0)
        bsz, n, bs, V = 2, 3, 5, 16
        logits_full = torch.randn(bsz, n, bs, V)
        target_full = torch.randint(0, V, (bsz, n, bs))
        block_mask_full = (torch.rand(bsz, n, bs) > 0.3).float()

        expected = self._weighted_mean(logits_full, target_full, block_mask_full, gamma)

        loss_fn = DFlashDecayLoss(loss_gamma=gamma, normalize="mean")
        pred_logits = logits_full[:, :, 1:, :]  # [bsz, n, bs-1, V]
        pred_targets = target_full[:, :, 1:]
        pred_mask = block_mask_full[:, :, 1:]
        got = loss_fn(pred_logits, pred_targets, pred_mask, num_tokens=None).total_loss

        assert torch.isclose(expected, got, atol=1e-6)

    def test_default_normalize_is_tokens(self):
        assert DFlashDecayLoss().normalize == "tokens"

    def test_invalid_normalize_raises(self):
        with pytest.raises(ValueError, match="normalize must be"):
            DFlashDecayLoss(normalize="bogus")


class TestBlockDiffusionCrossEntropyLoss:
    def test_returns_dllm_loss_output(self, dummy_inputs):
        logits, target_ids, noise_mask, p_mask, loss_mask = dummy_inputs
        result = BlockDiffusionCrossEntropyLoss()(logits, target_ids, noise_mask, p_mask, loss_mask)
        assert isinstance(result, DLLMLossOutput)

    def test_total_loss_equals_dllm_loss(self, dummy_inputs):
        """No AR component: total_loss == dllm_loss."""
        logits, target_ids, noise_mask, p_mask, loss_mask = dummy_inputs
        result = BlockDiffusionCrossEntropyLoss()(logits, target_ids, noise_mask, p_mask, loss_mask)
        assert torch.allclose(result.total_loss, result.dllm_loss, atol=1e-6)

    def test_scores_all_supervised_canvas_even_without_noise(self, dummy_inputs):
        """Loss support is ALL supervised canvas positions (matches Google's
        canvas_mask, NOT noise-gated), so zero corruption still yields a nonzero
        loss whenever supervised canvas tokens exist."""
        logits, target_ids, _, p_mask, loss_mask = dummy_inputs
        noise_mask = torch.zeros(B, L, dtype=torch.bool)
        result = BlockDiffusionCrossEntropyLoss()(logits, target_ids, noise_mask, p_mask, loss_mask)
        assert result.total_loss.item() > 0.0

    def test_flat_loss_ignores_p_mask(self, dummy_inputs):
        """Loss must NOT depend on p_mask (flat: no 1/p weighting)."""
        logits, target_ids, noise_mask, _, loss_mask = dummy_inputs
        loss_fn = BlockDiffusionCrossEntropyLoss()
        r_half = loss_fn(logits, target_ids, noise_mask, torch.full((B, L), 0.5), loss_mask)
        r_ones = loss_fn(logits, target_ids, noise_mask, torch.ones(B, L), loss_mask)
        r_tenth = loss_fn(logits, target_ids, noise_mask, torch.full((B, L), 0.1), loss_mask)
        assert torch.allclose(r_half.total_loss, r_ones.total_loss, atol=1e-6)
        assert torch.allclose(r_half.total_loss, r_tenth.total_loss, atol=1e-6)

    def test_differs_from_mdlm_when_p_not_one(self, dummy_inputs):
        """Sanity: with p_mask != 1, flat loss differs from MDLM's 1/p-weighted loss."""
        logits, target_ids, noise_mask, _, loss_mask = dummy_inputs
        p_mask = torch.full((B, L), 0.5)
        flat = BlockDiffusionCrossEntropyLoss()(logits, target_ids, noise_mask, p_mask, loss_mask)
        mdlm = MDLMCrossEntropyLoss()(logits, target_ids, noise_mask, p_mask, loss_mask)
        # MDLM scores corrupted positions only, multiplies by 1/0.5 = 2, and
        # normalizes by supervised count; the flat block-diffusion loss scores ALL
        # supervised canvas positions with no weight. They must differ.
        assert not torch.allclose(flat.total_loss, mdlm.total_loss, atol=1e-4)

    def test_numerical_correctness_against_reference(self):
        """loss = sum(CE over ALL supervised canvas) / num_supervised, no weighting,
        NOT noise-gated (matches Google's canvas-mask loss support)."""
        torch.manual_seed(123)
        B_t, L_t, V_t = 2, 4, 8
        logits = torch.randn(B_t, L_t, V_t)
        target_ids = torch.randint(0, V_t, (B_t, L_t))
        loss_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
        noise_mask = torch.tensor([[True, False, True, False], [False, True, False, False]])
        p_mask = torch.ones(B_t, L_t)

        ce = F.cross_entropy(logits.reshape(-1, V_t), target_ids.reshape(-1), reduction="none").reshape(B_t, L_t)
        mask = loss_mask.bool()  # ALL supervised positions, regardless of noise
        num_supervised = int(mask.sum().item())  # 5
        expected = (ce * mask.float()).sum() / num_supervised

        result = BlockDiffusionCrossEntropyLoss()(logits, target_ids, noise_mask, p_mask, loss_mask)
        assert torch.allclose(result.total_loss, expected, atol=1e-6)

    def test_all_supervised_canvas_contribute_corruption_agnostic(self):
        """ALL supervised positions contribute regardless of corruption; a
        corrupted-but-NOT-supervised position is still excluded (loss support =
        supervised canvas, matching Google — not the noise mask)."""
        torch.manual_seed(99)
        logits = torch.randn(1, 6, 16)
        target_ids = torch.randint(0, 16, (1, 6))
        loss_mask = torch.tensor([[1, 1, 1, 0, 0, 0]])
        # pos 0,1,2: supervised -> ALL included (corrupted or not)
        # pos 3: corrupted but NOT supervised -> excluded
        noise_mask = torch.tensor([[False, False, True, True, False, False]])
        p_mask = torch.ones(1, 6)

        ce = F.cross_entropy(logits.reshape(-1, 16), target_ids.reshape(-1), reduction="none").reshape(1, 6)
        mask = loss_mask.bool()
        expected = (ce * mask.float()).sum() / int(mask.sum().item())  # pos 0,1,2 / 3
        result = BlockDiffusionCrossEntropyLoss()(logits, target_ids, noise_mask, p_mask, loss_mask)
        assert torch.allclose(result.total_loss, expected, atol=1e-6)

    def test_global_normalization_denominator(self):
        """num_diffusion_tokens overrides the local supervised-canvas count as denominator."""
        torch.manual_seed(1)
        logits = torch.randn(1, 4, 8)
        target_ids = torch.randint(0, 8, (1, 4))
        loss_mask = torch.ones(1, 4, dtype=torch.long)
        noise_mask = torch.tensor([[True, True, False, False]])
        p_mask = torch.ones(1, 4)

        ce = F.cross_entropy(logits.reshape(-1, 8), target_ids.reshape(-1), reduction="none").reshape(1, 4)
        summed = (ce * loss_mask.float()).sum()  # ALL supervised positions

        loss_fn = BlockDiffusionCrossEntropyLoss()
        # local denominator = 4 supervised
        r_local = loss_fn(logits, target_ids, noise_mask, p_mask, loss_mask)
        assert torch.allclose(r_local.total_loss, summed / 4, atol=1e-6)
        # global denominator = 10
        r_global = loss_fn(logits, target_ids, noise_mask, p_mask, loss_mask, num_diffusion_tokens=10)
        assert torch.allclose(r_global.total_loss, summed / 10, atol=1e-6)

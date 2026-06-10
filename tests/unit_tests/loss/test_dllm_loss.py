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

B_D, T_D, V_D = 2, 15, 32  # batch, block_size-1 (15 predicted per block_size=16), vocab


@pytest.fixture
def dflash_inputs():
    torch.manual_seed(7)
    logits = torch.randn(B_D, T_D, V_D)
    target_ids = torch.randint(0, V_D, (B_D, T_D))
    block_mask = torch.ones(B_D, T_D)
    return logits, target_ids, block_mask


class TestDFlashDecayLoss:
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

    def test_gamma_controls_decay_rate(self):
        """Larger γ → slower decay → different total loss than small γ."""
        torch.manual_seed(2)
        T, V = 10, 16
        logits = torch.randn(1, T, V)
        target_ids = torch.randint(0, V, (1, T))
        block_mask = torch.ones(1, T)

        loss_fast = DFlashDecayLoss(loss_gamma=1.0)(logits, target_ids, block_mask).total_loss
        loss_slow = DFlashDecayLoss(loss_gamma=100.0)(logits, target_ids, block_mask).total_loss

        assert not torch.allclose(loss_fast, loss_slow, atol=1e-3)


class TestDFlashDraftAccuracy:
    """Per-position draft top-1 accuracy (correct, count) sums.

    The loss returns per-rank raw (correct, count) sums per block offset;
    the recipe SUM-allreduces both and divides post-reduction, so the
    reduction works for arbitrary per-rank token distributions without
    smuggling a per-rank denominator into the numerator.
    """

    def test_none_when_block_size_unknown(self, dflash_inputs):
        """Without block_size the per-position split is undefined -> both fields None."""
        logits, target_ids, block_mask = dflash_inputs
        result = DFlashDecayLoss(loss_gamma=7.0)(logits, target_ids, block_mask)
        assert result.draft_correct_per_pos is None
        assert result.draft_count_per_pos is None

    def test_perfect_predictions_give_full_counts(self):
        """argmax == target everywhere -> per-pos correct equals per-pos count."""
        B, N, bs, V = 2, 3, 5, 8  # T = N * (bs - 1) = 12
        T_per = bs - 1
        T = N * T_per
        target_ids = torch.randint(0, V, (B, T))
        logits = torch.full((B, T, V), -10.0)
        logits.scatter_(2, target_ids.unsqueeze(-1), 10.0)  # peak at the target
        block_mask = torch.ones(B, T)
        result = DFlashDecayLoss(loss_gamma=7.0)(logits, target_ids, block_mask, block_size=bs)
        assert result.draft_correct_per_pos.shape == (T_per,)
        assert torch.equal(result.draft_correct_per_pos, result.draft_count_per_pos)
        # Each of the T_per offsets has B * N valid positions.
        assert torch.all(result.draft_count_per_pos == B * N)

    def test_counts_exclude_masked_positions(self):
        """Positions with block_mask=0 must not contribute to correct OR count."""
        B, N, bs, V = 1, 1, 5, 8  # T = 4
        T = N * (bs - 1)
        target_ids = torch.zeros(B, T, dtype=torch.long)
        logits = torch.full((B, T, V), -10.0)
        logits[..., 0] = 10.0  # always predicts class 0 == target
        logits[0, 3] = 0.0
        logits[0, 3, 1] = 10.0  # offset k=4 predicts wrong
        block_mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        result = DFlashDecayLoss(loss_gamma=7.0)(logits, target_ids, block_mask, block_size=bs)
        # k=1,2,3 are correct + counted; k=4 is masked -> zero count, zero correct
        assert result.draft_correct_per_pos.tolist() == [1.0, 1.0, 1.0, 0.0]
        assert result.draft_count_per_pos.tolist() == [1.0, 1.0, 1.0, 0.0]

    def test_none_when_block_size_does_not_partition_tokens(self):
        """Irregular T cannot be reshaped into [B, N, block_size-1]."""
        correct = torch.ones(1, 5, dtype=torch.bool)
        block_mask = torch.ones(1, 5)

        correct_per_pos, count_per_pos = DFlashDecayLoss._draft_acc_per_pos(
            correct,
            block_mask,
            block_size=4,
        )

        assert correct_per_pos is None
        assert count_per_pos is None

    def test_fused_matches_nonfused(self):
        """forward_fused and forward must agree on loss and per-position sums."""
        torch.manual_seed(3)
        B, N, bs, D, V = 2, 2, 4, 16, 32
        T = N * (bs - 1)
        hidden = torch.randn(B, T, D)
        weight = torch.randn(V, D)
        bias = torch.randn(V)
        target_ids = torch.randint(0, V, (B, T))
        block_mask = torch.ones(B, T)
        loss_fn = DFlashDecayLoss(loss_gamma=7.0, use_fused_linear_ce=True, chunk_size=4)

        logits = torch.nn.functional.linear(hidden, weight, bias)
        ref = loss_fn(logits, target_ids, block_mask, num_tokens=B * T, block_size=bs)
        fused = loss_fn.forward_fused(
            hidden,
            weight,
            target_ids,
            block_mask,
            num_tokens=B * T,
            block_size=bs,
            lm_head_bias=bias,
        )

        assert torch.allclose(ref.total_loss, fused.total_loss, atol=1e-4)
        assert torch.equal(ref.draft_correct_per_pos, fused.draft_correct_per_pos)
        assert torch.equal(ref.draft_count_per_pos, fused.draft_count_per_pos)

    def test_paper_default_first_offset_weight_is_one(self):
        """The first predicted position of every block must have decay weight 1.0
        for the paper's (block_size, gamma) defaults. This locks Eq. 4 and the
        published triples (16/7, 10/5, 8/4) — if anyone retunes _decay_weights
        and accidentally shifts the start point, every block's k=1 supervision
        gets the wrong weight."""
        for block_size, gamma in [(16, 7.0), (10, 5.0), (8, 4.0)]:
            loss_fn = DFlashDecayLoss(loss_gamma=gamma)
            T_per = block_size - 1
            n_blocks = 3
            w = loss_fn._decay_weights(n_blocks * T_per, block_size, torch.device("cpu"), torch.float32)
            assert w.shape == (n_blocks * T_per,), f"block_size={block_size}: weights shape mismatch"
            # First weight of every block must be 1.0; weights must decay within a block.
            for b in range(n_blocks):
                start = b * T_per
                assert torch.isclose(w[start], torch.tensor(1.0)), (
                    f"block_size={block_size}, block {b}: first weight {w[start].item()} != 1.0"
                )
                assert w[start] > w[start + T_per - 1], (
                    f"block_size={block_size}, block {b}: weights do not decay within block"
                )

    def test_recipe_per_pos_metrics_dict_construction(self):
        """Lock the recipe-side contract: given the loss's per-rank
        ``(draft_correct_per_pos, draft_count_per_pos)`` tensors and the
        post-reduction divide it performs, the metrics dict must contain
        ``draft_acc`` plus one ``draft_acc_k{k}`` key per offset with the
        correct value. Mirrors train_ft.py:_run_train_optim_step verbatim
        so it catches drift in the recipe's reduction shape."""
        B, N, bs, V = 2, 2, 5, 8
        T = N * (bs - 1)
        torch.manual_seed(11)
        target_ids = torch.randint(0, V, (B, T))
        logits = torch.randn(B, T, V)
        block_mask = torch.ones(B, T)
        loss_fn = DFlashDecayLoss(loss_gamma=7.0)
        result = loss_fn(logits, target_ids, block_mask, block_size=bs)

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
        B, N, bs, V = 1, 2, 4, 8  # T = 6 (3 offsets x 2 blocks)
        T = N * (bs - 1)
        # Two uneven "shards" with the same shape but different content.
        t0 = torch.randint(0, V, (B, T))
        t1 = torch.randint(0, V, (B, T))
        l0 = torch.randn(B, T, V)
        l1 = torch.randn(B, T, V)
        m0 = torch.ones(B, T)
        m1 = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]])  # one position masked

        loss_fn = DFlashDecayLoss(loss_gamma=7.0)
        r0 = loss_fn(l0, t0, m0, block_size=bs)
        r1 = loss_fn(l1, t1, m1, block_size=bs)

        # Recipe pattern: SUM-allreduce across shards, then divide.
        correct_global = r0.draft_correct_per_pos + r1.draft_correct_per_pos
        count_global = r0.draft_count_per_pos + r1.draft_count_per_pos
        per_pos_acc = correct_global / count_global.clamp_min(1.0)
        overall_acc = correct_global.sum() / count_global.sum()

        # Hand-computed reference
        c0 = ((l0.argmax(-1) == t0).float() * m0).view(B, N, bs - 1).sum(dim=(0, 1))
        c1 = ((l1.argmax(-1) == t1).float() * m1).view(B, N, bs - 1).sum(dim=(0, 1))
        n0 = m0.view(B, N, bs - 1).sum(dim=(0, 1))
        n1 = m1.view(B, N, bs - 1).sum(dim=(0, 1))
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


def _init_single_process_group():
    """Trivial single-process gloo group for DTensor tests (mirrors test_kd_loss)."""
    if not torch.distributed.is_available():
        return None
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="gloo", init_method="tcp://127.0.0.1:29507", rank=0, world_size=1)
    return torch.distributed.group.WORLD


@pytest.fixture(scope="module")
def trivial_pg():
    pg = _init_single_process_group()
    if pg is None:
        pytest.skip("torch.distributed not available")
    return pg


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
        pred_logits = logits_full[:, :, 1:, :].reshape(bsz, n * (bs - 1), V)
        pred_targets = target_full[:, :, 1:].reshape(bsz, n * (bs - 1))
        pred_mask = block_mask_full[:, :, 1:].reshape(bsz, n * (bs - 1))
        got = loss_fn(pred_logits, pred_targets, pred_mask, num_tokens=None, block_size=bs).total_loss

        assert torch.isclose(expected, got, atol=1e-6)

    def test_default_normalize_is_tokens(self):
        assert DFlashDecayLoss().normalize == "tokens"

    def test_invalid_normalize_raises(self):
        with pytest.raises(ValueError, match="normalize must be"):
            DFlashDecayLoss(normalize="bogus")

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

"""Tests for dLLM strategies (MDLMStrategy, HybridStrategy, DFlashStrategy) and get_dllm_strategy."""

import types

import pytest
import torch

from nemo_automodel.recipes.dllm.strategy import (
    DFlashStrategy,
    HybridStrategy,
    MDLMStrategy,
    get_dllm_strategy,
)


def test_get_dllm_strategy_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unknown dllm.mode"):
        get_dllm_strategy("unknown")


# ---------------------------------------------------------------------------
# MDLMStrategy tests
# ---------------------------------------------------------------------------


class TestMDLMStrategy:
    @pytest.fixture
    def strategy(self):
        return MDLMStrategy()

    def test_apply_corruption_uses_uniform(self, strategy):
        """MDLM always uses uniform corruption (p_mask constant per sequence)."""
        torch.manual_seed(42)
        B, L = 4, 32
        input_ids = torch.randint(0, 100, (B, L))
        loss_mask = torch.ones(B, L, dtype=torch.long)
        _, _, p_mask = strategy.apply_corruption(
            input_ids,
            loss_mask,
            mask_token_id=999,
            eps=0.001,
            block_size=None,
            half_life_ratio=None,
        )
        for b in range(B):
            assert (p_mask[b] == p_mask[b, 0]).all()

    def test_prepare_batch_sets_noisy_input_ids(self, strategy):
        """MDLM sets input_ids to noisy tokens and removes attention_mask (bidirectional)."""
        batch = {"input_ids": torch.zeros(2, 4, dtype=torch.long), "attention_mask": torch.ones(2, 4)}
        noisy = torch.ones(2, 4, dtype=torch.long) * 999
        noise_mask = torch.ones(2, 4, dtype=torch.bool)
        clean = torch.zeros(2, 4, dtype=torch.long)

        result = strategy.prepare_batch(batch, noisy, noise_mask, clean)
        assert (result["input_ids"] == noisy).all()
        assert "attention_mask" not in result


# ---------------------------------------------------------------------------
# LLaDA-specific integration tests
# ---------------------------------------------------------------------------


class TestLLaDAIntegration:
    """Tests specific to LLaDA model integration with MDLM strategy."""

    LLADA_MASK_TOKEN_ID = 126336

    def test_corruption_with_llada_mask_token(self):
        """Corrupted positions get LLaDA's mask token; uncorrupted positions are unchanged."""
        torch.manual_seed(42)
        strategy = MDLMStrategy()
        B, L = 2, 16
        input_ids = torch.randint(0, 1000, (B, L))
        loss_mask = torch.ones(B, L, dtype=torch.long)

        noisy, noise_mask, p_mask = strategy.apply_corruption(
            input_ids,
            loss_mask,
            mask_token_id=self.LLADA_MASK_TOKEN_ID,
            eps=0.001,
            block_size=None,
            half_life_ratio=None,
        )
        assert (noisy[noise_mask] == self.LLADA_MASK_TOKEN_ID).all()
        assert (noisy[~noise_mask] == input_ids[~noise_mask]).all()

    def test_prepare_batch_passes_extra_keys_for_recipe_filtering(self):
        """Strategy keeps extra collator keys (input_lengths); the recipe filters
        them against the LLaDA forward signature (which does not accept **kwargs)."""
        strategy = MDLMStrategy()
        batch = {
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "attention_mask": torch.ones(2, 4),
            "input_lengths": torch.tensor([3, 4]),  # extra key from collator
        }
        noisy = torch.ones(2, 4, dtype=torch.long) * 126336
        noise_mask = torch.ones(2, 4, dtype=torch.bool)
        clean = torch.zeros(2, 4, dtype=torch.long)

        result = strategy.prepare_batch(batch, noisy, noise_mask, clean)
        assert (result["input_ids"] == noisy).all()
        assert "attention_mask" not in result
        assert "input_lengths" in result  # passed through; recipe filters it


# ---------------------------------------------------------------------------
# HybridStrategy tests
# ---------------------------------------------------------------------------


class TestHybridStrategy:
    @pytest.fixture
    def strategy(self):
        return HybridStrategy()

    def test_create_loss_fn_reads_alpha_from_config(self, strategy):
        assert strategy.create_loss_fn({"ar_loss_alpha": 0.3}).alpha == 0.3
        assert strategy.create_loss_fn({}).alpha == 1.0  # default

    def test_apply_corruption_uniform_when_no_block_size(self, strategy):
        """block_size=None should select uniform corruption (constant p_mask per row)."""
        torch.manual_seed(42)
        B, L = 2, 16
        input_ids = torch.randint(0, 100, (B, L))
        loss_mask = torch.ones(B, L, dtype=torch.long)
        _, _, p_mask = strategy.apply_corruption(
            input_ids,
            loss_mask,
            mask_token_id=999,
            eps=0.001,
            block_size=None,
            half_life_ratio=None,
        )
        for b in range(B):
            assert torch.allclose(p_mask[b], p_mask[b, 0].expand_as(p_mask[b]))

    def test_prepare_batch_passes_clean_input_ids(self, strategy):
        """Hybrid models receive clean tokens plus a masked_indices sidecar."""
        batch = {
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "attention_mask": torch.ones(2, 4),
            "use_cache": True,
        }
        noisy = torch.full((2, 4), 100, dtype=torch.long)
        noise_mask = torch.tensor([[True, False, True, False], [False, True, False, True]])
        clean = torch.arange(8, dtype=torch.long).reshape(2, 4)

        result = strategy.prepare_batch(batch, noisy, noise_mask, clean)

        assert (result["input_ids"] == clean).all()
        assert (result["masked_indices"] == noise_mask).all()
        assert (result["labels"] == clean).all()
        assert result["skip_loss"] is True
        assert "attention_mask" not in result
        assert "use_cache" not in result


# ---------------------------------------------------------------------------
# DFlashStrategy — anchor-block sampling (CPU, no model loading)
# ---------------------------------------------------------------------------

MASK_ID = 999
BLOCK_SIZE = 16


def _make_recipe(mask_token_id=MASK_ID):
    """Minimal recipe stub with the fields DFlashStrategy methods need."""
    return types.SimpleNamespace(mask_token_id=mask_token_id)


def _make_strategy(block_size=BLOCK_SIZE, overlap_anchors=False):
    """DFlashStrategy stub with block_size set; defaults to the
    non-overlapping sampler for backward compatibility with existing tests.
    """
    s = DFlashStrategy()
    s.block_size = block_size
    s.overlap_anchors = overlap_anchors
    return s


class TestDFlashSampleAnchorBlocks:
    """Tests for the non-overlapping (legacy) _sample_anchor_blocks path.

    Returns the per-sample 5-tuple
    ``(anchor_positions [B,N], block_keep_mask [B,N], block_output_ids,
    block_targets, block_mask)``.
    """

    def _make_inputs(self, seq_len, batch_size=2):
        torch.manual_seed(42)
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attn = torch.ones(batch_size, seq_len, dtype=torch.long)
        return input_ids, attn

    def test_shapes(self):
        s = _make_strategy(block_size=8)
        recipe = _make_recipe()
        for n in (1, 4):
            input_ids, attn = self._make_inputs(128)
            ap, keep, boi, bt, bm = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=n)
            assert ap.shape == (2, n)
            assert keep.shape == (2, n)
            assert boi.shape == (2, n * 8)
            assert bt.shape == (2, n * 7)
            assert bm.shape == (2, n * 7)

    def test_blocks_are_non_overlapping(self):
        s = _make_strategy(block_size=8)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(128)
        for _ in range(10):
            ap, *_ = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=4)
            starts = ap[0].tolist()  # batch-shared in non-overlap mode
            assert starts == sorted(starts)
            for i in range(len(starts) - 1):
                assert starts[i + 1] >= starts[i] + s.block_size, f"blocks overlap: {starts}"

    def test_blocks_fit_in_sequence(self):
        seq_len = 64
        s = _make_strategy(block_size=8)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(seq_len)
        for _ in range(10):
            ap, *_ = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=4)
            assert (ap >= 1).all() and (ap + s.block_size <= seq_len).all()

    def test_anchor_token_is_clean(self):
        """First token of each kept block must be the real token at its anchor."""
        s = _make_strategy(block_size=8)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(128)
        ap, keep, boi, *_ = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=3)
        B, n = ap.shape
        for b in range(B):
            for i in range(n):
                if keep[b, i]:
                    assert boi[b, i * s.block_size] == input_ids[b, ap[b, i]]

    def test_non_anchor_tokens_are_mask(self):
        """All positions after the anchor in each block should be MASK_ID."""
        s = _make_strategy(block_size=8)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(128)
        ap, _, boi, *_ = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=3)
        n = ap.shape[1]
        for b in range(n):
            noise_slice = boi[:, b * s.block_size + 1 : (b + 1) * s.block_size]
            assert (noise_slice == MASK_ID).all()

    def test_loss_mask_zeros_block_mask(self):
        """block_mask must be zero wherever loss_mask is zero."""
        torch.manual_seed(7)
        B, L, bs = 2, 64, 8
        s = _make_strategy(block_size=bs)
        recipe = _make_recipe()
        input_ids = torch.randint(0, 100, (B, L))
        attn = torch.ones(B, L, dtype=torch.long)
        # Zero the entire loss_mask — every predicted position should be masked out.
        loss_mask = torch.zeros(B, L, dtype=torch.long)
        _, _, _, _, bm = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=3, loss_mask=loss_mask)
        assert bm.sum().item() == 0


class TestDFlashSampleAnchorBlocksOverlapping:
    """Tests for the paper-default per-sample overlap_anchors=True sampler.

    Each sample draws ``num_blocks`` anchors independently (Appendix A.1), so
    anchor_positions is ``[B, N]`` with potentially different rows.
    """

    def _make_inputs(self, seq_len, batch_size=2):
        torch.manual_seed(42)
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attn = torch.ones(batch_size, seq_len, dtype=torch.long)
        return input_ids, attn

    def test_anchors_in_valid_range(self):
        """Every anchor must satisfy 1 <= a <= valid_len - block_size."""
        seq_len, bs = 64, 8
        s = _make_strategy(block_size=bs, overlap_anchors=True)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(seq_len)
        for _ in range(10):
            ap, *_ = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=16)
            assert (ap >= 1).all() and (ap <= seq_len - bs).all()

    def test_can_exceed_non_overlapping_cap(self):
        """N > seq_len // block_size succeeds (impossible in non-overlap mode)."""
        seq_len, bs = 32, 8  # non-overlap cap = 4
        s = _make_strategy(block_size=bs, overlap_anchors=True)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(seq_len)
        ap, keep, boi, bt, bm = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=20)
        assert ap.shape == (2, 20)
        assert boi.shape == (2, 20 * bs)

    def test_per_sample_diversity(self):
        """Different samples should (with high probability) get different anchors."""
        torch.manual_seed(0)
        s = _make_strategy(block_size=8, overlap_anchors=True)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(256, batch_size=2)
        ap, *_ = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=16)
        # Two independently-sampled rows of 16 anchors should not be identical.
        assert not torch.equal(ap[0], ap[1])

    def test_anchor_token_is_clean_per_sample(self):
        """Each kept block's first token is the real token at that sample's anchor."""
        s = _make_strategy(block_size=8, overlap_anchors=True)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(128)
        ap, keep, boi, *_ = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=10)
        B, n = ap.shape
        for b in range(B):
            for i in range(n):
                if keep[b, i]:
                    assert boi[b, i * s.block_size] == input_ids[b, ap[b, i]]

    def test_min_token_filter_drops_short_samples(self):
        """Samples with < 2*block_size supervised tokens get block_keep_mask=False."""
        bs = 8
        s = _make_strategy(block_size=bs, overlap_anchors=True)
        recipe = _make_recipe()
        torch.manual_seed(1)
        B, L = 2, 128
        input_ids = torch.randint(0, 100, (B, L))
        attn = torch.ones(B, L, dtype=torch.long)
        # Sample 0 long enough, sample 1 too short (< 2*bs valid tokens).
        attn[1, 2 * bs - 1 :] = 0
        ap, keep, _, _, bm = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=4)
        assert keep[0].all()  # long sample kept
        assert (~keep[1]).all()  # short sample fully dropped
        assert bm[1].sum().item() == 0  # short sample contributes no loss

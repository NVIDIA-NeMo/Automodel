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

"""Tests for dLLM strategies (MDLMStrategy, DFlashStrategy) and get_dllm_strategy."""

import types

import pytest
import torch

from nemo_automodel.components.loss.dllm_loss import (
    MDLMCrossEntropyLoss,
)
from nemo_automodel.recipes.dllm.strategy import (
    DLLM_STRATEGIES,
    DFlashStrategy,
    DLLMStrategy,
    MDLMStrategy,
    get_dllm_strategy,
)

# ---------------------------------------------------------------------------
# Strategy registry tests
# ---------------------------------------------------------------------------


class TestDLLMStrategyRegistry:
    def test_mdlm_in_strategies(self):
        assert "mdlm" in DLLM_STRATEGIES

    def test_dflash_in_strategies(self):
        assert "dflash" in DLLM_STRATEGIES

    def test_get_dflash_strategy(self):
        s = get_dllm_strategy("dflash")
        assert isinstance(s, DFlashStrategy)

    def test_get_mdlm_strategy(self):
        s = get_dllm_strategy("mdlm")
        assert isinstance(s, MDLMStrategy)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown dllm.mode"):
            get_dllm_strategy("unknown")

    def test_all_strategies_are_subclasses(self):
        for name, cls in DLLM_STRATEGIES.items():
            assert issubclass(cls, DLLMStrategy)

    def test_all_strategies_have_valid_normalization_mode(self):
        for name, cls in DLLM_STRATEGIES.items():
            s = cls()
            assert s.normalization_mode in ("supervised", "noise"), (
                f"Strategy {name} has invalid normalization_mode: {s.normalization_mode}"
            )


# ---------------------------------------------------------------------------
# MDLMStrategy tests
# ---------------------------------------------------------------------------


class TestMDLMStrategy:
    @pytest.fixture
    def strategy(self):
        return MDLMStrategy()

    def test_normalization_mode_default(self, strategy):
        assert strategy.normalization_mode == "supervised"

    def test_create_loss_fn_type(self, strategy):
        loss_fn = strategy.create_loss_fn({})
        assert isinstance(loss_fn, MDLMCrossEntropyLoss)

    def test_apply_corruption_shapes(self, strategy):
        torch.manual_seed(42)
        B, L = 2, 16
        input_ids = torch.randint(0, 100, (B, L))
        loss_mask = torch.ones(B, L, dtype=torch.long)
        noisy, noise_mask, p_mask = strategy.apply_corruption(
            input_ids,
            loss_mask,
            mask_token_id=999,
            eps=0.001,
            block_size=None,
            half_life_ratio=None,
        )
        assert noisy.shape == (B, L)
        assert noise_mask.shape == (B, L)
        assert p_mask.shape == (B, L)

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
        # Uniform corruption: p_mask is constant per sequence
        for b in range(B):
            assert (p_mask[b] == p_mask[b, 0]).all()

    def test_prepare_batch_sets_noisy_input_ids(self, strategy):
        """MDLM sets input_ids to noisy tokens and removes attention_mask."""
        batch = {"input_ids": torch.zeros(2, 4, dtype=torch.long), "attention_mask": torch.ones(2, 4)}
        noisy = torch.ones(2, 4, dtype=torch.long) * 999
        noise_mask = torch.ones(2, 4, dtype=torch.bool)
        clean = torch.zeros(2, 4, dtype=torch.long)

        result = strategy.prepare_batch(batch, noisy, noise_mask, clean)
        assert (result["input_ids"] == noisy).all()
        # attention_mask should be removed for MDLM (bidirectional)
        assert "attention_mask" not in result


# ---------------------------------------------------------------------------
# LLaDA-specific integration tests
# ---------------------------------------------------------------------------


class TestLLaDAIntegration:
    """Tests specific to LLaDA model integration with MDLM strategy."""

    LLADA_MASK_TOKEN_ID = 126336

    def test_mdlm_strategy_for_llada(self):
        """LLaDA uses MDLM mode."""
        strategy = get_dllm_strategy("mdlm")
        assert isinstance(strategy, MDLMStrategy)

    def test_corruption_with_llada_mask_token(self):
        """Verify corruption works with LLaDA's mask token ID."""
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
        # Corrupted positions should have LLaDA's mask token
        assert (noisy[noise_mask] == self.LLADA_MASK_TOKEN_ID).all()
        # Uncorrupted positions unchanged
        assert (noisy[~noise_mask] == input_ids[~noise_mask]).all()

    def test_mdlm_loss_with_llada_outputs(self):
        """Test MDLM loss with shapes matching LLaDA output."""
        torch.manual_seed(42)
        strategy = MDLMStrategy()
        loss_fn = strategy.create_loss_fn({})

        B, L, V_test = 2, 16, 100
        logits = torch.randn(B, L, V_test)
        target_ids = torch.randint(0, V_test, (B, L))
        loss_mask = torch.ones(B, L, dtype=torch.long)
        noise_mask = torch.rand(B, L) > 0.5
        p_mask = torch.full((B, L), 0.5)

        result = loss_fn(logits, target_ids, noise_mask, p_mask, loss_mask)
        assert result.total_loss.item() > 0
        # MDLM: total_loss == dllm_loss (no AR component)
        assert torch.allclose(result.total_loss, result.dllm_loss, atol=1e-6)

    def test_prepare_batch_for_llada_forward(self):
        """Verify batch prepared by MDLM strategy is compatible with LLaDA.

        LLaDA forward() accepts: input_ids, inputs_embeds, attention_mask,
        attention_bias, past_key_values, labels, use_cache, output_attentions,
        output_hidden_states, return_dict, cache_position.
        It does NOT accept **kwargs.
        """
        strategy = MDLMStrategy()
        batch = {
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "attention_mask": torch.ones(2, 4),
            "input_lengths": torch.tensor([3, 4]),  # Extra key from collator
        }
        noisy = torch.ones(2, 4, dtype=torch.long) * 126336
        noise_mask = torch.ones(2, 4, dtype=torch.bool)
        clean = torch.zeros(2, 4, dtype=torch.long)

        result = strategy.prepare_batch(batch, noisy, noise_mask, clean)

        # input_ids should be noisy
        assert (result["input_ids"] == noisy).all()
        # attention_mask should be removed by strategy
        assert "attention_mask" not in result
        # input_lengths is still present (filtering is done by the recipe)
        assert "input_lengths" in result

        # Simulate recipe-level filtering for LLaDA
        llada_params = {
            "input_ids",
            "inputs_embeds",
            "attention_mask",
            "attention_bias",
            "past_key_values",
            "labels",
            "use_cache",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
            "cache_position",
        }
        filtered = {k: v for k, v in result.items() if k in llada_params}
        assert "input_lengths" not in filtered
        assert "input_ids" in filtered


# ---------------------------------------------------------------------------
# normalization_mode override tests
# ---------------------------------------------------------------------------


class TestNormalizationModeOverride:
    """Verify that a strategy subclass can override normalization_mode."""

    def test_custom_noise_mode(self):
        class NoiseModeStrategy(DLLMStrategy):
            @property
            def normalization_mode(self):
                return "noise"

            def create_loss_fn(self, dllm_cfg):
                return MDLMCrossEntropyLoss()

            def apply_corruption(self, input_ids, loss_mask, mask_token_id, *, eps, block_size, half_life_ratio):
                from nemo_automodel.components.datasets.dllm.corruption import corrupt_uniform

                return corrupt_uniform(input_ids, loss_mask, mask_token_id, eps=eps)

            def prepare_batch(self, batch, noisy_input_ids, noise_mask, clean_input_ids):
                batch["input_ids"] = noisy_input_ids
                return batch

        s = NoiseModeStrategy()
        assert s.normalization_mode == "noise"


# ---------------------------------------------------------------------------
# DFlashStrategy — unit tests (CPU, no model loading)
# ---------------------------------------------------------------------------

MASK_ID = 999
BLOCK_SIZE = 16


def _make_recipe(mask_token_id=MASK_ID):
    """Minimal recipe stub with the fields DFlashStrategy methods need."""
    return types.SimpleNamespace(mask_token_id=mask_token_id)


def _make_strategy(block_size=BLOCK_SIZE):
    s = DFlashStrategy()
    s.block_size = block_size
    return s


class TestDFlashStrategy:
    @pytest.fixture
    def strategy(self):
        s = DFlashStrategy()
        s.block_size = BLOCK_SIZE
        return s

    @pytest.fixture
    def recipe(self):
        return _make_recipe()

    def test_loss_log_key(self, strategy):
        assert strategy.loss_log_key == "Loss/Train_DFlash"

    def test_defaults(self):
        s = DFlashStrategy()
        assert s.num_blocks_per_sample == 1
        assert s.block_size == 0


class TestDFlashSampleAnchorBlocks:
    """Tests for _sample_anchor_blocks (CPU, no GPU required)."""

    def _make_inputs(self, seq_len, batch_size=2):
        torch.manual_seed(42)
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        attn = torch.ones(batch_size, seq_len, dtype=torch.long)
        return input_ids, attn

    def test_single_block_shapes(self):
        s = _make_strategy(block_size=8)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(64)
        starts, boi, bt, bm = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=1)
        assert len(starts) == 1
        assert boi.shape == (2, 8)
        assert bt.shape == (2, 7)
        assert bm.shape == (2, 7)

    def test_multi_block_shapes(self):
        s = _make_strategy(block_size=8)
        recipe = _make_recipe()
        n = 4
        input_ids, attn = self._make_inputs(128)
        starts, boi, bt, bm = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=n)
        assert len(starts) == n
        assert boi.shape == (2, n * 8)
        assert bt.shape == (2, n * 7)
        assert bm.shape == (2, n * 7)

    def test_starts_are_sorted(self):
        s = _make_strategy(block_size=8)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(128)
        for _ in range(10):
            starts, *_ = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=4)
            assert starts == sorted(starts)

    def test_blocks_are_non_overlapping(self):
        s = _make_strategy(block_size=8)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(128)
        for _ in range(10):
            starts, *_ = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=4)
            for i in range(len(starts) - 1):
                assert starts[i + 1] >= starts[i] + s.block_size, (
                    f"blocks overlap: starts={starts}"
                )

    def test_blocks_fit_in_sequence(self):
        seq_len = 64
        s = _make_strategy(block_size=8)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(seq_len)
        for _ in range(10):
            starts, *_ = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=4)
            for start in starts:
                assert start >= 1
                assert start + s.block_size <= seq_len

    def test_fallback_when_sequence_too_short(self):
        """If sequence can't fit N blocks, gracefully returns fewer."""
        s = _make_strategy(block_size=16)
        recipe = _make_recipe()
        # seq_len=20 → only 1 block of size 16 fits (start ∈ [1,4])
        input_ids, attn = self._make_inputs(20)
        starts, boi, bt, bm = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=4)
        assert len(starts) == 1
        assert boi.shape[1] == 16

    def test_anchor_token_is_clean(self):
        """First token of each block in block_output_ids should be the real token."""
        s = _make_strategy(block_size=8)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(128)
        starts, boi, bt, bm = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=3)
        for i, start in enumerate(starts):
            assert (boi[:, i * s.block_size] == input_ids[:, start]).all()

    def test_non_anchor_tokens_are_mask(self):
        """All positions after the anchor in each block should be MASK_ID."""
        s = _make_strategy(block_size=8)
        recipe = _make_recipe()
        input_ids, attn = self._make_inputs(128)
        starts, boi, *_ = s._sample_anchor_blocks(recipe, input_ids, attn, num_blocks=3)
        n = len(starts)
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
        _, _, _, bm = s._sample_anchor_block(recipe, input_ids, attn, loss_mask=loss_mask)
        assert bm.sum().item() == 0


class TestDFlashBuildBlockAttentionMask:
    """Tests for _build_block_attention_mask (CPU, static method)."""

    def test_output_shape(self):
        starts = [10, 30, 55]
        block_size, ctx_len = 8, 55
        mask = DFlashStrategy._build_block_attention_mask(
            starts, block_size, ctx_len, torch.float32, torch.device("cpu")
        )
        n = len(starts)
        assert mask.shape == (1, 1, n * block_size, ctx_len + n * block_size)

    def test_context_prefix_attended(self):
        """Block b attends to context positions 0..starts[b]-1."""
        starts = [5, 20]
        block_size, ctx_len = 4, 20
        mask = DFlashStrategy._build_block_attention_mask(
            starts, block_size, ctx_len, torch.float32, torch.device("cpu")
        )
        assert (mask[0, 0, :block_size, :starts[0]] == 0.0).all()
        assert (mask[0, 0, block_size:, :starts[1]] == 0.0).all()

    def test_own_block_attended(self):
        """Each block's rows attend to its own noise columns."""
        starts = [5, 20]
        block_size, ctx_len = 4, 20
        mask = DFlashStrategy._build_block_attention_mask(
            starts, block_size, ctx_len, torch.float32, torch.device("cpu")
        )
        assert (mask[0, 0, :block_size, ctx_len : ctx_len + block_size] == 0.0).all()
        assert (mask[0, 0, block_size:, ctx_len + block_size : ctx_len + 2 * block_size] == 0.0).all()

    def test_cross_block_masked(self):
        """Block b must not attend to another block's noise columns."""
        starts = [5, 20]
        block_size, ctx_len = 4, 20
        mask = DFlashStrategy._build_block_attention_mask(
            starts, block_size, ctx_len, torch.float32, torch.device("cpu")
        )
        assert (mask[0, 0, :block_size, ctx_len + block_size : ctx_len + 2 * block_size] == float("-inf")).all()
        assert (mask[0, 0, block_size:, ctx_len : ctx_len + block_size] == float("-inf")).all()

    def test_future_context_masked(self):
        """Block b must not attend to context positions >= starts[b]."""
        starts = [5, 20]
        block_size, ctx_len = 4, 20
        mask = DFlashStrategy._build_block_attention_mask(
            starts, block_size, ctx_len, torch.float32, torch.device("cpu")
        )
        assert (mask[0, 0, :block_size, starts[0]:ctx_len] == float("-inf")).all()

    def test_single_block_full_context_attended(self):
        """Single block attends to its entire context and own noise."""
        start = 15
        block_size, ctx_len = 8, 15
        mask = DFlashStrategy._build_block_attention_mask(
            [start], block_size, ctx_len, torch.float32, torch.device("cpu")
        )
        assert (mask[0, 0, :, :ctx_len] == 0.0).all()
        assert (mask[0, 0, :, ctx_len:] == 0.0).all()

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

"""Tests for dLLM strategies (MDLMStrategy) and get_dllm_strategy."""

import pytest
import torch

from nemo_automodel.components.loss.dllm_loss import (
    MDLMCrossEntropyLoss,
)
from nemo_automodel.recipes.dllm.strategy import (
    DLLM_STRATEGIES,
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

    def test_get_mdlm_strategy(self):
        s = get_dllm_strategy("mdlm")
        assert isinstance(s, MDLMStrategy)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown dllm.mode"):
            get_dllm_strategy("unknown")

    def test_all_strategies_are_subclasses(self):
        for name, cls in DLLM_STRATEGIES.items():
            assert issubclass(cls, DLLMStrategy)


# ---------------------------------------------------------------------------
# MDLMStrategy tests
# ---------------------------------------------------------------------------


class TestMDLMStrategy:
    @pytest.fixture
    def strategy(self):
        return MDLMStrategy()

    def test_create_loss_fn_type(self, strategy):
        loss_fn = strategy.create_loss_fn({})
        assert isinstance(loss_fn, MDLMCrossEntropyLoss)

    def test_apply_corruption_shapes(self, strategy):
        torch.manual_seed(42)
        B, L = 2, 16
        input_ids = torch.randint(0, 100, (B, L))
        loss_mask = torch.ones(B, L, dtype=torch.long)
        noisy, noise_mask, p_mask = strategy.apply_corruption(
            input_ids, loss_mask, mask_token_id=999, eps=0.001,
            block_size=None, half_life_ratio=None,
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
            input_ids, loss_mask, mask_token_id=999, eps=0.001,
            block_size=None, half_life_ratio=None,
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
            input_ids, loss_mask, mask_token_id=self.LLADA_MASK_TOKEN_ID,
            eps=0.001, block_size=None, half_life_ratio=None,
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
            "input_ids", "inputs_embeds", "attention_mask", "attention_bias",
            "past_key_values", "labels", "use_cache", "output_attentions",
            "output_hidden_states", "return_dict", "cache_position",
        }
        filtered = {k: v for k, v in result.items() if k in llada_params}
        assert "input_lengths" not in filtered
        assert "input_ids" in filtered

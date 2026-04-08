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

"""Tests for dLLM corruption functions (corrupt_uniform, corrupt_blockwise)."""

import pytest
import torch

from nemo_automodel.components.datasets.dllm.corruption import (
    corrupt_blockwise,
    corrupt_uniform,
    gumbel_topk,
)

B, L = 4, 32
MASK_TOKEN_ID = 999


@pytest.fixture
def inputs():
    torch.manual_seed(0)
    input_ids = torch.randint(0, 100, (B, L))
    # Supervised positions: first 24 of 32
    loss_mask = torch.zeros(B, L, dtype=torch.long)
    loss_mask[:, :24] = 1
    return input_ids, loss_mask


# ---------------------------------------------------------------------------
# gumbel_topk
# ---------------------------------------------------------------------------


class TestGumbelTopk:
    def test_exact_k_selected(self):
        torch.manual_seed(0)
        log_w = torch.zeros(50)
        for k in [1, 5, 25, 50]:
            mask = gumbel_topk(log_w, k)
            assert mask.sum().item() == k

    def test_output_is_bool(self):
        mask = gumbel_topk(torch.zeros(10), 3)
        assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# corrupt_uniform
# ---------------------------------------------------------------------------


class TestCorruptUniform:
    def test_output_shapes(self, inputs):
        input_ids, loss_mask = inputs
        noisy, noise_mask, p_mask = corrupt_uniform(input_ids, loss_mask, MASK_TOKEN_ID)
        assert noisy.shape == (B, L)
        assert noise_mask.shape == (B, L)
        assert p_mask.shape == (B, L)

    def test_noise_mask_is_subset_of_loss_mask(self, inputs):
        input_ids, loss_mask = inputs
        _, noise_mask, _ = corrupt_uniform(input_ids, loss_mask, MASK_TOKEN_ID)
        # Every corrupted position must be a supervised position
        assert (noise_mask & ~loss_mask.bool()).sum() == 0

    def test_corrupted_positions_have_mask_token(self, inputs):
        input_ids, loss_mask = inputs
        noisy, noise_mask, _ = corrupt_uniform(input_ids, loss_mask, MASK_TOKEN_ID)
        assert (noisy[noise_mask] == MASK_TOKEN_ID).all()

    def test_uncorrupted_positions_unchanged(self, inputs):
        input_ids, loss_mask = inputs
        noisy, noise_mask, _ = corrupt_uniform(input_ids, loss_mask, MASK_TOKEN_ID)
        assert (noisy[~noise_mask] == input_ids[~noise_mask]).all()

    def test_p_mask_in_valid_range(self, inputs):
        input_ids, loss_mask = inputs
        _, _, p_mask = corrupt_uniform(input_ids, loss_mask, MASK_TOKEN_ID, eps=0.01)
        # p = (1-eps)*t + eps, t in [0,1], so p in [eps, 1]
        assert (p_mask >= 0.01 - 1e-6).all()
        assert (p_mask <= 1.0 + 1e-6).all()

    def test_p_mask_constant_per_sequence(self, inputs):
        """Uniform corruption uses a single t per sequence."""
        input_ids, loss_mask = inputs
        _, _, p_mask = corrupt_uniform(input_ids, loss_mask, MASK_TOKEN_ID)
        for b in range(B):
            assert (p_mask[b] == p_mask[b, 0]).all()

    def test_no_corruption_outside_loss_mask(self, inputs):
        """Positions with loss_mask=0 should never be corrupted."""
        input_ids, loss_mask = inputs
        noisy, _, _ = corrupt_uniform(input_ids, loss_mask, MASK_TOKEN_ID)
        unsupervised = ~loss_mask.bool()
        assert (noisy[unsupervised] == input_ids[unsupervised]).all()


# ---------------------------------------------------------------------------
# corrupt_blockwise
# ---------------------------------------------------------------------------


class TestCorruptBlockwise:
    def test_output_shapes(self, inputs):
        input_ids, loss_mask = inputs
        noisy, noise_mask, p_mask = corrupt_blockwise(
            input_ids, loss_mask, MASK_TOKEN_ID, block_size=8
        )
        assert noisy.shape == (B, L)
        assert noise_mask.shape == (B, L)
        assert p_mask.shape == (B, L)

    def test_noise_mask_is_subset_of_loss_mask(self, inputs):
        input_ids, loss_mask = inputs
        _, noise_mask, _ = corrupt_blockwise(
            input_ids, loss_mask, MASK_TOKEN_ID, block_size=8
        )
        assert (noise_mask & ~loss_mask.bool()).sum() == 0

    def test_corrupted_positions_have_mask_token(self, inputs):
        input_ids, loss_mask = inputs
        noisy, noise_mask, _ = corrupt_blockwise(
            input_ids, loss_mask, MASK_TOKEN_ID, block_size=8
        )
        assert (noisy[noise_mask] == MASK_TOKEN_ID).all()

    def test_uncorrupted_positions_unchanged(self, inputs):
        input_ids, loss_mask = inputs
        noisy, noise_mask, _ = corrupt_blockwise(
            input_ids, loss_mask, MASK_TOKEN_ID, block_size=8
        )
        assert (noisy[~noise_mask] == input_ids[~noise_mask]).all()

    def test_p_mask_in_valid_range(self, inputs):
        input_ids, loss_mask = inputs
        _, _, p_mask = corrupt_blockwise(
            input_ids, loss_mask, MASK_TOKEN_ID, block_size=8, eps=0.01
        )
        assert (p_mask >= 0.01 - 1e-6).all()
        assert (p_mask <= 1.0 + 1e-6).all()

    def test_sequence_level_when_no_block_size(self, inputs):
        """block_size=None should do per-sequence sampling."""
        input_ids, loss_mask = inputs
        noisy, noise_mask, p_mask = corrupt_blockwise(
            input_ids, loss_mask, MASK_TOKEN_ID, block_size=None
        )
        # p_mask should be constant per sequence (single m sampled)
        for b in range(B):
            assert (p_mask[b] == p_mask[b, 0]).all()

    def test_blockwise_p_mask_varies_across_blocks(self, inputs):
        """With block_size, different blocks can have different p values."""
        input_ids, loss_mask = inputs
        torch.manual_seed(123)
        _, _, p_mask = corrupt_blockwise(
            input_ids, loss_mask, MASK_TOKEN_ID, block_size=8
        )
        # With 4 blocks of size 8, p values should differ across blocks
        # (not guaranteed per run, but with seed=123 and 4 blocks it's very likely)
        block_ps = [p_mask[0, i * 8].item() for i in range(4)]
        assert len(set(block_ps)) > 1, "Expected different p values across blocks"

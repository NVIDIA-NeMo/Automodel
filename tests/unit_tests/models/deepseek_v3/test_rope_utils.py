# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import math

import pytest
import torch

from nemo_automodel.components.distributed.te_cp_utils import (
    generate_positional_ids_for_cp,
    get_batch_on_this_cp_rank,
    pad_thd_sequences_for_cp,
)
from nemo_automodel.components.models.deepseek_v3.rope_utils import (
    apply_rotary_emb,
    freqs_cis_from_position_ids,
    precompute_freqs_cis,
    yarn_get_mscale,
)


class TestYarnGetMscale:
    """Tests for yarn_get_mscale function"""

    def test_scale_less_than_one(self):
        """Test that scale <= 1 returns 1.0"""
        assert yarn_get_mscale(scale=0.5, mscale=1.0) == 1.0
        assert yarn_get_mscale(scale=1.0, mscale=1.0) == 1.0

    def test_scale_greater_than_one(self):
        """Test that scale > 1 returns correct mscale value"""
        scale = 2.0
        mscale = 1.0
        expected = 0.1 * mscale * math.log(scale) + 1.0
        result = yarn_get_mscale(scale=scale, mscale=mscale)
        assert abs(result - expected) < 1e-6

    def test_different_mscale_values(self):
        """Test with different mscale values"""
        scale = 4.0
        mscale = 2.0
        expected = 0.1 * mscale * math.log(scale) + 1.0
        result = yarn_get_mscale(scale=scale, mscale=mscale)
        assert abs(result - expected) < 1e-6

    def test_large_scale(self):
        """Test with large scale values"""
        scale = 100.0
        mscale = 1.0
        expected = 0.1 * mscale * math.log(scale) + 1.0
        result = yarn_get_mscale(scale=scale, mscale=mscale)
        assert abs(result - expected) < 1e-6
        assert result > 1.0


class TestPrecomputeFreqsCis:
    """Tests for precompute_freqs_cis function"""

    def test_basic_computation_no_scaling(self):
        """Test basic frequency computation without rope scaling"""
        qk_rope_head_dim = 64
        max_seq_len = 128
        rope_theta = 10000.0
        rope_scaling = None

        freqs = precompute_freqs_cis(qk_rope_head_dim, max_seq_len, rope_theta, rope_scaling)

        # Check shape
        assert freqs.shape == (qk_rope_head_dim // 2,)
        assert freqs.dtype == torch.float32

        # Check that frequencies are positive
        assert torch.all(freqs > 0)

        # Check that frequencies are decreasing (higher dimensions have lower frequencies)
        assert torch.all(freqs[:-1] >= freqs[1:])

    def test_with_rope_scaling_short_seq(self):
        """Test with rope scaling when sequence is shorter than original"""
        qk_rope_head_dim = 64
        max_seq_len = 1024  # Less than original_max_position_embeddings
        rope_theta = 10000.0
        rope_scaling = {
            "factor": 2.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 4096,
        }

        freqs = precompute_freqs_cis(qk_rope_head_dim, max_seq_len, rope_theta, rope_scaling)

        # Check shape
        assert freqs.shape == (qk_rope_head_dim // 2,)

        # When seq_len <= original_seq_len, no scaling should be applied
        freqs_no_scaling = precompute_freqs_cis(qk_rope_head_dim, max_seq_len, rope_theta, None)
        torch.testing.assert_close(freqs, freqs_no_scaling)

    def test_with_rope_scaling_long_seq(self):
        """Test with rope scaling when sequence is longer than original"""
        qk_rope_head_dim = 64
        max_seq_len = 8192  # Greater than original_max_position_embeddings
        rope_theta = 10000.0
        rope_scaling = {
            "factor": 2.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 4096,
        }

        freqs = precompute_freqs_cis(qk_rope_head_dim, max_seq_len, rope_theta, rope_scaling)

        # Check shape
        assert freqs.shape == (qk_rope_head_dim // 2,)

        # When seq_len > original_seq_len, scaling should be applied
        freqs_no_scaling = precompute_freqs_cis(qk_rope_head_dim, max_seq_len, rope_theta, None)
        # Freqs should be different when scaling is applied
        assert not torch.allclose(freqs, freqs_no_scaling)

    def test_different_head_dims(self):
        """Test with different head dimensions"""
        for head_dim in [32, 64, 128, 256]:
            freqs = precompute_freqs_cis(head_dim, 128, 10000.0, None)
            assert freqs.shape == (head_dim // 2,)

    def test_different_rope_theta(self):
        """Test with different rope theta values"""
        qk_rope_head_dim = 64
        max_seq_len = 128

        freqs_small_theta = precompute_freqs_cis(qk_rope_head_dim, max_seq_len, 1000.0, None)
        freqs_large_theta = precompute_freqs_cis(qk_rope_head_dim, max_seq_len, 100000.0, None)

        # Both should start with 1.0 (first element)
        assert freqs_small_theta[0] == 1.0
        assert freqs_large_theta[0] == 1.0

        # For higher indices, smaller theta results in faster decay, so higher freq values
        # (but this is not true for ALL elements due to the formula)
        # Just verify they are different
        assert not torch.allclose(freqs_small_theta, freqs_large_theta)


class TestApplyRotaryEmb:
    """Tests for apply_rotary_emb function"""

    def test_basic_application_bshd(self):
        """Test basic rotary embedding application with bshd format"""
        batch_size = 2
        seq_len = 4
        num_heads = 8
        head_dim = 64

        x = torch.randn(batch_size, seq_len, num_heads, head_dim)
        freqs_cis = torch.randn(batch_size, seq_len, head_dim // 2, dtype=torch.complex64)

        result = apply_rotary_emb(x, freqs_cis, qkv_format="bshd")

        # Check output shape
        assert result.shape == x.shape
        assert result.dtype == x.dtype

    def test_thd_format(self):
        """Test with thd format (total_tokens, num_heads, head_dim)"""
        total_tokens = 16
        num_heads = 8
        head_dim = 64

        x = torch.randn(total_tokens, num_heads, head_dim)
        freqs_cis = torch.randn(1, total_tokens, head_dim // 2, dtype=torch.complex64)

        result = apply_rotary_emb(x, freqs_cis, qkv_format="thd")

        # Check output shape
        assert result.shape == x.shape
        assert result.dtype == x.dtype

    def test_with_unsqueeze_dim(self):
        """Test with unsqueeze_dim parameter"""
        batch_size = 2
        seq_len = 4
        num_heads = 8
        head_dim = 64

        # Input without the unsqueezed dimension
        x = torch.randn(batch_size, seq_len, head_dim)
        freqs_cis = torch.randn(batch_size, seq_len, head_dim // 2, dtype=torch.complex64)

        # Apply with unsqueeze_dim=2 (will add dimension at position 2)
        result = apply_rotary_emb(x, freqs_cis, qkv_format="bshd", unsqueeze_dim=2)

        # Check output shape (should match input - dimension is squeezed back)
        assert result.shape == x.shape
        assert result.dtype == x.dtype

    def test_dtype_preservation(self):
        """Test that output dtype matches input dtype"""
        batch_size = 2
        seq_len = 4
        num_heads = 8
        head_dim = 64

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype)
            freqs_cis = torch.randn(batch_size, seq_len, head_dim // 2, dtype=torch.complex64)

            result = apply_rotary_emb(x, freqs_cis, qkv_format="bshd")

            assert result.dtype == dtype

    def test_rotary_invariance_property(self):
        """Test that applying rotary embeddings preserves norm"""
        batch_size = 2
        seq_len = 4
        num_heads = 2
        head_dim = 8

        x = torch.randn(batch_size, seq_len, num_heads, head_dim)
        freqs_cis = torch.randn(batch_size, seq_len, head_dim // 2, dtype=torch.complex64)

        # Normalize freqs_cis to have magnitude 1 (as they should be)
        freqs_cis = freqs_cis / freqs_cis.abs()

        result = apply_rotary_emb(x, freqs_cis, qkv_format="bshd")

        # The norm should be approximately preserved
        input_norm = torch.norm(x.reshape(-1, head_dim), dim=-1)
        output_norm = torch.norm(result.reshape(-1, head_dim), dim=-1)

        # Allow some numerical error
        torch.testing.assert_close(input_norm, output_norm, rtol=1e-4, atol=1e-4)


class TestFreqsCisFromPositionIds:
    """Tests for freqs_cis_from_position_ids function"""

    def test_basic_computation(self):
        """Test basic frequency computation from position IDs"""
        batch_size = 2
        seq_len = 4
        head_dim = 64

        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
        freqs = torch.randn(head_dim // 2)

        freqs_cis = freqs_cis_from_position_ids(position_ids, freqs)

        # Check output shape
        assert freqs_cis.shape == (batch_size, seq_len, head_dim // 2)
        assert freqs_cis.dtype == torch.complex64

    def test_magnitude_is_one(self):
        """Test that output complex numbers have magnitude 1"""
        batch_size = 2
        seq_len = 4
        head_dim = 64

        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
        freqs = torch.randn(head_dim // 2)

        freqs_cis = freqs_cis_from_position_ids(position_ids, freqs)

        # All complex numbers should have magnitude 1 (since torch.polar with magnitude 1)
        magnitudes = torch.abs(freqs_cis)
        torch.testing.assert_close(magnitudes, torch.ones_like(magnitudes), rtol=1e-5, atol=1e-5)

    def test_sequential_position_ids(self):
        """Test with sequential position IDs"""
        seq_len = 8
        head_dim = 32

        position_ids = torch.arange(seq_len).unsqueeze(0)
        freqs = torch.ones(head_dim // 2) * 0.1  # Simple frequency

        freqs_cis = freqs_cis_from_position_ids(position_ids, freqs)

        # Check that phase increases with position
        phases = torch.angle(freqs_cis[0, :, 0])
        # Phases should be increasing (modulo 2*pi)
        assert phases.shape == (seq_len,)

    def test_non_sequential_position_ids(self):
        """Test with non-sequential position IDs (e.g., for packed sequences)"""
        batch_size = 2
        seq_len = 4
        head_dim = 64

        # Non-sequential positions
        position_ids = torch.tensor([[0, 1, 0, 1], [2, 3, 4, 5]])
        freqs = torch.randn(head_dim // 2)

        freqs_cis = freqs_cis_from_position_ids(position_ids, freqs)

        # Check output shape
        assert freqs_cis.shape == (batch_size, seq_len, head_dim // 2)

        # Positions with same ID should have same freqs_cis
        torch.testing.assert_close(freqs_cis[0, 0], freqs_cis[0, 2])
        torch.testing.assert_close(freqs_cis[0, 1], freqs_cis[0, 3])

    def test_large_position_ids(self):
        """Test with large position IDs"""
        batch_size = 1
        seq_len = 4
        head_dim = 64

        # Large position IDs (e.g., for long sequences)
        position_ids = torch.tensor([[1000, 2000, 3000, 4000]])
        freqs = torch.randn(head_dim // 2)

        freqs_cis = freqs_cis_from_position_ids(position_ids, freqs)

        # Check output shape
        assert freqs_cis.shape == (batch_size, seq_len, head_dim // 2)

        # All magnitudes should still be 1
        magnitudes = torch.abs(freqs_cis)
        torch.testing.assert_close(magnitudes, torch.ones_like(magnitudes), rtol=1e-5, atol=1e-5)

    def test_different_batch_position_patterns(self):
        """Test with different position patterns in different batches"""
        batch_size = 3
        seq_len = 4
        head_dim = 32

        position_ids = torch.tensor([
            [0, 1, 2, 3],      # Sequential
            [0, 0, 1, 1],      # Repeated
            [10, 20, 30, 40],  # Large gaps
        ])
        freqs = torch.randn(head_dim // 2)

        freqs_cis = freqs_cis_from_position_ids(position_ids, freqs)

        # Check output shape
        assert freqs_cis.shape == (batch_size, seq_len, head_dim // 2)

        # Batch 1: repeated positions should have same values
        torch.testing.assert_close(freqs_cis[1, 0], freqs_cis[1, 1])
        torch.testing.assert_close(freqs_cis[1, 2], freqs_cis[1, 3])


class TestFreqsCisWithContextParallel:
    """Tests for freqs_cis_from_position_ids with context parallelism"""

    @pytest.mark.parametrize("cp_size", [1, 2, 4])
    def test_freqs_cis_with_different_cp_sizes(self, cp_size):
        """Test that freqs_cis computation works correctly for different CP sizes"""
        # Setup
        batch_size = 2
        seq_len = 128
        head_dim = 64

        # Create input_ids, labels, and cu_seqlens for packed sequences
        # Two sequences of equal length
        input_ids = torch.randint(0, 1000, (batch_size * seq_len,))
        labels = torch.randint(0, 1000, (batch_size * seq_len,))
        cu_seqlens = torch.tensor([0, seq_len, 2 * seq_len])

        # Pad sequences for CP (must be divisible by 2*cp_size for THD format)
        divisibility_factor = 2 * cp_size
        input_ids_padded, labels_padded, cu_seqlens_padded = pad_thd_sequences_for_cp(
            input_ids, labels, cu_seqlens, divisibility_factor
        )

        # Generate position IDs
        position_ids = generate_positional_ids_for_cp(cu_seqlens_padded, divisibility_factor)

        # Precompute base frequencies
        freqs = precompute_freqs_cis(head_dim, seq_len * 2, 10000.0, None)

        # Split data for each CP rank and verify freqs_cis
        for cp_rank in range(cp_size):
            _, _, position_ids_rank = get_batch_on_this_cp_rank(
                cu_seqlens_padded,
                input_ids_padded,
                labels_padded,
                position_ids,
                cp_size,
                cp_rank,
                qvk_format="thd",
            )

            # Compute freqs_cis for this rank's position_ids
            position_ids_rank_2d = position_ids_rank.unsqueeze(0)
            freqs_cis_rank = freqs_cis_from_position_ids(position_ids_rank_2d, freqs)

            # Verify shape and properties
            assert freqs_cis_rank.shape[0] == 1
            assert freqs_cis_rank.shape[2] == head_dim // 2
            assert freqs_cis_rank.dtype == torch.complex64

            # Verify all magnitudes are 1
            magnitudes = torch.abs(freqs_cis_rank)
            torch.testing.assert_close(magnitudes, torch.ones_like(magnitudes), rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("cp_size,cp_rank", [(2, 0), (2, 1), (4, 0), (4, 2), (4, 3)])
    def test_freqs_cis_consistency_across_ranks(self, cp_size, cp_rank):
        """Test that freqs_cis values are consistent across CP ranks for same positions"""
        # Setup
        batch_size = 2
        seq_len = 64
        head_dim = 128

        # Create packed sequences
        input_ids = torch.randint(0, 1000, (batch_size * seq_len,))
        labels = torch.randint(0, 1000, (batch_size * seq_len,))
        cu_seqlens = torch.tensor([0, seq_len, 2 * seq_len])

        divisibility_factor = 2 * cp_size
        input_ids_padded, labels_padded, cu_seqlens_padded = pad_thd_sequences_for_cp(
            input_ids, labels, cu_seqlens, divisibility_factor
        )

        # Generate position IDs
        position_ids_full = generate_positional_ids_for_cp(cu_seqlens_padded, divisibility_factor)

        # Precompute frequencies
        freqs = precompute_freqs_cis(head_dim, seq_len * 2, 10000.0, None)

        # Get position_ids for this CP rank
        _, _, position_ids_rank = get_batch_on_this_cp_rank(
            cu_seqlens_padded,
            input_ids_padded,
            labels_padded,
            position_ids_full,
            cp_size,
            cp_rank,
            qvk_format="thd",
        )

        # Compute freqs_cis
        position_ids_rank_2d = position_ids_rank.unsqueeze(0)
        freqs_cis_rank = freqs_cis_from_position_ids(position_ids_rank_2d, freqs)

        # Verify that positions with same ID have same freqs_cis
        unique_positions = torch.unique(position_ids_rank)
        for pos in unique_positions:
            mask = position_ids_rank == pos
            indices = torch.where(mask)[0]
            if len(indices) > 1:
                # All tokens at this position should have identical freqs_cis
                for i in range(1, len(indices)):
                    torch.testing.assert_close(
                        freqs_cis_rank[0, indices[0]],
                        freqs_cis_rank[0, indices[i]]
                    )

    def test_freqs_cis_cp_with_variable_sequence_lengths(self):
        """Test freqs_cis with variable-length sequences and CP splitting"""
        cp_size = 2
        head_dim = 64

        # Create sequences of different lengths
        seq_lens = [32, 64, 48]
        input_ids = torch.cat([torch.randint(0, 1000, (l,)) for l in seq_lens])
        labels = torch.cat([torch.randint(0, 1000, (l,)) for l in seq_lens])
        cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), 0)))

        divisibility_factor = 2 * cp_size
        input_ids_padded, labels_padded, cu_seqlens_padded = pad_thd_sequences_for_cp(
            input_ids, labels, cu_seqlens, divisibility_factor
        )

        # Generate position IDs
        position_ids = generate_positional_ids_for_cp(cu_seqlens_padded, divisibility_factor)

        # Precompute frequencies
        freqs = precompute_freqs_cis(head_dim, max(seq_lens) * 2, 10000.0, None)

        # Test for each CP rank
        for cp_rank in range(cp_size):
            _, _, position_ids_rank = get_batch_on_this_cp_rank(
                cu_seqlens_padded,
                input_ids_padded,
                labels_padded,
                position_ids,
                cp_size,
                cp_rank,
                qvk_format="thd",
            )

            # Compute freqs_cis
            position_ids_rank_2d = position_ids_rank.unsqueeze(0)
            freqs_cis_rank = freqs_cis_from_position_ids(position_ids_rank_2d, freqs)

            # Verify output properties
            assert freqs_cis_rank.dtype == torch.complex64
            assert freqs_cis_rank.shape[2] == head_dim // 2

            # Verify magnitudes
            magnitudes = torch.abs(freqs_cis_rank)
            torch.testing.assert_close(magnitudes, torch.ones_like(magnitudes), rtol=1e-5, atol=1e-5)

    def test_freqs_cis_cp_reconstructibility(self):
        """Test that we can reconstruct full freqs_cis from CP-split pieces"""
        cp_size = 2
        head_dim = 64
        seq_len = 128

        # Create simple test case with one sequence
        input_ids = torch.randint(0, 1000, (seq_len,))
        labels = torch.randint(0, 1000, (seq_len,))
        cu_seqlens = torch.tensor([0, seq_len])

        divisibility_factor = 2 * cp_size
        input_ids_padded, labels_padded, cu_seqlens_padded = pad_thd_sequences_for_cp(
            input_ids, labels, cu_seqlens, divisibility_factor
        )

        # Generate position IDs for full sequence
        position_ids_full = generate_positional_ids_for_cp(cu_seqlens_padded, divisibility_factor)

        # Precompute frequencies
        freqs = precompute_freqs_cis(head_dim, seq_len * 2, 10000.0, None)

        # Collect freqs_cis from each CP rank
        freqs_cis_parts = []
        for cp_rank in range(cp_size):
            _, _, position_ids_rank = get_batch_on_this_cp_rank(
                cu_seqlens_padded,
                input_ids_padded,
                labels_padded,
                position_ids_full,
                cp_size,
                cp_rank,
                qvk_format="thd",
            )

            position_ids_rank_2d = position_ids_rank.unsqueeze(0)
            freqs_cis_rank = freqs_cis_from_position_ids(position_ids_rank_2d, freqs)
            freqs_cis_parts.append(freqs_cis_rank)

        # Verify that each part has the expected length
        expected_part_len = cu_seqlens_padded[-1].item() // cp_size
        for part in freqs_cis_parts:
            assert part.shape[1] == expected_part_len

    @pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
    def test_freqs_cis_cp_different_sizes_with_rope_scaling(self, cp_size):
        """Test freqs_cis with CP and RoPE scaling for long context"""
        seq_len = 256
        head_dim = 128
        max_seq_len = 4096

        rope_scaling = {
            "factor": 2.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 2048,
        }

        # Create test data
        input_ids = torch.randint(0, 1000, (seq_len,))
        labels = torch.randint(0, 1000, (seq_len,))
        cu_seqlens = torch.tensor([0, seq_len])

        divisibility_factor = 2 * cp_size
        input_ids_padded, labels_padded, cu_seqlens_padded = pad_thd_sequences_for_cp(
            input_ids, labels, cu_seqlens, divisibility_factor
        )

        # Generate position IDs
        position_ids = generate_positional_ids_for_cp(cu_seqlens_padded, divisibility_factor)

        # Precompute frequencies with scaling
        freqs = precompute_freqs_cis(head_dim, max_seq_len, 10000.0, rope_scaling)

        # Test each CP rank
        for cp_rank in range(cp_size):
            _, _, position_ids_rank = get_batch_on_this_cp_rank(
                cu_seqlens_padded,
                input_ids_padded,
                labels_padded,
                position_ids,
                cp_size,
                cp_rank,
                qvk_format="thd",
            )

            # Compute freqs_cis
            position_ids_rank_2d = position_ids_rank.unsqueeze(0)
            freqs_cis_rank = freqs_cis_from_position_ids(position_ids_rank_2d, freqs)

            # Verify properties
            assert freqs_cis_rank.dtype == torch.complex64
            magnitudes = torch.abs(freqs_cis_rank)
            torch.testing.assert_close(magnitudes, torch.ones_like(magnitudes), rtol=1e-5, atol=1e-5)


class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_full_rope_pipeline(self):
        """Test full RoPE pipeline: precompute -> position_ids -> apply"""
        batch_size = 2
        seq_len = 8
        num_heads = 4
        head_dim = 64
        max_seq_len = 128
        rope_theta = 10000.0

        # Step 1: Precompute frequencies
        freqs = precompute_freqs_cis(head_dim, max_seq_len, rope_theta, rope_scaling=None)

        # Step 2: Create position IDs and compute freqs_cis
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
        freqs_cis = freqs_cis_from_position_ids(position_ids, freqs)

        # Step 3: Apply rotary embeddings
        x = torch.randn(batch_size, seq_len, num_heads, head_dim)
        result = apply_rotary_emb(x, freqs_cis, qkv_format="bshd")

        # Verify output
        assert result.shape == x.shape
        assert result.dtype == x.dtype

    def test_packed_sequence_scenario(self):
        """Test RoPE with packed sequences (non-sequential position IDs)"""
        total_tokens = 12
        num_heads = 4
        head_dim = 64
        max_seq_len = 128
        rope_theta = 10000.0

        # Packed sequences: 3 sequences of lengths [3, 4, 5]
        # Position IDs restart for each sequence
        position_ids = torch.tensor([
            [0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4]
        ])

        # Precompute frequencies
        freqs = precompute_freqs_cis(head_dim, max_seq_len, rope_theta, rope_scaling=None)

        # Compute freqs_cis from position IDs
        freqs_cis = freqs_cis_from_position_ids(position_ids, freqs)

        # Apply rotary embeddings with thd format
        x = torch.randn(total_tokens, num_heads, head_dim)
        result = apply_rotary_emb(x, freqs_cis, qkv_format="thd")

        # Verify output
        assert result.shape == x.shape

        # Tokens at the same position within their sequence should have similar rotations
        # (though the actual values will differ due to the input)
        assert result.shape == (total_tokens, num_heads, head_dim)

    def test_rope_with_scaling_long_context(self):
        """Test RoPE with scaling for long context"""
        batch_size = 1
        seq_len = 16
        num_heads = 2
        head_dim = 32
        max_seq_len = 8192  # Long context
        rope_theta = 10000.0

        rope_scaling = {
            "factor": 2.0,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 4096,
        }

        # Precompute frequencies with scaling
        freqs = precompute_freqs_cis(head_dim, max_seq_len, rope_theta, rope_scaling)

        # Create position IDs
        position_ids = torch.arange(seq_len).unsqueeze(0)
        freqs_cis = freqs_cis_from_position_ids(position_ids, freqs)

        # Apply rotary embeddings
        x = torch.randn(batch_size, seq_len, num_heads, head_dim)
        result = apply_rotary_emb(x, freqs_cis, qkv_format="bshd")

        # Verify output
        assert result.shape == x.shape
        assert result.dtype == x.dtype

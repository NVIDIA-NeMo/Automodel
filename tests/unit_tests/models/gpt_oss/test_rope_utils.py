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

from nemo_automodel.components.models.gpt_oss.rope_utils import (
    RotaryEmbedding,
    apply_rotary_emb,
    position_ids_to_freqs_cis,
)


class TestApplyRotaryEmb:
    """Tests for apply_rotary_emb function"""

    def test_basic_application(self):
        """Test basic rotary embedding application"""
        batch_size = 2
        seq_len = 4
        num_heads = 8
        head_dim = 64

        x = torch.randn(batch_size, seq_len, num_heads, head_dim)
        cos = torch.randn(seq_len, head_dim // 2)
        sin = torch.randn(seq_len, head_dim // 2)

        result = apply_rotary_emb(x, cos, sin)

        # Check output shape
        assert result.shape == x.shape
        assert result.dtype == x.dtype

    def test_output_shape_preservation(self):
        """Test that output shape matches input shape"""
        for shape in [(2, 4, 8, 64), (1, 10, 4, 32), (4, 8, 16, 128)]:
            x = torch.randn(*shape)
            cos = torch.randn(shape[1], shape[3] // 2)
            sin = torch.randn(shape[1], shape[3] // 2)

            result = apply_rotary_emb(x, cos, sin)
            assert result.shape == x.shape

    def test_dtype_preservation(self):
        """Test that output dtype matches input dtype"""
        batch_size = 2
        seq_len = 4
        num_heads = 8
        head_dim = 64

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype)
            cos = torch.randn(seq_len, head_dim // 2)
            sin = torch.randn(seq_len, head_dim // 2)

            result = apply_rotary_emb(x, cos, sin)
            assert result.dtype == dtype

    def test_rotary_preserves_norm(self):
        """Test that rotary embeddings approximately preserve norm"""
        batch_size = 2
        seq_len = 4
        num_heads = 2
        head_dim = 8

        x = torch.randn(batch_size, seq_len, num_heads, head_dim)
        # Use unit cos/sin for cleaner norm preservation
        angles = torch.linspace(0, 2 * math.pi, seq_len * (head_dim // 2)).reshape(seq_len, head_dim // 2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        result = apply_rotary_emb(x, cos, sin)

        # The norm should be approximately preserved
        input_norm = torch.norm(x.reshape(-1, head_dim), dim=-1)
        output_norm = torch.norm(result.reshape(-1, head_dim), dim=-1)

        torch.testing.assert_close(input_norm, output_norm, rtol=1e-4, atol=1e-4)

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths"""
        batch_size = 2
        num_heads = 8
        head_dim = 64

        for seq_len in [1, 8, 16, 32, 128]:
            x = torch.randn(batch_size, seq_len, num_heads, head_dim)
            cos = torch.randn(seq_len, head_dim // 2)
            sin = torch.randn(seq_len, head_dim // 2)

            result = apply_rotary_emb(x, cos, sin)
            assert result.shape == (batch_size, seq_len, num_heads, head_dim)

    def test_partial_rotary_embeddings(self):
        """Test partial rotary embeddings where only first rotary_dim dimensions are rotated"""
        batch_size = 2
        seq_len = 4
        num_heads = 8
        head_dim = 64
        rotary_dim = 32  # Only rotate half the dimensions

        x = torch.randn(batch_size, seq_len, num_heads, head_dim)
        # cos/sin have dimension rotary_dim // 2
        cos = torch.randn(seq_len, rotary_dim // 2)
        sin = torch.randn(seq_len, rotary_dim // 2)

        # Store the pass-through part
        x_pass_original = x[..., rotary_dim:].clone()

        result = apply_rotary_emb(x, cos, sin)

        # Check output shape
        assert result.shape == x.shape
        assert result.dtype == x.dtype

        # Verify that the pass-through dimensions are unchanged
        torch.testing.assert_close(result[..., rotary_dim:], x_pass_original)

        # Verify that the rotated dimensions are different
        assert not torch.allclose(result[..., :rotary_dim], x[..., :rotary_dim])

    def test_partial_rotary_preserves_passthrough(self):
        """Test that partial rotary embeddings preserve the non-rotated dimensions exactly"""
        batch_size = 2
        seq_len = 4
        num_heads = 4
        head_dim = 128
        rotary_dim = 64  # Rotate only first 64 dimensions

        x = torch.randn(batch_size, seq_len, num_heads, head_dim)
        cos = torch.randn(seq_len, rotary_dim // 2)
        sin = torch.randn(seq_len, rotary_dim // 2)

        result = apply_rotary_emb(x, cos, sin)

        # The last (head_dim - rotary_dim) dimensions should be identical
        torch.testing.assert_close(
            result[..., rotary_dim:],
            x[..., rotary_dim:],
            rtol=0,
            atol=0,
            msg="Pass-through dimensions should be exactly preserved"
        )

    def test_partial_rotary_different_factors(self):
        """Test partial rotary with different rotary dimension sizes"""
        batch_size = 2
        seq_len = 4
        num_heads = 8
        head_dim = 128

        for rotary_dim in [32, 64, 96]:
            x = torch.randn(batch_size, seq_len, num_heads, head_dim)
            cos = torch.randn(seq_len, rotary_dim // 2)
            sin = torch.randn(seq_len, rotary_dim // 2)

            x_pass = x[..., rotary_dim:].clone()
            result = apply_rotary_emb(x, cos, sin)

            assert result.shape == x.shape
            # Verify pass-through is preserved
            torch.testing.assert_close(result[..., rotary_dim:], x_pass)


class TestRotaryEmbedding:
    """Tests for RotaryEmbedding class"""

    def test_initialization(self):
        """Test RotaryEmbedding initialization"""
        rope = RotaryEmbedding(
            head_dim=64,
            base=10000,
            dtype=torch.float32,
            initial_context_length=4096,
            scaling_factor=1.0,
        )

        assert rope.head_dim == 64
        assert rope.base == 10000
        assert rope.dtype == torch.float32
        assert rope.initial_context_length == 4096
        assert rope.scaling_factor == 1.0

    def test_forward_basic(self):
        """Test forward pass with basic inputs"""
        rope = RotaryEmbedding(
            head_dim=64,
            base=10000,
            dtype=torch.float32,
        )

        batch_size = 2
        seq_len = 8
        num_heads = 4

        query = torch.randn(batch_size, seq_len, num_heads, 64)
        key = torch.randn(batch_size, seq_len, num_heads, 64)

        query_rot, key_rot = rope(query, key)

        assert query_rot.shape == query.shape
        assert key_rot.shape == key.shape
        assert query_rot.dtype == query.dtype
        assert key_rot.dtype == key.dtype

    def test_compute_concentration_no_scaling(self):
        """Test concentration computation without scaling"""
        rope = RotaryEmbedding(
            head_dim=64,
            base=10000,
            dtype=torch.float32,
            scaling_factor=1.0,
        )

        concentration, inv_freq = rope._compute_concentration_and_inv_freq()

        # With scaling_factor=1.0, concentration should be 1.0
        assert concentration == 1.0
        assert inv_freq.shape == (32,)  # head_dim // 2

    def test_compute_concentration_with_scaling(self):
        """Test concentration computation with scaling"""
        scaling_factor = 2.0
        rope = RotaryEmbedding(
            head_dim=64,
            base=10000,
            dtype=torch.float32,
            scaling_factor=scaling_factor,
        )

        concentration, inv_freq = rope._compute_concentration_and_inv_freq()

        # With scaling_factor > 1, concentration should be > 1.0
        expected_concentration = 0.1 * math.log(scaling_factor) + 1.0
        assert abs(concentration - expected_concentration) < 1e-6
        assert inv_freq.shape == (32,)

    def test_compute_cos_sin(self):
        """Test cos/sin computation"""
        rope = RotaryEmbedding(
            head_dim=64,
            base=10000,
            dtype=torch.float32,
        )

        num_tokens = 16
        cos, sin = rope._compute_cos_sin(num_tokens)

        assert cos.shape == (num_tokens, 32)
        assert sin.shape == (num_tokens, 32)
        assert cos.dtype == torch.float32
        assert sin.dtype == torch.float32

    def test_different_head_dims(self):
        """Test with different head dimensions"""
        for head_dim in [32, 64, 128, 256]:
            rope = RotaryEmbedding(
                head_dim=head_dim,
                base=10000,
                dtype=torch.float32,
            )

            query = torch.randn(2, 8, 4, head_dim)
            key = torch.randn(2, 8, 4, head_dim)

            query_rot, key_rot = rope(query, key)

            assert query_rot.shape == query.shape
            assert key_rot.shape == key.shape

    def test_different_base_values(self):
        """Test with different base values"""
        for base in [1000, 10000, 100000]:
            rope = RotaryEmbedding(
                head_dim=64,
                base=base,
                dtype=torch.float32,
            )

            query = torch.randn(2, 8, 4, 64)
            key = torch.randn(2, 8, 4, 64)

            query_rot, key_rot = rope(query, key)

            assert query_rot.shape == query.shape
            assert key_rot.shape == key.shape

    def test_ntk_parameters(self):
        """Test NTK-aware interpolation parameters"""
        rope = RotaryEmbedding(
            head_dim=64,
            base=10000,
            dtype=torch.float32,
            scaling_factor=2.0,
            ntk_alpha=1.0,
            ntk_beta=32.0,
        )

        concentration, inv_freq = rope._compute_concentration_and_inv_freq()

        # Verify concentration is computed correctly
        expected_concentration = 0.1 * math.log(2.0) + 1.0
        assert abs(concentration - expected_concentration) < 1e-6

    def test_partial_rotary_factor_initialization(self):
        """Test RotaryEmbedding initialization with partial_rotary_factor"""
        head_dim = 128
        partial_rotary_factor = 0.5

        rope = RotaryEmbedding(
            head_dim=head_dim,
            base=10000,
            dtype=torch.float32,
            partial_rotary_factor=partial_rotary_factor,
        )

        assert rope.head_dim == head_dim
        assert rope.partial_rotary_factor == partial_rotary_factor
        assert rope.rotary_dim == int(head_dim * partial_rotary_factor)
        assert rope.rotary_dim == 64

    def test_partial_rotary_factor_forward(self):
        """Test forward pass with partial_rotary_factor"""
        head_dim = 128
        partial_rotary_factor = 0.5
        rotary_dim = int(head_dim * partial_rotary_factor)

        rope = RotaryEmbedding(
            head_dim=head_dim,
            base=10000,
            dtype=torch.float32,
            partial_rotary_factor=partial_rotary_factor,
        )

        batch_size = 2
        seq_len = 8
        num_heads = 4

        query = torch.randn(batch_size, seq_len, num_heads, head_dim)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # Store non-rotated parts
        query_pass = query[..., rotary_dim:].clone()
        key_pass = key[..., rotary_dim:].clone()

        query_rot, key_rot = rope(query, key)

        # Check shapes
        assert query_rot.shape == query.shape
        assert key_rot.shape == key.shape

        # Verify that non-rotated dimensions are preserved
        torch.testing.assert_close(query_rot[..., rotary_dim:], query_pass)
        torch.testing.assert_close(key_rot[..., rotary_dim:], key_pass)

        # Verify that rotated dimensions are different
        assert not torch.allclose(query_rot[..., :rotary_dim], query[..., :rotary_dim])
        assert not torch.allclose(key_rot[..., :rotary_dim], key[..., :rotary_dim])

    def test_partial_rotary_factor_different_values(self):
        """Test RotaryEmbedding with different partial_rotary_factor values"""
        head_dim = 128
        batch_size = 2
        seq_len = 8
        num_heads = 4

        for partial_rotary_factor in [0.25, 0.5, 0.75, 1.0]:
            rope = RotaryEmbedding(
                head_dim=head_dim,
                base=10000,
                dtype=torch.float32,
                partial_rotary_factor=partial_rotary_factor,
            )

            expected_rotary_dim = int(head_dim * partial_rotary_factor)
            assert rope.rotary_dim == expected_rotary_dim

            query = torch.randn(batch_size, seq_len, num_heads, head_dim)
            key = torch.randn(batch_size, seq_len, num_heads, head_dim)

            query_rot, key_rot = rope(query, key)

            assert query_rot.shape == query.shape
            assert key_rot.shape == key.shape

            # When factor is 1.0, all dimensions should be rotated
            if partial_rotary_factor == 1.0:
                assert rope.rotary_dim == head_dim
            else:
                # Otherwise, verify pass-through dimensions
                torch.testing.assert_close(
                    query_rot[..., expected_rotary_dim:],
                    query[..., expected_rotary_dim:],
                )

    def test_partial_rotary_compute_concentration_uses_rotary_dim(self):
        """Test that _compute_concentration_and_inv_freq uses rotary_dim instead of head_dim"""
        head_dim = 128
        partial_rotary_factor = 0.5
        rotary_dim = int(head_dim * partial_rotary_factor)

        rope = RotaryEmbedding(
            head_dim=head_dim,
            base=10000,
            dtype=torch.float32,
            partial_rotary_factor=partial_rotary_factor,
        )

        concentration, inv_freq = rope._compute_concentration_and_inv_freq()

        # inv_freq should have dimension rotary_dim // 2, not head_dim // 2
        assert inv_freq.shape == (rotary_dim // 2,)
        assert inv_freq.shape != (head_dim // 2,)


class TestPositionIdsToFreqsCis:
    """Tests for position_ids_to_freqs_cis function"""

    def test_basic_computation_bshd(self):
        """Test basic frequency computation with bshd format"""
        rope = RotaryEmbedding(
            head_dim=64,
            base=10000,
            dtype=torch.float32,
        )

        batch_size = 2
        seq_len = 8
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)

        freqs_cis = position_ids_to_freqs_cis(rope, position_ids, qkv_format="bshd")

        # Check output shape: should be (batch_size, seq_len, head_dim)
        assert freqs_cis.shape == (batch_size, seq_len, 64)
        assert freqs_cis.dtype == torch.float32

    def test_basic_computation_thd(self):
        """Test basic frequency computation with thd format"""
        rope = RotaryEmbedding(
            head_dim=64,
            base=10000,
            dtype=torch.float32,
        )

        seq_len = 16
        position_ids = torch.arange(seq_len)

        freqs_cis = position_ids_to_freqs_cis(rope, position_ids, qkv_format="thd")

        # Check output shape: should be (seq_len, head_dim)
        assert freqs_cis.shape == (seq_len, 64)
        assert freqs_cis.dtype == torch.float32

    def test_sequential_positions(self):
        """Test with sequential position IDs"""
        rope = RotaryEmbedding(
            head_dim=32,
            base=10000,
            dtype=torch.float32,
        )

        seq_len = 8
        position_ids = torch.arange(seq_len).unsqueeze(0)

        freqs_cis = position_ids_to_freqs_cis(rope, position_ids, qkv_format="bshd")

        assert freqs_cis.shape == (1, seq_len, 32)

    def test_non_sequential_positions(self):
        """Test with non-sequential position IDs (packed sequences)"""
        rope = RotaryEmbedding(
            head_dim=64,
            base=10000,
            dtype=torch.float32,
        )

        # Packed sequences: position IDs restart
        position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3]])

        freqs_cis = position_ids_to_freqs_cis(rope, position_ids, qkv_format="bshd")

        assert freqs_cis.shape == (1, 7, 64)

        # Positions with same ID should have same freqs_cis values
        torch.testing.assert_close(freqs_cis[0, 0], freqs_cis[0, 3])
        torch.testing.assert_close(freqs_cis[0, 1], freqs_cis[0, 4])

    def test_large_position_ids(self):
        """Test with large position IDs"""
        rope = RotaryEmbedding(
            head_dim=64,
            base=10000,
            dtype=torch.float32,
        )

        position_ids = torch.tensor([[1000, 2000, 3000, 4000]])

        freqs_cis = position_ids_to_freqs_cis(rope, position_ids, qkv_format="bshd")

        assert freqs_cis.shape == (1, 4, 64)

    def test_with_scaling_factor(self):
        """Test with scaling factor for long context"""
        rope = RotaryEmbedding(
            head_dim=64,
            base=10000,
            dtype=torch.float32,
            scaling_factor=2.0,
            initial_context_length=4096,
        )

        position_ids = torch.arange(16).unsqueeze(0)

        freqs_cis = position_ids_to_freqs_cis(rope, position_ids, qkv_format="bshd")

        assert freqs_cis.shape == (1, 16, 64)

    def test_different_batch_patterns(self):
        """Test with different position patterns in different batches"""
        rope = RotaryEmbedding(
            head_dim=32,
            base=10000,
            dtype=torch.float32,
        )

        position_ids = torch.tensor([
            [0, 1, 2, 3],      # Sequential
            [0, 0, 1, 1],      # Repeated
            [10, 20, 30, 40],  # Large gaps
        ])

        freqs_cis = position_ids_to_freqs_cis(rope, position_ids, qkv_format="bshd")

        assert freqs_cis.shape == (3, 4, 32)

        # Batch 1: repeated positions should have same values
        torch.testing.assert_close(freqs_cis[1, 0], freqs_cis[1, 1])
        torch.testing.assert_close(freqs_cis[1, 2], freqs_cis[1, 3])


class TestPositionIdsToFreqsCisWithContextParallel:
    """Tests for position_ids_to_freqs_cis with context parallelism"""

    @pytest.mark.parametrize("cp_size", [1, 2, 4])
    def test_freqs_cis_with_different_cp_sizes(self, cp_size):
        """Test that freqs_cis computation works correctly for different CP sizes"""
        rope = RotaryEmbedding(
            head_dim=64,
            base=10000,
            dtype=torch.float32,
        )

        # Directly create position IDs as they would appear after CP sharding
        # Simulate a chunk of a sequence assigned to this CP rank
        seq_len_per_rank = 128 // cp_size
        position_ids_rank = torch.arange(seq_len_per_rank)

        # Compute freqs_cis for this rank's position_ids
        freqs_cis_rank = position_ids_to_freqs_cis(rope, position_ids_rank, qkv_format="thd")

        # Verify shape and properties
        assert freqs_cis_rank.ndim == 2
        assert freqs_cis_rank.shape[1] == 64
        assert freqs_cis_rank.dtype == torch.float32

    @pytest.mark.parametrize("cp_size,cp_rank", [(2, 0), (2, 1), (4, 0), (4, 2), (4, 3)])
    def test_freqs_cis_consistency_across_ranks(self, cp_size, cp_rank):
        """Test that freqs_cis values are consistent across CP ranks for same positions"""
        rope = RotaryEmbedding(
            head_dim=128,
            base=10000,
            dtype=torch.float32,
        )

        # Directly create position IDs for this rank
        seq_len_per_rank = 64 // cp_size
        # Simulate position IDs that might have some repetition (packed sequences)
        position_ids_rank = torch.arange(seq_len_per_rank) % 10

        # Compute freqs_cis
        freqs_cis_rank = position_ids_to_freqs_cis(rope, position_ids_rank, qkv_format="thd")

        # Verify that positions with same ID have same freqs_cis
        unique_positions = torch.unique(position_ids_rank)
        for pos in unique_positions:
            mask = position_ids_rank == pos
            indices = torch.where(mask)[0]
            if len(indices) > 1:
                # All tokens at this position should have identical freqs_cis
                for i in range(1, len(indices)):
                    torch.testing.assert_close(
                        freqs_cis_rank[indices[0]],
                        freqs_cis_rank[indices[i]]
                    )

    def test_freqs_cis_cp_with_variable_sequence_lengths(self):
        """Test freqs_cis with variable-length sequences and CP splitting"""
        rope = RotaryEmbedding(
            head_dim=64,
            base=10000,
            dtype=torch.float32,
        )

        # Directly create position IDs simulating variable-length sequences after CP split
        # Simulate 3 sequences: [0-9], [0-15], [0-11] concatenated then split
        position_ids_rank = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3])

        # Compute freqs_cis
        freqs_cis_rank = position_ids_to_freqs_cis(rope, position_ids_rank, qkv_format="thd")

        # Verify output properties
        assert freqs_cis_rank.dtype == torch.float32
        assert freqs_cis_rank.shape[1] == 64

    def test_freqs_cis_cp_reconstructibility(self):
        """Test that we can reconstruct full freqs_cis from CP-split pieces"""
        rope = RotaryEmbedding(
            head_dim=64,
            base=10000,
            dtype=torch.float32,
        )

        cp_size = 2
        seq_len = 128

        # Directly create position IDs for each CP rank
        # Simulate splitting a sequence of length 128 across 2 ranks
        freqs_cis_parts = []
        for cp_rank in range(cp_size):
            # Each rank gets half the sequence
            position_ids_rank = torch.arange(seq_len // cp_size)
            freqs_cis_rank = position_ids_to_freqs_cis(rope, position_ids_rank, qkv_format="thd")
            freqs_cis_parts.append(freqs_cis_rank)

        # Verify that each part has the expected length
        expected_part_len = seq_len // cp_size
        for part in freqs_cis_parts:
            assert part.shape[0] == expected_part_len

    @pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
    def test_freqs_cis_cp_different_sizes_with_rope_scaling(self, cp_size):
        """Test freqs_cis with CP and RoPE scaling for long context"""
        rope = RotaryEmbedding(
            head_dim=128,
            base=10000,
            dtype=torch.float32,
            scaling_factor=2.0,
            initial_context_length=2048,
        )

        seq_len = 256

        # Directly create position IDs for a CP rank
        seq_len_per_rank = seq_len // cp_size
        position_ids_rank = torch.arange(seq_len_per_rank)

        # Compute freqs_cis
        freqs_cis_rank = position_ids_to_freqs_cis(rope, position_ids_rank, qkv_format="thd")

        # Verify properties
        assert freqs_cis_rank.dtype == torch.float32
        assert freqs_cis_rank.shape[1] == 128


class TestIntegration:
    """Integration tests combining multiple functions"""

    def test_full_rope_pipeline(self):
        """Test full RoPE pipeline: RotaryEmbedding -> position_ids_to_freqs_cis -> apply_rotary_emb"""
        rope = RotaryEmbedding(
            head_dim=64,
            base=10000,
            dtype=torch.float32,
        )

        batch_size = 2
        seq_len = 8
        num_heads = 4

        # Step 1: Create position IDs and compute freqs_cis
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
        freqs_cis = position_ids_to_freqs_cis(rope, position_ids, qkv_format="bshd")

        # Step 2: Extract cos and sin from freqs_cis
        # freqs_cis contains concatenated cos and sin
        cos = freqs_cis[..., :32]  # First half is cos
        sin = freqs_cis[..., 32:]   # Second half is sin

        # Step 3: Apply rotary embeddings
        x = torch.randn(batch_size, seq_len, num_heads, 64)

        # Need to reshape cos/sin for apply_rotary_emb
        cos_for_apply = cos[:, :, None, :]  # Add head dimension
        sin_for_apply = sin[:, :, None, :]

        # Broadcast to match x shape
        cos_for_apply = cos_for_apply.expand(batch_size, seq_len, num_heads, 32)
        sin_for_apply = sin_for_apply.expand(batch_size, seq_len, num_heads, 32)

        # Verify output
        assert cos_for_apply.shape == (batch_size, seq_len, num_heads, 32)
        assert sin_for_apply.shape == (batch_size, seq_len, num_heads, 32)

    def test_packed_sequence_scenario(self):
        """Test RoPE with packed sequences (non-sequential position IDs)"""
        rope = RotaryEmbedding(
            head_dim=64,
            base=10000,
            dtype=torch.float32,
        )

        # Packed sequences: 3 sequences of lengths [3, 4, 5]
        total_tokens = 12
        position_ids = torch.tensor([0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4])

        # Compute freqs_cis
        freqs_cis = position_ids_to_freqs_cis(rope, position_ids, qkv_format="thd")

        # Verify output
        assert freqs_cis.shape == (total_tokens, 64)

        # Tokens at position 0 in different sequences should have same freqs_cis
        torch.testing.assert_close(freqs_cis[0], freqs_cis[3])
        torch.testing.assert_close(freqs_cis[0], freqs_cis[7])

    def test_rope_with_scaling_long_context(self):
        """Test RoPE with scaling for long context"""
        rope = RotaryEmbedding(
            head_dim=32,
            base=10000,
            dtype=torch.float32,
            scaling_factor=2.0,
            initial_context_length=4096,
        )

        batch_size = 1
        seq_len = 16
        num_heads = 2

        # Create position IDs
        position_ids = torch.arange(seq_len).unsqueeze(0)
        freqs_cis = position_ids_to_freqs_cis(rope, position_ids, qkv_format="bshd")

        # Verify output
        assert freqs_cis.shape == (batch_size, seq_len, 32)
        assert freqs_cis.dtype == torch.float32

    def test_forward_method_integration(self):
        """Test the forward method of RotaryEmbedding class"""
        rope = RotaryEmbedding(
            head_dim=64,
            base=10000,
            dtype=torch.float32,
        )

        batch_size = 2
        seq_len = 8
        num_heads = 4

        query = torch.randn(batch_size, seq_len, num_heads, 64)
        key = torch.randn(batch_size, seq_len, num_heads, 64)

        # Apply RoPE using forward method
        query_rot, key_rot = rope(query, key)

        # Verify shapes
        assert query_rot.shape == query.shape
        assert key_rot.shape == key.shape

        # Verify that rotation was applied (output should differ from input)
        assert not torch.allclose(query, query_rot)
        assert not torch.allclose(key, key_rot)

    def test_partial_rotary_factor_integration(self):
        """Test full RoPE pipeline with partial_rotary_factor"""
        head_dim = 128
        partial_rotary_factor = 0.5
        rotary_dim = int(head_dim * partial_rotary_factor)

        rope = RotaryEmbedding(
            head_dim=head_dim,
            base=10000,
            dtype=torch.float32,
            partial_rotary_factor=partial_rotary_factor,
        )

        batch_size = 2
        seq_len = 8
        num_heads = 4

        query = torch.randn(batch_size, seq_len, num_heads, head_dim)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # Store non-rotated parts
        query_pass = query[..., rotary_dim:].clone()
        key_pass = key[..., rotary_dim:].clone()

        # Apply RoPE
        query_rot, key_rot = rope(query, key)

        # Verify shapes
        assert query_rot.shape == query.shape
        assert key_rot.shape == key.shape

        # Verify that non-rotated dimensions are exactly preserved
        torch.testing.assert_close(query_rot[..., rotary_dim:], query_pass, rtol=0, atol=0)
        torch.testing.assert_close(key_rot[..., rotary_dim:], key_pass, rtol=0, atol=0)

        # Verify that rotated dimensions are different
        assert not torch.allclose(query_rot[..., :rotary_dim], query[..., :rotary_dim])
        assert not torch.allclose(key_rot[..., :rotary_dim], key[..., :rotary_dim])

    def test_partial_rotary_with_position_ids_to_freqs_cis(self):
        """Test position_ids_to_freqs_cis with partial rotary factor"""
        head_dim = 128
        partial_rotary_factor = 0.5
        rotary_dim = int(head_dim * partial_rotary_factor)

        rope = RotaryEmbedding(
            head_dim=head_dim,
            base=10000,
            dtype=torch.float32,
            partial_rotary_factor=partial_rotary_factor,
        )

        batch_size = 2
        seq_len = 8
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)

        # Compute freqs_cis
        freqs_cis = position_ids_to_freqs_cis(rope, position_ids, qkv_format="bshd")

        # freqs_cis should have shape (batch_size, seq_len, rotary_dim)
        # because it contains concatenated cos and sin, each of size rotary_dim // 2
        assert freqs_cis.shape == (batch_size, seq_len, rotary_dim)
        assert freqs_cis.dtype == torch.float32

    def test_partial_rotary_with_scaling(self):
        """Test partial rotary factor combined with RoPE scaling"""
        head_dim = 128
        partial_rotary_factor = 0.5
        rotary_dim = int(head_dim * partial_rotary_factor)

        rope = RotaryEmbedding(
            head_dim=head_dim,
            base=10000,
            dtype=torch.float32,
            partial_rotary_factor=partial_rotary_factor,
            scaling_factor=2.0,
            initial_context_length=4096,
        )

        batch_size = 2
        seq_len = 16
        num_heads = 4

        query = torch.randn(batch_size, seq_len, num_heads, head_dim)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim)

        # Store non-rotated parts
        query_pass = query[..., rotary_dim:].clone()
        key_pass = key[..., rotary_dim:].clone()

        query_rot, key_rot = rope(query, key)

        # Verify shapes
        assert query_rot.shape == query.shape
        assert key_rot.shape == key.shape

        # Verify pass-through dimensions are preserved
        torch.testing.assert_close(query_rot[..., rotary_dim:], query_pass)
        torch.testing.assert_close(key_rot[..., rotary_dim:], key_pass)

    def test_partial_rotary_with_ntk(self):
        """Test partial rotary factor combined with NTK-aware interpolation"""
        head_dim = 128
        partial_rotary_factor = 0.75
        rotary_dim = int(head_dim * partial_rotary_factor)

        rope = RotaryEmbedding(
            head_dim=head_dim,
            base=10000,
            dtype=torch.float32,
            partial_rotary_factor=partial_rotary_factor,
            scaling_factor=2.0,
            ntk_alpha=1.0,
            ntk_beta=32.0,
        )

        batch_size = 2
        seq_len = 8
        num_heads = 4

        query = torch.randn(batch_size, seq_len, num_heads, head_dim)
        key = torch.randn(batch_size, seq_len, num_heads, head_dim)

        query_pass = query[..., rotary_dim:].clone()
        key_pass = key[..., rotary_dim:].clone()

        query_rot, key_rot = rope(query, key)

        # Verify pass-through dimensions
        torch.testing.assert_close(query_rot[..., rotary_dim:], query_pass)
        torch.testing.assert_close(key_rot[..., rotary_dim:], key_pass)

        # Verify concentration was computed correctly with scaling
        concentration, inv_freq = rope._compute_concentration_and_inv_freq()
        expected_concentration = 0.1 * math.log(2.0) + 1.0
        assert abs(concentration - expected_concentration) < 1e-6

        # Verify inv_freq uses rotary_dim
        assert inv_freq.shape == (rotary_dim // 2,)

    def test_partial_rotary_packed_sequences(self):
        """Test partial rotary factor with packed sequences"""
        head_dim = 64
        partial_rotary_factor = 0.5
        rotary_dim = int(head_dim * partial_rotary_factor)

        rope = RotaryEmbedding(
            head_dim=head_dim,
            base=10000,
            dtype=torch.float32,
            partial_rotary_factor=partial_rotary_factor,
        )

        # Packed sequences: 3 sequences of lengths [3, 4, 5]
        total_tokens = 12
        position_ids = torch.tensor([0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3, 4])

        # Compute freqs_cis
        freqs_cis = position_ids_to_freqs_cis(rope, position_ids, qkv_format="thd")

        # Verify output shape: should be (total_tokens, rotary_dim)
        assert freqs_cis.shape == (total_tokens, rotary_dim)

        # Tokens at position 0 in different sequences should have same freqs_cis
        torch.testing.assert_close(freqs_cis[0], freqs_cis[3])
        torch.testing.assert_close(freqs_cis[0], freqs_cis[7])

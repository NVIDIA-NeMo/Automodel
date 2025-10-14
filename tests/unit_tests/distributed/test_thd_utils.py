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

import torch
import pytest

from nemo_automodel.components.distributed.thd_utils import process_input_for_thd, split_batch_into_thd_chunks


class TestProcessInputForTHD:
    """Tests for process_input_for_thd function."""

    def test_basic_conversion(self):
        """Test basic conversion from BSHD to THD format with 2D token IDs."""
        batch_size, seq_len = 2, 6
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]),
            "labels": torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]),
            "seq_lens": torch.tensor([[6], [6]]),
            "seq_lens_padded": torch.tensor([[6], [6]]),
        }

        result = process_input_for_thd(batch)

        # Check shapes - for 2D input [batch, seq], output is [batch*seq] (squeezed)
        assert result["input_ids"].shape == (12,)
        assert result["position_ids"].shape == (12,)
        assert result["labels"].shape == (12,)

        # Check values are preserved
        assert torch.equal(result["input_ids"], torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))

        # Check cu_seqlens
        assert "cu_seqlens" in result
        assert "cu_seqlens_padded" in result
        assert torch.equal(result["cu_seqlens"], torch.tensor([0, 6, 12], dtype=torch.int32))

    def test_with_multiple_packed_sequences(self):
        """Test with multiple packed sequences per example."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 99, 4, 5], [6, 7, 99, 8, 9, 10]]),
            "labels": torch.tensor([[2, 3, 99, 4, 5, 6], [7, 8, 9, 10, 11, 12]]),
            "position_ids": torch.tensor([[0, 1, 2, 0, 0, 1], [0, 1, 0, 0, 1, 2]]),
            "seq_lens": torch.tensor([[3, 2], [2, 3]]),
            "seq_lens_padded": torch.tensor([[4, 2], [3, 3]]),
        }

        result = process_input_for_thd(batch)

        # Check shapes - 2D input [batch, seq] becomes [batch*seq] (squeezed)
        assert result["input_ids"].shape == (12,)
        assert result["position_ids"].shape == (12,)

        # Check cu_seqlens: [0, 3, 5, 7, 10]
        # First batch: seq 1 (len 3), seq 2 (len 2) -> cumsum [0, 3, 5]
        # Second batch: seq 1 (len 2), seq 2 (len 3) -> cumsum [5, 7, 10]
        expected_cu_seqlens = torch.tensor([0, 3, 5, 7, 10], dtype=torch.int32)
        assert torch.equal(result["cu_seqlens"], expected_cu_seqlens)

        # Check cu_seqlens_padded: [0, 4, 6, 9, 12]
        expected_cu_seqlens_padded = torch.tensor([0, 4, 6, 9, 12], dtype=torch.int32)
        assert torch.equal(result["cu_seqlens_padded"], expected_cu_seqlens_padded)

    def test_with_variable_num_sequences_and_padding(self):
        """Test with variable number of sequences per example (seq_lens padding with -1000)."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 99, 4, 5], [6, 7, 8, 9, 10, 11]]),
            "labels": torch.tensor([[2, 3, 99, 4, 5, 6], [7, 8, 9, 10, 11, 12]]),
            "position_ids": torch.tensor([[0, 1, 2, 0, 0, 1], [0, 1, 2, 3, 4, 5]]),
            "seq_lens": torch.tensor([[3, 2], [6, -1000]]),  # -1000 is padding
            "seq_lens_padded": torch.tensor([[4, 2], [6, -1000]]),
        }

        result = process_input_for_thd(batch)

        # Check shapes - 2D input [batch, seq] becomes [batch*seq] (squeezed)
        assert result["input_ids"].shape == (12,)
        assert result["position_ids"].shape == (12,)

        # Check cu_seqlens: [0, 3, 5, 11] (filters out -1000)
        expected_cu_seqlens = torch.tensor([0, 3, 5, 11], dtype=torch.int32)
        assert torch.equal(result["cu_seqlens"], expected_cu_seqlens)

        # Check cu_seqlens_padded: [0, 4, 6, 12]
        expected_cu_seqlens_padded = torch.tensor([0, 4, 6, 12], dtype=torch.int32)
        assert torch.equal(result["cu_seqlens_padded"], expected_cu_seqlens_padded)

    def test_with_qkv_format_preservation(self):
        """Test that non-tensor keys like qkv_format are preserved."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7]]),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]]),
            "seq_lens": torch.tensor([[3], [3]]),
            "seq_lens_padded": torch.tensor([[3], [3]]),
            "qkv_format": "thd",  # Non-tensor key
        }

        result = process_input_for_thd(batch)

        # Check that qkv_format is preserved
        assert "qkv_format" in result
        assert result["qkv_format"] == "thd"

    def test_dtype_preservation(self):
        """Test that dtypes are preserved correctly."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
            "seq_lens": torch.tensor([[3], [3]], dtype=torch.long),
            "seq_lens_padded": torch.tensor([[3], [3]], dtype=torch.long),
        }

        result = process_input_for_thd(batch)

        assert result["input_ids"].dtype == torch.long
        assert result["position_ids"].dtype == torch.long
        assert result["labels"].dtype == torch.long
        assert result["cu_seqlens"].dtype == torch.int32
        assert result["cu_seqlens_padded"].dtype == torch.int32

    def test_with_embeddings_3d_input(self):
        """Test with 3D embeddings input (pipeline parallelism scenario)."""
        batch_size, seq_len, hidden_dim = 2, 6, 128
        batch = {
            "input_ids": torch.randn(batch_size, seq_len, hidden_dim),  # 3D embeddings
            "labels": torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]),
            "seq_lens": torch.tensor([[6], [6]]),
            "seq_lens_padded": torch.tensor([[6], [6]]),
        }

        result = process_input_for_thd(batch)

        # Check shapes - 3D input [batch, seq, hidden] becomes [batch*seq, hidden]
        assert result["input_ids"].shape == (12, hidden_dim)
        assert result["position_ids"].shape == (12,)
        assert result["cu_seqlens"].shape == (3,)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_different_batch_sizes(self, batch_size):
        """Test with different batch sizes."""
        seq_len = 16
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            "seq_lens": torch.full((batch_size, 1), seq_len),
            "seq_lens_padded": torch.full((batch_size, 1), seq_len),
        }

        result = process_input_for_thd(batch)

        # Check output shapes
        expected_total_tokens = batch_size * seq_len
        assert result["input_ids"].shape == (expected_total_tokens,)
        assert result["position_ids"].shape == (expected_total_tokens,)
        assert result["labels"].shape == (expected_total_tokens,)
        assert result["cu_seqlens"].shape == (batch_size + 1,)

    @pytest.mark.parametrize("seq_len", [32, 64, 128, 256, 512, 1024])
    def test_different_sequence_lengths(self, seq_len):
        """Test with different sequence lengths."""
        batch_size = 4
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            "seq_lens": torch.full((batch_size, 1), seq_len),
            "seq_lens_padded": torch.full((batch_size, 1), seq_len),
        }

        result = process_input_for_thd(batch)

        # Check output shapes
        expected_total_tokens = batch_size * seq_len
        assert result["input_ids"].shape == (expected_total_tokens,)
        assert result["cu_seqlens"][-1].item() == expected_total_tokens

    def test_single_batch_single_sequence(self):
        """Test edge case: single batch with single sequence."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[2, 3, 4, 5]]),
            "position_ids": torch.tensor([[0, 1, 2, 3]]),
            "seq_lens": torch.tensor([[4]]),
            "seq_lens_padded": torch.tensor([[4]]),
        }

        result = process_input_for_thd(batch)

        assert result["input_ids"].shape == (4,)
        assert torch.equal(result["cu_seqlens"], torch.tensor([0, 4], dtype=torch.int32))

    def test_large_batch_with_packing(self):
        """Test with large batch size and multiple packed sequences."""
        batch_size = 16
        num_packs = 3
        seq_len_per_pack = 128
        total_seq_len = num_packs * seq_len_per_pack

        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, total_seq_len)),
            "labels": torch.randint(0, 1000, (batch_size, total_seq_len)),
            "position_ids": torch.arange(total_seq_len).unsqueeze(0).expand(batch_size, -1),
            "seq_lens": torch.full((batch_size, num_packs), seq_len_per_pack),
            "seq_lens_padded": torch.full((batch_size, num_packs), seq_len_per_pack),
        }

        result = process_input_for_thd(batch)

        expected_total_tokens = batch_size * total_seq_len
        expected_num_sequences = batch_size * num_packs

        assert result["input_ids"].shape == (expected_total_tokens,)
        assert result["cu_seqlens"].shape == (expected_num_sequences + 1,)
        assert result["cu_seqlens"][-1].item() == expected_total_tokens

    def test_chunking_basic(self):
        """Test basic chunking functionality."""
        batch_size, seq_len = 4, 6
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]]),
            "labels": torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19], [20, 21, 22, 23, 24, 25]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]),
            "seq_lens": torch.tensor([[6], [6], [6], [6]]),
            "seq_lens_padded": torch.tensor([[6], [6], [6], [6]]),
        }

        # Process with 2 chunks (2 batch items per chunk)
        result = split_batch_into_thd_chunks(batch, num_chunks=2)

        # Check shapes - should be [num_chunks, tokens_per_chunk]
        assert result["input_ids"].shape == (2, 12), f"Expected shape (2, 12), got {result['input_ids'].shape}"
        assert result["labels"].shape == (2, 12)
        assert result["position_ids"].shape == (2, 12)

        # Check cu_seqlens has correct shape [num_chunks, seqs_per_chunk+1]
        assert result["cu_seqlens"].shape[0] == 2

        # Each chunk should have cumulative lengths [0, 6, 12]
        assert torch.equal(result["cu_seqlens"][0], torch.tensor([0, 6, 12], dtype=torch.int32))
        assert torch.equal(result["cu_seqlens"][1], torch.tensor([0, 6, 12], dtype=torch.int32))

    def test_chunking_with_packed_sequences(self):
        """Test chunking with multiple packed sequences per example."""
        batch = {
            "input_ids": torch.tensor([
                [1, 2, 3, 99, 4, 5],
                [6, 7, 99, 8, 9, 10],
                [11, 12, 13, 99, 14, 15],
                [16, 17, 99, 18, 19, 20]
            ]),
            "labels": torch.tensor([
                [2, 3, 99, 4, 5, 6],
                [7, 8, 9, 10, 11, 12],
                [12, 13, 99, 14, 15, 16],
                [17, 18, 19, 20, 21, 22]
            ]),
            "position_ids": torch.tensor([
                [0, 1, 2, 0, 0, 1],
                [0, 1, 0, 0, 1, 2],
                [0, 1, 2, 0, 0, 1],
                [0, 1, 0, 0, 1, 2]
            ]),
            "seq_lens": torch.tensor([[3, 2], [2, 3], [3, 2], [2, 3]]),
            "seq_lens_padded": torch.tensor([[4, 2], [3, 3], [4, 2], [3, 3]]),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2)

        # Check shapes
        assert result["input_ids"].shape == (2, 12)
        assert result["labels"].shape == (2, 12)

        # Check cu_seqlens for first chunk: [0, 3, 5, 7, 10]
        expected_cu_seqlens_0 = torch.tensor([0, 3, 5, 7, 10], dtype=torch.int32)
        assert torch.equal(result["cu_seqlens"][0], expected_cu_seqlens_0)

        # Check cu_seqlens for second chunk: [0, 3, 5, 7, 10]
        expected_cu_seqlens_1 = torch.tensor([0, 3, 5, 7, 10], dtype=torch.int32)
        assert torch.equal(result["cu_seqlens"][1], expected_cu_seqlens_1)

    def test_chunking_with_embeddings(self):
        """Test chunking with 3D embeddings input."""
        batch_size, seq_len, hidden_dim = 4, 6, 128
        batch = {
            "input_ids": torch.randn(batch_size, seq_len, hidden_dim),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            "seq_lens": torch.full((batch_size, 1), seq_len),
            "seq_lens_padded": torch.full((batch_size, 1), seq_len),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2)

        # Check shapes - should be [num_chunks, tokens_per_chunk, hidden_dim]
        assert result["input_ids"].shape == (2, 12, hidden_dim)
        assert result["position_ids"].shape == (2, 12)
        assert result["cu_seqlens"].shape[0] == 2


class TestProcessInputForTHDWithChunks:
    """Comprehensive tests for process_input_for_thd_with_chunks function."""

    def test_variable_length_cu_seqlens_padding(self):
        """Test that cu_seqlens with different lengths are padded correctly."""
        # Create a batch where different chunks will have different cu_seqlens lengths
        batch = {
            "input_ids": torch.tensor([
                [1, 2, 3, 99, 4, 5],  # 2 sequences (lengths 3, 2)
                [6, 7, 8, 9, 10, 11],  # 1 sequence (length 6)
                [12, 13, 14, 15, 99, 99],  # 1 sequence (length 4)
                [16, 17, 99, 18, 19, 20]  # 2 sequences (lengths 2, 3)
            ]),
            "labels": torch.randint(0, 100, (4, 6)),
            "position_ids": torch.tensor([
                [0, 1, 2, 0, 0, 1],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 0, 0],
                [0, 1, 0, 0, 1, 2]
            ]),
            "seq_lens": torch.tensor([[3, 2, -1000], [6, -1000, -1000], [4, -1000, -1000], [2, 3, -1000]]),
            "seq_lens_padded": torch.tensor([[4, 2, -1000], [6, -1000, -1000], [4, -1000, -1000], [3, 3, -1000]]),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2)

        # First chunk has 2+1=3 sequences, second chunk has 1+2=3 sequences
        # cu_seqlens should be [num_chunks, max_seqs_across_chunks+1]
        assert result["cu_seqlens"].shape[0] == 2

        # First chunk: [0, 3, 5, 11] (2 seqs from first batch, 1 seq from second batch)
        expected_cu_seqlens_0 = torch.tensor([0, 3, 5, 11], dtype=torch.int32)
        assert torch.equal(result["cu_seqlens"][0], expected_cu_seqlens_0)

        # Second chunk: [0, 4, 6, 9] (1 seq from third batch, 2 seqs from fourth batch)
        expected_cu_seqlens_1 = torch.tensor([0, 4, 6, 9], dtype=torch.int32)
        assert torch.equal(result["cu_seqlens"][1], expected_cu_seqlens_1)

    def test_single_chunk(self):
        """Test with num_chunks=1 (no actual chunking)."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7]]),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]]),
            "seq_lens": torch.tensor([[3], [3]]),
            "seq_lens_padded": torch.tensor([[3], [3]]),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=1)

        # With num_chunks=1, returns same format as process_input_for_thd (no chunk dimension)
        assert result["input_ids"].shape == (6,)
        assert result["labels"].shape == (6,)
        assert result["position_ids"].shape == (6,)
        assert result["cu_seqlens"].shape == (3,)
        assert torch.equal(result["cu_seqlens"], torch.tensor([0, 3, 6], dtype=torch.int32))

    def test_many_chunks(self):
        """Test with many chunks (num_chunks=8)."""
        batch_size = 16
        seq_len = 32
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            "seq_lens": torch.full((batch_size, 1), seq_len),
            "seq_lens_padded": torch.full((batch_size, 1), seq_len),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=8)

        # With 8 chunks, each chunk processes 2 batch items (16 / 8)
        tokens_per_chunk = 2 * seq_len
        assert result["input_ids"].shape == (8, tokens_per_chunk)
        assert result["labels"].shape == (8, tokens_per_chunk)
        assert result["cu_seqlens"].shape[0] == 8

        # Each chunk should have 3 cu_seqlens values: [0, 32, 64]
        for i in range(8):
            assert torch.equal(result["cu_seqlens"][i], torch.tensor([0, 32, 64], dtype=torch.int32))

    def test_non_tensor_key_preservation(self):
        """Test that non-tensor keys are preserved from the batch."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]),
            "seq_lens": torch.full((4, 1), 3),
            "seq_lens_padded": torch.full((4, 1), 3),
            "qkv_format": "thd",
            "metadata": {"batch_id": 123},
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2)

        # Non-tensor keys should be preserved
        assert "qkv_format" in result
        assert result["qkv_format"] == "thd"
        assert "metadata" in result
        assert result["metadata"] == {"batch_id": 123}

    def test_custom_seq_lens_padding_value(self):
        """Test with custom seq_lens_padding_value."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]),
            "seq_lens": torch.tensor([[3, -999], [3, -999], [3, -999], [3, -999]]),
            "seq_lens_padded": torch.tensor([[3, -999], [3, -999], [3, -999], [3, -999]]),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2, seq_lens_padding_value=-999)

        # Check that -999 is used for padding in cu_seqlens
        # cu_seqlens for each chunk should be [0, 3, 6] initially
        # After padding they remain [0, 3, 6]
        assert result["cu_seqlens"].shape == (2, 3)

    def test_cu_seqlens_padded_handling(self):
        """Test that cu_seqlens_padded is properly handled when present."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 99, 4, 5], [6, 7, 8, 9, 10, 11]] * 2),
            "labels": torch.tensor([[2, 3, 99, 4, 5, 6], [7, 8, 9, 10, 11, 12]] * 2),
            "position_ids": torch.tensor([[0, 1, 2, 0, 0, 1], [0, 1, 2, 3, 4, 5]] * 2),
            "seq_lens": torch.tensor([[3, 2], [6, -1000]] * 2),
            "seq_lens_padded": torch.tensor([[4, 2], [6, -1000]] * 2),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2)

        # Both cu_seqlens and cu_seqlens_padded should be present
        assert "cu_seqlens" in result
        assert "cu_seqlens_padded" in result
        assert result["cu_seqlens_padded"] is not None

        # They should have the same first dimension (num_chunks)
        assert result["cu_seqlens"].shape[0] == 2
        assert result["cu_seqlens_padded"].shape[0] == 2

    def test_chunks_equivalence_to_no_chunks(self):
        """Test that chunking with 1 chunk is equivalent to process_input_for_thd."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7]]),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]]),
            "seq_lens": torch.tensor([[3], [3]]),
            "seq_lens_padded": torch.tensor([[3], [3]]),
        }

        result_no_chunk = process_input_for_thd(batch)
        result_with_chunk = split_batch_into_thd_chunks(batch, num_chunks=1)

        # Results should match exactly (no batch dimension when num_chunks=1)
        assert torch.equal(result_with_chunk["input_ids"], result_no_chunk["input_ids"])
        assert torch.equal(result_with_chunk["labels"], result_no_chunk["labels"])
        assert torch.equal(result_with_chunk["position_ids"], result_no_chunk["position_ids"])
        assert torch.equal(result_with_chunk["cu_seqlens"], result_no_chunk["cu_seqlens"])

    def test_dtype_preservation_in_chunks(self):
        """Test that dtypes are preserved correctly through chunking."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.long),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=torch.long),
            "seq_lens": torch.tensor([[3], [3], [3], [3]], dtype=torch.long),
            "seq_lens_padded": torch.tensor([[3], [3], [3], [3]], dtype=torch.long),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2)

        assert result["input_ids"].dtype == torch.long
        assert result["labels"].dtype == torch.long
        assert result["position_ids"].dtype == torch.long
        assert result["cu_seqlens"].dtype == torch.int32
        assert result["cu_seqlens_padded"].dtype == torch.int32

    def test_position_ids_none_handling(self):
        """Test handling when position_ids is None."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]),
            "position_ids": None,
            "seq_lens": torch.tensor([[3], [3], [3], [3]]),
            "seq_lens_padded": torch.tensor([[3], [3], [3], [3]]),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2)

        # position_ids should be None in result
        assert result["position_ids"] is None

    def test_padding_mask_correctness(self):
        """Test that padding_mask is correctly generated."""
        batch = {
            "input_ids": torch.tensor([[0, 1, 2], [3, 0, 5], [0, 0, 8], [9, 10, 11]]),
            "labels": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]),
            "seq_lens": torch.tensor([[3], [3], [3], [3]]),
            "seq_lens_padded": torch.tensor([[3], [3], [3], [3]]),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2, padding_token_id=0)

        # Check padding_mask shape
        assert result["padding_mask"].shape == (2, 6)

        # Verify padding mask identifies 0s as padding
        # First chunk: [0, 1, 2, 3, 0, 5] -> mask: [T, F, F, F, T, F]
        expected_mask_0 = torch.tensor([True, False, False, False, True, False])
        assert torch.equal(result["padding_mask"][0], expected_mask_0)

    @pytest.mark.parametrize("num_chunks", [2, 4, 8])
    def test_different_chunk_sizes(self, num_chunks):
        """Test with different numbers of chunks."""
        batch_size = 16
        seq_len = 8
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            "seq_lens": torch.full((batch_size, 1), seq_len),
            "seq_lens_padded": torch.full((batch_size, 1), seq_len),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=num_chunks)

        items_per_chunk = batch_size // num_chunks
        tokens_per_chunk = items_per_chunk * seq_len

        assert result["input_ids"].shape == (num_chunks, tokens_per_chunk)
        assert result["cu_seqlens"].shape[0] == num_chunks

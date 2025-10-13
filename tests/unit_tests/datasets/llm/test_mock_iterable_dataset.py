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

import pytest
import torch

from nemo_automodel.components.datasets.llm.mock_iterable_dataset import MockIterableDataset


class TestMockIterableDataset:
    """Test suite for MockIterableDataset."""

    def test_initialization(self):
        """Test dataset initialization with default parameters."""
        dataset = MockIterableDataset(vocab_size=1000, seq_len=512)
        assert dataset.vocab_size == 1000
        assert dataset.seq_len == 512
        assert dataset.num_samples == 1000000
        assert dataset.batch_size == 1

    def test_initialization_with_custom_params(self):
        """Test dataset initialization with custom parameters."""
        dataset = MockIterableDataset(vocab_size=5000, seq_len=1024, num_samples=100, batch_size=4)
        assert dataset.vocab_size == 5000
        assert dataset.seq_len == 1024
        assert dataset.num_samples == 100
        assert dataset.batch_size == 4

    def test_len(self):
        """Test __len__ method returns correct number of samples."""
        dataset = MockIterableDataset(vocab_size=1000, seq_len=512, num_samples=50)
        assert len(dataset) == 50

    def test_iter_yields_correct_number_of_samples(self):
        """Test that iteration yields the expected number of samples."""
        num_samples = 10
        dataset = MockIterableDataset(vocab_size=1000, seq_len=512, num_samples=num_samples)
        samples = list(dataset)
        assert len(samples) == num_samples

    def test_sample_structure(self):
        """Test that each sample has the correct structure and keys."""
        dataset = MockIterableDataset(vocab_size=1000, seq_len=512, batch_size=2)
        sample = next(iter(dataset))

        # Check that all required keys are present
        assert "input_ids" in sample
        assert "labels" in sample
        assert "position_ids" in sample

    def test_sample_shapes_unbatched(self):
        """Test tensor shapes for unbatched samples (batch_size=1)."""
        vocab_size = 1000
        seq_len = 512
        batch_size = 1
        dataset = MockIterableDataset(vocab_size=vocab_size, seq_len=seq_len, batch_size=batch_size)
        sample = next(iter(dataset))

        assert sample["input_ids"].shape == (batch_size, seq_len)
        assert sample["labels"].shape == (batch_size, seq_len)
        assert sample["position_ids"].shape == (batch_size, seq_len)

    def test_sample_shapes_batched(self):
        """Test tensor shapes for batched samples."""
        vocab_size = 1000
        seq_len = 512
        batch_size = 4
        dataset = MockIterableDataset(vocab_size=vocab_size, seq_len=seq_len, batch_size=batch_size)
        sample = next(iter(dataset))

        assert sample["input_ids"].shape == (batch_size, seq_len)
        assert sample["labels"].shape == (batch_size, seq_len)
        assert sample["position_ids"].shape == (batch_size, seq_len)

    def test_input_ids_within_vocab_range(self):
        """Test that input_ids are within the valid vocabulary range."""
        vocab_size = 100
        seq_len = 50
        dataset = MockIterableDataset(vocab_size=vocab_size, seq_len=seq_len)
        sample = next(iter(dataset))

        assert sample["input_ids"].min() >= 0
        assert sample["input_ids"].max() < vocab_size

    def test_labels_are_shifted_correctly(self):
        """Test that labels are correctly shifted versions of input_ids."""
        vocab_size = 1000
        seq_len = 512
        batch_size = 2
        dataset = MockIterableDataset(vocab_size=vocab_size, seq_len=seq_len, batch_size=batch_size)
        sample = next(iter(dataset))

        input_ids = sample["input_ids"]
        labels = sample["labels"]

        # Labels should be input_ids shifted left by 1, with last position as -100
        assert torch.all(labels[:, :-1] == input_ids[:, 1:])
        assert torch.all(labels[:, -1] == -100)

    def test_position_ids_sequential(self):
        """Test that position_ids are sequential from 0 to seq_len-1."""
        vocab_size = 1000
        seq_len = 512
        batch_size = 2
        dataset = MockIterableDataset(vocab_size=vocab_size, seq_len=seq_len, batch_size=batch_size)
        sample = next(iter(dataset))

        expected_positions = torch.arange(seq_len)
        for batch_idx in range(batch_size):
            assert torch.all(sample["position_ids"][batch_idx] == expected_positions)

    def test_tensor_dtypes(self):
        """Test that tensors have the correct data types."""
        dataset = MockIterableDataset(vocab_size=1000, seq_len=512)
        sample = next(iter(dataset))

        # input_ids and labels should be integer types
        assert sample["input_ids"].dtype == torch.long or sample["input_ids"].dtype == torch.int64
        assert sample["labels"].dtype == torch.long or sample["labels"].dtype == torch.int64
        assert sample["position_ids"].dtype == torch.long or sample["position_ids"].dtype == torch.int64

    def test_multiple_iterations(self):
        """Test that multiple iterations through the dataset work correctly."""
        num_samples = 5
        dataset = MockIterableDataset(vocab_size=1000, seq_len=512, num_samples=num_samples)

        # First iteration
        samples1 = list(dataset)
        assert len(samples1) == num_samples

        # Second iteration
        samples2 = list(dataset)
        assert len(samples2) == num_samples

    def test_different_samples_have_different_tokens(self):
        """Test that consecutive samples generate different random tokens."""
        dataset = MockIterableDataset(vocab_size=1000, seq_len=512, num_samples=3)
        samples = list(dataset)

        # With high probability, random samples should be different
        # Check that not all samples are identical
        assert not torch.all(samples[0]["input_ids"] == samples[1]["input_ids"])
        assert not torch.all(samples[1]["input_ids"] == samples[2]["input_ids"])

    def test_large_vocab_size(self):
        """Test with a large vocabulary size."""
        vocab_size = 50000
        seq_len = 256
        dataset = MockIterableDataset(vocab_size=vocab_size, seq_len=seq_len, num_samples=2)
        sample = next(iter(dataset))

        assert sample["input_ids"].min() >= 0
        assert sample["input_ids"].max() < vocab_size

    def test_large_sequence_length(self):
        """Test with a large sequence length."""
        vocab_size = 1000
        seq_len = 8192
        dataset = MockIterableDataset(vocab_size=vocab_size, seq_len=seq_len, num_samples=1)
        sample = next(iter(dataset))

        assert sample["input_ids"].shape[1] == seq_len
        assert sample["labels"].shape[1] == seq_len
        assert sample["position_ids"].shape[1] == seq_len

    def test_large_batch_size(self):
        """Test with a large batch size."""
        vocab_size = 1000
        seq_len = 512
        batch_size = 32
        dataset = MockIterableDataset(vocab_size=vocab_size, seq_len=seq_len, batch_size=batch_size, num_samples=1)
        sample = next(iter(dataset))

        assert sample["input_ids"].shape[0] == batch_size
        assert sample["labels"].shape[0] == batch_size
        assert sample["position_ids"].shape[0] == batch_size

    def test_edge_case_single_token_sequence(self):
        """Test edge case with sequence length of 1."""
        vocab_size = 100
        seq_len = 1
        dataset = MockIterableDataset(vocab_size=vocab_size, seq_len=seq_len, num_samples=1)
        sample = next(iter(dataset))

        assert sample["input_ids"].shape[1] == 1
        assert sample["labels"].shape[1] == 1
        assert sample["position_ids"].shape[1] == 1
        # For seq_len=1, label should be -100 (padding)
        assert sample["labels"][0, 0] == -100

    def test_edge_case_vocab_size_one(self):
        """Test edge case with vocabulary size of 1."""
        vocab_size = 1
        seq_len = 10
        dataset = MockIterableDataset(vocab_size=vocab_size, seq_len=seq_len, num_samples=1)
        sample = next(iter(dataset))

        # All tokens should be 0 (the only valid token)
        assert torch.all(sample["input_ids"] == 0)

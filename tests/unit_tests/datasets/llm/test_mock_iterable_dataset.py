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

from nemo_automodel.components.datasets.llm.mock_iterable_dataset import MockIterableDataset, MockIterableDatasetConfig


class TestMockIterableDataset:
    """Test suite for MockIterableDataset."""

    def test_initialization(self):
        """Test dataset initialization with default parameters."""
        dataset = MockIterableDataset(vocab_size=1000, seq_len=512)
        assert dataset.vocab_size == 1000
        assert dataset.seq_len == 512
        assert dataset.num_samples == 1000000
        assert dataset.batch_size == 1
        assert dataset.seed == 0

    def test_initialization_with_custom_params(self):
        """Test dataset initialization with custom parameters."""
        dataset = MockIterableDataset(vocab_size=5000, seq_len=1024, num_samples=100, batch_size=4, seed=123)
        assert dataset.vocab_size == 5000
        assert dataset.seq_len == 1024
        assert dataset.num_samples == 100
        assert dataset.batch_size == 4
        assert dataset.seed == 123

    def test_config_build_forwards_seed(self):
        """Test that the typed config owns deterministic dataset construction."""
        dataset = MockIterableDatasetConfig(vocab_size=50, seq_len=8, num_samples=2, batch_size=3, seed=456).build()

        assert dataset.seed == 456
        assert dataset.batch_size == 3

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
        """Test that each iterator replays the deterministic stream from its beginning."""
        num_samples = 5
        dataset = MockIterableDataset(vocab_size=1000, seq_len=512, num_samples=num_samples)

        # First iteration
        samples1 = list(dataset)
        assert len(samples1) == num_samples

        # Second iteration
        samples2 = list(dataset)
        assert len(samples2) == num_samples
        for first, second in zip(samples1, samples2):
            assert torch.equal(first["input_ids"], second["input_ids"])

    def test_seed_controls_generated_stream(self):
        """Test that equal seeds replay and different seeds distinguish token streams."""
        first = next(iter(MockIterableDataset(vocab_size=1000, seq_len=64, batch_size=2, seed=17)))
        replay = next(iter(MockIterableDataset(vocab_size=1000, seq_len=64, batch_size=2, seed=17)))
        different = next(iter(MockIterableDataset(vocab_size=1000, seq_len=64, batch_size=2, seed=18)))

        assert torch.equal(first["input_ids"], replay["input_ids"])
        assert first["mock_data_fingerprint"] == replay["mock_data_fingerprint"]
        assert not torch.equal(first["input_ids"], different["input_ids"])
        assert first["mock_data_fingerprint"] != different["mock_data_fingerprint"]

    def test_shard_replays_for_peers_and_distinguishes_dp_ranks(self):
        """Test that peers on one DP rank agree while separate DP ranks receive unique data."""
        dataset = MockIterableDataset(vocab_size=1000, seq_len=64, batch_size=2, seed=123)

        peer_a = next(iter(dataset.shard(4, 2)))
        peer_b = next(iter(dataset.shard(4, 2)))
        other_rank = next(iter(dataset.shard(4, 3)))

        assert torch.equal(peer_a["input_ids"], peer_b["input_ids"])
        assert not torch.equal(peer_a["input_ids"], other_rank["input_ids"])

    def test_shard_does_not_mutate_source_dataset(self):
        """Test that deriving a DP shard leaves the source stream unchanged."""
        dataset = MockIterableDataset(vocab_size=1000, seq_len=64, batch_size=2, seed=123)
        expected = next(iter(MockIterableDataset(vocab_size=1000, seq_len=64, batch_size=2, seed=123)))

        dataset.shard(4, 2)
        actual = next(iter(dataset))

        assert torch.equal(actual["input_ids"], expected["input_ids"])

    @pytest.mark.parametrize(("num_shards", "index"), [(0, 0), (4, -1), (4, 4), (True, 0), (4, False)])
    def test_shard_rejects_invalid_coordinates(self, num_shards, index):
        """Test that invalid data-parallel shard coordinates fail clearly."""
        dataset = MockIterableDataset()

        with pytest.raises(ValueError):
            dataset.shard(num_shards, index)

    def test_iteration_does_not_consume_global_torch_rng(self):
        """Test that the dataset-owned generator leaves the process RNG untouched."""
        torch.manual_seed(987)
        state = torch.get_rng_state()

        next(iter(MockIterableDataset(vocab_size=1000, seq_len=64, batch_size=2, seed=123)))

        assert torch.equal(torch.get_rng_state(), state)

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

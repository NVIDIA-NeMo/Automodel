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
from datasets import Dataset

from nemo_automodel.components.datasets.llm.neat_packing import (
    CROSS_ENTROPY_IGNORE_IDX,
    _build_packed_sample,
    greedy_knapsack,
    neat_pack_dataset,
)
from nemo_automodel.components.datasets.utils import (
    _indexed_mask_to_4d_block_causal,
    neat_packed_collater,
)


class TestGreedyKnapsack:
    def test_basic(self):
        """All samples fit into bins correctly."""
        lengths = [3, 2, 4, 1]
        bins = greedy_knapsack(lengths, max_length=5)

        # Check every index appears exactly once
        all_indices = sorted(idx for b in bins for idx in b)
        assert all_indices == [0, 1, 2, 3]

        # Check no bin exceeds capacity
        for b in bins:
            assert sum(lengths[i] for i in b) <= 5

    def test_efficiency(self):
        """Greedy knapsack should produce fewer bins than sequential first-fit."""
        lengths = [3, 3, 2, 2, 1, 1, 1, 1]
        max_length = 5

        bins = greedy_knapsack(lengths, max_length)

        # Optimal: [3,2], [3,2], [1,1,1,1] = 3 bins
        # Sequential first-fit: [3], [3,2], [2,1,1], [1,1] = 4 bins
        assert len(bins) <= 4  # should be 3 or better

    def test_single_sample_per_bin(self):
        """Each sample equals max_length."""
        lengths = [5, 5, 5]
        bins = greedy_knapsack(lengths, max_length=5)
        assert len(bins) == 3

    def test_skip_oversized(self):
        """Samples larger than max_length are skipped."""
        lengths = [3, 10, 2]
        bins = greedy_knapsack(lengths, max_length=5)
        all_indices = sorted(idx for b in bins for idx in b)
        assert 1 not in all_indices  # index 1 (length=10) skipped
        assert 0 in all_indices
        assert 2 in all_indices

    def test_empty(self):
        bins = greedy_knapsack([], max_length=5)
        assert bins == []


class TestBuildPackedSample:
    def test_basic(self):
        samples = [
            {"input_ids": [1, 2, 3], "labels": [-100, -100, 10]},
            {"input_ids": [4, 5], "labels": [20, 30]},
        ]
        result = _build_packed_sample(samples, pack_size=8, padding_idx=0)

        assert result["input_ids"].shape == (8,)
        assert result["labels"].shape == (8,)
        assert result["attention_mask"].shape == (8,)
        assert result["position_ids"].shape == (8,)

        # Check input_ids: [1,2,3, 4,5, 0,0,0]
        assert result["input_ids"].tolist() == [1, 2, 3, 4, 5, 0, 0, 0]

        # Check labels: [-100,-100,10, 20,30, -100,-100,-100]
        assert result["labels"].tolist() == [-100, -100, 10, 20, 30, -100, -100, -100]

        # Check indexed attention mask: [1,1,1, 2,2, 0,0,0]
        assert result["attention_mask"].tolist() == [1, 1, 1, 2, 2, 0, 0, 0]

        # Check position_ids reset: [0,1,2, 0,1, 0,0,0]
        assert result["position_ids"].tolist() == [0, 1, 2, 0, 1, 0, 0, 0]


class TestNeatPackDataset:
    def _make_dataset(self, samples):
        return Dataset.from_dict(
            {
                "input_ids": [s["input_ids"] for s in samples],
                "labels": [s["labels"] for s in samples],
            }
        )

    def test_end_to_end(self):
        samples = [
            {"input_ids": [1, 2, 3], "labels": [10, 20, 30]},
            {"input_ids": [4, 5], "labels": [40, 50]},
            {"input_ids": [6, 7, 8, 9], "labels": [60, 70, 80, 90]},
        ]
        ds = self._make_dataset(samples)

        packed = neat_pack_dataset(ds, split="train", pack_size=6, padding_idx=0)

        # Total tokens = 3+2+4 = 9, pack_size=6 -> at least 2 packs
        assert len(packed) >= 2

        # Check all packs have correct size
        for i in range(len(packed)):
            assert len(packed[i]["input_ids"]) == 6
            assert len(packed[i]["labels"]) == 6
            assert len(packed[i]["attention_mask"]) == 6
            assert len(packed[i]["position_ids"]) == 6

    def test_indexed_attention_mask(self):
        """Attention mask uses 1-based indexing, 0 for padding."""
        samples = [
            {"input_ids": [1, 2], "labels": [10, 20]},
            {"input_ids": [3, 4], "labels": [30, 40]},
        ]
        ds = self._make_dataset(samples)

        packed = neat_pack_dataset(ds, split="train", pack_size=6, padding_idx=0)

        # Both samples should fit in one pack
        assert len(packed) == 1
        mask = packed[0]["attention_mask"]
        if isinstance(mask, torch.Tensor):
            mask = mask.tolist()

        # [1,1, 2,2, 0,0] — 1-based, 0=padding
        assert mask == [1, 1, 2, 2, 0, 0]

    def test_position_ids_reset(self):
        """Position IDs reset at each sample boundary."""
        samples = [
            {"input_ids": [1, 2, 3], "labels": [10, 20, 30]},
            {"input_ids": [4, 5], "labels": [40, 50]},
        ]
        ds = self._make_dataset(samples)

        packed = neat_pack_dataset(ds, split="train", pack_size=6, padding_idx=0)

        pos = packed[0]["position_ids"]
        if isinstance(pos, torch.Tensor):
            pos = pos.tolist()
        assert pos == [0, 1, 2, 0, 1, 0]

    def test_labels_preserved(self):
        """Labels with -100 masking are preserved through packing."""
        samples = [
            {"input_ids": [1, 2, 3], "labels": [-100, -100, 30]},
        ]
        ds = self._make_dataset(samples)

        packed = neat_pack_dataset(ds, split="train", pack_size=4, padding_idx=0)

        labels = packed[0]["labels"]
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        assert labels[:3] == [-100, -100, 30]
        assert labels[3] == CROSS_ENTROPY_IGNORE_IDX

    def test_drop_long_samples(self):
        samples = [
            {"input_ids": [1, 2, 3, 4, 5, 6], "labels": [10, 20, 30, 40, 50, 60]},
            {"input_ids": [7, 8], "labels": [70, 80]},
        ]
        ds = self._make_dataset(samples)

        packed = neat_pack_dataset(
            ds,
            split="train",
            pack_size=4,
            padding_idx=0,
            drop_long_samples=True,
        )
        # Only the short sample should remain
        assert len(packed) == 1

    def test_drop_long_samples_raises(self):
        samples = [
            {"input_ids": [1, 2, 3, 4, 5], "labels": [10, 20, 30, 40, 50]},
        ]
        ds = self._make_dataset(samples)

        with pytest.raises(ValueError, match="too long"):
            neat_pack_dataset(ds, split="train", pack_size=3, padding_idx=0)

    def test_max_packs(self):
        samples = [{"input_ids": [i], "labels": [i * 10]} for i in range(20)]
        ds = self._make_dataset(samples)

        packed = neat_pack_dataset(
            ds,
            split="train",
            pack_size=3,
            padding_idx=0,
            max_packs=2,
        )
        assert len(packed) == 2

    def test_loss_mask(self):
        """loss_mask should set corresponding labels to -100."""
        ds = Dataset.from_dict(
            {
                "input_ids": [[1, 2, 3]],
                "labels": [[10, 20, 30]],
                "loss_mask": [[0, 1, 1]],
            }
        )

        packed = neat_pack_dataset(ds, split="train", pack_size=4, padding_idx=0)
        labels = packed[0]["labels"]
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        assert labels[0] == -100  # masked by loss_mask
        assert labels[1] == 20
        assert labels[2] == 30


class TestIndexedMaskTo4dBlockCausal:
    def test_single_sample(self):
        """Single sample, no padding."""
        mask = torch.tensor([[1, 1, 1]])  # [1, 3]
        result = _indexed_mask_to_4d_block_causal(mask)
        assert result.shape == (1, 1, 3, 3)

        expected = torch.tensor(
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ]
        )
        assert torch.equal(result[0, 0], expected)

    def test_two_sequences_with_padding(self):
        """Two sub-sequences and padding."""
        mask = torch.tensor([[1, 1, 2, 2, 0]])  # [1, 5]
        result = _indexed_mask_to_4d_block_causal(mask)
        assert result.shape == (1, 1, 5, 5)

        r = result[0, 0]
        # First sub-seq (positions 0,1) — causal within block
        assert r[0, 0] == True
        assert r[1, 0] == True
        assert r[1, 1] == True

        # Second sub-seq (positions 2,3) — causal within block
        assert r[2, 2] == True
        assert r[3, 2] == True
        assert r[3, 3] == True

        # Cross-block should be False
        assert r[2, 0] == False
        assert r[2, 1] == False
        assert r[0, 2] == False

        # Padding (position 4) — all False
        assert r[4, :].sum() == 0
        assert r[:, 4].sum() == 0

    def test_batch(self):
        mask = torch.tensor(
            [
                [1, 1, 2, 0],
                [1, 1, 1, 1],
            ]
        )
        result = _indexed_mask_to_4d_block_causal(mask)
        assert result.shape == (2, 1, 4, 4)

    def test_cross_sample_fully_invisible(self):
        """Exhaustive check: no position in sample X can attend to any position in sample Y."""
        # 3 sub-sequences of lengths 3, 2, 1 + 2 padding
        mask = torch.tensor([[1, 1, 1, 2, 2, 3, 0, 0]])
        result = _indexed_mask_to_4d_block_causal(mask)
        r = result[0, 0]  # [8, 8]

        # Define sample boundaries
        sample_ranges = {
            "A": range(0, 3),  # positions 0,1,2
            "B": range(3, 5),  # positions 3,4
            "C": range(5, 6),  # position 5
            "pad": range(6, 8),
        }

        # Cross-sample: every pair (X, Y) where X != Y must be fully invisible
        for name_q, range_q in sample_ranges.items():
            for name_k, range_k in sample_ranges.items():
                if name_q == name_k:
                    continue
                for i in range_q:
                    for j in range_k:
                        assert r[i, j] == False, (
                            f"Position {i} ({name_q}) should NOT attend to position {j} ({name_k}), but mask is True"
                        )

        # Within-sample: causal holds (i can attend to j iff j <= i)
        for name, rng in sample_ranges.items():
            if name == "pad":
                continue
            for i in rng:
                for j in rng:
                    if j <= i:
                        assert r[i, j] == True, f"Position {i} ({name}) should attend to {j}, but mask is False"
                    else:
                        assert r[i, j] == False

    def test_position_ids_and_mask_consistency(self):
        """Position IDs reset + mask isolation: full end-to-end check."""
        samples = [
            {"input_ids": [10, 20, 30], "labels": [-100, -100, 100]},
            {"input_ids": [40, 50], "labels": [200, 300]},
        ]
        ds = Dataset.from_dict(
            {
                "input_ids": [s["input_ids"] for s in samples],
                "labels": [s["labels"] for s in samples],
            }
        )
        packed = neat_pack_dataset(ds, split="train", pack_size=7, padding_idx=0)

        pos = packed[0]["position_ids"]
        mask = packed[0]["attention_mask"]
        if isinstance(pos, torch.Tensor):
            pos = pos.tolist()
        if isinstance(mask, torch.Tensor):
            mask = mask.tolist()

        # Position IDs reset: [0,1,2, 0,1, 0,0]
        assert pos == [0, 1, 2, 0, 1, 0, 0]

        # Indexed mask: [1,1,1, 2,2, 0,0]
        assert mask == [1, 1, 1, 2, 2, 0, 0]

        # 4D mask: cross-sample invisible
        mask_t = torch.tensor([mask])
        mask_4d = _indexed_mask_to_4d_block_causal(mask_t)[0, 0]

        # Sample B (pos 3,4) cannot see Sample A (pos 0,1,2)
        assert mask_4d[3, 0] == False
        assert mask_4d[3, 1] == False
        assert mask_4d[3, 2] == False
        assert mask_4d[4, 0] == False

        # Sample A cannot see Sample B
        assert mask_4d[0, 3] == False
        assert mask_4d[2, 3] == False


class TestNeatPackedCollater:
    def test_collater_output(self):
        batch = [
            {
                "input_ids": torch.tensor([1, 2, 3, 0]),
                "labels": torch.tensor([10, 20, 30, -100]),
                "attention_mask": torch.tensor([1, 1, 1, 0]),
                "position_ids": torch.tensor([0, 1, 2, 0]),
            },
            {
                "input_ids": torch.tensor([4, 5, 6, 7]),
                "labels": torch.tensor([40, 50, 60, 70]),
                "attention_mask": torch.tensor([1, 1, 1, 1]),
                "position_ids": torch.tensor([0, 1, 2, 3]),
            },
        ]
        result = neat_packed_collater(batch)

        assert result["input_ids"].shape == (2, 4)
        assert result["labels"].shape == (2, 4)
        assert result["position_ids"].shape == (2, 4)
        assert result["attention_mask"].shape == (2, 1, 4, 4)
        assert result["attention_mask"].dtype == torch.bool

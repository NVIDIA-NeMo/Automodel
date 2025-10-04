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

import pytest
from datasets import Dataset

from nemo_automodel.components.datasets.llm.packed_sequence import pack_dataset


@pytest.fixture
def base_dataset():
    """Sample dataset with 4 sequences of varying lengths"""
    return Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3], [4, 5, 6, 7], [8, 9], [10, 11, 12, 13, 14]],
            "labels": [[1, 2, 3], [4, 5, 6, 7], [8, 9], [10, 11, 12, 13, 14]],
        }
    )


def test_basic_packing(base_dataset):
    """Test basic packing without splitting across packs"""
    packed_ds = pack_dataset(
        base_dataset, split="train", packed_sequence_size=10, split_across_pack=False, max_packs=None
    )

    assert len(packed_ds) == 2
    # Check packed_ds[0] is [1,2,3,4,5,6,7,8,9] plus [0] for padding
    assert packed_ds[0]["input_ids"] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
    # seq_lens contains only attention-participating lengths; trailing padding doesn't add a new sequence
    assert packed_ds[0]["seq_lens"] == [3, 4, 2]
    # seq_lens_padded augments the last sequence span with trailing pack padding
    assert packed_ds[0]["seq_lens_padded"] == [3, 4, 3]
    # pos_ids of the last seq continue into padded tokens.
    # See packed sequence implementation: nemo_automodel/datasets/llm/packed_sequence.py#L228-L234
    assert packed_ds[0]["position_ids"] == [0, 1, 2, 0, 1, 2, 3, 0, 1, 2]
    assert packed_ds[1]["input_ids"] == [10, 11, 12, 13, 14, 0, 0, 0, 0, 0]
    # labels are padded with CROSS_ENTROPY_IGNORE_IDX i.e -100
    assert packed_ds[1]["labels"] == [10, 11, 12, 13, 14, -100, -100, -100, -100, -100]


@pytest.mark.parametrize(
    "split_across_pack,max_packs,expected",
    [
        (True, 2, 2),
        (False, 3, 3),
    ],
)
def test_split_across_pack(base_dataset, split_across_pack, max_packs, expected):
    """Test splitting sequences across packs with different configurations"""
    packed_ds = pack_dataset(
        base_dataset, split="train", packed_sequence_size=5, split_across_pack=split_across_pack, max_packs=max_packs
    )
    assert len(packed_ds) == expected


def test_loss_mask_handling():
    """Test handling of loss masks with different configurations"""
    ds_with_mask = Dataset.from_dict(
        {"input_ids": [[1, 2, 3], [4, 5, 6]], "labels": [[1, 2, 3], [4, 5, 6]], "loss_mask": [[1, 1, 0], [1, 1, 1]]}
    )

    packed_ds = pack_dataset(
        ds_with_mask, split="train", packed_sequence_size=5, split_across_pack=False, max_packs=None
    )
    assert packed_ds[0]["labels"][-3:] == [-100] * 3
    assert packed_ds[0]["labels"][:2] != [-100] * 2
    assert packed_ds[1]["labels"][:3] != [-100] * 3
    assert packed_ds[1]["labels"][-2:] == [-100] * 2


def test_position_id_wrapping(base_dataset):
    """Test position ID generation with wrapping"""
    packed_ds = pack_dataset(
        base_dataset, split="train", packed_sequence_size=5, split_across_pack=False, max_packs=None
    )
    assert packed_ds[0]["position_ids"] == [0, 1, 2, 3, 4]


def test_exact_fit():
    """Test sequence that exactly fills pack size"""
    exact_fit_ds = Dataset.from_dict({"input_ids": [[1, 2, 3, 4, 5]], "labels": [[1, 2, 3, 4, 5]]})

    packed_ds = pack_dataset(
        exact_fit_ds, split="train", packed_sequence_size=5, split_across_pack=False, max_packs=None
    )
    assert len(packed_ds) == 1
    assert packed_ds[0]["input_ids"] == [1, 2, 3, 4, 5]


def test_error_on_oversized_sequence():
    """Test error when sequence is too long and split disabled"""
    oversized_ds = Dataset.from_dict({"input_ids": [[1, 2, 3, 4, 5, 6]], "labels": [[1, 2, 3, 4, 5, 6]]})

    with pytest.raises(ValueError):
        pack_dataset(oversized_ds, split="train", packed_sequence_size=5, split_across_pack=False, max_packs=None)


def test_seq_lens_padded():
    """
    Test seq_lens_padded handling for identifier tokens that don't participate in attention.

    Example: [a a a 1 b b 2 2 c c c c 3]
    - seq_lens: [3, 2, 4] (actual attention tokens)
    - seq_lens_padded: [4, 4, 5] (includes identifier tokens 1, 2 2, 3)

    Uses identifier_token_id=99 to identify non-attention tokens.
    """
    # Dataset with sequences and identifier tokens
    ds_with_identifiers = Dataset.from_dict(
        {
            "input_ids": [
                [1, 2, 3, 99],  # [a a a 1] - 3 tokens + 1 identifier (99)
                [4, 5, 99, 99],  # [b b 2 2] - 2 tokens + 2 identifiers (99)
                [6, 7, 8, 9, 99],  # [c c c c 3] - 4 tokens + 1 identifier (99)
            ],
            "labels": [[1, 2, 3, -100], [4, 5, -100, -100], [6, 7, 8, 9, -100]],
            "seq_len_padded": [4, 4, 5],  # Includes identifier tokens
        }
    )

    packed_ds = pack_dataset(
        ds_with_identifiers,
        split="train",
        packed_sequence_size=13,
        split_across_pack=False,
        max_packs=None,
        padding_idx=99,
    )

    assert len(packed_ds) == 1
    # Verify seq_lens (attention tokens only: 3 + 2 + 4 = 9 tokens)
    # Note: packed_ds is a HF Dataset, so items are already lists/arrays not tensors
    import torch
    seq_lens = packed_ds[0]["seq_lens"]
    if isinstance(seq_lens, torch.Tensor):
        seq_lens = seq_lens.tolist()
    assert seq_lens == [3, 2, 4]

    # Verify seq_lens_padded (includes identifiers: 4 + 4 + 5 = 13 tokens)
    assert "seq_lens_padded" in packed_ds[0]
    seq_lens_padded = packed_ds[0]["seq_lens_padded"]
    if isinstance(seq_lens_padded, torch.Tensor):
        seq_lens_padded = seq_lens_padded.tolist()
    assert seq_lens_padded == [4, 4, 5]

    # Verify all tokens are packed correctly
    input_ids = packed_ds[0]["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids.tolist()
    assert input_ids == [1, 2, 3, 99, 4, 5, 99, 99, 6, 7, 8, 9, 99]


def test_seq_lens_padded_with_split():
    """Test seq_lens_padded when splitting across packs"""
    ds_with_identifiers = Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 99], [4, 5, 6, 7, 8, 99]],
            "labels": [[1, 2, 3, -100], [4, 5, 6, 7, 8, -100]],
            "seq_len_padded": [4, 6],
        }
    )

    packed_ds = pack_dataset(
        ds_with_identifiers,
        split="train",
        packed_sequence_size=7,
        split_across_pack=True,
        max_packs=None,
        padding_idx=99,
    )

    # First pack should have seq_lens and seq_lens_padded
    assert "seq_lens_padded" in packed_ds[0]
    # Second pack should also have seq_lens_padded
    assert "seq_lens_padded" in packed_ds[1]


def test_seq_lens_padded_always_present(base_dataset):
    """seq_lens_padded is always generated and present in outputs"""
    packed_ds = pack_dataset(
        base_dataset,
        split="train",
        packed_sequence_size=10,
        split_across_pack=False,
        max_packs=None,
    )

    # Verify seq_lens_padded is in output
    assert "seq_lens_padded" in packed_ds[0]

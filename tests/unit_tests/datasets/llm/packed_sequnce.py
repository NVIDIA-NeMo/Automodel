# Copyright (c) 2025, NVIDIA CORPORATION.
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
from nemo_automodel.datasets.llm.hf_dataset_packed_sequence import HFDatasetPackedSequenceHelper

@pytest.fixture
def base_dataset():
    return Dataset.from_dict({
        "input_ids": [[1,2,3], [4,5,6,7], [8,9], [10,11,12,13,14]],
        "labels": [[1,2,3], [4,5,6,7], [8,9], [10,11,12,13,14]],
    })

def test_basic_packing(base_dataset):
    """Test basic packing without splitting across packs"""
    helper = HFDatasetPackedSequenceHelper(base_dataset, "train")
    packed_ds = helper.pack(
        packed_sequence_size=10,
        split_across_pack=False,
        max_packs=None
    )

    assert len(packed_ds) == 2
    # Check packed_ds[0] is [1,2,3,4,5,6,7,8,9] plus [0] for padding
    assert packed_ds[0]["input_ids"] == [1,2,3,4,5,6,7,8,9,0]
    # Padding len is included in the seq_lens array. padded tokens are treated as part of the seq len, ref: 
    # https://github.com/NVIDIA-NeMo/NeMo-Automodel/blob/athitten/generic_packed_seq/nemo_automodel/datasets/llm/hf_dataset_packed_sequence.py#L221-L226
    assert packed_ds[0]["seq_lens"] == [3,4,2,1]
    # pos_ids of the last seq continue into padded tokens. ref: https://github.com/NVIDIA-NeMo/NeMo-Automodel/blob/athitten/generic_packed_seq/nemo_automodel/datasets/llm/hf_dataset_packed_sequence.py#L228-L234
    assert packed_ds[0]["position_ids"] == [0, 1, 2, 0, 1, 2, 3, 0, 1, 2]
    assert packed_ds[1]["input_ids"] == [10,11,12,13,14,0,0,0,0,0]
    # labels are padded with CROSS_ENTROPY_IGNORE_IDX i.e -100
    assert packed_ds[1]["labels"] == [10,11,12,13,14,-100,-100,-100,-100,-100]


@pytest.mark.parametrize("split_across_pack,max_packs,expected", [
    (True, 2, 2),
    (False, 3, 3),
])
def test_split_across_pack(base_dataset, split_across_pack, max_packs, expected):
    """Test splitting sequences across packs with different configurations"""
    helper = HFDatasetPackedSequenceHelper(base_dataset, "train")
    packed_ds = helper.pack(
        packed_sequence_size=5,
        split_across_pack=split_across_pack,
        max_packs=max_packs
    )
    assert len(packed_ds) == expected


def test_loss_mask_handling():
    """Test handling of loss masks with different configurations"""
    ds_with_mask = Dataset.from_dict({
        "input_ids": [[1,2,3], [4,5,6]],
        "labels": [[1,2,3], [4,5,6]],
        "loss_mask": [[1,1,0], [1,1,1]]
    })
    
    helper = HFDatasetPackedSequenceHelper(ds_with_mask, "train")
    packed_ds = helper.pack(
        packed_sequence_size=5,
        split_across_pack=False,
        max_packs=None
    )
    
    assert packed_ds[0]["loss_mask"] == [1,1,0,0,0]
    assert packed_ds[1]["loss_mask"] == [1,1,1,0,0]


def test_position_id_wrapping(base_dataset):
    """Test position ID generation with wrapping"""
    helper = HFDatasetPackedSequenceHelper(base_dataset, "train")
    packed_ds = helper.pack(
        packed_sequence_size=5,
        split_across_pack=False,
        max_packs=None
    )
    assert packed_ds[0]["position_ids"] == [0,1,2,3,4]


def test_exact_fit():
    """Test sequence that exactly fills pack size"""
    exact_fit_ds = Dataset.from_dict({
        "input_ids": [[1,2,3,4,5]],
        "labels": [[1,2,3,4,5]]
    })
    
    helper = HFDatasetPackedSequenceHelper(exact_fit_ds, "train")
    packed_ds = helper.pack(
        packed_sequence_size=5,
        split_across_pack=False,
        max_packs=None
    )
    assert len(packed_ds) == 1
    assert packed_ds[0]["input_ids"] == [1,2,3,4,5]

def test_error_on_oversized_sequence():
    """Test error when sequence is too long and split disabled"""
    oversized_ds = Dataset.from_dict({
        "input_ids": [[1,2,3,4,5,6]],
        "labels": [[1,2,3,4,5,6]]
    })
    
    helper = HFDatasetPackedSequenceHelper(oversized_ds, "train")
    with pytest.raises(ValueError):
        helper.pack(
            packed_sequence_size=5,
            split_across_pack=False,
            max_packs=None
        )

test_exact_fit()
test_error_on_oversized_sequence()
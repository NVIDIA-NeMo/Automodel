# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import pickle

import numpy
import pytest
import torch
from torch.utils.data import DataLoader

from nemo_automodel.components.datasets.llm.megatron.indexed_dataset import (
    IndexedDataset,
    IndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)


@pytest.fixture
def indexed_dataset_prefix(tmp_path):
    prefix = str(tmp_path / "tiny")
    builder = IndexedDatasetBuilder(get_bin_path(prefix))
    builder.add_item(torch.tensor([11, 12, 13], dtype=torch.int64))
    builder.end_document()
    builder.add_item(torch.tensor([21, 22], dtype=torch.int64))
    builder.end_document()
    builder.finalize(get_idx_path(prefix))
    return prefix


@pytest.mark.parametrize("mmap", [True, False])
def test_pickle_round_trip_reopens_readers(indexed_dataset_prefix, mmap):
    dataset = IndexedDataset(indexed_dataset_prefix, mmap=mmap)

    restored = pickle.loads(pickle.dumps(dataset))  # noqa: S301

    assert restored.bin_reader is not dataset.bin_reader
    assert restored.index is not dataset.index
    numpy.testing.assert_array_equal(restored[0], dataset[0])
    numpy.testing.assert_array_equal(restored[1], dataset[1])


def test_spawn_dataloader_worker_reads_mmap_dataset(indexed_dataset_prefix):
    dataset = IndexedDataset(indexed_dataset_prefix, mmap=True)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        multiprocessing_context="spawn",
    )

    samples = list(dataloader)

    assert len(samples) == 2
    torch.testing.assert_close(samples[0], torch.tensor([11, 12, 13], dtype=torch.int32))
    torch.testing.assert_close(samples[1], torch.tensor([21, 22], dtype=torch.int32))

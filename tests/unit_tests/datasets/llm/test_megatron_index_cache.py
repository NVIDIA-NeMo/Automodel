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

"""Cache reuse accepts numeric Megatron indices and rejects pickled files."""

import pickle
from types import SimpleNamespace

import multiprocess
import numpy as np
import pytest

from nemo_automodel.components.datasets.llm.megatron.builder import BlendedDataset
from nemo_automodel.components.datasets.llm.megatron.gpt_dataset import GPTDataset, Split


def _read_worker_indices(dataset, names):
    """Read reopened numeric indices and report their mapping contract in a worker."""
    return [
        (isinstance(getattr(dataset, name), np.memmap), getattr(dataset, name).mode, getattr(dataset, name).tolist())
        for name in names
    ]


@pytest.mark.parametrize(
    "dataset_class,names",
    [
        (GPTDataset, ("document_index", "sample_index", "shuffle_index")),
        (BlendedDataset, ("dataset_index", "dataset_sample_index")),
    ],
)
def test_worker_reopens_cached_indices(tmp_path, dataset_class, names):
    dataset = dataset_class.__new__(dataset_class)
    for name in names:
        path = tmp_path / f"{name}.npy"
        values = np.arange(4, dtype=np.int32)
        if name == "sample_index":
            values = values.reshape(2, 2)
        np.save(path, values, allow_pickle=False)
        setattr(dataset, name, np.load(path, allow_pickle=False, mmap_mode="r"))
    with multiprocess.Pool(1) as pool:
        result = pool.apply(_read_worker_indices, (dataset, names))
    assert result == [(True, "r", [[0, 1], [2, 3]] if name == "sample_index" else [0, 1, 2, 3]) for name in names]
    assert _read_worker_indices(dataset, names) == result


@pytest.mark.parametrize("poisoned_suffix", ["document_index", "sample_index", "shuffle_index"])
def test_gpt_index_cache_rejects_pickle(tmp_path, poisoned_suffix):
    dataset = GPTDataset.__new__(GPTDataset)
    dataset.config = SimpleNamespace(path_to_cache=str(tmp_path))
    dataset.unique_description_hash = "test"
    dataset.index_split = Split.train

    prefix = "test-GPTDataset-train"
    (tmp_path / f"{prefix}-description.txt").write_text("test")
    arrays = {
        "document_index": np.array([0], dtype=np.int32),
        "sample_index": np.array([[0, 0], [0, 1]], dtype=np.int32),
        "shuffle_index": np.array([0], dtype=np.uint32),
    }
    for suffix, array in arrays.items():
        np.save(tmp_path / f"{prefix}-{suffix}.npy", array, allow_pickle=False)

    loaded = dataset._build_document_sample_shuffle_indices()
    for actual, expected in zip(loaded, arrays.values()):
        np.testing.assert_array_equal(actual, expected)

    (tmp_path / f"{prefix}-{poisoned_suffix}.npy").write_bytes(pickle.dumps(arrays[poisoned_suffix]))
    with pytest.raises(ValueError, match="pickled"):
        dataset._build_document_sample_shuffle_indices()


@pytest.mark.parametrize("poisoned_suffix", ["dataset_index", "dataset_sample_index"])
def test_blended_index_cache_rejects_pickle(tmp_path, poisoned_suffix):
    dataset = BlendedDataset.__new__(BlendedDataset)
    dataset.config = SimpleNamespace(path_to_cache=str(tmp_path))
    dataset.unique_description_hash = "test"
    dataset.split = Split.train

    prefix = "test-BlendedDataset-train"
    (tmp_path / f"{prefix}-description.txt").write_text("test")
    arrays = {
        "dataset_index": np.array([0], dtype=np.int16),
        "dataset_sample_index": np.array([0], dtype=np.int64),
    }
    for suffix, array in arrays.items():
        np.save(tmp_path / f"{prefix}-{suffix}.npy", array, allow_pickle=False)

    loaded = dataset._build_indices()
    for actual, expected in zip(loaded, arrays.values()):
        np.testing.assert_array_equal(actual, expected)

    (tmp_path / f"{prefix}-{poisoned_suffix}.npy").write_bytes(pickle.dumps(arrays[poisoned_suffix]))
    with pytest.raises(ValueError, match="pickled"):
        dataset._build_indices()

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

import pickle

from nemo_automodel.components.datasets.lazy_mapped_dataset import LazyMappedDataset


class FakeDataset:
    """Minimal map-style dataset for testing"""

    def __init__(self, n: int = 10):
        self._data = [{"idx": i, "value": i * 2} for i in range(n)]

    def __len__(self) -> int:
        """returns the number of item"""
        return len(self._data)

    def __getitem__(self, idx: int):
        """Return the item at the given index"""
        return self._data[idx]


def _add_transformed(x):
    """Add a 'transformed' key to the example"""
    return {**x, "transformed": True}


def test_len():
    """Test __len__ returns the correct dataset size"""
    ds = LazyMappedDataset(FakeDataset(5), lambda x: x)
    assert len(ds) == 5


def test_getitem_applies_map_fn():
    """Test __getitem__ applies the transform function to the item"""
    ds = LazyMappedDataset(FakeDataset(5), _add_transformed)
    item = ds[0]
    assert item["transformed"] is True
    assert item["idx"] == 0


def test_map_fn_called_once_per_index_with_cache():
    """Test map_fn is only called once per index when caching is enabled"""
    call_counts = {}

    def counting_fn(x):
        idx = x["idx"]
        call_counts[idx] = call_counts.get(idx, 0) + 1
        return x

    ds = LazyMappedDataset(FakeDataset(5), counting_fn)
    ds[0]
    ds[0]
    ds[0]
    assert call_counts[0] == 1


def test_map_fn_called_every_time_without_cache():
    """test map_fn is called on every access when caching is disabled"""
    call_count = [0]

    def counting_fn(x):
        call_count[0] += 1
        return x

    ds = LazyMappedDataset(FakeDataset(5), counting_fn, cache_size=0)
    ds[0]
    ds[0]
    ds[0]
    assert call_count[0] == 3


def test_cache_size_zero_disables_cache():
    """test cache_size=0 disables caching and cache_info return None"""
    ds = LazyMappedDataset(FakeDataset(5), lambda x: x, cache_size=0)
    assert ds.cache_info is None


def test_cache_info_available_when_caching():
    """test cache_info returns LRU statistics when caching is enable"""
    ds = LazyMappedDataset(FakeDataset(5), lambda x: x)
    ds[0]
    info = ds.cache_info
    assert info is not None
    assert info.hits + info.misses > 0


def test_pickle_roundtrip():
    """test that the dataset can be pickled and unpickled with caching enabled"""
    ds = LazyMappedDataset(FakeDataset(5), _add_transformed)
    ds_restored = pickle.loads(pickle.dumps(ds)) # noqa: S301
    assert len(ds_restored) == 5
    assert ds_restored[0]["transformed"] is True


def test_pickle_roundtrip_no_cache():
    """test that the dataset can be pickled and unpickled with caching disabled"""
    ds = LazyMappedDataset(FakeDataset(5), _add_transformed, cache_size=0)
    ds_restored = pickle.loads(pickle.dumps(ds))  # noqa: S301
    assert len(ds_restored) == 5
    assert ds_restored[2]["transformed"] is True

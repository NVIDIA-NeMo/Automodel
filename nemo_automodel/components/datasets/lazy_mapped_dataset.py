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

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LazyMappedDataset(Dataset):
    """
    Dataset wrapper that applies a transform function on-the-fly instead of
    preprocessing the whole dataset upfront with .map(fn).

    Args:
        dataset: Any object that supports ``__len__`` and ``__getitem__``
            (e.g. a Hugging Face ``datasets.Dataset``).
        map_fn: A callable that accepts a single example and returns the
            transformed example.
        cache_size: Number of processed items to cache. Defaults to the full
            dataset size. Set to 0 to disable caching.

    Returns:
        A map-style dataset that applies map_fn lazily on each item access.
    """

    def __init__(self, dataset, map_fn, cache_size=None):
        self._dataset = dataset
        self._map_fn = map_fn

        if cache_size is None:
            cache_size = len(dataset)

        self._cache_size = cache_size
        self._build_get_item()

    def _build_get_item(self) -> None:
        """Build the internal item accessor, with or without LRU caching"""
        if self._cache_size > 0:

            @lru_cache(maxsize=self._cache_size)
            def _cached_transform(idx: int) -> Any:
                return self._map_fn(self._dataset[idx])

            self._get_item = _cached_transform
            logger.debug("LazyMappedDataset: LRU cache enabled (maxsize=%d)", self._cache_size)
        else:
            self._get_item = lambda idx: self._map_fn(self._dataset[idx])
            logger.debug("LazyMappedDataset: caching disabled")

    def __getstate__(self) -> dict:
        """Returns pickable state by dropping the unpicklable _get_item function"""
        state = self.__dict__.copy()
        state["_get_item"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Restores state and rebuild _get_item after unpickling"""
        self.__dict__.update(state)
        self._build_get_item()

    def __len__(self) -> int:
        """Returns the number of items in the dataset"""
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Any:
        """Returns the transformed item at the given index"""
        return self._get_item(idx)

    @property
    def cache_info(self) -> Any | None:
        """Return LRU cache statistics, or ``None`` if caching is disabled."""
        fn = self._get_item
        if hasattr(fn, "cache_info"):
            return fn.cache_info()
        return None

    def __repr__(self) -> str:
        """returns a string representation of the dataset"""
        cache = (
            f", cache_size={self._get_item.cache_info().maxsize}"
            if hasattr(self._get_item, "cache_info")
            else ", cache_size=0"
        )

        return f"{self.__class__.__name__}(dataset={self._dataset!r}, map_fn={self._map_fn!r}{cache})"

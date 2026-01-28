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

from typing import Any, Dict, Iterable, Iterator, List

import pytest

from nemo_automodel.components.datasets.reservoir_sampler import ReservoirSampler


def _make_items(n: int) -> list[dict[str, int]]:
    return [{"id": i} for i in range(n)]

def _collect(iterable: Iterable[Dict[str, Any]]) -> list[Dict[str, Any]]:
    # Avoid `list(iterable)` since list() may call __len__ (length_hint).
    out: list[Dict[str, Any]] = []
    for x in iterable:
        out.append(x)
    return out


class _ReIterable:
    """Iterable that is not itself an iterator (DeltaLakeIterator-like)."""

    def __init__(self, items: list[dict[str, int]]):
        self._items = items

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        yield from self._items


@pytest.mark.parametrize("buffer_size", [1, 2, 3, 7, 20], ids=lambda x: f"buffer_size={x}")
def test_reservoir_sampler_yields_all_items_exactly_once(buffer_size: int) -> None:
    items = _make_items(10)
    out = _collect(ReservoirSampler(_ReIterable(items), buffer_size=buffer_size, seed=123))

    assert len(out) == len(items)
    out_ids = [x["id"] for x in out]
    assert sorted(out_ids) == list(range(10))
    assert len(set(out_ids)) == len(out_ids)


def test_reservoir_sampler_empty_iterable_yields_nothing() -> None:
    out = _collect(ReservoirSampler([], buffer_size=4, seed=0))
    assert out == []


def test_reservoir_sampler_is_reproducible_with_seed() -> None:
    items = _make_items(50)
    out1 = [x["id"] for x in _collect(ReservoirSampler(_ReIterable(items), buffer_size=7, seed=42))]
    out2 = [x["id"] for x in _collect(ReservoirSampler(_ReIterable(items), buffer_size=7, seed=42))]
    assert out1 == out2


def test_reservoir_sampler_len_and_getitem_raise() -> None:
    sampler = ReservoirSampler(_ReIterable(_make_items(3)), buffer_size=2, seed=0)
    with pytest.raises(RuntimeError, match="__len__ is not supported"):
        _ = len(sampler)
    with pytest.raises(RuntimeError, match="__getitem__ is not supported"):
        _ = sampler[0]  # type: ignore[index]


@pytest.mark.parametrize("buffer_size", [0, -1], ids=lambda x: f"buffer_size={x}")
def test_reservoir_sampler_invalid_buffer_size_raises(buffer_size: int) -> None:
    with pytest.raises(ValueError, match="buffer_size must be > 0"):
        _ = ReservoirSampler(_ReIterable(_make_items(3)), buffer_size=buffer_size, seed=0)


def test_reservoir_sampler_missing_iterator_raises() -> None:
    with pytest.raises(ValueError, match="iterator must be provided"):
        _ = ReservoirSampler(None, buffer_size=3, seed=0)  # type: ignore[arg-type]


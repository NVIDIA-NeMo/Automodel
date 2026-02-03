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

from nemo_automodel.components.checkpoint.checkpointing import _equally_divide_layers


def _make_keys(count: int) -> list[str]:
    return [f"layer.{i}" for i in range(count)]


def _count_by_shard(mapping: dict[str, int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for shard_index in mapping.values():
        counts[shard_index] = counts.get(shard_index, 0) + 1
    return counts


def test_equally_divide_layers_num_shards_gt_num_layers():
    keys = _make_keys(3)

    mapping = _equally_divide_layers(5, keys)

    assert mapping == {keys[0]: 1, keys[1]: 2, keys[2]: 3}
    assert set(mapping.values()) == {1, 2, 3}


def test_equally_divide_layers_num_shards_eq_num_layers():
    keys = _make_keys(4)

    mapping = _equally_divide_layers(4, keys)

    assert mapping == {keys[0]: 1, keys[1]: 2, keys[2]: 3, keys[3]: 4}


def test_equally_divide_layers_num_shards_lt_num_layers():
    keys = _make_keys(10)

    mapping = _equally_divide_layers(3, keys)

    assert _count_by_shard(mapping) == {1: 4, 2: 3, 3: 3}
    assert [mapping[key] for key in keys] == [1, 1, 1, 1, 2, 2, 2, 3, 3, 3]


def test_equally_divide_layers_num_shards_one():
    keys = _make_keys(5)

    mapping = _equally_divide_layers(1, keys)

    assert len(mapping) == len(keys)
    assert set(mapping.values()) == {1}

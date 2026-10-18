# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Record sharding invariants required by collective-bearing model forwards."""

import math

import pytest

from nemo_automodel.shared.decision_records import pad_and_shard_records


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@pytest.mark.parametrize("size", [0, 1, 2, 3, 4, 9, 17])
def test_all_ranks_receive_equal_forward_counts_and_cover_records(size, world_size):
    records = list(range(size))
    shards = [pad_and_shard_records(records, rank=rank, world_size=world_size) for rank in range(world_size)]
    expected = math.ceil(size / world_size)
    assert [len(shard) for shard in shards] == [expected] * world_size
    assert set(item for shard in shards for item in shard) == set(records)
    for batch_size in (1, 2, 3):
        assert len({math.ceil(len(shard) / batch_size) for shard in shards}) == 1


def test_single_record_is_repeated_for_every_rank():
    assert [pad_and_shard_records(["only"], rank=rank, world_size=8) for rank in range(8)] == [["only"]] * 8


def test_shuffle_is_repeatable_and_changes_order():
    records = list(range(32))
    first = [pad_and_shard_records(records, rank=rank, world_size=4, shuffle_seed=7) for rank in range(4)]
    again = [pad_and_shard_records(records, rank=rank, world_size=4, shuffle_seed=7) for rank in range(4)]
    other = [pad_and_shard_records(records, rank=rank, world_size=4, shuffle_seed=8) for rank in range(4)]
    assert first == again
    assert first != other
    assert sorted(item for shard in first for item in shard) == records


@pytest.mark.parametrize("rank,world_size", [(0, 0), (-1, 2), (2, 2)])
def test_invalid_parallel_group_is_rejected(rank, world_size):
    with pytest.raises(ValueError, match="world_size"):
        pad_and_shard_records([1], rank=rank, world_size=world_size)

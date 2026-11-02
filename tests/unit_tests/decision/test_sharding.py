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

import json
import math
from pathlib import Path

import pytest

from nemo_automodel.components.datasets.decision.records import load_records, pad_and_shard_records


@pytest.mark.parametrize("record_ids", [["same", "same"], [1, "1"]])
def test_loading_rejects_duplicate_canonical_record_ids(tmp_path: Path, record_ids: list[str | int]) -> None:
    path = tmp_path / "records.jsonl"
    records = [{"id": record_id, "state": str(index)} for index, record_id in enumerate(record_ids)]
    path.write_text("\n".join(json.dumps(record) for record in records))
    with pytest.raises(ValueError, match="duplicate record ID"):
        load_records(path)


@pytest.mark.parametrize("record", [{}, {"id": None}, {"id": ""}, {"id": "   "}])
def test_loading_rejects_missing_or_empty_record_ids(tmp_path: Path, record: dict[str, object]) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="nonempty record ID"):
        load_records(path)


def test_valid_loaded_records_can_still_be_padded_across_ranks(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text('{"id": 0}\n{"id": "other"}\n')
    records = load_records(path)
    assert records == [{"id": 0}, {"id": "other"}]
    shards = [pad_and_shard_records(records, rank=rank, world_size=4) for rank in range(4)]
    assert shards == [[records[0]], [records[1]], [records[0]], [records[1]]]
    assert load_records(path, limit=1) == records[:1]


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

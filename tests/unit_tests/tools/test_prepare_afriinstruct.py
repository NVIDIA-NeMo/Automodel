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

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.prepare_afriinstruct import _PreparationConfig, _prepare_dataset


def _config(output_dir: Path, **overrides) -> _PreparationConfig:
    values = {
        "dataset_id": "test/afriinstruct",
        "dataset_config": None,
        "source_split": "train",
        "output_dir": output_dir,
        "validation_fraction": 0.2,
        "sample_fraction": 1.0,
        "seed": 42,
        "shard_size": 7,
        "max_samples": None,
        "languages": frozenset(),
        "tasks": frozenset(),
        "sources": frozenset(),
    }
    values.update(overrides)
    return _PreparationConfig(**values)


def _records(count: int = 100) -> list[dict[str, str | None]]:
    languages = ("swa", "hau", "yor")
    tasks = ("translation", "qa")
    return [
        {
            "instruction": f"Instruction {index}",
            "output": f"Answer {index}",
            "lang": languages[index % len(languages)],
            "task": tasks[index % len(tasks)],
            "source": "unit-test",
        }
        for index in range(count)
    ]


def test_prepare_dataset_writes_conversations_shards_and_manifests(tmp_path: Path) -> None:
    records = _records()
    records.append(dict(records[0]))
    records.append({"instruction": "", "output": "invalid", "lang": "swa", "task": "qa", "source": "test"})

    summary = _prepare_dataset(_config(tmp_path / "prepared"), records)

    assert summary["stats"]["input_records"] == 102
    assert summary["stats"]["written_records"] == 100
    assert summary["stats"]["duplicate_records"] == 1
    assert summary["stats"]["malformed_records"] == 1
    assert summary["stats"]["train_records"] > 0
    assert summary["stats"]["validation_records"] > 0

    output_dir = tmp_path / "prepared"
    for split in ("train", "validation"):
        manifest = json.loads((output_dir / f"{split}_meta.json").read_text(encoding="utf-8"))
        assert manifest
        for entry in manifest.values():
            shard_path = output_dir / entry["file_name"]
            lines = shard_path.read_text(encoding="utf-8").splitlines()
            assert 1 <= len(lines) <= 7
            assert entry["columns"] == {"messages": "messages"}

    first_record = json.loads(next(output_dir.glob("processed/*/*.jsonl")).read_text(encoding="utf-8").splitlines()[0])
    assert first_record["messages"][0]["role"] == "user"
    assert first_record["messages"][1]["role"] == "assistant"
    assert first_record["lang"] in {"swa", "hau", "yor"}
    assert not (output_dir / ".dedup.sqlite3").exists()


def test_prepare_dataset_applies_filters_and_max_samples(tmp_path: Path) -> None:
    config = _config(
        tmp_path / "filtered",
        languages=frozenset({"swa"}),
        tasks=frozenset({"translation"}),
        max_samples=5,
    )

    summary = _prepare_dataset(config, _records())

    assert summary["stats"]["written_records"] == 5
    assert summary["stats"]["by_language"] == {"swa": 5}
    assert summary["stats"]["by_task"] == {"translation": 5}


def test_prepare_dataset_applies_deterministic_hash_sampling(tmp_path: Path) -> None:
    config = _config(tmp_path / "sampled", sample_fraction=0.25)

    summary = _prepare_dataset(config, _records(400))

    assert 70 <= summary["stats"]["written_records"] <= 130
    assert summary["stats"]["sampled_out_records"] + summary["stats"]["written_records"] == 400
    assert len(summary["stats"]["by_language"]) == 3
    assert len(summary["stats"]["by_task"]) == 2


def test_prepare_dataset_is_deterministic(tmp_path: Path) -> None:
    records = _records(40)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    _prepare_dataset(_config(first_dir), records)
    _prepare_dataset(_config(second_dir), records)

    for relative_path in ("train_meta.json", "validation_meta.json", "summary.json"):
        assert (first_dir / relative_path).read_text(encoding="utf-8") == (second_dir / relative_path).read_text(
            encoding="utf-8"
        )
    first_shards = {
        path.relative_to(first_dir): path.read_text(encoding="utf-8")
        for path in sorted(first_dir.glob("processed/*/*.jsonl"))
    }
    second_shards = {
        path.relative_to(second_dir): path.read_text(encoding="utf-8")
        for path in sorted(second_dir.glob("processed/*/*.jsonl"))
    }
    assert first_shards == second_shards


def test_prepare_dataset_rejects_nonempty_output_directory(tmp_path: Path) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("user data", encoding="utf-8")

    with pytest.raises(FileExistsError, match="not empty"):
        _prepare_dataset(_config(output_dir), _records(1))

    assert (output_dir / "keep.txt").read_text(encoding="utf-8") == "user data"

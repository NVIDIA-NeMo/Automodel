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

import csv
import importlib.util
import json
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[3] / "examples" / "llm_pretrain"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RULER = load_module("titans_ruler_test_module", "titans_ruler.py")
PREDICT = load_module("titans_ruler_predict_test_module", "titans_ruler_predict.py")
REGISTRY = load_module("titans_eval_registry_test_module", "titans_eval_registry.py")


def write_jsonl(path: Path, rows: list[dict], *, blank_after_first: bool = False) -> None:
    lines = [json.dumps(row) for row in rows]
    if blank_after_first:
        lines.insert(1, "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def sample(index: int) -> dict:
    return {"index": index, "input": f"prompt {index}", "outputs": [f"answer {index}"]}


def prediction(ordinal: int, *, index: int = 0, pred: str = "answer") -> dict:
    return {
        **sample(index),
        "_sample_ordinal": ordinal,
        "pred": pred,
        "prompt_tokens": 10,
        "generated_tokens": 2,
    }


def test_range_selection_uses_nonblank_zero_based_ordinals(tmp_path: Path) -> None:
    data = tmp_path / "validation.jsonl"
    write_jsonl(data, [sample(7), sample(7), sample(9), sample(10)], blank_after_first=True)

    selected = list(PREDICT.iter_samples(data, 1, 3))

    assert [ordinal for ordinal, _ in selected] == [1, 2]
    assert [row["index"] for _, row in selected] == [7, 9]


def test_resume_uses_ordinal_and_rejects_duplicate_ordinal(tmp_path: Path) -> None:
    output = tmp_path / "shard.jsonl"
    write_jsonl(output, [prediction(1, index=4), prediction(2, index=4)])
    assert PREDICT.load_completed_samples(output, 1, 3) == {1, 2}

    write_jsonl(output, [prediction(1), prediction(1)])
    with pytest.raises(ValueError, match="duplicate ordinal 1"):
        PREDICT.load_completed_samples(output, 1, 3)


def test_resume_rejects_legacy_output_without_ordinal(tmp_path: Path) -> None:
    output = tmp_path / "legacy.jsonl"
    write_jsonl(output, [{**sample(0), "pred": "answer"}])

    with pytest.raises(ValueError, match="unsafe resume"):
        PREDICT.load_completed_samples(output, 0, 1)


def test_resume_rejects_predictions_without_manifest(tmp_path: Path) -> None:
    output = tmp_path / "shard.jsonl"
    write_jsonl(output, [prediction(0)])

    with pytest.raises(ValueError, match="no manifest"):
        PREDICT.write_manifest(output, {"task": "niah_single_1"})


def test_merge_sorts_rows_and_validates_unique_complete_ordinals(tmp_path: Path) -> None:
    first = tmp_path / "00000002-00000004.jsonl"
    second = tmp_path / "00000000-00000002.jsonl"
    merged = tmp_path / "merged.jsonl"
    write_jsonl(first, [prediction(3), prediction(2)])
    write_jsonl(second, [prediction(1), prediction(0)])

    rows = RULER.merge_prediction_shards([first, second], merged, 0, 4)

    assert [row["_sample_ordinal"] for row in rows] == [0, 1, 2, 3]
    assert [json.loads(line)["_sample_ordinal"] for line in merged.read_text().splitlines()] == [0, 1, 2, 3]

    write_jsonl(first, [prediction(1), prediction(2)])
    with pytest.raises(ValueError, match="duplicate prediction ordinal 1"):
        RULER.merge_prediction_shards([first, second], merged, 0, 4)


def test_merge_rejects_missing_rows(tmp_path: Path) -> None:
    shard = tmp_path / "00000000-00000002.jsonl"
    write_jsonl(shard, [prediction(0), prediction(2)])

    with pytest.raises(ValueError, match=r"missing=\[1\]"):
        RULER.merge_prediction_shards([shard], tmp_path / "merged.jsonl", 0, 3)


def test_merge_rejects_mismatched_shard_manifest(tmp_path: Path) -> None:
    shard = tmp_path / "00000000-00000001.jsonl"
    write_jsonl(shard, [prediction(0)])
    Path(f"{shard}.manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": "/checkpoint/a",
                "task": "niah_single_1",
                "shard_range": {"start": 0, "end": 1},
            }
        )
    )

    with pytest.raises(ValueError, match="checkpoint"):
        RULER.validate_shard_manifests(
            [shard],
            {"checkpoint": "/checkpoint/b", "task": "niah_single_1"},
        )


def test_scoring_reads_merged_raw_jsonl(tmp_path: Path) -> None:
    predictions = tmp_path / "pred"
    write_jsonl(
        predictions / "niah_single_1.jsonl",
        [
            {**prediction(0, pred="The ANSWER 0 is here"), "outputs": ["answer 0"]},
            {**prediction(1, pred=""), "outputs": ["answer 1"]},
        ],
    )

    rows = RULER.score_s_niah_predictions(predictions, ["niah_single_1"])

    assert rows == [{"task": "niah_single_1", "score": 50.0, "nulls": 1, "num_samples": 2}]
    with (predictions / "summary.csv").open(newline="") as stream:
        summary = list(csv.DictReader(stream))
    assert summary == [{"task": "niah_single_1", "score": "50.0", "nulls": "1", "num_samples": "2"}]


def test_prepare_dataset_forwards_and_records_explicit_seed(tmp_path: Path, monkeypatch) -> None:
    commands = []

    def fake_run(command, *, cwd):
        commands.append((command, cwd))
        output = Path(command[command.index("--save_dir") + 1])
        task = command[command.index("--task") + 1]
        write_jsonl(output / task / "validation.jsonl", [sample(0), sample(1)])

    monkeypatch.setattr(RULER, "run", fake_run)
    prepared = RULER.prepare_dataset(
        scripts=tmp_path / "RULER" / "scripts",
        data_dir=tmp_path / "data",
        task="niah_single_1",
        tokenizer="tokenizer",
        context_length=2048,
        num_samples=2,
        model_template_type="base",
        ruler_revision="revision",
        seed=43,
    )

    assert prepared.exists()
    assert commands[0][0][commands[0][0].index("--random_seed") + 1] == "43"
    manifest = json.loads((prepared.parent / "manifest.json").read_text())
    assert manifest["random_seed"] == 43


def test_eval_registry_resolves_enabled_models_and_ttt_modes(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    memory_root = outputs / "memory"
    control_root = outputs / "control"
    for root in (memory_root, control_root):
        checkpoint = root / "checkpoint_step_1" / "model" / "consolidated"
        checkpoint.mkdir(parents=True)
        (root / "LATEST").symlink_to("checkpoint_step_1")
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "models": [
                    {"name": "memory", "run_root": "memory", "memory_model": True, "enabled": True},
                    {"name": "control", "run_root": "control", "memory_model": False, "enabled": True},
                    {"name": "planned", "run_root": "missing", "memory_model": True, "enabled": False},
                ],
            }
        )
    )

    rows = REGISTRY.evaluation_rows(registry, tmp_path)

    assert [(name, enabled) for name, _, enabled in rows] == [
        ("memory", True),
        ("memory", False),
        ("control", True),
    ]

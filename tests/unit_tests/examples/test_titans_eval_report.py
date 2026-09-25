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
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[3] / "examples" / "llm_pretrain" / "titans_eval_report.py"
SPEC = importlib.util.spec_from_file_location("titans_eval_report", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


def _write_ruler_fixture(root: Path, mode: str, task: str, context: int, *, complete: bool = True) -> None:
    prediction_dir = root / mode / "lmm" / str(context) / "pred"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    sample_count = 2
    summary_path = prediction_dir / "summary.csv"
    existing = []
    if summary_path.exists():
        with summary_path.open(newline="") as stream:
            existing = list(csv.DictReader(stream))
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task", "score", "nulls", "num_samples"])
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow({"task": task, "score": 75.0, "nulls": 0, "num_samples": sample_count})
    merged = prediction_dir / f"{task}.jsonl"
    rows = [
        {"_sample_ordinal": 0, "pred": "first"},
        {"_sample_ordinal": 1, "pred": "second"},
    ]
    if not complete:
        rows.pop()
    merged.write_text("".join(json.dumps(row) + "\n" for row in rows))
    Path(f"{merged}.manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": "/checkpoints/lmm",
                "code_git_revision": "abc123",
                "task": task,
                "context_length": context,
                "requested_sample_count": sample_count,
                "generation_settings": {"enable_ttt_updates": mode == "ttt_on"},
            }
        )
    )


def _write_lm_fixture(root: Path, mode: str, benchmark: str) -> None:
    output_dir = root / mode / "lm" / "lmm"
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw = raw_dir / f"{benchmark}.000000000000-000000000002.jsonl"
    raw.write_text('{"index":0}\n{"index":1}\n')
    summary = {
        "schema_version": 1,
        "benchmark": benchmark,
        "metadata": {
            "checkpoint": "/checkpoints/lmm",
            "enable_ttt_updates": mode == "ttt_on",
        },
        "metrics": {
            "samples": 2,
            "token_count": 11,
            "perplexity": 4.5,
            "bits_per_byte": 1.25,
            "last_word_exact_accuracy": 0.5 if benchmark == "lambada" else None,
        },
        "source_indices": [0, 1],
    }
    summary_path = output_dir / f"{benchmark}.summary.json"
    summary_path.write_text(json.dumps(summary))
    manifest = {
        "schema_version": 1,
        "benchmark": benchmark,
        "partials": [
            {
                "path": str(raw.relative_to(output_dir)),
                "sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "rows": 2,
            }
        ],
        "summary": {
            "path": summary_path.name,
            "sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        },
    }
    (output_dir / f"{benchmark}.manifest.json").write_text(json.dumps(manifest))


def test_collects_tidy_ruler_and_lm_rows_with_integrity(tmp_path: Path) -> None:
    ruler_root = tmp_path / "ruler"
    lm_root = tmp_path / "lm"
    _write_ruler_fixture(ruler_root, "ttt_on", "s_niah_pk", 2048)
    _write_lm_fixture(lm_root, "ttt_off", "wikitext103")

    ruler_rows = REPORT.collect_ruler([ruler_root])
    lm_rows = REPORT.collect_lm([lm_root])

    assert len(ruler_rows) == 1
    assert ruler_rows[0].value == 75.0
    assert ruler_rows[0].ttt_mode == "on"
    assert ruler_rows[0].valid is True
    assert ruler_rows[0].complete is True
    assert {row.metric for row in lm_rows} == {"perplexity", "bits_per_byte"}
    assert all(row.ttt_mode == "off" and row.valid and row.complete for row in lm_rows)
    assert all(row.code_revision is None for row in lm_rows)


def test_marks_incomplete_ruler_cell_without_inventing_value(tmp_path: Path) -> None:
    root = tmp_path / "ruler"
    _write_ruler_fixture(root, "ttt_off", "s_niah_n", 2048, complete=False)

    row = REPORT.collect_ruler([root])[0]

    assert row.value == 75.0
    assert row.valid is True
    assert row.complete is False
    assert "incomplete" in row.status


def test_writes_outputs_and_self_contained_report(tmp_path: Path) -> None:
    ruler_root = tmp_path / "ruler"
    lm_root = tmp_path / "lm"
    _write_ruler_fixture(ruler_root, "ttt_on", "s_niah_pk", 2048)
    _write_ruler_fixture(ruler_root, "ttt_off", "s_niah_pk", 2048)
    _write_ruler_fixture(ruler_root, "ttt_on", "s_niah_n", 2048)
    for mode in ("ttt_on", "ttt_off"):
        for benchmark in ("wikitext103", "fineweb_edu", "lambada"):
            _write_lm_fixture(lm_root, mode, benchmark)
    rows = REPORT.collect_ruler([ruler_root]) + REPORT.collect_lm([lm_root])
    output_dir = tmp_path / "report"

    csv_path, json_path = REPORT.write_tidy_outputs(
        rows,
        output_dir,
        {"ruler": [ruler_root], "lm": [lm_root]},
    )
    plt = REPORT._matplotlib()
    if plt is None:
        pytest.skip("matplotlib is not installed")
    figures = REPORT.plot_ruler(rows, output_dir, {}, plt) + REPORT.plot_lm(rows, output_dir, plt)
    html_path = REPORT.write_html(rows, output_dir, figures)

    assert csv_path.is_file()
    payload = json.loads(json_path.read_text())
    assert len(payload["records"]) == len(rows)
    assert all(record["value"] is not None for record in payload["records"])
    assert (output_dir / "titans_s_niah_pk_noise.png").is_file()
    assert (output_dir / "titans_s_niah_pk_noise.svg").is_file()
    assert (output_dir / "titans_lm_lambada.png").is_file()
    report = html_path.read_text()
    assert "<svg" in report
    assert "No compatible parameter-matched floor or full-attention control" in report
    assert "essay at context length 512 is unavailable" in report
    assert 'src="' not in report

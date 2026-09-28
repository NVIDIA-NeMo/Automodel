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


def _write_ruler_fixture(
    root: Path,
    mode: str,
    task: str,
    context: int,
    *,
    architecture: str = "lmm",
    complete: bool = True,
    score: float = 75.0,
    seed: int | None = None,
) -> None:
    if seed is None:
        prediction_dir = root / mode / architecture / str(context) / "pred"
        sample_count = 2
    else:
        prediction_dir = root / architecture / mode.replace("_", "-") / str(context) / f"seed-{seed}" / "pred"
        sample_count = 100
    prediction_dir.mkdir(parents=True, exist_ok=True)
    summary_path = prediction_dir / "summary.csv"
    existing = []
    if summary_path.exists():
        with summary_path.open(newline="") as stream:
            existing = list(csv.DictReader(stream))
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task", "score", "nulls", "num_samples"])
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow({"task": task, "score": score, "nulls": 0, "num_samples": sample_count})
    merged = prediction_dir / f"{task}.jsonl"
    rows = [{"_sample_ordinal": ordinal, "pred": f"prediction-{ordinal}"} for ordinal in range(sample_count)]
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
                **({"data_seed": seed} if seed is not None else {}),
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


def test_aggregates_new_seeded_layout_with_provenance(tmp_path: Path) -> None:
    root = tmp_path / "ruler"
    for seed, score in ((42, 60.0), (43, 75.0), (44, 90.0)):
        _write_ruler_fixture(root, "ttt_on", "s_niah_pk", 4096, score=score, seed=seed)

    rows = REPORT.collect_ruler([root])

    assert len(rows) == 1
    row = rows[0]
    assert row.architecture == "lmm"
    assert row.ttt_mode == "on"
    assert row.value == pytest.approx(75.0)
    assert row.sample_count == 300
    assert row.seed_count == 3
    assert row.seeds == "42,43,44"
    assert row.valid is True
    assert row.complete is True
    assert len(json.loads(row.artifacts)) == 3
    assert all("seed-" in artifact for artifact in json.loads(row.artifacts))


def test_preserves_but_does_not_plot_incomplete_seeded_cell(tmp_path: Path) -> None:
    root = tmp_path / "ruler"
    for seed in (42, 43):
        _write_ruler_fixture(root, "ttt_off", "s_niah_n", 8192, seed=seed)

    row = REPORT.collect_ruler([root])[0]

    assert row.value == 75.0
    assert row.sample_count == 200
    assert row.seed_count == 2
    assert row.valid is True
    assert row.complete is False
    assert "expected 3 seeds, found 2" in row.status


def test_plots_mac_recovery_only_from_exact_matched_control_cells(tmp_path: Path) -> None:
    root = tmp_path / "ruler"
    _write_ruler_fixture(root, "ttt_off", "s_niah_pk", 2048, architecture="local_matched", score=20.0)
    _write_ruler_fixture(root, "ttt_off", "s_niah_pk", 2048, architecture="full_matched", score=80.0)
    _write_ruler_fixture(root, "ttt_on", "s_niah_pk", 2048, architecture="mac", score=50.0)
    rows = REPORT.collect_ruler([root])
    plt = REPORT._matplotlib()
    if plt is None:
        pytest.skip("matplotlib is not installed")

    assert (
        REPORT.plot_mac_recovery([row for row in rows if row.architecture != "full_matched"], tmp_path, {}, plt) == []
    )
    figures = REPORT.plot_mac_recovery(rows, tmp_path, {}, plt)

    assert {path.name for path in figures} == {
        "titans_mac_bridge_recovery.png",
        "titans_mac_bridge_recovery.svg",
    }


def test_collapses_only_globally_identical_ttt_pairs() -> None:
    base = {
        "source": "lm",
        "architecture": "mac",
        "task_or_benchmark": "wikitext103",
        "context_length": None,
        "metric": "perplexity",
        "unit": "ratio",
        "sample_count": 2,
        "token_count": 11,
        "valid": True,
        "complete": True,
        "status": "ok",
        "checkpoint": None,
        "code_revision": None,
        "artifact": "summary.json",
        "artifacts": "[]",
    }
    on = REPORT.ReportRow(ttt_mode="on", value=4.5, **base)
    off = REPORT.ReportRow(ttt_mode="off", value=4.5, **base)

    collapsed, did_collapse = REPORT._collapse_identical_ttt_rows([on, off])

    assert did_collapse is True
    assert collapsed == [on]
    differing = REPORT.ReportRow(ttt_mode="off", value=4.6, **base)
    unchanged, did_collapse = REPORT._collapse_identical_ttt_rows([on, differing])
    assert did_collapse is False
    assert unchanged == [on, differing]


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
    assert "FSDP2 zero-fast-weight initialization bug" in report
    assert "Parameter-matched no-memory controls now exist" in report
    assert "Missing task/context cells are not interpolated" in report
    assert "Every matched aggregate metric is exactly identical with TTT updates on and off" in report
    assert 'src="' not in report

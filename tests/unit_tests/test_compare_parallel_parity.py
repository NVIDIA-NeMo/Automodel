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

import json
import sys

import pytest

from tests.functional_tests.parallelism import compare_parallel_parity


def _write_metrics(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _run_comparison(monkeypatch, baseline_path, parallel_path, *, metric=None):
    argv = ["compare_parallel_parity.py", str(baseline_path), str(parallel_path), "--axis", "pp"]
    if metric is not None:
        argv += ["--metric", metric]
    monkeypatch.setattr(sys, "argv", argv)
    compare_parallel_parity.main()


def _run_validation_comparison(monkeypatch, baseline_path, parallel_path):
    _run_comparison(monkeypatch, baseline_path, parallel_path, metric="val_loss")


def _run_training_comparison(monkeypatch, baseline_path, parallel_path):
    _run_comparison(monkeypatch, baseline_path, parallel_path)


def test_validation_parity_accepts_finite_matching_steps(tmp_path, monkeypatch):
    baseline_path = tmp_path / "baseline.jsonl"
    parallel_path = tmp_path / "parallel.jsonl"
    _write_metrics(baseline_path, [{"step": 2, "val_loss": 1.0}, {"step": 4, "val_loss": 0.9}])
    _write_metrics(parallel_path, [{"step": 2, "val_loss": 1.01}, {"step": 4, "val_loss": 0.91}])

    _run_validation_comparison(monkeypatch, baseline_path, parallel_path)


def test_grad_norm_guard_applies_to_training_only(tmp_path, monkeypatch):
    """Validation logs carry no grad_norm, so only ``--metric loss`` may demand one."""
    baseline_path = tmp_path / "baseline.jsonl"
    parallel_path = tmp_path / "parallel.jsonl"
    records = [{"step": 2, "loss": 2.0, "val_loss": 1.0}, {"step": 4, "loss": 1.8, "val_loss": 0.9}]
    _write_metrics(baseline_path, records)
    _write_metrics(parallel_path, records)

    _run_validation_comparison(monkeypatch, baseline_path, parallel_path)

    with pytest.raises(AssertionError, match="Neither log reported grad_norm"):
        _run_training_comparison(monkeypatch, baseline_path, parallel_path)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), float("-inf")])
def test_validation_parity_rejects_nonfinite_loss(tmp_path, monkeypatch, nonfinite):
    baseline_path = tmp_path / "baseline.jsonl"
    parallel_path = tmp_path / "parallel.jsonl"
    _write_metrics(baseline_path, [{"step": 2, "val_loss": nonfinite}])
    _write_metrics(parallel_path, [{"step": 2, "val_loss": nonfinite}])

    with pytest.raises(AssertionError, match="non-finite val_loss"):
        _run_validation_comparison(monkeypatch, baseline_path, parallel_path)


def test_validation_parity_requires_identical_step_sets(tmp_path, monkeypatch):
    baseline_path = tmp_path / "baseline.jsonl"
    parallel_path = tmp_path / "parallel.jsonl"
    _write_metrics(
        baseline_path,
        [
            {"step": 2, "val_loss": 1.0},
            {"step": 4, "val_loss": 0.9},
        ],
    )
    _write_metrics(parallel_path, [{"step": 2, "val_loss": 1.0}])

    with pytest.raises(AssertionError, match="Validation steps differ"):
        _run_validation_comparison(monkeypatch, baseline_path, parallel_path)


def test_validation_parity_rejects_flat_validation_curve(tmp_path, monkeypatch):
    """Two runs that agree on a frozen val_loss prove nothing about validation."""
    baseline_path = tmp_path / "baseline.jsonl"
    parallel_path = tmp_path / "parallel.jsonl"
    records = [{"step": 2, "val_loss": 1.0}, {"step": 4, "val_loss": 1.0}]
    _write_metrics(baseline_path, records)
    _write_metrics(parallel_path, records)

    with pytest.raises(AssertionError, match="baseline val_loss is flat.*not reading the trained model"):
        _run_validation_comparison(monkeypatch, baseline_path, parallel_path)


def test_training_parity_rejects_nonfinite_gradient_norm(tmp_path, monkeypatch):
    baseline_path = tmp_path / "baseline.jsonl"
    parallel_path = tmp_path / "parallel.jsonl"
    record = {"step": 1, "loss": 1.0, "grad_norm": float("nan")}
    _write_metrics(baseline_path, [record])
    _write_metrics(parallel_path, [record])

    with pytest.raises(AssertionError, match="non-finite gradient norm"):
        _run_training_comparison(monkeypatch, baseline_path, parallel_path)

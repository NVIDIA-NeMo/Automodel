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

"""Ensure corrupt or incomplete distributed runs cannot pass the parity harness."""

import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from tests.functional_tests.parallelism.compare_parallel_parity import main, read_metrics
from tests.functional_tests.parallelism_multigpu.recipe_parity import compare_recipe_runs
from tests.functional_tests.parallelism_multigpu.run_recipe import initialize_parity_weights


@pytest.mark.parametrize("metric", ["loss", "grad_norm"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_reject_nonfinite_metrics(tmp_path: Path, metric: str, value: float) -> None:
    """NaN differences must not silently satisfy a greater-than tolerance check."""
    path = tmp_path / "training.jsonl"
    path.write_text(json.dumps({"step": 0, "loss": 3.0, "grad_norm": 1.0, metric: value}) + "\n")
    with pytest.raises(AssertionError, match="non-finite"):
        read_metrics(str(path))


@pytest.mark.parametrize("steps", [1, 2, 3])
def test_require_complete_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, steps: int) -> None:
    """Even two agreeing truncated runs must fail; a complete finite trajectory passes."""
    path = tmp_path / "training.jsonl"
    path.write_text("\n".join(json.dumps({"step": i, "loss": 3.0 - 0.1 * i, "grad_norm": 1.0}) for i in range(steps)))
    monkeypatch.setattr(sys, "argv", ["compare", str(path), str(path), "--axis", "pp", "--expected-steps", "3"])
    if steps == 3:
        main()
    else:
        with pytest.raises(AssertionError, match="expected steps"):
            main()


@pytest.mark.parametrize("count", [None, 4, 8])
def test_ci_exposes_requested_gpus(tmp_path: Path, count: int | None) -> None:
    """Exercise the real shell entrypoint, replacing only the expensive pytest runner."""
    coverage = tmp_path / "coverage"
    coverage.write_text('#!/bin/bash\nif [[ "$1" == run ]]; then echo "VISIBLE=$CUDA_VISIBLE_DEVICES"; fi\n')
    coverage.chmod(0o755)
    root = Path(__file__).resolve().parents[3]
    args = ["bash", str(root / "tests/run_test.sh"), "--TEST_NAME=parallelism_multigpu,moe_multigpu"]
    if count is not None:
        args.append(f"--GPU_COUNT={count}")
    result = subprocess.run(
        args,
        env={**os.environ, "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=True,
    )
    expected = ",".join(str(i) for i in range(count or 2))
    assert result.stdout.strip() == f"VISIBLE={expected}"


@pytest.mark.parametrize(
    "validation",
    [[{"step": 2, "val_loss": 3.0}], [], [{"step": 2, "val_loss": float("nan")}], [{"step": 1, "val_loss": 3.0}]],
)
def test_recipe_validation_schema(tmp_path: Path, validation: list[dict[str, float]]) -> None:
    """Accept real recipe validation records and reject missing, corrupt or early output."""
    (tmp_path / "training.jsonl").write_text(
        "\n".join(json.dumps({"step": i, "loss": 3.0 - 0.1 * i, "grad_norm": 1.0}) for i in range(3))
    )
    (tmp_path / "validation.jsonl").write_text("\n".join(json.dumps(record) for record in validation))
    if validation == [{"step": 2, "val_loss": 3.0}]:
        compare_recipe_runs(baseline=tmp_path, parallel=tmp_path, loss_tol=0.05, grad_norm_rtol=0.05)
    else:
        with pytest.raises(AssertionError):
            compare_recipe_runs(baseline=tmp_path, parallel=tmp_path, loss_tol=0.05, grad_norm_rtol=0.05)


def test_initial_weights_do_not_depend_on_stage_or_visit_order() -> None:
    """PP stages omit different parameters; global RNG order is not a valid oracle."""
    model = torch.nn.ModuleDict({str(i): torch.nn.Linear(8, 8, dtype=torch.bfloat16) for i in range(4)})
    stage = deepcopy(model)
    del stage["0"]
    del stage["1"]
    stage = torch.nn.ModuleDict(reversed(list(stage.items())))
    reference = initialize_parity_weights([model])
    actual = initialize_parity_weights([stage])
    assert actual == {name: reference[name] for name in actual}
    assert reference["2.weight"] != reference["3.weight"]
    for name, parameter in stage.named_parameters():
        torch.testing.assert_close(parameter, dict(model.named_parameters())[name], rtol=0, atol=0)

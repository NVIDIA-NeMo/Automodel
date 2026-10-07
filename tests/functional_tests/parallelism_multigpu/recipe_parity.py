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

"""Bounded subprocesses and training/validation parity for tiny GPU recipes."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def gpu_count() -> int:
    """Require a four- or eight-GPU allocation; a wrong runner must fail."""
    import torch

    count = torch.cuda.device_count()
    assert count in (4, 8), f"This suite needs four or eight visible GPUs, got {count}"
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        assert count == len(visible.split(",")), f"Requested GPUs {visible}, but only {count} are available"
    return count


def run_distributed(ranks: int, args: list[str], log: Path) -> None:
    """Run real NCCL workers with a hard timeout and retain their combined log."""
    command = [
        "timeout",
        "--kill-after=10",
        "180",
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={ranks}",
        *args,
    ]
    env = {**os.environ, "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"}
    # Import this checkout, including when pytest was invoked from another cwd.
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    with log.open("w") as output:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=output, stderr=subprocess.STDOUT)
    assert result.returncode == 0, f"{' '.join(command)}\n{log.read_text()}"


def compare_recipe_runs(*, baseline: Path, parallel: Path, loss_tol: float, grad_norm_rtol: float) -> None:
    """Require all three training steps and final validation to agree and be finite."""
    subprocess.run(
        [
            sys.executable,
            str(REPO / "tests/functional_tests/parallelism/compare_parallel_parity.py"),
            str(baseline / "training.jsonl"),
            str(parallel / "training.jsonl"),
            "--axis",
            "pp",
            "--expected-steps",
            "3",
            "--loss-tol",
            str(loss_tol),
            "--grad-norm-rtol",
            str(grad_norm_rtol),
        ],
        check=True,
    )
    validation = []
    for directory in (baseline, parallel):
        records = [
            json.loads(line) for line in (directory / "validation.jsonl").read_text().splitlines() if line.strip()
        ]
        assert len(records) == 1 and records[0]["step"] == 2, records
        loss = float(records[0]["val_loss"])
        assert math.isfinite(loss), f"{directory}: non-finite validation loss {loss}"
        validation.append(loss)
    assert abs(validation[0] - validation[1]) <= loss_tol, validation


def run_recipe_pair(
    *,
    ranks: int,
    pp_size: int,
    config: Path,
    overrides: list[str],
    output: Path,
    loss_tol: float,
    grad_norm_rtol: float,
) -> None:
    """Compare PP against the same TP/CP/EP/DP topology with PP disabled."""
    for name, pp in (("baseline", 1), ("parallel", pp_size)):
        run_distributed(
            ranks // pp_size * pp,
            [
                "examples/llm_finetune/finetune.py",
                "--config",
                str(config),
                *overrides,
                "--distributed.pp_size",
                str(pp),
                "--checkpoint.checkpoint_dir",
                str(output / name),
            ],
            output / f"{name}.log",
        )
    # A successful numerical comparison must actually have used the static PP path.
    log = (output / "parallel.log").read_text()
    assert "Precomputed pipeline stage shapes" in log
    assert "dynamic metadata inference" not in log
    compare_recipe_runs(
        baseline=output / "baseline", parallel=output / "parallel", loss_tol=loss_tol, grad_norm_rtol=grad_norm_rtol
    )

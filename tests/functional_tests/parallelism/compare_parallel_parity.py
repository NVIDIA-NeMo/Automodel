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

"""Parallel-vs-single-rank training parity validator.

Compares two ``training.jsonl`` logs produced by the same recipe and seed: a
single-rank baseline and a run with one parallelism axis enabled (TP, PP, CP, or
EP). Both runs must follow the same loss and gradient-norm trajectory.

This is the generic net for parallelism correctness. A smoke test only fails on
a crash or a hang, but wrong stage metadata, a gradient that syncs over the
wrong group, a mis-sharded context-parallel sequence, or a missing tensor-
parallel reduction all keep the run alive and silently change the numbers. Those
show up here as a diverging loss or gradient norm.

Gradient norm is checked separately from loss because the two fail
independently: loss is reduced before the gradient all-reduce, so a broken
gradient sync leaves the loss curve intact while the norm drifts.

Called by the ``L2_Parallelism_*`` scripts after two torchrun invocations.

Usage:
    python compare_parallel_parity.py baseline.jsonl pp2.jsonl --axis pp
"""

from __future__ import annotations

import argparse
import json

# Both runs share a seed and a data order, so step 1 differs only by floating-
# point reduction order. Later steps accumulate that difference through the
# optimizer, so the bound is deliberately loose enough to absorb bf16 reduction
# noise while still catching a real divergence, which moves the loss by whole
# nats rather than hundredths.
DEFAULT_LOSS_TOL = 0.05
# Gradient norm is a sum over every parameter, so it carries more accumulated
# rounding than a single loss value; compared as a relative delta.
DEFAULT_GRAD_NORM_RTOL = 0.05


def read_metrics(jsonl_path: str) -> dict[int, dict[str, float]]:
    """Read per-step training metrics from a ``training.jsonl`` log.

    Args:
        jsonl_path: Path to a ``training.jsonl`` written by ``MetricLogger``.

    Returns:
        Mapping of step index to a dict with the ``loss`` key and, when the
        recipe reported it, ``grad_norm``.
    """
    entries: dict[int, dict[str, float]] = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "step" not in record or "loss" not in record:
                continue
            sample: dict[str, float] = {"loss": float(record["loss"])}
            grad_norm = record.get("grad_norm")
            if grad_norm is not None:
                sample["grad_norm"] = float(grad_norm)
            entries[int(record["step"])] = sample
    return entries


def main() -> None:
    """Compare a single-rank baseline log against a parallel-run log."""
    parser = argparse.ArgumentParser(description="Compare single-rank vs parallel training parity")
    parser.add_argument("baseline_jsonl", help="training.jsonl from the single-rank baseline run")
    parser.add_argument("parallel_jsonl", help="training.jsonl from the parallel run")
    parser.add_argument("--axis", required=True, help="Parallelism axis under test, e.g. pp/tp/cp/ep")
    parser.add_argument("--loss-tol", type=float, default=DEFAULT_LOSS_TOL, help="Absolute per-step loss tolerance")
    parser.add_argument(
        "--grad-norm-rtol",
        type=float,
        default=DEFAULT_GRAD_NORM_RTOL,
        help="Relative per-step gradient-norm tolerance",
    )
    args = parser.parse_args()

    baseline = read_metrics(args.baseline_jsonl)
    parallel = read_metrics(args.parallel_jsonl)

    assert len(baseline) > 0, f"No training records in {args.baseline_jsonl}"
    assert len(parallel) > 0, f"No training records in {args.parallel_jsonl}"

    common_steps = sorted(set(baseline) & set(parallel))
    assert len(common_steps) > 0, (
        f"No overlapping steps between {args.baseline_jsonl} (steps {sorted(baseline)}) "
        f"and {args.parallel_jsonl} (steps {sorted(parallel)})"
    )

    loss_failures: list[str] = []
    grad_norm_failures: list[str] = []
    compared_grad_norms = 0

    print(f"=== {args.axis} parity: {len(common_steps)} common steps ===")
    print(f"{'step':>6}  {'baseline':>12}  {'parallel':>12}  {'delta':>12}")
    for step in common_steps:
        base_loss = baseline[step]["loss"]
        par_loss = parallel[step]["loss"]
        delta = abs(base_loss - par_loss)
        print(f"{step:>6}  {base_loss:>12.6f}  {par_loss:>12.6f}  {delta:>12.6f}")
        if delta > args.loss_tol:
            loss_failures.append(f"step {step}: baseline={base_loss:.6f} {args.axis}={par_loss:.6f} delta={delta:.6f}")

        base_norm = baseline[step].get("grad_norm")
        par_norm = parallel[step].get("grad_norm")
        if base_norm is None or par_norm is None:
            continue
        compared_grad_norms += 1
        scale = max(abs(base_norm), 1e-8)
        norm_delta = abs(base_norm - par_norm) / scale
        if norm_delta > args.grad_norm_rtol:
            grad_norm_failures.append(
                f"step {step}: baseline={base_norm:.6f} {args.axis}={par_norm:.6f} rel_delta={norm_delta:.6f}"
            )

    assert not loss_failures, (
        f"{args.axis} run diverged from the single-rank baseline beyond {args.loss_tol} "
        f"in loss:\n  " + "\n  ".join(loss_failures)
    )
    assert not grad_norm_failures, (
        f"{args.axis} run diverged from the single-rank baseline beyond {args.grad_norm_rtol} "
        f"relative in gradient norm:\n  " + "\n  ".join(grad_norm_failures)
    )

    # A log without grad_norm would silently reduce this to a loss-only check.
    assert compared_grad_norms > 0, (
        "Neither log reported grad_norm, so the gradient-sync half of this check did not run. "
        "Confirm the recipe logs grad_norm to training.jsonl."
    )

    print(f"{args.axis} parity OK: {len(common_steps)} steps, {compared_grad_norms} gradient norms compared")


if __name__ == "__main__":
    main()

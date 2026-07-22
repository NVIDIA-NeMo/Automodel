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

"""Compare shared- and separate-mesh KD loss traces and publish the result."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

METRICS = ("loss", "ce_loss", "kd_loss", "grad_norm")
RUN_SETTINGS = ("kd_ratio", "temperature")


def _read_trace(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No metrics found in {path}")
    for row in rows:
        for field in (*METRICS, *RUN_SETTINGS):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Non-finite {field} in {path}: {row}")
        expected = (1.0 - row["kd_ratio"]) * row["ce_loss"] + row["kd_ratio"] * row["kd_loss"]
        if not math.isclose(row["loss"], expected, rel_tol=0.0, abs_tol=1.0e-5):
            raise ValueError(f"Mixed loss identity failed in {path}: {row}")
    return rows


def _compare_pair(
    label: str,
    baseline: list[dict],
    candidate: list[dict],
    absolute_tolerance: float,
    relative_tolerance: float,
) -> list[dict]:
    if len(baseline) != len(candidate):
        raise ValueError(f"{label}: step count differs: {len(baseline)} != {len(candidate)}")
    comparisons = []
    for baseline_row, candidate_row in zip(baseline, candidate):
        if baseline_row["step"] != candidate_row["step"]:
            raise ValueError(f"{label}: step mismatch: {baseline_row['step']} != {candidate_row['step']}")
        mismatched_settings = {
            setting: (baseline_row[setting], candidate_row[setting])
            for setting in RUN_SETTINGS
            if float(baseline_row[setting]) != float(candidate_row[setting])
        }
        if mismatched_settings:
            raise ValueError(f"{label} step {baseline_row['step']} has mismatched run settings: {mismatched_settings}")
        comparison = {"layout": label, "step": baseline_row["step"]}
        failed = {}
        for metric in METRICS:
            baseline_value = float(baseline_row[metric])
            candidate_value = float(candidate_row[metric])
            absolute_difference = abs(candidate_value - baseline_value)
            relative_difference = absolute_difference / max(abs(baseline_value), float.fromhex("0x1.0p-1022"))
            comparison[f"{metric}_abs_diff"] = absolute_difference
            comparison[f"{metric}_rel_diff"] = relative_difference
            if not math.isclose(
                candidate_value,
                baseline_value,
                rel_tol=relative_tolerance,
                abs_tol=absolute_tolerance,
            ):
                failed[metric] = {
                    "baseline": baseline_value,
                    "candidate": candidate_value,
                    "abs_diff": absolute_difference,
                    "rel_diff": relative_difference,
                }
        comparisons.append(comparison)
        if failed:
            raise ValueError(
                f"{label} step {baseline_row['step']} exceeds tolerances "
                f"(atol={absolute_tolerance}, rtol={relative_tolerance}): {failed}"
            )
    return comparisons


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared-dense", type=Path, required=True)
    parser.add_argument("--separate-tp", type=Path, required=True)
    parser.add_argument("--separate-pp", type=Path, required=True)
    parser.add_argument("--shared-moe", type=Path, required=True)
    parser.add_argument("--separate-ep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-2)
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    shared_dense = _read_trace(args.shared_dense)
    shared_moe = _read_trace(args.shared_moe)
    comparisons = [
        *_compare_pair(
            "teacher_tp2",
            shared_dense,
            _read_trace(args.separate_tp),
            args.absolute_tolerance,
            args.relative_tolerance,
        ),
        *_compare_pair(
            "teacher_pp3",
            shared_dense,
            _read_trace(args.separate_pp),
            args.absolute_tolerance,
            args.relative_tolerance,
        ),
        *_compare_pair(
            "teacher_ep8",
            shared_moe,
            _read_trace(args.separate_ep),
            args.absolute_tolerance,
            args.relative_tolerance,
        ),
    ]
    summary = {
        "pass": True,
        "absolute_tolerance": args.absolute_tolerance,
        "relative_tolerance": args.relative_tolerance,
        "max_abs_diff": {metric: max(row[f"{metric}_abs_diff"] for row in comparisons) for metric in METRICS},
        "max_rel_diff": {metric: max(row[f"{metric}_rel_diff"] for row in comparisons) for metric in METRICS},
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if args.wandb:
        wandb_dir = args.output.parent / "wandb_summary"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WANDB_DATA_DIR", str(wandb_dir / "data"))
        os.environ.setdefault("WANDB_DIR", str(wandb_dir))

        import wandb

        run = wandb.init(
            entity="Nemo-automodel",
            project="kd_sep_mesh",
            name="loss_parity_summary",
            job_type="validation",
            tags=["kd", "loss-parity", "cw-dfw"],
            dir=str(wandb_dir),
            config={
                "absolute_tolerance": args.absolute_tolerance,
                "relative_tolerance": args.relative_tolerance,
            },
        )
        table_columns = ["layout", "step"]
        for metric in METRICS:
            table_columns.extend((f"{metric}_abs_diff", f"{metric}_rel_diff"))
        table = wandb.Table(columns=table_columns)
        for row in comparisons:
            table.add_data(row["layout"], row["step"], *(row[column] for column in table_columns[2:]))
        run.log({"loss_parity": table})
        for metric, value in summary["max_abs_diff"].items():
            run.summary[f"max_abs_diff/{metric}"] = value
        for metric, value in summary["max_rel_diff"].items():
            run.summary[f"max_rel_diff/{metric}"] = value
        run.summary["pass"] = True
        run.finish()

    print("KD_SEP_MESH_PARITY", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
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

"""Downstream-eval gate for the Tulu-3 weekly convergence CI.

Reads a recipe's ``ci.downstream_eval`` block, runs IFEval on the trained
consolidated checkpoint via ``examples/convergence/tulu3/eval/run_eval.sh``, and
asserts the score is within ``k * stderr`` of the recorded baseline.

PASS (exit 0) iff ``abs(ci_score - baseline) < k * stderr``.

Invoked by convergence_tests_launcher.sh after training + eval-env setup:

    python3 convergence_eval.py \
        --recipe examples/convergence/tulu3/models/moonlight-16b/moonlight_16b_ep8_te_fusedadam.yaml \
        --checkpoint-dir checkpoints_convergence/moonlight_16b_te_fusedadam \
        --output-dir "$TEST_DIR"
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import yaml

AUTOMODEL_DIR = "/opt/Automodel"
RUN_EVAL = "examples/convergence/tulu3/eval/run_eval.sh"


def _load_downstream_eval(recipe_path: str) -> dict:
    with open(recipe_path, "r", encoding="utf-8") as f:
        recipe = yaml.safe_load(f) or {}
    cfg = ((recipe.get("ci") or {}).get("downstream_eval")) or {}
    if not cfg:
        sys.exit(f"[convergence_eval] no ci.downstream_eval block in {recipe_path}")
    for req in ("metric", "baseline", "stderr"):
        if req not in cfg:
            sys.exit(f"[convergence_eval] ci.downstream_eval missing required key '{req}'")
    return cfg


def _find_consolidated_checkpoint(checkpoint_dir: str) -> str:
    """Resolve <checkpoint_dir>/<latest step>/model/consolidated."""
    latest = os.path.join(checkpoint_dir, "LATEST")
    if os.path.exists(latest):
        step_dir = os.path.realpath(latest)
    else:
        # Fall back to the highest epoch_*/step_* directory.
        candidates = sorted(glob.glob(os.path.join(checkpoint_dir, "epoch_*_step_*")))
        if not candidates:
            sys.exit(f"[convergence_eval] no checkpoint under {checkpoint_dir}")
        step_dir = candidates[-1]
    consolidated = os.path.join(step_dir, "model", "consolidated")
    if not os.path.isdir(consolidated):
        sys.exit(f"[convergence_eval] consolidated checkpoint not found at {consolidated}")
    return consolidated


def _run_eval(cfg: dict, checkpoint: str, output_dir: str) -> None:
    tokenizer = cfg.get("tokenizer", "")
    # `checkpoint` sentinel -> use the trained checkpoint's own tokenizer (it carries
    # the chat template), e.g. gemma4 whose base ships none.
    if tokenizer == "checkpoint":
        tokenizer = checkpoint

    cmd = [
        "bash",
        RUN_EVAL,
        "--model-path",
        checkpoint,
        "--tokenizer",
        tokenizer,
        "--tasks",
        str(cfg.get("tasks", "ifeval")),
        "--tp-size",
        str(cfg.get("tp_size", 1)),
        "--dp-size",
        str(cfg.get("dp_size", 1)),
        "--output-path",
        os.path.join(output_dir, "eval_results"),
    ]
    if cfg.get("thinking"):
        cmd.append("--thinking")
    if cfg.get("extra_model_args"):
        cmd += ["--extra-model-args", str(cfg["extra_model_args"])]
    if cfg.get("gen_kwargs"):
        cmd += ["--gen-kwargs", str(cfg["gen_kwargs"])]

    print("[convergence_eval] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=AUTOMODEL_DIR, check=True)


def _read_metric(output_dir: str, task: str, metric: str) -> float:
    results = sorted(glob.glob(os.path.join(output_dir, "eval_results", "**", "results_*.json"), recursive=True))
    if not results:
        sys.exit(f"[convergence_eval] no results_*.json under {output_dir}/eval_results")
    with open(results[-1], "r", encoding="utf-8") as f:
        data = json.load(f)
    task_res = data.get("results", {}).get(task, {})
    # lm-eval keys are "<metric>,<filter>" (filter is "none" for ifeval).
    for key, value in task_res.items():
        if key.split(",")[0] == metric:
            return float(value)
    sys.exit(f"[convergence_eval] metric '{metric}' not found in {results[-1]} (have: {list(task_res)})")


def main() -> int:
    p = argparse.ArgumentParser(description="Convergence downstream-eval threshold gate.")
    p.add_argument("--recipe", required=True, help="Recipe YAML with the ci.downstream_eval block")
    p.add_argument("--checkpoint-dir", required=True, help="checkpoint.checkpoint_dir from the resolved config")
    p.add_argument("--output-dir", required=True, help="Directory for eval outputs")
    args = p.parse_args()

    cfg = _load_downstream_eval(args.recipe)
    checkpoint = _find_consolidated_checkpoint(args.checkpoint_dir)
    _run_eval(cfg, checkpoint, args.output_dir)

    score = _read_metric(args.output_dir, str(cfg.get("tasks", "ifeval")), str(cfg["metric"]))
    baseline = float(cfg["baseline"])
    stderr = float(cfg["stderr"])
    k = float(cfg.get("k", 2))
    tol = k * stderr
    diff = abs(score - baseline)
    passed = diff < tol

    print(
        f"[convergence_eval] metric={cfg['metric']} score={score:.4f} baseline={baseline:.4f} "
        f"|diff|={diff:.4f} tol={k}*{stderr:.4f}={tol:.4f} -> {'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

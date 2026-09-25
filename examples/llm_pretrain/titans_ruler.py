#!/usr/bin/env python3
"""Run official NVIDIA RULER data generation/scoring with Titans inference."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ruler-repo", type=Path, required=True)
    parser.add_argument("--ruler-revision", default="ab17b785")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[2048])
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["niah_single_1", "niah_single_2", "niah_single_3"],
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--model-template-type", default="base")
    return parser.parse_args()


def run(command: list[str], *, cwd: Path) -> None:
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def verify_ruler_revision(repo: Path, expected: str) -> None:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()
    if not revision.startswith(expected):
        raise RuntimeError(
            f"RULER checkout is {revision}, expected a revision beginning with {expected}"
        )


def score_s_niah_predictions(prediction_dir: Path, tasks: list[str]) -> None:
    """Write the official S-NIAH substring metric without NeMo Toolkit."""
    rows = []
    for task in tasks:
        predictions = [
            json.loads(line)
            for line in (prediction_dir / f"{task}.jsonl").read_text().splitlines()
            if line.strip()
        ]
        sample_scores = []
        for prediction in predictions:
            answers = prediction["outputs"]
            text = prediction["pred"].lower()
            sample_scores.append(sum(answer.lower() in text for answer in answers) / len(answers))
        rows.append(
            {
                "task": task,
                "score": 100.0 * sum(sample_scores) / len(sample_scores),
                "nulls": sum(not prediction["pred"].strip() for prediction in predictions),
                "num_samples": len(predictions),
            }
        )
    with (prediction_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task", "score", "nulls", "num_samples"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    ruler_repo = args.ruler_repo.resolve()
    scripts = ruler_repo / "scripts"
    predictor = Path(__file__).with_name("titans_ruler_predict.py").resolve()
    verify_ruler_revision(ruler_repo, args.ruler_revision)

    for context_length in args.context_lengths:
        length_root = args.output_dir.resolve() / str(context_length)
        data_dir = length_root / "data"
        prediction_dir = length_root / "pred"
        data_dir.mkdir(parents=True, exist_ok=True)
        prediction_dir.mkdir(parents=True, exist_ok=True)

        for task in args.tasks:
            run(
                [
                    sys.executable,
                    "data/prepare.py",
                    "--save_dir",
                    str(data_dir),
                    "--benchmark",
                    "synthetic",
                    "--task",
                    task,
                    "--tokenizer_path",
                    args.tokenizer,
                    "--tokenizer_type",
                    "hf",
                    "--max_seq_length",
                    str(context_length),
                    "--model_template_type",
                    args.model_template_type,
                    "--num_samples",
                    str(args.num_samples),
                ],
                cwd=scripts,
            )
            run(
                [
                    sys.executable,
                    str(predictor),
                    "--checkpoint",
                    str(args.checkpoint),
                    "--tokenizer",
                    args.tokenizer,
                    "--data-jsonl",
                    str(data_dir / task / "validation.jsonl"),
                    "--output-jsonl",
                    str(prediction_dir / f"{task}.jsonl"),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                ],
                cwd=scripts,
            )

        try:
            run(
                [
                    sys.executable,
                    "eval/evaluate.py",
                    "--data_dir",
                    str(prediction_dir),
                    "--benchmark",
                    "synthetic",
                ],
                cwd=scripts,
            )
        except subprocess.CalledProcessError:
            print(
                "Official scorer dependencies are unavailable; "
                "writing the equivalent S-NIAH substring metric locally.",
                flush=True,
            )
            score_s_niah_predictions(prediction_dir, args.tasks)


if __name__ == "__main__":
    main()

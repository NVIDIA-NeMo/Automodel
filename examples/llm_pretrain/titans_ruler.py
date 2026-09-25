#!/usr/bin/env python3
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
"""Prepare, shard, merge, and score official NVIDIA RULER evaluations."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse RULER orchestration arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ruler-repo", type=Path, required=True)
    parser.add_argument("--ruler-revision", default="ab17b785")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, help="Shared prepared-data root; context length is appended")
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[2048])
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["niah_single_1", "niah_single_2", "niah_single_3"],
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--stop-word", action="append", default=[])
    parser.add_argument("--disable-ttt-updates", action="store_true")
    parser.add_argument("--model-template-type", default="base")
    parser.add_argument("--mode", choices=("all", "prepare", "predict", "merge"), default="all")
    return parser.parse_args()


def run(command: list[str], *, cwd: Path) -> None:
    """Run a subprocess and log its command."""
    LOGGER.info("+ %s", " ".join(str(part) for part in command))
    subprocess.run(command, cwd=cwd, check=True)


def git_revision(repo: Path) -> str:
    """Return the full Git revision for a checkout."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def verify_ruler_revision(repo: Path, expected: str) -> str:
    """Verify and return the RULER checkout revision."""
    revision = git_revision(repo)
    if not revision.startswith(expected):
        raise RuntimeError(f"RULER checkout is {revision}, expected a revision beginning with {expected}")
    return revision


def count_jsonl_rows(path: Path) -> int:
    """Count nonblank records in a JSONL file."""
    with path.open() as stream:
        return sum(bool(line.strip()) for line in stream)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    """Atomically write formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def prepare_dataset(
    *,
    scripts: Path,
    data_dir: Path,
    task: str,
    tokenizer: str,
    context_length: int,
    num_samples: int,
    model_template_type: str,
    ruler_revision: str,
) -> Path:
    """Prepare one task dataset under an inter-process lock and safely reuse it."""
    task_dir = data_dir / task
    validation_path = task_dir / "validation.jsonl"
    manifest_path = task_dir / "manifest.json"
    expected_manifest = {
        "tokenizer": tokenizer,
        "ruler_revision": ruler_revision,
        "task": task,
        "context_length": context_length,
        "requested_sample_count": num_samples,
        "generation_settings": {
            "benchmark": "synthetic",
            "model_template_type": model_template_type,
            "tokenizer_type": "hf",
        },
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    lock_path = data_dir / f".{task}.prepare.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (
            validation_path.exists()
            and manifest_path.exists()
            and json.loads(manifest_path.read_text()) == expected_manifest
            and count_jsonl_rows(validation_path) == num_samples
        ):
            return validation_path

        with tempfile.TemporaryDirectory(prefix=f".{task}.prepare-", dir=data_dir) as temporary:
            temporary_dir = Path(temporary)
            run(
                [
                    sys.executable,
                    "data/prepare.py",
                    "--save_dir",
                    str(temporary_dir),
                    "--benchmark",
                    "synthetic",
                    "--task",
                    task,
                    "--tokenizer_path",
                    tokenizer,
                    "--tokenizer_type",
                    "hf",
                    "--max_seq_length",
                    str(context_length),
                    "--model_template_type",
                    model_template_type,
                    "--num_samples",
                    str(num_samples),
                ],
                cwd=scripts,
            )
            generated = temporary_dir / task / "validation.jsonl"
            generated_count = count_jsonl_rows(generated)
            if generated_count != num_samples:
                raise ValueError(f"RULER generated {generated_count} rows for {task}, expected {num_samples}")
            task_dir.mkdir(parents=True, exist_ok=True)
            generated.replace(validation_path)
            write_json_atomic(manifest_path, expected_manifest)
    return validation_path


def merge_prediction_shards(
    shard_paths: list[Path],
    output_path: Path,
    expected_start: int,
    expected_end: int,
) -> list[dict[str, Any]]:
    """Validate and deterministically merge prediction shards by ordinal."""
    rows_by_ordinal: dict[int, dict[str, Any]] = {}
    for shard_path in sorted(shard_paths, key=lambda path: path.name):
        with shard_path.open() as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                ordinal = row.get("_sample_ordinal")
                if not isinstance(ordinal, int):
                    raise ValueError(f"{shard_path}:{line_number}: missing or invalid '_sample_ordinal'")
                if not expected_start <= ordinal < expected_end:
                    raise ValueError(
                        f"{shard_path}:{line_number}: ordinal {ordinal} outside [{expected_start}, {expected_end})"
                    )
                if ordinal in rows_by_ordinal:
                    raise ValueError(f"duplicate prediction ordinal {ordinal} in {shard_path}")
                rows_by_ordinal[ordinal] = row

    expected_ordinals = set(range(expected_start, expected_end))
    actual_ordinals = set(rows_by_ordinal)
    if actual_ordinals != expected_ordinals:
        missing = sorted(expected_ordinals - actual_ordinals)
        unexpected = sorted(actual_ordinals - expected_ordinals)
        raise ValueError(
            f"cannot merge {len(actual_ordinals)} rows; expected {len(expected_ordinals)} unique ordinals "
            f"(missing={missing[:10]}, unexpected={unexpected[:10]})"
        )

    rows = [rows_by_ordinal[ordinal] for ordinal in range(expected_start, expected_end)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output_path.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        for row in rows:
            temporary.write(json.dumps(row) + "\n")
    temporary_path.replace(output_path)
    return rows


def validate_shard_manifests(shard_paths: list[Path], expected: dict[str, Any]) -> None:
    """Reject shards produced by a different evaluation run."""
    for shard_path in shard_paths:
        manifest_path = Path(f"{shard_path}.manifest.json")
        if not manifest_path.exists():
            raise ValueError(f"{shard_path} has no manifest")
        manifest = json.loads(manifest_path.read_text())
        for key, expected_value in expected.items():
            if isinstance(expected_value, dict) and isinstance(manifest.get(key), dict):
                actual = manifest[key]
                mismatches = {
                    nested_key: (actual.get(nested_key), nested_value)
                    for nested_key, nested_value in expected_value.items()
                    if actual.get(nested_key) != nested_value
                }
                if mismatches:
                    raise ValueError(f"{manifest_path}: {key} mismatches {mismatches!r}")
                continue
            if manifest.get(key) != expected_value:
                raise ValueError(f"{manifest_path}: {key}={manifest.get(key)!r}, expected {expected_value!r}")
        shard_range = manifest.get("shard_range")
        if (
            not isinstance(shard_range, dict)
            or not isinstance(shard_range.get("start"), int)
            or not isinstance(shard_range.get("end"), int)
            or shard_range["start"] >= shard_range["end"]
        ):
            raise ValueError(f"{manifest_path}: invalid shard_range {shard_range!r}")


def score_s_niah_predictions(prediction_dir: Path, tasks: list[str]) -> list[dict[str, Any]]:
    """Score merged raw JSONL with the official S-NIAH substring metric."""
    rows: list[dict[str, Any]] = []
    for task in tasks:
        predictions = [
            json.loads(line) for line in (prediction_dir / f"{task}.jsonl").read_text().splitlines() if line.strip()
        ]
        if not predictions:
            raise ValueError(f"cannot score empty predictions for {task}")
        sample_scores = []
        for prediction in predictions:
            answers = prediction["outputs"]
            if not answers:
                raise ValueError(f"prediction ordinal {prediction.get('_sample_ordinal')} has no answers")
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
    return rows


def main() -> None:
    """Run the requested RULER orchestration phase."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    sample_end = args.sample_end if args.sample_end is not None else args.num_samples
    if args.sample_start < 0 or sample_end < args.sample_start or sample_end > args.num_samples:
        raise ValueError(f"invalid sample range [{args.sample_start}, {sample_end}) for {args.num_samples} samples")
    if args.mode == "all" and (args.sample_start != 0 or sample_end != args.num_samples):
        raise ValueError("partial ranges require --mode predict; run --mode merge after all shards complete")

    ruler_repo = args.ruler_repo.resolve()
    scripts = ruler_repo / "scripts"
    predictor = Path(__file__).with_name("titans_ruler_predict.py").resolve()
    ruler_revision = verify_ruler_revision(ruler_repo, args.ruler_revision)
    code_revision = git_revision(Path(__file__).resolve().parents[2])

    for context_length in args.context_lengths:
        length_root = args.output_dir.resolve() / str(context_length)
        data_dir = args.data_dir.resolve() / str(context_length) if args.data_dir is not None else length_root / "data"
        prediction_dir = length_root / "pred"
        prediction_dir.mkdir(parents=True, exist_ok=True)

        for task in args.tasks:
            validation_path = data_dir / task / "validation.jsonl"
            if args.mode in {"all", "prepare", "predict"}:
                validation_path = prepare_dataset(
                    scripts=scripts,
                    data_dir=data_dir,
                    task=task,
                    tokenizer=args.tokenizer,
                    context_length=context_length,
                    num_samples=args.num_samples,
                    model_template_type=args.model_template_type,
                    ruler_revision=ruler_revision,
                )
            if args.mode in {"all", "predict"}:
                shard_dir = prediction_dir / "shards" / task
                shard_path = shard_dir / f"{args.sample_start:08d}-{sample_end:08d}.jsonl"
                command = [
                    sys.executable,
                    str(predictor),
                    "--checkpoint",
                    str(args.checkpoint),
                    "--tokenizer",
                    args.tokenizer,
                    "--data-jsonl",
                    str(validation_path),
                    "--output-jsonl",
                    str(shard_path),
                    "--sample-start",
                    str(args.sample_start),
                    "--sample-end",
                    str(sample_end),
                    "--requested-samples",
                    str(args.num_samples),
                    "--task",
                    task,
                    "--context-length",
                    str(context_length),
                    "--ruler-revision",
                    ruler_revision,
                    "--code-revision",
                    code_revision,
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                ]
                if args.disable_ttt_updates:
                    command.append("--disable-ttt-updates")
                for stop_word in args.stop_word:
                    command.extend(["--stop-word", stop_word])
                run(command, cwd=scripts)

        if args.mode in {"all", "merge"}:
            for task in args.tasks:
                shard_paths = list((prediction_dir / "shards" / task).glob("*.jsonl"))
                merged_path = prediction_dir / f"{task}.jsonl"
                validate_shard_manifests(
                    shard_paths,
                    {
                        "checkpoint": str(args.checkpoint.resolve()),
                        "tokenizer": args.tokenizer,
                        "ruler_revision": ruler_revision,
                        "code_git_revision": code_revision,
                        "task": task,
                        "context_length": context_length,
                        "requested_sample_count": args.num_samples,
                        "generation_settings": {
                            "max_new_tokens": args.max_new_tokens,
                            "stop_words": args.stop_word,
                            "enable_ttt_updates": not args.disable_ttt_updates,
                        },
                    },
                )
                merge_prediction_shards(shard_paths, merged_path, 0, args.num_samples)
                write_json_atomic(
                    Path(f"{merged_path}.manifest.json"),
                    {
                        "checkpoint": str(args.checkpoint.resolve()),
                        "tokenizer": args.tokenizer,
                        "ruler_revision": ruler_revision,
                        "code_git_revision": code_revision,
                        "task": task,
                        "context_length": context_length,
                        "requested_sample_count": args.num_samples,
                        "shard_range": {"start": 0, "end": args.num_samples},
                        "generation_settings": {
                            "max_new_tokens": args.max_new_tokens,
                            "stop_words": args.stop_word,
                            "enable_ttt_updates": not args.disable_ttt_updates,
                        },
                        "source_shards": [path.name for path in sorted(shard_paths)],
                    },
                )
            score_s_niah_predictions(prediction_dir, args.tasks)


if __name__ == "__main__":
    main()

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
"""Generate a resumable prediction shard for an official NVIDIA RULER task."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse prediction arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--data-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--sample-end", type=int)
    parser.add_argument("--requested-samples", type=int)
    parser.add_argument("--task")
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--ruler-revision", default="unknown")
    parser.add_argument("--code-revision")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--stop-word", action="append", default=[])
    parser.add_argument("--disable-ttt-updates", action="store_true")
    parser.add_argument("--device")
    return parser.parse_args()


def count_samples(path: Path) -> int:
    """Count nonblank JSONL records."""
    with path.open() as stream:
        return sum(bool(line.strip()) for line in stream)


def validate_sample_range(sample_start: int, sample_end: int, sample_count: int) -> None:
    """Validate a zero-based, half-open sample range."""
    if sample_start < 0 or sample_end < sample_start:
        raise ValueError(f"invalid sample range [{sample_start}, {sample_end})")
    if sample_end > sample_count:
        raise ValueError(f"sample range ends at {sample_end}, but dataset has only {sample_count} records")


def iter_samples(path: Path, sample_start: int, sample_end: int) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield records selected by zero-based ordinal, ignoring blank lines."""
    ordinal = 0
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            if ordinal >= sample_end:
                break
            if ordinal >= sample_start:
                sample = json.loads(line)
                missing = {"index", "input", "outputs"} - sample.keys()
                if missing:
                    raise ValueError(f"{path}:{line_number}: missing fields {sorted(missing)}")
                yield ordinal, sample
            ordinal += 1


def load_completed_samples(path: Path, sample_start: int, sample_end: int) -> set[int]:
    """Read completed ordinals, rejecting output that is unsafe to resume."""
    if not path.exists():
        return set()
    completed: set[int] = set()
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            ordinal = row.get("_sample_ordinal")
            if not isinstance(ordinal, int):
                raise ValueError(
                    f"{path}:{line_number}: missing or invalid '_sample_ordinal'; "
                    "refusing an unsafe resume because RULER 'index' values are not unique"
                )
            if not sample_start <= ordinal < sample_end:
                raise ValueError(
                    f"{path}:{line_number}: ordinal {ordinal} is outside shard range [{sample_start}, {sample_end})"
                )
            if ordinal in completed:
                raise ValueError(f"{path}:{line_number}: duplicate ordinal {ordinal}; refusing unsafe resume")
            completed.add(ordinal)
    return completed


def git_revision(path: Path) -> str:
    """Return the code checkout revision, or ``unknown`` outside Git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Create or validate immutable shard metadata."""
    manifest_path = Path(f"{path}.manifest.json")
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest:
            raise ValueError(f"{manifest_path} does not match this prediction run")
        return
    if path.exists() and path.stat().st_size:
        raise ValueError(f"{path} has predictions but no manifest; refusing an unsafe resume")
    temporary = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)


def main() -> None:
    """Generate the requested prediction shard."""
    args = parse_args()
    import torch

    sample_count = count_samples(args.data_jsonl)
    requested_samples = args.requested_samples if args.requested_samples is not None else sample_count
    sample_end = args.sample_end if args.sample_end is not None else requested_samples
    if requested_samples != sample_count:
        raise ValueError(
            f"prepared dataset has {sample_count} records, expected --requested-samples={requested_samples}"
        )
    validate_sample_range(args.sample_start, sample_end, sample_count)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": str(args.checkpoint.resolve()),
        "tokenizer": args.tokenizer,
        "ruler_revision": args.ruler_revision,
        "code_git_revision": args.code_revision or git_revision(Path(__file__).resolve().parent),
        "task": args.task or args.data_jsonl.parent.name,
        "context_length": args.context_length,
        "requested_sample_count": requested_samples,
        "shard_range": {"start": args.sample_start, "end": sample_end},
        "generation_settings": {
            "max_new_tokens": args.max_new_tokens,
            "stop_words": args.stop_word,
            "enable_ttt_updates": not args.disable_ttt_updates,
        },
        "device": device,
    }
    write_manifest(args.output_jsonl, manifest)
    completed = load_completed_samples(args.output_jsonl, args.sample_start, sample_end)
    if len(completed) == sample_end - args.sample_start:
        return

    from titans_generate import TitansGenerator

    generator = TitansGenerator(args.checkpoint, args.tokenizer, device)

    with args.output_jsonl.open("a", buffering=1) as output:
        for ordinal, sample in iter_samples(args.data_jsonl, args.sample_start, sample_end):
            if ordinal in completed:
                continue
            result = generator.generate(
                sample["input"],
                max_new_tokens=args.max_new_tokens,
                enable_ttt_updates=not args.disable_ttt_updates,
            )
            prediction = result["text"]
            for stop_word in args.stop_word:
                prediction = prediction.split(stop_word, maxsplit=1)[0]
            output.write(
                json.dumps(
                    {
                        **sample,
                        "_sample_ordinal": ordinal,
                        "pred": prediction,
                        "prompt_tokens": result["prompt_tokens"],
                        "generated_tokens": result["generated_tokens"],
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate predictions for an official NVIDIA RULER task JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from titans_generate import TitansGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--data-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--stop-word", action="append", default=[])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_completed_samples(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed = set()
    with path.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "_sample_ordinal" not in row:
                raise ValueError(
                    f"{path}:{line_number}: missing '_sample_ordinal'; "
                    "refusing an unsafe resume because RULER 'index' values are not unique"
                )
            completed.add(row["_sample_ordinal"])
    return completed


def main() -> None:
    args = parse_args()
    completed = load_completed_samples(args.output_jsonl)
    generator = TitansGenerator(args.checkpoint, args.tokenizer, args.device)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with args.data_jsonl.open() as source, args.output_jsonl.open("a", buffering=1) as output:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            missing = {"index", "input", "outputs"} - sample.keys()
            if missing:
                raise ValueError(f"{args.data_jsonl}:{line_number}: missing fields {sorted(missing)}")
            if line_number in completed:
                continue

            result = generator.generate(
                sample["input"],
                max_new_tokens=args.max_new_tokens,
            )
            prediction = result["text"]
            for stop_word in args.stop_word:
                prediction = prediction.split(stop_word, maxsplit=1)[0]

            output.write(
                json.dumps(
                    {
                        **sample,
                        "_sample_ordinal": line_number,
                        "pred": prediction,
                        "prompt_tokens": result["prompt_tokens"],
                        "generated_tokens": result["generated_tokens"],
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()

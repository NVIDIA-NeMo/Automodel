# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe whether a Titans checkpoint stores an answer even when free generation fails."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from examples.llm_pretrain.titans_generate import TitansGenerator


def _target_metrics(
    generator: TitansGenerator,
    prompt: str,
    target: str,
    *,
    enable_ttt_updates: bool,
) -> dict[str, float | int]:
    tokenizer = generator.tokenizer
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(generator.device)
    target_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids.to(generator.device)
    input_ids = torch.cat((prompt_ids, target_ids), dim=1)
    with torch.no_grad():
        logits = generator.model(input_ids, enable_ttt_updates=enable_ttt_updates).logits.float()
    target_logits = logits[:, prompt_ids.shape[1] - 1 : input_ids.shape[1] - 1]
    losses = F.cross_entropy(
        target_logits.reshape(-1, target_logits.shape[-1]),
        target_ids.reshape(-1),
        reduction="none",
    )
    predictions = target_logits.argmax(dim=-1)
    first_logits = target_logits[0, 0]
    first_target = target_ids[0, 0]
    first_rank = int((first_logits > first_logits[first_target]).sum().item()) + 1
    return {
        "target_tokens": int(target_ids.numel()),
        "target_nll": float(losses.mean().item()),
        "target_perplexity": float(math.exp(min(losses.mean().item(), 30.0))),
        "target_token_accuracy": float((predictions == target_ids).float().mean().item()),
        "first_target_rank": first_rank,
    }


def probe_sample(
    generator: TitansGenerator,
    sample: dict[str, Any],
    *,
    enable_ttt_updates: bool,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Measure prefixed generation and teacher-forced answer likelihood."""
    answer = str(sample["outputs"][0])
    answer_prefix = str(sample.get("answer_prefix", ""))
    prefixed_prompt = str(sample["input"]) + answer_prefix
    generation = generator.generate(
        prefixed_prompt,
        max_new_tokens=max_new_tokens,
        enable_ttt_updates=enable_ttt_updates,
    )
    prediction = generation["text"]
    return {
        "index": sample["index"],
        "answer": answer,
        "answer_prefix": answer_prefix,
        "prefixed_prediction": prediction,
        "prefixed_exact_contains": answer.lower() in prediction.lower(),
        "prompt_tokens": generation["prompt_tokens"],
        **_target_metrics(
            generator,
            prefixed_prompt,
            answer,
            enable_ttt_updates=enable_ttt_updates,
        ),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--disable-ttt-updates", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Run retrieval probes and write per-example and aggregate metrics."""
    args = parse_args()
    generator = TitansGenerator(args.checkpoint, args.tokenizer, args.device)
    samples = []
    with args.data_jsonl.open() as stream:
        for line in stream:
            if line.strip():
                samples.append(json.loads(line))
            if len(samples) == args.samples:
                break
    if len(samples) != args.samples:
        raise ValueError(f"requested {args.samples} samples, found {len(samples)}")
    records = [
        probe_sample(
            generator,
            sample,
            enable_ttt_updates=not args.disable_ttt_updates,
            max_new_tokens=args.max_new_tokens,
        )
        for sample in samples
    ]
    aggregate = {
        "checkpoint": str(args.checkpoint),
        "samples": len(records),
        "prefixed_exact_accuracy": sum(record["prefixed_exact_contains"] for record in records) / len(records),
        "mean_target_nll": sum(record["target_nll"] for record in records) / len(records),
        "mean_target_token_accuracy": sum(record["target_token_accuracy"] for record in records) / len(records),
        "mean_first_target_rank": sum(record["first_target_rank"] for record in records) / len(records),
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in aggregate.items() if key != "records"}, sort_keys=True))


if __name__ == "__main__":
    main()

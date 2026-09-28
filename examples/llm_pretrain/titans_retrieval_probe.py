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
) -> dict[str, Any]:
    tokenizer = generator.tokenizer
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(generator.device)
    response_text = " " + target
    input_ids = tokenizer(prompt + response_text, return_tensors="pt", add_special_tokens=True).input_ids.to(
        generator.device
    )
    prefix_ok = torch.equal(input_ids[:, : prompt_ids.shape[1]], prompt_ids)
    if prefix_ok:
        target_ids = input_ids[:, prompt_ids.shape[1] :]
    else:
        target_ids = tokenizer(response_text, return_tensors="pt", add_special_tokens=False).input_ids.to(
            generator.device
        )
        input_ids = torch.cat((prompt_ids, target_ids), dim=1)
    with torch.no_grad():
        logits = generator.model(input_ids, enable_ttt_updates=enable_ttt_updates).logits.float()
        prompt_last_logits = generator.model(
            prompt_ids,
            logits_to_keep=1,
            enable_ttt_updates=enable_ttt_updates,
        ).logits[:, -1].float()
    target_logits = logits[:, prompt_ids.shape[1] - 1 : input_ids.shape[1] - 1]
    losses = F.cross_entropy(
        target_logits.reshape(-1, target_logits.shape[-1]),
        target_ids.reshape(-1),
        reduction="none",
    )
    predictions = target_logits.argmax(dim=-1)
    answer_offset = next(
        (
            index
            for index, token_id in enumerate(target_ids[0])
            if tokenizer.decode(token_id).strip()
        ),
        None,
    )
    if answer_offset is None:
        raise ValueError(f"target contains no non-whitespace tokens: {target!r}")
    answer_losses = losses[answer_offset:]
    answer_predictions = predictions[:, answer_offset:]
    answer_ids = target_ids[:, answer_offset:]
    first_answer_logits = target_logits[0, answer_offset]
    first_answer_token = answer_ids[0, 0]
    first_answer_rank = (
        int((first_answer_logits > first_answer_logits[first_answer_token]).sum().item()) + 1
    )
    first_answer_prediction = first_answer_logits.argmax()
    boundary_teacher_prediction = target_logits[0, 0].argmax()
    boundary_prompt_prediction = prompt_last_logits[0].argmax()
    return {
        "answer_token_offset": answer_offset,
        "answer_tokens": int(answer_ids.numel()),
        "answer_nll": float(answer_losses.mean().item()),
        "answer_perplexity": float(math.exp(min(answer_losses.mean().item(), 30.0))),
        "answer_token_accuracy": float((answer_predictions == answer_ids).float().mean().item()),
        "tokenization_prefix_ok": prefix_ok,
        "first_answer_rank": first_answer_rank,
        "first_answer_token_id": int(first_answer_token.item()),
        "first_answer_token_text": tokenizer.decode(first_answer_token),
        "first_answer_prediction_id": int(first_answer_prediction.item()),
        "first_answer_prediction_text": tokenizer.decode(first_answer_prediction),
        "boundary_teacher_prediction_id": int(boundary_teacher_prediction.item()),
        "boundary_teacher_prediction_text": tokenizer.decode(boundary_teacher_prediction),
        "boundary_prompt_prediction_id": int(boundary_prompt_prediction.item()),
        "boundary_prompt_prediction_text": tokenizer.decode(boundary_prompt_prediction),
        "prefix_logit_max_abs_diff": float(
            (target_logits[0, 0] - prompt_last_logits[0]).abs().max().item()
        ),
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
        "mean_answer_nll": sum(record["answer_nll"] for record in records) / len(records),
        "mean_answer_token_accuracy": sum(record["answer_token_accuracy"] for record in records) / len(records),
        "mean_first_answer_rank": sum(record["first_answer_rank"] for record in records) / len(records),
        "records": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in aggregate.items() if key != "records"}, sort_keys=True))


if __name__ == "__main__":
    main()

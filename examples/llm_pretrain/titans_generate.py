#!/usr/bin/env python3
"""Correctness-first generation for consolidated AutoModel Titans checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import nemo_automodel.components.models.titans  # noqa: F401 - registers Titans with HF Auto*


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", default="NousResearch/Llama-2-7b-hf")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    return parser.parse_args()


def load_requests(args: argparse.Namespace) -> list[dict]:
    if args.prompt is not None:
        return [{"id": "prompt", "prompt": args.prompt}]
    requests = []
    with args.input_jsonl.open() as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            request = json.loads(line)
            if "prompt" not in request:
                raise ValueError(f"{args.input_jsonl}:{line_number}: missing required 'prompt' field")
            request.setdefault("id", str(line_number))
            requests.append(request)
    return requests


def main() -> None:
    args = parse_args()
    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()
    if not hasattr(model, "generate_full_prefix"):
        raise TypeError(f"{args.checkpoint} did not load an AutoModel Titans causal LM")

    results = []
    for request in load_requests(args):
        encoded = tokenizer(request["prompt"], return_tensors="pt", add_special_tokens=True)
        input_ids = encoded.input_ids.to(args.device)
        max_new_tokens = int(request.get("max_new_tokens", args.max_new_tokens))
        generated = model.generate_full_prefix(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        continuation = generated[:, input_ids.shape[1] :]
        results.append(
            {
                "id": request["id"],
                "prompt_tokens": input_ids.shape[1],
                "generated_tokens": continuation.shape[1],
                "text": tokenizer.decode(continuation[0], skip_special_tokens=True),
            }
        )

    if args.output_jsonl is not None:
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.output_jsonl.open("w") as stream:
            for result in results:
                stream.write(json.dumps(result) + "\n")
    else:
        for result in results:
            print(json.dumps(result))


if __name__ == "__main__":
    main()

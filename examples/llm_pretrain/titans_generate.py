#!/usr/bin/env python3
"""Correctness-first generation for consolidated AutoModel Titans checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import nemo_automodel.components.models.titans  # noqa: F401 - registers Titans with HF Auto*


class TitansGenerator:
    """Reusable correctness-first generator for JSONL and benchmark adapters."""

    def __init__(self, checkpoint: Path, tokenizer: str, device: str) -> None:
        self.device = device
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        self.model, loading_info = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=dtype,
            trust_remote_code=True,
            key_mapping={
                r"^(.*\.memory)\.A_log$": r"\1._fp32_params.A_log",
                r"^(.*\.memory)\.dt_bias$": r"\1._fp32_params.dt_bias",
            },
            output_loading_info=True,
        )
        load_errors = {
            name: loading_info.get(name, [])
            for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
            if loading_info.get(name)
        }
        if load_errors:
            raise RuntimeError(f"Checkpoint did not load strictly from {checkpoint}: {load_errors}")
        self.model.to(device)
        self.model.eval()
        if not hasattr(self.model, "generate_full_prefix"):
            raise TypeError(f"{checkpoint} did not load an AutoModel Titans causal LM")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> dict:
        encoded = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded.input_ids.to(self.device)
        generated = self.model.generate_full_prefix(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
        )
        continuation = generated[:, input_ids.shape[1] :]
        return {
            "prompt_tokens": input_ids.shape[1],
            "generated_tokens": continuation.shape[1],
            "text": self.tokenizer.decode(continuation[0], skip_special_tokens=True),
        }


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
    generator = TitansGenerator(args.checkpoint, args.tokenizer, args.device)

    results = []
    for request in load_requests(args):
        max_new_tokens = int(request.get("max_new_tokens", args.max_new_tokens))
        result = generator.generate(
            request["prompt"],
            max_new_tokens=max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        results.append({"id": request["id"], **result})

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

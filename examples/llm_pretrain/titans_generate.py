#!/usr/bin/env python3
"""Correctness-first generation for consolidated AutoModel Titans checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import nemo_automodel.components.models.titans  # noqa: F401 - registers Titans with HF Auto*


def load_consolidated_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    """Load every tensor from a consolidated HF safetensors checkpoint."""
    index_path = checkpoint / "model.safetensors.index.json"
    single_path = checkpoint / "model.safetensors"
    if index_path.is_file():
        with index_path.open() as stream:
            weight_map = json.load(stream)["weight_map"]
        shard_names = sorted(set(weight_map.values()))
        expected_keys = set(weight_map)
    elif single_path.is_file():
        shard_names = [single_path.name]
        expected_keys = None
    else:
        raise FileNotFoundError(f"No Hugging Face safetensors found in {checkpoint}")

    state_dict: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        shard = load_file(str(checkpoint / shard_name), device="cpu")
        duplicate_keys = state_dict.keys() & shard.keys()
        if duplicate_keys:
            raise RuntimeError(f"Duplicate checkpoint tensors: {sorted(duplicate_keys)[:10]}")
        state_dict.update(shard)

    if expected_keys is not None and set(state_dict) != expected_keys:
        raise RuntimeError(
            f"Safetensors index mismatch: missing={sorted(expected_keys - state_dict.keys())[:10]}, "
            f"unexpected={sorted(state_dict.keys() - expected_keys)[:10]}"
        )
    return state_dict


class TitansGenerator:
    """Reusable correctness-first generator for JSONL and benchmark adapters."""

    def __init__(self, checkpoint: Path, tokenizer: str, device: str) -> None:
        self.device = device
        self.enable_ttt_updates = True
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True)
        config.torch_dtype = dtype
        self.model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        if not hasattr(self.model, "state_dict_adapter"):
            raise TypeError(f"{checkpoint} did not construct an AutoModel Titans causal LM")

        hf_state_dict = load_consolidated_state_dict(checkpoint)
        native_state_dict = self.model.state_dict_adapter.from_hf(hf_state_dict)
        self.model.load_state_dict(native_state_dict, strict=True)
        self.model.tie_weights()
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
        enable_ttt_updates: bool = True,
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
            enable_ttt_updates=enable_ttt_updates,
        )
        continuation = generated[:, input_ids.shape[1] :]
        return {
            "prompt_tokens": input_ids.shape[1],
            "generated_tokens": continuation.shape[1],
            "text": self.tokenizer.decode(continuation[0], skip_special_tokens=True),
        }


def parse_args() -> argparse.Namespace:
    """Parse generation command-line arguments."""
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
    parser.add_argument("--disable-ttt-updates", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    return parser.parse_args()


def load_requests(args: argparse.Namespace) -> list[dict]:
    """Load one prompt or a JSONL batch of generation requests."""
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
    """Load a consolidated checkpoint and generate requested continuations."""
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
            enable_ttt_updates=not args.disable_ttt_updates,
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

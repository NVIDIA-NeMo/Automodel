# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Build a tiny, real-text 100K--256K DiffusionGemma SFT dataset.

The source text is streamed from FineWeb-Edu, so only enough documents for the
requested examples are downloaded. Each example is a single-turn passkey task:
the long document is context, while the supervised response stays one short
DiffusionGemma canvas.
"""

from __future__ import annotations

import argparse
import json

from datasets import load_dataset
from transformers import AutoTokenizer


def _render_length(tokenizer, user: str, answer: str) -> int:
    messages = [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))


def main() -> None:
    """Stream FineWeb-Edu and write exact-length passkey chat examples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="diffusion_gemma_long_100k.jsonl")
    parser.add_argument("--target-tokens", type=int, default=100_000, choices=(100_000, 131_072, 256_000))
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--model", default="google/diffusiongemma-26B-A4B-it")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    documents = iter(load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True))

    with open(args.output, "w", encoding="utf-8") as output:
        for sample_idx in range(args.num_samples):
            passkey = f"DG-LONG-{sample_idx:04d}-7391"
            prefix = (
                f"Remember this passkey: {passkey}. Read the document below. "
                "At the end, return only the passkey.\n\nDOCUMENT START\n"
            )
            suffix = "\nDOCUMENT END\nReturn only the passkey."
            answer = passkey

            corpus_tokens: list[int] = []
            # Collect with margin for the chat template and instructions.
            while len(corpus_tokens) < args.target_tokens:
                text = next(documents)["text"]
                corpus_tokens.extend(tokenizer.encode(text + "\n\n", add_special_tokens=False))

            lo, hi = 0, min(len(corpus_tokens), args.target_tokens)
            best_user, best_length = "", 0
            while lo <= hi:
                mid = (lo + hi) // 2
                corpus = tokenizer.decode(corpus_tokens[:mid], skip_special_tokens=True)
                user = prefix + corpus + suffix
                length = _render_length(tokenizer, user, answer)
                if length <= args.target_tokens:
                    best_user, best_length = user, length
                    lo = mid + 1
                else:
                    hi = mid - 1

            row = {
                "messages": [
                    {"role": "user", "content": best_user},
                    {"role": "assistant", "content": answer},
                ],
                "source": "HuggingFaceFW/fineweb-edu/sample-10BT",
                "rendered_tokens": best_length,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"sample={sample_idx} rendered_tokens={best_length} passkey={passkey}", flush=True)


if __name__ == "__main__":
    main()

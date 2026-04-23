# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Single-GPU TE attention injection smoke test.

Loads a small HF model with ``attn_implementation="te"``, runs a few forward
passes on random input, then prints the dispatch counters so you can see how
often the TE kernel actually ran versus falling back to native SDPA.

Works on A100 (TE falls back to FA2 / cuDNN FMHA); use this to validate the
injection logic independent of the FA3 kernel.

    python tools/debug_te_attention.py --model meta-llama/Llama-3.2-1B
"""

import argparse
import logging

import torch

from nemo_automodel._transformers.te_attention import (
    get_te_attention_stats,
    inject_te_attention,
    reset_te_attention_stats,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument(
        "--with-mask",
        action="store_true",
        help="Pass an explicit attention_mask (reproduces the VLM/Gemma4 fallback case).",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from transformers import AutoModelForImageTextToText

    model = (
        AutoModelForImageTextToText.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .cuda()
        .eval()
    )

    inject_te_attention(model)
    reset_te_attention_stats()

    torch.manual_seed(0)
    # vocab_size may live under text_config for VLMs (e.g. Gemma4).
    vocab = getattr(model.config, "vocab_size", None) or getattr(
        getattr(model.config, "text_config", None), "vocab_size", None
    )
    if vocab is None:
        raise RuntimeError("Cannot infer vocab_size from model.config. Check model config structure.")
    input_ids = torch.randint(0, vocab, (args.batch, args.seq_len), device="cuda")
    attn_mask = torch.ones_like(input_ids) if args.with_mask else None

    with torch.no_grad():
        for _ in range(args.steps):
            model(input_ids=input_ids, attention_mask=attn_mask)

    stats = get_te_attention_stats()
    total = sum(stats.values())
    print("=" * 60)
    print(f"model={args.model}  with_mask={args.with_mask}  steps={args.steps}")
    print(f"te_sdpa calls: {total}")
    for k, v in stats.items():
        pct = 100.0 * v / total if total else 0.0
        print(f"  {k:<28s} {v:>6d}  ({pct:5.1f}%)")
    print("=" * 60)
    if stats["te_hits"] == 0:
        print("[!] TE kernel NEVER ran. Check fallback_mask / fallback_scale_mismatch.")
    elif stats["fallback_mask"] > stats["te_hits"]:
        print("[!] Fallback dominates — block-causal mask conversion not implemented.")
    else:
        print("[OK] TE path is running.")


if __name__ == "__main__":
    main()

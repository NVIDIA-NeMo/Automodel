# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""TE attention injection smoke test (single-GPU or multi-GPU via device_map).

Loads a HF model with ``attn_implementation="te"``, runs a few forward passes
on random input, then prints the dispatch counters so you can see how often the
TE kernel actually ran versus falling back to native SDPA.

Works on A100 (TE falls back to FA2 / cuDNN FMHA); use this to validate the
injection logic independent of the FA3 kernel.

Single GPU:
    python tools/debug_te_attention.py --model meta-llama/Llama-3.2-1B

Multi-GPU (e.g. 31B across 8×A100):
    python tools/debug_te_attention.py --model /path/to/gemma-4-31B-it --device-map auto
"""

import argparse
import logging

import torch

from nemo_automodel._transformers.te_attention import (
    get_te_attention_stats,
    inject_te_attention,
    reset_te_attention_stats,
)

logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument(
        "--device-map",
        default=None,
        help="HF device_map for multi-GPU loading (e.g. 'auto'). Default: load on cuda:0.",
    )
    p.add_argument(
        "--with-mask",
        action="store_true",
        help="Pass an explicit attention_mask (reproduces the VLM/Gemma4 fallback case).",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from transformers import AutoModelForImageTextToText

    load_kwargs: dict = {"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if args.device_map is not None:
        load_kwargs["device_map"] = args.device_map

    model = AutoModelForImageTextToText.from_pretrained(args.model, **load_kwargs)
    if args.device_map is None:
        model = model.cuda()
    model = model.eval()

    inject_te_attention(model)
    reset_te_attention_stats()

    # Determine which device to create input tensors on.
    # With device_map="auto" the first layer is on cuda:0.
    input_device = next(model.parameters()).device

    torch.manual_seed(0)
    # vocab_size may live under text_config for VLMs (e.g. Gemma4).
    vocab = getattr(model.config, "vocab_size", None) or getattr(
        getattr(model.config, "text_config", None), "vocab_size", None
    )
    if vocab is None:
        raise RuntimeError("Cannot infer vocab_size from model.config. Check model config structure.")
    input_ids = torch.randint(0, vocab, (args.batch, args.seq_len), device=input_device)
    attn_mask = torch.ones_like(input_ids) if args.with_mask else None

    with torch.no_grad():
        for _ in range(args.steps):
            model(input_ids=input_ids, attention_mask=attn_mask)

    stats = get_te_attention_stats()
    total = sum(stats.values())
    sep = "=" * 60
    logger.info(sep)
    logger.info("model=%s  with_mask=%s  steps=%d", args.model, args.with_mask, args.steps)
    logger.info("te_sdpa calls: %d", total)
    for k, v in stats.items():
        pct = 100.0 * v / total if total else 0.0
        logger.info("  %-28s %6d  (%5.1f%%)", k, v, pct)
    logger.info(sep)
    if stats["te_hits"] == 0:
        logger.warning("TE kernel NEVER ran. Check fallback_mask / fallback_scale_mismatch.")
    elif stats["fallback_mask"] > stats["te_hits"]:
        logger.warning("Fallback dominates — block-causal mask conversion not implemented.")
    else:
        logger.info("TE path is running.")


if __name__ == "__main__":
    main()

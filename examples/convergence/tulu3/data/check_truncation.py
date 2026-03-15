#!/usr/bin/env python3
"""Calculate truncation percentage for a given dataset and sequence length.

Usage:
    python scripts/check_truncation.py --dataset allenai/tulu-3-sft-mixture --seq_length 1024
    python scripts/check_truncation.py --dataset allenai/tulu-3-sft-mixture --seq_length 1024 2048 4096
    python scripts/check_truncation.py --dataset allenai/tulu-3-sft-mixture --seq_length 2048 --model Qwen/Qwen3-30B-A3B
    python scripts/check_truncation.py --dataset allenai/tulu-3-sft-mixture --seq_length 2048 --chat_template path/to/template.json
    python scripts/check_truncation.py --dataset allenai/tulu-3-sft-mixture --seq_length 2048 --num_samples 5000
"""

import argparse
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate truncation percentage for a dataset and sequence length.")
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset ID or local path")
    parser.add_argument("--seq_length", type=int, nargs="+", required=True, help="Sequence length(s) to check")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B", help="Model/tokenizer name")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--chat_template", type=str, default=None, help="Path to chat template JSON file")
    parser.add_argument("--num_samples", type=int, default=0, help="Number of samples to check (0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--hf_home", type=str, default=None, help="Override HF_HOME cache directory")
    return parser.parse_args()


def load_dataset_messages(dataset_id, split):
    """Load dataset and return it."""
    from datasets import VerificationMode, load_dataset

    dataset = load_dataset(dataset_id, split=split, streaming=False, verification_mode=VerificationMode.NO_CHECKS)
    return dataset


def compute_lengths(dataset, tokenizer, num_samples, seed):
    """Tokenize samples and return list of token lengths."""
    import random

    indices = list(range(len(dataset)))
    if num_samples > 0 and num_samples < len(dataset):
        random.seed(seed)
        random.shuffle(indices)
        indices = indices[:num_samples]

    from tqdm import tqdm

    lengths = []
    errors = 0

    for idx in tqdm(indices, desc="Tokenizing", unit="sample"):
        row = dataset[idx]
        msgs = row.get("messages", [])
        if not msgs:
            continue
        try:
            result = tokenizer.apply_chat_template(
                msgs,
                tokenize=True,
                return_dict=True,
                padding=False,
                truncation=False,
            )
            lengths.append(len(result["input_ids"]))
        except Exception:
            errors += 1

    logger.info(f"  {len(lengths)} samples tokenized, {errors} errors")
    return lengths


def print_report(lengths, seq_lengths):
    """Print truncation statistics."""
    if not lengths:
        logger.info("No samples to analyze.")
        return

    lengths_sorted = sorted(lengths)
    n = len(lengths)

    logger.info(f"\nDataset statistics ({n} samples):")
    logger.info(f"  Min:    {lengths_sorted[0]:>8,} tokens")
    logger.info(f"  p25:    {lengths_sorted[n // 4]:>8,} tokens")
    logger.info(f"  Median: {lengths_sorted[n // 2]:>8,} tokens")
    logger.info(f"  p75:    {lengths_sorted[3 * n // 4]:>8,} tokens")
    logger.info(f"  p95:    {lengths_sorted[int(n * 0.95)]:>8,} tokens")
    logger.info(f"  p99:    {lengths_sorted[int(n * 0.99)]:>8,} tokens")
    logger.info(f"  Max:    {lengths_sorted[-1]:>8,} tokens")
    logger.info(f"  Mean:   {sum(lengths) / n:>8,.0f} tokens")

    logger.info("\nTruncation at each seq_length:")
    logger.info(f"  {'seq_length':>10}  {'truncated':>10}  {'pct':>7}  {'kept':>10}  {'kept_pct':>7}")
    logger.info(f"  {'-' * 10}  {'-' * 10}  {'-' * 7}  {'-' * 10}  {'-' * 7}")
    for sl in sorted(seq_lengths):
        truncated = sum(1 for length in lengths if length > sl)
        kept = n - truncated
        logger.info(
            f"  {sl:>10,}  {truncated:>10,}  {100 * truncated / n:>6.1f}%  {kept:>10,}  {100 * kept / n:>6.1f}%"
        )


def main():
    args = parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    # Allow running with cached models/datasets when the Hub is unreachable.
    # Set before any HF imports to avoid proxy/connection issues.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.chat_template:
        import json
        from pathlib import Path

        p = Path(args.chat_template)
        content = p.read_text(encoding="utf-8")
        try:
            content = json.loads(content)["chat_template"]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        tokenizer.chat_template = content
        logger.info(f"Using custom chat template from: {args.chat_template}")
    else:
        logger.info("Using default tokenizer chat template")

    logger.info(f"Loading dataset: {args.dataset} (split={args.split})")
    dataset = load_dataset_messages(args.dataset, args.split)
    logger.info(f"  Total samples: {len(dataset):,}")

    sample_desc = f"{args.num_samples:,}" if args.num_samples > 0 else f"all {len(dataset):,}"
    logger.info(f"\nTokenizing {sample_desc} samples...")
    lengths = compute_lengths(dataset, tokenizer, args.num_samples, args.seed)

    print_report(lengths, args.seq_length)


if __name__ == "__main__":
    main()

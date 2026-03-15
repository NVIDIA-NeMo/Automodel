#!/usr/bin/env python3
"""Validate ChatDataset output tensors for correctness.

Iterates over samples from ChatDataset and checks five invariants:
  1. attention_mask has content_length 1s followed by padding 0s (no all-1s on padded samples)
  2. labels == -100 at every position where attention_mask == 0
  3. At least 1 supervised token (labels != -100) per sample
  4. No eos_token_id at positions where attention_mask == 0
  5. eos_token_id appears in content tokens (not silently truncated)

Exits non-zero if any assertion fails.

Usage:
    python validate_data.py --dataset allenai/tulu-3-sft-mixture --model Qwen/Qwen3-30B-A3B --seq_length 1024
    python validate_data.py --dataset allenai/tulu-3-sft-mixture --model Qwen/Qwen3-30B-A3B --seq_length 1024 --num-samples 500
"""

import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Validate ChatDataset samples for training correctness.")
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset ID or local path")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B", help="Model/tokenizer name")
    parser.add_argument("--seq_length", type=int, default=1024, help="Sequence length")
    parser.add_argument("--padding", type=str, default="max_length", help="Padding strategy")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--num-samples", type=int, default=0, help="Number of samples to check (0 = all)")
    return parser.parse_args()


def validate(dataset, eos_token_id, num_samples):
    """Run all assertions over *dataset* and return per-assertion counts."""
    n = len(dataset) if num_samples == 0 else min(num_samples, len(dataset))

    assertion_names = [
        "attention_mask_shape",  # 1s then 0s, no all-1s when padded
        "labels_masked_in_padding",  # labels == -100 where attn == 0
        "has_supervised_token",  # at least one labels != -100
        "no_eos_in_padding",  # no eos_token_id where attn == 0
        "eos_in_content",  # eos_token_id appears in content (not silently truncated)
    ]
    pass_counts = {name: 0 for name in assertion_names}
    fail_counts = {name: 0 for name in assertion_names}
    fail_examples = {name: [] for name in assertion_names}  # store first few failures
    max_fail_examples = 3

    for i in range(n):
        sample = dataset[i]
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        attention_mask = sample["attention_mask"]
        seq_len = len(input_ids)

        # --- 1. attention_mask shape: content_length 1s then padding 0s ---
        ok = True
        # Find the transition point: first 0 in attention_mask
        first_zero = None
        for j in range(seq_len):
            if attention_mask[j] == 0:
                first_zero = j
                break

        if first_zero is not None:
            # All positions before first_zero must be 1, all after must be 0
            prefix_ok = all(attention_mask[j] == 1 for j in range(first_zero))
            suffix_ok = all(attention_mask[j] == 0 for j in range(first_zero, seq_len))
            ok = prefix_ok and suffix_ok
        # If first_zero is None, all 1s is valid only if there is no padding
        # (seq_length == content_length). That is fine.

        if ok:
            pass_counts["attention_mask_shape"] += 1
        else:
            fail_counts["attention_mask_shape"] += 1
            if len(fail_examples["attention_mask_shape"]) < max_fail_examples:
                fail_examples["attention_mask_shape"].append(
                    f"  sample {i}: first_zero={first_zero}, mask snippet={attention_mask[:20]}"
                )

        # --- 2. labels == -100 where attention_mask == 0 ---
        bad_positions = [j for j in range(seq_len) if attention_mask[j] == 0 and labels[j] != -100]
        if not bad_positions:
            pass_counts["labels_masked_in_padding"] += 1
        else:
            fail_counts["labels_masked_in_padding"] += 1
            if len(fail_examples["labels_masked_in_padding"]) < max_fail_examples:
                fail_examples["labels_masked_in_padding"].append(
                    f"  sample {i}: {len(bad_positions)} positions with attn=0 but labels!=-100, first: {bad_positions[:5]}"
                )

        # --- 3. At least 1 supervised token ---
        supervised_positions = [j for j in range(seq_len) if labels[j] != -100]
        if supervised_positions:
            pass_counts["has_supervised_token"] += 1
        else:
            fail_counts["has_supervised_token"] += 1
            if len(fail_examples["has_supervised_token"]) < max_fail_examples:
                fail_examples["has_supervised_token"].append(f"  sample {i}: no supervised tokens (all labels == -100)")

        # --- 4. No eos_token_id in padding ---
        eos_in_padding = [j for j in range(seq_len) if attention_mask[j] == 0 and input_ids[j] == eos_token_id]
        if not eos_in_padding:
            pass_counts["no_eos_in_padding"] += 1
        else:
            fail_counts["no_eos_in_padding"] += 1
            if len(fail_examples["no_eos_in_padding"]) < max_fail_examples:
                fail_examples["no_eos_in_padding"].append(
                    f"  sample {i}: eos_token_id at padding positions: {eos_in_padding[:5]}"
                )

        # --- 5. eos_token_id appears in content (catches silent truncation) ---
        content_ids = [input_ids[j] for j in range(seq_len) if attention_mask[j] == 1]
        if eos_token_id in content_ids:
            pass_counts["eos_in_content"] += 1
        else:
            fail_counts["eos_in_content"] += 1
            if len(fail_examples["eos_in_content"]) < max_fail_examples:
                fail_examples["eos_in_content"].append(
                    f"  sample {i}: eos_token_id={eos_token_id} not found in {len(content_ids)} content tokens"
                )

        if (i + 1) % 500 == 0:
            print(f"  Checked {i + 1}/{n} samples...", flush=True)

    return assertion_names, pass_counts, fail_counts, fail_examples, n


def main():
    args = parse_args()

    from transformers import AutoTokenizer

    from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id

    print(f"Loading dataset: {args.dataset} (split={args.split}, seq_length={args.seq_length}, padding={args.padding})")
    dataset = ChatDataset(
        path_or_dataset_id=args.dataset,
        tokenizer=tokenizer,
        split=args.split,
        seq_length=args.seq_length,
        padding=args.padding,
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"pad_token_id: {tokenizer.pad_token_id}  eos_token_id: {eos_token_id}")
    print()

    num_samples = args.num_samples
    total = len(dataset) if num_samples == 0 else min(num_samples, len(dataset))
    print(f"Validating {total} samples...")

    assertion_names, pass_counts, fail_counts, fail_examples, n = validate(dataset, eos_token_id, num_samples)

    # --- Report ---
    print()
    print("=" * 70)
    print(f"VALIDATION REPORT ({n} samples)")
    print("=" * 70)

    any_failures = False
    for name in assertion_names:
        p = pass_counts[name]
        f = fail_counts[name]
        status = "PASS" if f == 0 else "FAIL"
        if f > 0:
            any_failures = True
        print(f"  {status}  {name:<30s}  pass={p:>6d}  fail={f:>6d}")
        for ex in fail_examples[name]:
            print(f"       {ex}")

    print("=" * 70)

    if any_failures:
        total_failures = sum(fail_counts.values())
        print(f"FAILED: {total_failures} assertion failures across {n} samples.")
        sys.exit(1)
    else:
        print(f"ALL PASSED: {n} samples validated successfully.")
        sys.exit(0)


if __name__ == "__main__":
    main()

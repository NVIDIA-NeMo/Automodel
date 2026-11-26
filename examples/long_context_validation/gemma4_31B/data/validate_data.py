#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate ChatDataset label/loss-mask correctness for the CoderForge SFT cache.

Loads the prefiltered JSONL through ``ChatDataset`` (the exact training path) and
checks the invariants that determine training correctness. The checks are
**padding-side agnostic** (the Gemma4 tokenizer left-pads) and focus on labels
rather than attention-mask shape:

  1. has_supervised_token       -- at least one label != -100. Catches the
     all-masked failure mode, a live risk for Gemma4 since its chat template has
     no real ``{% generation %}`` block (masking uses the multiturn fallback).
  2. stop_token_in_supervised   -- a Gemma4 stop token appears among supervised
     tokens: the turn terminator ``<turn|>`` (id 106), or ``<|tool_response>`` (id 50),
     which the model emits after its tool calls. ``eos_token_id`` (1 = ``<eos>``) is
     NOT the learned stop token; generation_config stops on 106 and 50.
  3. final_turn_terminated      -- a stop token is among the last few supervised
     tokens (``<turn|>`` then ``\\n``, or ``<|tool_response>`` when the trajectory ends
     on a tool call such as ``finish``), so the final model turn is complete and
     was not truncated mid-turn.
  4. pad_positions_masked       -- wherever input_ids == pad_token_id, labels are
     -100 (padding never contributes to loss), regardless of padding side.

Exits non-zero if any assertion fails.

Usage:
    python validate_data.py \
        --dataset ./cached/togethercomputer_CoderForge-Preview_filtered_reward1_seq32768/data.jsonl \
        --model /path/to/hf_gemma4_31b \
        --seq_length 32768 \
        --num-samples 200
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

# The generation-block template lives one directory up from this script (data/../).
# Resolve it absolutely so the default works regardless of the caller's CWD.
_DEFAULT_CHAT_TEMPLATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gemma4_coderforge_chat_template.jinja"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for CoderForge ChatDataset validation."""
    parser = argparse.ArgumentParser(description="Validate the CoderForge SFT cache for training correctness.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to prefiltered JSONL (or HF dataset ID)")
    parser.add_argument("--model", type=str, required=True, help="Tokenizer path or HF id (Gemma4 checkpoint dir)")
    parser.add_argument("--seq_length", type=int, default=32768, help="Sequence length")
    parser.add_argument(
        "--padding",
        type=str,
        default="do_not_pad",
        help="Padding strategy. Training uses do_not_pad (the collator pads the batch).",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument(
        "--chat_template",
        type=str,
        default=_DEFAULT_CHAT_TEMPLATE,
        help="Chat template (Jinja file or string) with {%% generation %%} blocks for direct assistant "
        "masking. Empty string uses the tokenizer's own template (the prefix-arithmetic fallback).",
    )
    parser.add_argument(
        "--stop_token_id",
        type=int,
        nargs="+",
        default=[106, 50],
        help="Gemma4 stop tokens a model turn may end on: <turn|> (106) and <|tool_response> (50).",
    )
    parser.add_argument(
        "--terminator_window",
        type=int,
        default=2,
        help="Stop token must appear within this many trailing supervised tokens (turns end with <turn|> then newline).",
    )
    parser.add_argument("--num-samples", type=int, default=0, help="Number of samples to check (0 = all)")
    return parser.parse_args()


def validate(
    dataset, pad_token_id: int, stop_token_ids: List[int], terminator_window: int, num_samples: int
) -> Tuple[List[str], Dict[str, int], Dict[str, int], Dict[str, list], int]:
    """Run all assertions over *dataset* and return per-assertion counts."""
    n = len(dataset) if num_samples == 0 else min(num_samples, len(dataset))

    assertion_names = [
        "has_supervised_token",
        "stop_token_in_supervised",
        "final_turn_terminated",
        "pad_positions_masked",
    ]
    pass_counts = {name: 0 for name in assertion_names}
    fail_counts = {name: 0 for name in assertion_names}
    fail_examples = {name: [] for name in assertion_names}
    max_fail_examples = 3

    def record(name: str, ok: bool, msg: str) -> None:
        if ok:
            pass_counts[name] += 1
        else:
            fail_counts[name] += 1
            if len(fail_examples[name]) < max_fail_examples:
                fail_examples[name].append(f"  {msg}")

    for i in range(n):
        sample = dataset[i]
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        supervised_ids = [labels[j] for j in range(len(labels)) if labels[j] != -100]

        record("has_supervised_token", bool(supervised_ids), f"sample {i}: no supervised tokens (all -100)")

        record(
            "stop_token_in_supervised",
            any(t in supervised_ids for t in stop_token_ids),
            f"sample {i}: stop tokens {stop_token_ids} absent from {len(supervised_ids)} supervised tokens",
        )

        tail = supervised_ids[-terminator_window:] if supervised_ids else []
        record(
            "final_turn_terminated",
            any(t in tail for t in stop_token_ids),
            f"sample {i}: final supervised tokens {tail} lack a terminator {stop_token_ids} (possible mid-turn truncation)",
        )

        bad_pad = [j for j in range(len(input_ids)) if input_ids[j] == pad_token_id and labels[j] != -100]
        record(
            "pad_positions_masked",
            not bad_pad,
            f"sample {i}: {len(bad_pad)} pad positions with labels != -100, first {bad_pad[:5]}",
        )

        if (i + 1) % 500 == 0:
            print(f"  Checked {i + 1}/{n} samples...", flush=True)

    return assertion_names, pass_counts, fail_counts, fail_examples, n


def main() -> None:
    """Run CoderForge ChatDataset validation from the command line."""
    args = parse_args()

    from nemo_automodel import NeMoAutoTokenizer
    from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset

    print(f"Loading tokenizer: {args.model}")
    tokenizer = NeMoAutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    print(f"Loading dataset: {args.dataset} (seq_length={args.seq_length}, padding={args.padding})")
    dataset = ChatDataset(
        path_or_dataset_id=args.dataset,
        tokenizer=tokenizer,
        split=args.split,
        seq_length=args.seq_length,
        padding=args.padding,
        chat_template=args.chat_template or None,
    )
    print(f"Dataset size: {len(dataset)}")
    print(
        f"stop_token_ids: {args.stop_token_id}  pad_token_id: {pad_token_id}  (tokenizer.eos_token_id={tokenizer.eos_token_id})"
    )
    print()

    total = len(dataset) if args.num_samples == 0 else min(args.num_samples, len(dataset))
    print(f"Validating {total} samples...")

    assertion_names, pass_counts, fail_counts, fail_examples, n = validate(
        dataset, pad_token_id, args.stop_token_id, args.terminator_window, args.num_samples
    )

    print()
    print("=" * 70)
    print(f"VALIDATION REPORT ({n} samples)")
    print("=" * 70)
    any_failures = False
    for name in assertion_names:
        f = fail_counts[name]
        status = "PASS" if f == 0 else "FAIL"
        if f > 0:
            any_failures = True
        print(f"  {status}  {name:<28s}  pass={pass_counts[name]:>6d}  fail={f:>6d}")
        for ex in fail_examples[name]:
            print(f"       {ex}")
    print("=" * 70)

    if any_failures:
        print(f"FAILED: {sum(fail_counts.values())} assertion failures across {n} samples.")
        sys.exit(1)
    print(f"ALL PASSED: {n} samples validated successfully.")
    sys.exit(0)


if __name__ == "__main__":
    main()

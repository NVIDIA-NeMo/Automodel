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
"""Assert that ``last_turn_collate_fn`` supervises exactly the final assistant turn.

Runs on CPU and needs no GPU. This is the cheapest check that the SFT loss is pointed
at the right tokens, and it should pass before any multi-GPU run is launched.

For each sampled conversation it verifies:

* the shipped collator supervises **every** assistant turn (the baseline that motivates
  the wrapper),
* the wrapper leaves exactly one contiguous supervised run,
* that run decodes to the final assistant message and ends at ``<|im_end|>``,
* the run excludes the generation-prompt prefix (``<think>`` ...) that the chat
  template inserts ahead of assistant content.

Usage::

    export HF_TOKEN=...
    uv run --no-project --with "transformers>=5" --with datasets --with torch \\
        python examples/vlm_finetune/qwen3_5_moe/check_masking_v4_88k.py --n 8
"""

import argparse
import importlib.util
import pathlib

import torch
from datasets import load_dataset
from transformers import AutoProcessor

from nemo_automodel.components.datasets.vlm.collate_fns import default_collate_fn

IGNORE_INDEX = -100
_MODULE_PATH = pathlib.Path(__file__).resolve().parent / "v4_88k.py"


def _load_adapter():
    """Import the recipe-local dataset adapter module by path."""
    spec = importlib.util.spec_from_file_location("v4_88k_adapter", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runs(labels_row: torch.Tensor) -> list[tuple[int, int]]:
    """Return the ``[start, end)`` bounds of each contiguous supervised span."""
    supervised = labels_row.ne(IGNORE_INDEX).tolist()
    spans = []
    start = None
    for index, flag in enumerate(supervised):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(supervised)))
    return spans


def main() -> None:
    """Check last-turn masking against real rows and exit non-zero on failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="vuhaian/v4_88k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--n", type=int, default=8, help="Conversations to check.")
    args = parser.parse_args()

    adapter = _load_adapter()
    processor = AutoProcessor.from_pretrained(args.model, padding_side="right")
    tokenizer = getattr(processor, "tokenizer", processor)

    _, suffixes = adapter._resolve_markers(tokenizer)
    for suffix in suffixes:
        print(f"generation-prompt suffix after the assistant marker: {tokenizer.decode(suffix)!r} ids={suffix}")

    dataset = load_dataset(args.dataset, split=args.split).shuffle(seed=0).select(range(args.n))

    failures = 0
    for row in dataset:
        messages = row["messages"]
        conversation = [
            {"role": m["role"], "content": [{"type": "text", "text": m["content"] or ""}]} for m in messages
        ]
        examples = [{"conversation": conversation}]
        n_assistant = sum(1 for m in messages if m["role"] == "assistant")

        baseline = default_collate_fn(examples, processor)
        wrapped = adapter.last_turn_collate_fn(examples, processor)

        baseline_runs = _runs(baseline["labels"][0])
        wrapped_runs = _runs(wrapped["labels"][0])
        n_tokens = wrapped["input_ids"].shape[1]

        status = "ok"
        if len(baseline_runs) != n_assistant:
            status = f"FAIL baseline supervised {len(baseline_runs)} runs, expected {n_assistant}"
        elif len(wrapped_runs) != 1:
            status = f"FAIL wrapper left {len(wrapped_runs)} runs, expected 1"
        else:
            start, end = wrapped_runs[0]
            # The label values are the target token ids themselves, so decoding them
            # avoids re-deriving the next-token shift between labels and input_ids.
            supervised_ids = wrapped["labels"][0, start:end]
            decoded = tokenizer.decode(supervised_ids[supervised_ids.ne(IGNORE_INDEX)])
            expected = (messages[-1]["content"] or "").strip()
            if not decoded.replace("<|im_end|>", "").strip().endswith(expected[-120:]):
                status = "FAIL supervised span does not match the final assistant message"
            elif suffixes:
                # Labels are shifted against input_ids by default_collate_fn, so the
                # supervised span starting at label position `start` begins at token
                # `start + 1`; a generation-prompt suffix sits immediately before it.
                # Untrimmed, the tokens before the span end in "assistant\n", which
                # matches neither suffix.
                token_start = start + 1
                preceding = wrapped["input_ids"][0, :token_start].tolist()
                if not any(len(s) <= token_start and preceding[-len(s) :] == s for s in suffixes):
                    status = "FAIL generation-prompt prefix was not excluded from the loss"

        if status != "ok":
            failures += 1
        print(
            f"turns={len(messages):4d} assistant={n_assistant:4d} tokens={n_tokens:7d} "
            f"baseline_runs={len(baseline_runs):4d} wrapped_runs={len(wrapped_runs):2d} "
            f"supervised={int(wrapped['labels'][0].ne(IGNORE_INDEX).sum()):6d}  {status}"
        )

    if failures:
        raise SystemExit(f"{failures}/{args.n} conversations failed the masking check")
    print(f"\nOK - {args.n}/{args.n} conversations supervise exactly the final assistant turn")


if __name__ == "__main__":
    main()

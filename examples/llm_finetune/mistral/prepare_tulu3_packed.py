# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare native-tokenized Tulu3 THD packs for the Mistral Large 4 128K recipe.

Run with the same Automodel environment as training, for example::

    python examples/llm_finetune/mistral/prepare_tulu3_packed.py \
        --model-path /checkpoints/mistral-large4 \
        --output data/mistral_large4_tulu3_packed128k/train

The shared VLM collator supplies native formatting and assistant-only labels,
shifted once. The shared LLM packer preserves those labels and document bounds.
Documents and then packed rows are shuffled with seed 1234.
Conversations with empty assistant responses are excluded and counted because
the native Mistral tokenizer rejects them.
"""

import argparse
import json
import logging
import multiprocessing
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import TypedDict

import torch

from nemo_automodel import NeMoAutoTokenizer
from nemo_automodel.components.datasets.llm.packed_sequence import pack_dataset
from nemo_automodel.components.datasets.vlm.collate_fns import default_collate_fn
from nemo_automodel.components.datasets.vlm.datasets import make_tulu3_dataset

logger = logging.getLogger(__name__)


class _TextPart(TypedDict):
    type: str
    text: str


class _Turn(TypedDict):
    role: str
    content: list[_TextPart]


class _Document(TypedDict):
    input_ids: list[int]
    labels: list[int]


def _tokenize_batch(conversations: list[list[_Turn]], model_path: str) -> list[_Document]:
    """Format text conversations through the shared native-tokenizer collator."""
    torch.set_num_threads(1)
    tokenizer = NeMoAutoTokenizer.from_pretrained(model_path, mode="finetuning", padding_side="right")
    result = []
    for conversation in conversations:
        batch = default_collate_fn([{"conversation": conversation}], processor=tokenizer)
        result.append({"input_ids": batch["input_ids"][0].tolist(), "labels": batch["labels"][0].tolist()})
    return result


def main() -> None:
    """Save bounded, reproducible packs from real Tulu3 without model loading."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-packs", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {args.output}")
    if args.max_packs < 1 or args.num_workers < 1:
        raise ValueError("max-packs and num-workers must be positive")
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(1)
    source = make_tulu3_dataset().shuffle(seed=1234)
    counts = {
        "source_documents_scanned": 0,
        "dropped_empty_assistant_conversations": 0,
        "tokenized_documents_consumed": 0,
        "dropped_overlong": 0,
        "dropped_without_labels": 0,
    }

    def documents(pool: ProcessPoolExecutor) -> Iterator[_Document]:
        # At most one bounded wave is in flight, so max_packs does not tokenize
        # the remainder of the full corpus after the requested packs are filled.
        wave_size = 256 * args.num_workers
        for start in range(0, len(source), wave_size):
            rows = source[start : start + wave_size]["conversation"]
            valid_rows = []
            for conversation in rows:
                counts["source_documents_scanned"] += 1
                if any(
                    turn["role"] == "assistant" and not any(part["text"] for part in turn["content"])
                    for turn in conversation
                ):
                    counts["dropped_empty_assistant_conversations"] += 1
                    continue
                valid_rows.append(conversation)
            batches = [valid_rows[i : i + 256] for i in range(0, len(valid_rows), 256)]
            for batch in pool.map(_tokenize_batch, batches, repeat(args.model_path)):
                for sample in batch:
                    counts["tokenized_documents_consumed"] += 1
                    if len(sample["input_ids"]) > 131072:
                        counts["dropped_overlong"] += 1
                        continue
                    if not any(label != -100 for label in sample["labels"]):
                        counts["dropped_without_labels"] += 1
                        continue
                    yield sample
            logger.info("Processed %d documents", counts["tokenized_documents_consumed"])

    tokenizer = NeMoAutoTokenizer.from_pretrained(args.model_path, mode="finetuning", padding_side="right")
    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        packed = pack_dataset(
            documents(pool),
            split="train",
            packed_sequence_size=131072,
            max_packs=args.max_packs,
            padding_idx=tokenizer.pad_token_id or 0,
            cp_size=32,
            pad_to_multiple_of=64,
        ).shuffle(seed=1234)
    if len(packed) < args.max_packs:
        raise ValueError(f"Insufficient real-data packs: {len(packed)} < {args.max_packs}")
    packed.save_to_disk(str(args.output))
    manifest = {
        "dataset": "allenai/tulu-3-sft-mixture",
        "source_fingerprint": source._fingerprint,
        "packed_fingerprint": packed._fingerprint,
        "model_path": args.model_path,
        "seed": 1234,
        "packs": len(packed),
        "packed_sequence_size": 131072,
        "document_alignment": 64,
        "label_policy": "shared default_collate_fn; assistant-only; shifted once before packing",
        **counts,
    }
    args.output.parent.joinpath("preparation.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("Saved %d packs to %s", len(packed), args.output)


if __name__ == "__main__":
    main()

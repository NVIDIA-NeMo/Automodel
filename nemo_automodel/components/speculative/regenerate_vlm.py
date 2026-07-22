# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Regenerate image-conversation answers with a vision-language target model.

ViSpec's stage-2 corpus is built by throwing away the original (short) captions
of an image-caption dataset and having the *target VLM itself* answer each
prompt, with a length instruction appended so the answers are long enough to
train a draft on. Two properties matter:

* **On-policy.** The draft is supervised on the target's own output
  distribution, so the training text has to come from the target.
* **Long assistant turns.** Public multimodal SFT corpora answer in a sentence
  or two; a draft trained on those never sees the long-form decoding it has to
  accelerate. ViSpec appends "Please answer with at least 1000 words." to every
  prompt to force long generations.

The output is ShareGPT-style JSONL plus a meta JSON, which
``nemo_automodel.components.datasets.vlm.datasets.make_meta_dataset`` reads
directly, so the ViSpec recipe consumes it without a bespoke dataset class.

Example::

    uv run python -m nemo_automodel.components.speculative.regenerate_vlm \\
        --model Qwen/Qwen2.5-VL-7B-Instruct \\
        --dataset liuhaotian/LLaVA-Pretrain \\
        --image-root /data/LLaVA-Pretrain \\
        --output-dir /data/vispec_stage2 \\
        --end 68000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoProcessor

from nemo_automodel.shared.import_utils import safe_import

logger = logging.getLogger(__name__)

HAVE_PIL, PIL_Image = safe_import("PIL.Image")

LENGTH_INSTRUCTION = "Please answer with at least 1000 words."
IMAGE_PLACEHOLDER = "<image>"


@dataclass
class RegenerationConfig:
    """Settings for one regeneration run.

    Attributes:
        model: Target VLM id or path; it both prompts and answers.
        dataset: HuggingFace dataset id or local path of the source corpus.
        split: Split expression passed to ``load_dataset``.
        image_root: Directory the dataset's relative image paths resolve against.
        output_dir: Destination directory for ``data.jsonl`` and ``meta.json``.
        start: First source-row index to regenerate (after shuffling).
        end: One past the last source-row index to regenerate.
        max_new_tokens: Generation cap per answer.
        temperature: Sampling temperature; ``0`` means greedy.
        shuffle_seed: Seed for the source-corpus shuffle, so ``start``/``end``
            slices are reproducible across shards.
        length_instruction: Sentence appended to every prompt to force a long
            answer. Empty disables the append.
    """

    model: str
    dataset: str
    split: str
    image_root: str
    output_dir: str
    start: int
    end: int
    max_new_tokens: int
    temperature: float
    shuffle_seed: int
    length_instruction: str


def _extract_prompt(example: dict) -> tuple[str, str] | None:
    """Pull the first human turn and image path out of a ShareGPT-style row.

    Args:
        example: One source row, with a ``conversations`` list and an ``image`` path.

    Returns:
        ``(prompt_text, image_path)``, or ``None`` when the row has no usable
        human turn or no image.
    """
    image_path = example.get("image")
    conversations = example.get("conversations") or []
    if not image_path:
        return None
    for turn in conversations:
        if turn.get("from") != "human":
            continue
        text = turn.get("value", "").replace(IMAGE_PLACEHOLDER, "").strip()
        return text, image_path
    return None


def _build_messages(prompt_text: str, image_path: str, length_instruction: str) -> list[dict]:
    """Build the chat messages handed to the target's chat template.

    Args:
        prompt_text: The source row's human turn, image placeholder stripped.
        image_path: Absolute path to the image for this row.
        length_instruction: Sentence appended after the image, or empty.

    Returns:
        A single-user-turn message list in HuggingFace multimodal chat format.
    """
    content: list[dict] = []
    if prompt_text:
        content.append({"type": "text", "text": prompt_text})
    content.append({"type": "image", "image": image_path})
    if length_instruction:
        content.append({"type": "text", "text": length_instruction})
    return [{"role": "user", "content": content}]


@torch.no_grad()
def _generate_answer(model, processor, messages: list[dict], config: RegenerationConfig) -> str:
    """Run the target once and decode its answer.

    Args:
        model: The loaded target VLM.
        processor: The target's processor.
        messages: Chat messages from :func:`_build_messages`.
        config: The active regeneration settings.

    Returns:
        The decoded assistant answer, without the prompt prefix.
    """
    inputs = processor.apply_chat_template(
        [messages],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    generated = model.generate(
        **inputs,
        max_new_tokens=config.max_new_tokens,
        do_sample=config.temperature > 0,
        temperature=config.temperature if config.temperature > 0 else None,
    )
    prompt_len = inputs["input_ids"].shape[-1]
    return processor.tokenizer.decode(generated[0, prompt_len:], skip_special_tokens=True).strip()


def _write_meta(output_dir: Path, image_root: str) -> Path:
    """Write the meta JSON that ``make_meta_dataset`` reads.

    Args:
        output_dir: Directory holding the generated ``data.jsonl``.
        image_root: Directory the JSONL's relative image paths resolve against.

    Returns:
        Path to the written meta file.
    """
    meta_path = output_dir / "meta.json"
    meta = {
        "vispec_stage2": {
            "file_name": "data.jsonl",
            "columns": {"messages": "conversations", "images": "images"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
            "media_dir": image_root,
        }
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta_path


def _load_target_model(model_path: str) -> torch.nn.Module:
    """Load the target VLM and place it on the available local device.

    ``device_map="auto"`` requires the optional ``accelerate`` package. The
    regeneration utility only needs one local target replica, so explicit
    placement works in both minimal and Accelerate-enabled environments.

    Args:
        model_path: HuggingFace model identifier or local checkpoint directory.

    Returns:
        The evaluated target model on CUDA when available, otherwise CPU.
    """
    model = AutoModelForImageTextToText.from_pretrained(model_path, torch_dtype="auto").eval()
    return model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


def _load_source_dataset(dataset_path: str, split: str):
    """Load a source corpus from either a dataset ID or a local JSON file.

    Args:
        dataset_path: HuggingFace dataset identifier or local JSON/JSONL file.
        split: Dataset split name passed to ``datasets.load_dataset``.

    Returns:
        The loaded HuggingFace dataset split.
    """
    if Path(dataset_path).is_file():
        return load_dataset("json", data_files=dataset_path, split=split)
    return load_dataset(dataset_path, split=split)


def regenerate(config: RegenerationConfig) -> Path:
    """Regenerate answers for a slice of the source corpus and write the shard.

    Args:
        config: The regeneration settings.

    Returns:
        Path to the written ``data.jsonl``.
    """
    if not HAVE_PIL:
        raise ImportError("Pillow is required to load images for VLM regeneration (`uv pip install pillow`).")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "data.jsonl"

    processor = AutoProcessor.from_pretrained(config.model)
    model = _load_target_model(config.model)

    dataset = _load_source_dataset(config.dataset, config.split).shuffle(seed=config.shuffle_seed)
    dataset = dataset.select(range(config.start, min(config.end, len(dataset))))

    written = 0
    skipped = 0
    with data_path.open("w") as handle:
        for example in dataset:
            extracted = _extract_prompt(example)
            if extracted is None:
                skipped += 1
                continue
            prompt_text, image_path = extracted
            absolute_image = os.path.join(config.image_root, image_path)
            if not os.path.exists(absolute_image):
                skipped += 1
                continue
            messages = _build_messages(prompt_text, absolute_image, config.length_instruction)
            answer = _generate_answer(model, processor, messages, config)
            if not answer:
                skipped += 1
                continue
            # The prompt stored for training is the ORIGINAL one, without the
            # length instruction: the instruction exists only to elicit a long
            # answer, and training on it would teach the draft a prompt shape
            # that never appears at deployment.
            row = {
                "conversations": [
                    {"from": "human", "value": f"{IMAGE_PLACEHOLDER}\n{prompt_text}".strip()},
                    {"from": "gpt", "value": answer},
                ],
                "images": [image_path],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    meta_path = _write_meta(output_dir, config.image_root)
    logger.info("Wrote %d regenerated samples (%d skipped) to %s; meta at %s", written, skipped, data_path, meta_path)
    return data_path


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Regenerate VLM answers for ViSpec stage-2 training.")
    parser.add_argument("--model", required=True, help="Target VLM id or path.")
    parser.add_argument("--dataset", required=True, help="Source dataset id or path.")
    parser.add_argument("--split", default="train", help="Source split expression.")
    parser.add_argument("--image-root", required=True, help="Directory the dataset's image paths resolve against.")
    parser.add_argument("--output-dir", required=True, help="Destination directory for data.jsonl and meta.json.")
    parser.add_argument("--start", type=int, default=0, help="First source-row index (after shuffling).")
    parser.add_argument("--end", type=int, default=68000, help="One past the last source-row index.")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Generation cap per answer.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature; 0 means greedy.")
    parser.add_argument("--shuffle-seed", type=int, default=42, help="Seed for the source-corpus shuffle.")
    parser.add_argument(
        "--length-instruction",
        default=LENGTH_INSTRUCTION,
        help="Sentence appended to every prompt to force a long answer; empty disables it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Parses ``argv`` and returns the process exit code."""
    logging.basicConfig(level=logging.INFO)
    args = _build_parser().parse_args(argv)
    if args.end <= args.start:
        raise ValueError(f"--end ({args.end}) must be greater than --start ({args.start})")
    regenerate(
        RegenerationConfig(
            model=args.model,
            dataset=args.dataset,
            split=args.split,
            image_root=args.image_root,
            output_dir=args.output_dir,
            start=args.start,
            end=args.end,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            shuffle_seed=args.shuffle_seed,
            length_instruction=args.length_instruction,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

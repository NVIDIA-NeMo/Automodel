# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import io
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import torch
import torch.utils.data
from datasets import Image as HfImage
from datasets import load_dataset
from PIL import Image

if TYPE_CHECKING:
    from transformers import ProcessorMixin

from nemo_automodel.components.datasets.vlm.utils import (
    _build_video_metadata,
    _lmdb_env_cache,
    _preload_media,
    json2token,
)

logger = logging.getLogger(__name__)


@dataclass
class RdrDatasetConfig:
    """Construction-time configuration for the RDR dataset."""

    path_or_dataset: str = "quintend/rdr-items"
    """HuggingFace dataset id or local path for the RDR dataset."""
    split: str = "train"
    """Dataset split to load (e.g. ``"train"``, ``"test"``)."""

    def build(self) -> list[dict[str, object]]:
        """Build the RDR dataset from this config."""
        return make_rdr_dataset(path_or_dataset=self.path_or_dataset, split=self.split)


def make_rdr_dataset(path_or_dataset="quintend/rdr-items", split="train", **kwargs):
    """Load and preprocess the RDR dataset for image-to-text fine-tuning.

    Args:
        path_or_dataset (str): Path or identifier for the RDR dataset.
        split (str): Dataset split to load.
        **kwargs: Additional arguments.

    Returns:
        Dataset: The processed dataset.
    """
    dataset = load_dataset(path_or_dataset, split=split)

    def format(example):
        return {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": "Describe this image."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example["text"]}],
                },
            ],
        }

    return [format(example) for example in dataset]
    # return dataset.map(format, batched=False)


@dataclass
class CordV2DatasetConfig:
    """Construction-time configuration for the CORD-V2 dataset."""

    path_or_dataset: str = "naver-clova-ix/cord-v2"
    """HuggingFace dataset id or local path for the CORD-V2 dataset."""
    split: str = "train"
    """Dataset split to load (e.g. ``"train"``, ``"test"``)."""
    limit_dataset_samples: int | None = None
    """Optional maximum number of samples to load."""

    def build(self) -> list[dict[str, object]]:
        """Build the CORD-V2 dataset from this config."""
        return make_cord_v2_dataset(
            path_or_dataset=self.path_or_dataset,
            split=self.split,
            limit_dataset_samples=self.limit_dataset_samples,
        )


def make_cord_v2_dataset(
    path_or_dataset="naver-clova-ix/cord-v2",
    split="train",
    limit_dataset_samples: int | None = None,
    **kwargs,
):
    """Load and preprocess the CORD-V2 dataset for image-to-text fine-tuning."""
    if limit_dataset_samples is not None:
        split = f"{split}[:{limit_dataset_samples}]"
    dataset = load_dataset(path_or_dataset, split=split)

    def format(example):
        ground_truth = json.loads(example["ground_truth"])
        if "gt_parses" in ground_truth:  # when multiple ground truths are available, e.g., docvqa
            assert isinstance(ground_truth["gt_parses"], list)
            gt_jsons = ground_truth["gt_parses"]
        else:
            assert "gt_parse" in ground_truth and isinstance(
                ground_truth["gt_parse"],
                dict,
            )
            gt_jsons = [ground_truth["gt_parse"]]

        text = random.choice(
            [json2token(gt_json, sort_json_key=True) for gt_json in gt_jsons],
        )

        return {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image"]},
                        {"type": "text", "text": "Describe this image."},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": text}]},
            ],
        }

    return [format(example) for example in dataset]
    # return dataset.map(format, batched=False, num_proc=8,remove_columns=["ground_truth"])


@dataclass
class MedPixDatasetConfig:
    """Construction-time configuration for the MedPix dataset."""

    path_or_dataset: str = "medpix-dataset/medpix-dataset"
    """HuggingFace dataset id or local path for MedPix."""
    split: str = "train"
    """Dataset split to load."""

    def build(self) -> object:
        """Build the lazily decoded MedPix dataset."""
        return make_medpix_dataset(path_or_dataset=self.path_or_dataset, split=self.split)


def make_medpix_dataset(path_or_dataset="medpix-dataset/medpix-dataset", split="train", **kwargs):
    """Load and preprocess the MedPix dataset for image-to-text fine-tuning.

    Formatting is deferred to ``__getitem__`` via ``with_transform`` so the
    dataset stays Arrow-backed (no Python-side copy of every image). Images are
    loaded undecoded (``Image(decode=False)``) and wrapped as lazy ``PIL``
    handles (``Image.open`` reads only the header); the actual pixel decode then
    happens on demand in the DataLoader workers, rather than eagerly decoding the
    whole split up front.
    """
    dataset = load_dataset(path_or_dataset, split=split)
    if "image_id" in getattr(dataset, "features", {}):
        dataset = dataset.cast_column("image_id", HfImage(decode=False))

    def lazy_image(value):
        # ``Image(decode=False)`` yields a ``{"bytes": ..., "path": ...}`` dict.
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                return Image.open(io.BytesIO(value["bytes"]))
            if value.get("path"):
                return value["path"]
        return value

    def transform(batch):
        return {
            "conversation": [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": lazy_image(image)},
                            {"type": "text", "text": question},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                ]
                for image, question, answer in zip(batch["image_id"], batch["question"], batch["answer"])
            ]
        }

    return dataset.with_transform(transform)


@dataclass
class LlavaOnevisionDatasetConfig:
    """Construction-time configuration for the LLaVA-OneVision dataset."""

    path_or_dataset: str = "liuhaotian/LLaVA-Instruct-150K"
    """HuggingFace dataset id or local path."""
    split: str = "train"
    """Dataset split to load."""

    def build(self) -> list[dict[str, object]]:
        """Build the LLaVA-OneVision conversation dataset."""
        return make_llava_onevision_dataset(path_or_dataset=self.path_or_dataset, split=self.split)


def make_llava_onevision_dataset(
    path_or_dataset="liuhaotian/LLaVA-Instruct-150K",
    split="train",
    **kwargs,
):
    """Load and preprocess the LLaVA-Instruct-150K dataset for LLaVA-OneVision-1.5.

    This function loads conversation-format data with images and returns it in
    the standard NeMo VLM format expected by the collate function.

    Args:
        path_or_dataset: Path to the dataset on HuggingFace Hub or local path.
        split: Dataset split to load (e.g., "train", "train[:1000]").
        **kwargs: Additional arguments passed to load_dataset.

    Returns:
        List of dicts with "conversation" and "image" keys.
    """
    dataset = load_dataset(path_or_dataset, split=split)

    def format(example):
        conversations = example.get("conversations", [])
        image = example.get("image", None)

        # Convert conversations to NeMo VLM format
        nemo_conversation = []
        for turn in conversations:
            role = turn.get("from", "")
            content = turn.get("value", "")

            # Map role names
            if role == "human":
                role = "user"
            elif role == "gpt":
                role = "assistant"

            # Parse content for image placeholders
            content_items = []
            if "<image>" in content:
                content_items.append({"type": "image", "image": image})
                content = content.replace("<image>", "").strip()

            if content:
                content_items.append({"type": "text", "text": content})

            nemo_conversation.append({"role": role, "content": content_items})

        return {
            "conversation": nemo_conversation,
            "image": image,
        }

    return [format(example) for example in dataset]


@dataclass
class Tulu3MagicoderTextMixDatasetConfig:
    """Construction-time configuration for the Tulu-3/Magicoder text mixture."""

    tulu_split: str = "train"
    """Split expression for the Tulu-3 source."""
    magicoder_split: str = "train"
    """Split expression for the Magicoder source."""
    seed: int = 42
    """Sampling seed for interleaving the two sources."""
    max_turns: int = 16
    """Maximum number of turns retained from a Tulu-3 conversation."""
    limit_total: int | None = None
    """Optional cap on the merged row count."""

    def build(self) -> list[dict[str, object]]:
        """Build the text-only Tulu-3/Magicoder mixture."""
        return make_tulu3_magicoder_text_mix_dataset(
            tulu_split=self.tulu_split,
            magicoder_split=self.magicoder_split,
            seed=self.seed,
            max_turns=self.max_turns,
            limit_total=self.limit_total,
        )


def make_tulu3_magicoder_text_mix_dataset(
    tulu_split: str = "train",
    magicoder_split: str = "train",
    seed: int = 42,
    max_turns: int = 16,
    limit_total: int | None = None,
    **kwargs,
) -> list:
    """Build a text-only 80/20 mix of Tulu-3-SFT-mixture and Magicoder-OSS-Instruct-75K.

    Both datasets are converted into the NeMo VLM ``{"conversation": [...]}`` shape
    consumed by :func:`nemo_automodel.components.datasets.vlm.collate_fns.default_collate_fn`.
    Because ``default_collate_fn`` is image-aware only when a conversation turn
    contains an ``{"type": "image", ...}`` entry, returning text-only conversations
    here yields batches with no ``pixel_values`` / vision tensors -- which is what
    the Gemma 4 base+drafter composite expects for text-only training.

    Sources:
        - ``allenai/tulu-3-sft-mixture`` (multi-turn, ``messages`` field of
          ``{"role", "content"}`` dicts).
        - ``ise-uiuc/Magicoder-OSS-Instruct-75K`` (2-turn, ``problem`` and
          ``solution`` fields).

    Mixing uses ``datasets.interleave_datasets`` with probabilities ``[0.8, 0.2]``
    and ``stopping_strategy="all_exhausted"`` so both datasets are sampled until
    every example has been drawn at least once.

    Args:
        tulu_split: HF split expression for the Tulu-3 source (e.g. ``"train"``
            or ``"train[:50000]"``).
        magicoder_split: HF split expression for the Magicoder source.
        seed: Seed forwarded to ``interleave_datasets`` for reproducibility.
        max_turns: Drop Tulu-3 conversations with more than this many turns to
            keep memory bounded. Magicoder samples are always 2 turns.
        limit_total: If set, cap the merged dataset to this many rows.
        **kwargs: Additional arguments forwarded to ``load_dataset`` for both
            sources.

    Returns:
        List of ``{"conversation": [...]}`` dicts. Each conversation is a list of
        ``{"role": "user"|"assistant"|"system", "content": [{"type": "text", "text": ...}]}``
        turns with no ``image`` field anywhere in the structure.
    """
    from datasets import interleave_datasets

    tulu = load_dataset("allenai/tulu-3-sft-mixture", split=tulu_split, **kwargs)
    magicoder = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split=magicoder_split, **kwargs)

    valid_roles = {"system", "user", "assistant"}

    def _tulu_to_conversation(example):
        messages = example.get("messages") or []
        if not messages or len(messages) > max_turns:
            return None
        conversation = []
        for turn in messages:
            role = turn.get("role", "")
            text = turn.get("content", "") or ""
            if role not in valid_roles or not text.strip():
                continue
            conversation.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": text}],
                }
            )
        if len(conversation) < 2:
            return None
        # The chat template needs at least one assistant turn for labels.
        if not any(t["role"] == "assistant" for t in conversation):
            return None
        return {"conversation": conversation}

    def _magicoder_to_conversation(example):
        problem = (example.get("problem") or "").strip()
        solution = (example.get("solution") or "").strip()
        if not problem or not solution:
            return None
        return {
            "conversation": [
                {"role": "user", "content": [{"type": "text", "text": problem}]},
                {"role": "assistant", "content": [{"type": "text", "text": solution}]},
            ],
        }

    # Project both datasets to a single shared column ("conversation") via
    # ``datasets.Dataset.map`` so ``interleave_datasets`` can concatenate them.
    # Rows that fail conversion (too long, missing fields) are emitted with an
    # empty conversation and then filtered out before materialization.
    def _tulu_map(example):
        out = _tulu_to_conversation(example)
        return {"conversation": out["conversation"] if out is not None else []}

    def _magicoder_map(example):
        out = _magicoder_to_conversation(example)
        return {"conversation": out["conversation"] if out is not None else []}

    tulu = tulu.map(_tulu_map, remove_columns=tulu.column_names)
    magicoder = magicoder.map(_magicoder_map, remove_columns=magicoder.column_names)
    tulu = tulu.filter(lambda ex: len(ex["conversation"]) >= 2)
    magicoder = magicoder.filter(lambda ex: len(ex["conversation"]) >= 2)

    mixed = interleave_datasets(
        [tulu, magicoder],
        probabilities=[0.8, 0.2],
        stopping_strategy="all_exhausted",
        seed=seed,
    )

    out = []
    for ex in mixed:
        out.append({"conversation": ex["conversation"]})
        if limit_total is not None and len(out) >= limit_total:
            break
    return out


def make_tulu3_dataset(
    path_or_dataset: str = "allenai/tulu-3-sft-mixture",
    split: str = "train",
    **kwargs,
):
    """Load ``allenai/tulu-3-sft-mixture`` directly from the HF Hub as text-only conversations.

    This avoids the meta-JSON + JSONL data-prep step required by
    :func:`make_meta_dataset`: point the recipe's ``dataset._target_`` at this
    function and the Tulu-3 split is pulled straight from the Hub. Each row's
    ``messages`` field is converted with the same helper the meta-JSON path uses
    (:func:`_convert_sharegpt_to_conversation`), so the resulting data composition is
    **identical** to dumping the split to JSONL and loading it via
    :func:`make_meta_dataset`: no turn cap, ``system`` turns preserved, every row kept
    in the original split order. Conversations are text-only (no ``image`` entries),
    so batches carry no ``pixel_values`` / vision tensors.

    The returned dataset stays Arrow-backed (``map``) so the full ~939k-row split is
    not copied into a Python list.

    Args:
        path_or_dataset: HF Hub id (or local path) of the Tulu-3 SFT mixture.
        split: HF split expression (e.g. ``"train"`` or ``"train[:50000]"``).
        **kwargs: Ignored. Accepted so recipe-level dataset keys (e.g. ``truncate``)
            that are forwarded to the dataset target do not raise.

    Returns:
        datasets.Dataset: Rows with a single ``conversation`` column, each a list of
        ``{"role": "user"|"assistant", "content": [{"type": "text", "text": ...}]}`` turns.
    """
    dataset = load_dataset(path_or_dataset, split=split)
    return dataset.map(_convert_sharegpt_to_conversation, remove_columns=dataset.column_names)


@dataclass
class UnimmChatDatasetConfig:
    """Construction-time configuration for the UniMM-Chat dataset."""

    path_or_dataset: str = "Yirany/UniMM-Chat"
    """HuggingFace dataset id or local path for the UniMM-Chat dataset."""
    split: str = "train"
    """Dataset split to load (e.g. ``"train"``, ``"test"``)."""

    def build(self) -> list[dict[str, object]]:
        """Build the UniMM-Chat dataset from this config."""
        return make_unimm_chat_dataset(path_or_dataset=self.path_or_dataset, split=self.split)


def make_unimm_chat_dataset(path_or_dataset="Yirany/UniMM-Chat", split="train", **kwargs):
    """Load and preprocess the UniMM-Chat dataset for image-to-text fine-tuning."""
    dataset = load_dataset(path_or_dataset, split=split)
    image_placeholder_pattern = re.compile(r"<image\s*>", re.IGNORECASE)

    def convert_user_message(value, image):
        """Convert a human message with optional image placeholders into multimodal content."""
        segments = image_placeholder_pattern.split(value)
        placeholders = len(image_placeholder_pattern.findall(value))
        content = []

        for idx, segment in enumerate(segments):
            text = segment.strip()
            if text:
                content.append({"type": "text", "text": text})
            if idx < placeholders:
                content.append({"type": "image", "image": image})

        if not content:
            content.append({"type": "image", "image": image})

        return content

    def format(example):
        conversation = []
        image = example["image"]

        for turn in json.loads(example["conversation"]):
            speaker = turn.get("from")
            value = turn.get("value", "")

            if speaker == "human":
                content = convert_user_message(value, image)
                conversation.append({"role": "user", "content": content})
            elif speaker == "gpt":
                conversation.append(
                    {"role": "assistant", "content": [{"type": "text", "text": value.strip()}]},
                )
            else:
                # Skip unrecognized roles to keep dataset consistent
                continue

        return {"conversation": conversation}

    return [format(example) for example in dataset]


def _convert_sharegpt_to_conversation(
    example,
    columns=None,
    tags=None,
    media_dir=None,
):
    """Convert a single sharegpt-format example to Automodel conversation format.

    Args:
        example (dict): A single data example in sharegpt format.
        columns (dict): Column name mapping with keys 'messages', 'images', 'videos'.
        tags (dict): Tag mapping with keys 'role_tag', 'content_tag', 'user_tag', 'assistant_tag',
            'system_tag'.
        media_dir (str | None): Directory prefix for resolving relative media paths.

    Returns:
        dict: Example in Automodel conversation format.
    """
    columns = columns or {}
    tags = tags or {}

    messages_col = columns.get("messages", "messages")
    images_col = columns.get("images", "images")
    videos_col = columns.get("videos", "videos")

    role_tag = tags.get("role_tag", "role")
    content_tag = tags.get("content_tag", "content")
    user_tag = tags.get("user_tag", "user")
    assistant_tag = tags.get("assistant_tag", "assistant")
    system_tag = tags.get("system_tag", "system")

    messages = example.get(messages_col, [])
    images = list(example.get(images_col, []) or [])
    videos = list(example.get(videos_col, []) or [])

    image_idx = 0
    video_idx = 0
    conversation = []

    for msg in messages:
        role_value = msg.get(role_tag, "")
        content_text = msg.get(content_tag, "")

        if role_value == user_tag:
            role = "user"
        elif role_value == assistant_tag:
            role = "assistant"
        elif role_value == system_tag:
            role = "system"
        else:
            continue

        # Media placeholders are only meaningful in user turns; system and
        # assistant turns are emitted as plain text.
        if role in ("assistant", "system"):
            conversation.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": content_text}],
                }
            )
            continue

        # Parse user content: split on <image> and <video> placeholders
        content_parts = []
        pattern = re.compile(r"(<image>|<video>)", re.IGNORECASE)
        segments = pattern.split(content_text)

        for segment in segments:
            if segment.lower() == "<image>":
                if image_idx < len(images):
                    img_entry = images[image_idx]
                    # Handle dict-style image entries: {"path": "...", ...extra}
                    if isinstance(img_entry, dict):
                        img_path = img_entry["path"]
                        img_extra = {k: v for k, v in img_entry.items() if k != "path"}
                    else:
                        img_path = img_entry
                        img_extra = {}
                    if media_dir and not os.path.isabs(img_path):
                        img_path = os.path.join(media_dir, img_path)
                    # LMDB paths (e.g. "/data/db.lmdb::key") are kept as
                    # strings here; actual decoding is deferred to
                    # _preload_media() in __getitem__.
                    content_parts.append({"type": "image", "image": img_path, **img_extra})
                    image_idx += 1
            elif segment.lower() == "<video>":
                if video_idx < len(videos):
                    vid_entry = videos[video_idx]
                    # Handle dict-style video entries: {"path": "...", "frame_indices": [...], ...}
                    if isinstance(vid_entry, dict):
                        vid_path = vid_entry["path"]
                        vid_extra = {k: v for k, v in vid_entry.items() if k != "path"}
                    else:
                        vid_path = vid_entry
                        vid_extra = {}
                    if media_dir and not os.path.isabs(vid_path):
                        vid_path = os.path.join(media_dir, vid_path)
                    content_parts.append({"type": "video", "video": vid_path, **vid_extra})
                    video_idx += 1
            else:
                text = segment.strip()
                if text:
                    content_parts.append({"type": "text", "text": text})

        if not content_parts:
            content_parts.append({"type": "text", "text": ""})

        conversation.append({"role": "user", "content": content_parts})

    result = {"conversation": conversation}

    # Pass through metadata for downstream use
    if "mm_inputs_meta" in example:
        result["mm_inputs_meta"] = example["mm_inputs_meta"]
    if "_text_tokens" in example:
        result["_text_tokens"] = example["_text_tokens"]

    return result


def _load_json_or_jsonl(file_path):
    """Load data from a JSON or JSONL file.

    Args:
        file_path (str): Path to the JSON or JSONL file.

    Returns:
        list[dict]: List of data examples.
    """
    with open(file_path) as f:
        if file_path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            if isinstance(data, list):
                return data
            raise ValueError(f"Expected a JSON array in {file_path}, got {type(data).__name__}")


def _load_jsonl_for_rank(file_path, sample_ratio, rank, world_size):
    """Load only the JSONL lines needed for this rank, avoiding full json.loads on skipped lines.

    Handles sample_ratio and sharding so that each rank only parses and stores
    its own subset.  The semantics match the original load-all-then-slice approach:
        1. Apply ``sample_ratio`` (deterministic ``Random(42).sample``) on the full
           index range.
        2. Shard the resulting list with ``[rank::world_size]``.

    Returns:
        tuple[list[dict], int]: (parsed examples for this rank, total line count).
    """
    do_shard = world_size is not None and world_size > 1

    if sample_ratio == 1.0:
        # Fast path: single pass – only json.loads lines at idx % world_size == rank
        results = []
        total = 0
        with open(file_path) as f:
            for line in f:
                if not line.strip():
                    continue
                if not do_shard or total % world_size == rank:
                    results.append(json.loads(line))
                total += 1
        # Truncate so every rank has exactly the same count
        if do_shard:
            per_rank = total // world_size
            if per_rank == 0:
                logger.warning(
                    "Dataset '%s' has only %d samples but world_size=%d — "
                    "this dataset contributes 0 samples to every rank.",
                    file_path,
                    total,
                    world_size,
                )
            results = results[:per_rank]
        return results, total

    # sample_ratio != 1.0 — need total count first for deterministic sampling
    # Single pass: store raw line strings (without JSON parsing), then parse only sampled lines.
    raw_lines = []
    with open(file_path) as f:
        for line in f:
            if line.strip():
                raw_lines.append(line)

    total = len(raw_lines)

    if sample_ratio < 1.0:
        n_samples = max(1, math.floor(total * sample_ratio))
        # Deterministic sample – identical to Random(42).sample(raw_data, n) index-wise
        sampled_order = random.Random(42).sample(range(total), n_samples)
    else:
        # sample_ratio > 1.0 — upsample: floor(ratio) full copies + fractional remainder
        n_full_copies = int(sample_ratio)
        frac = sample_ratio - n_full_copies
        n_extra = math.floor(total * frac)
        sampled_order = list(range(total)) * n_full_copies
        if n_extra > 0:
            sampled_order += random.Random(42).sample(range(total), n_extra)

    n_samples = len(sampled_order)

    if do_shard:
        per_rank = n_samples // world_size
        if per_rank == 0:
            logger.warning(
                "Dataset '%s' has only %d samples after sampling (ratio=%.2f) "
                "but world_size=%d — this dataset contributes 0 samples to every rank.",
                file_path,
                n_samples,
                sample_ratio,
                world_size,
            )
        sampled_order = sampled_order[rank * per_rank : (rank + 1) * per_rank]

    # Parse only the sampled lines (no second file read)
    results = [json.loads(raw_lines[i]) for i in sampled_order]
    del raw_lines  # free raw strings immediately

    return results, total


def _collect_sample_stats(examples):
    """Count images, videos, text-only samples and estimate token counts.

    Token estimation mirrors the logic in LengthGroupedSampler._estimate_tokens:
    - Text tokens: uses pre-computed ``_text_tokens`` when present (written by
      ``scripts/precompute_tokens.py``), otherwise falls back to ``chars // 3``.
    - Media tokens: uses ``mm_inputs_meta`` image/video dimensions when present
      (populated by the precompute script), otherwise ``500`` per media item.

    Returns:
        dict with keys n_images, n_videos, n_text_only, n_text_tokens,
        n_media_tokens, n_missing_text_tokens, n_missing_mm_inputs_meta.
        ``n_text_tokens + n_media_tokens`` gives the best available estimate
        of total training tokens.
    """
    n_images = n_videos = n_text_only = 0
    n_text_tokens = n_media_tokens = 0
    n_missing_text_tokens = n_missing_mm_inputs_meta = 0
    for ex in examples:
        conv = ex.get("conversation", [])
        has_image = has_video = False
        n_chars = 0
        media_count = 0

        for msg in conv:
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        t = item.get("type")
                        if t == "image":
                            n_images += 1
                            has_image = True
                            media_count += 1
                        elif t == "video":
                            n_videos += 1
                            has_video = True
                            media_count += 1
                        elif t == "text":
                            n_chars += len(item.get("text", ""))
            elif isinstance(content, str):
                n_chars += len(content)

        if not has_image and not has_video:
            n_text_only += 1

        # ── text token estimate ──────────────────────────────────────
        precomputed = ex.get("_text_tokens")
        if precomputed is not None:
            n_text_tokens += int(precomputed)
        else:
            n_text_tokens += n_chars // 3
            n_missing_text_tokens += 1

        # ── media token estimate ─────────────────────────────────────
        mm_meta = ex.get("mm_inputs_meta")
        if mm_meta is not None:
            # Precomputed per-image/video token counts stored by precompute script
            sample_media = 0
            for img_meta in mm_meta.get("images_meta") or []:
                if img_meta is None:
                    pass
                elif isinstance(img_meta, dict):
                    sample_media += int(img_meta.get("n_tokens", 500))
                else:
                    # Legacy format: [height, width] — no image_cfg here, use default
                    sample_media += 500
            for vid_meta in mm_meta.get("videos_meta") or []:
                if vid_meta is None:
                    pass
                elif isinstance(vid_meta, dict):
                    sample_media += int(vid_meta.get("n_tokens", 500))
                else:
                    # Legacy format: [height, width] — no image_cfg here, use default
                    sample_media += 500
            n_media_tokens += sample_media
        else:
            n_media_tokens += media_count * 500
            if media_count > 0:
                n_missing_mm_inputs_meta += 1

    return {
        "n_images": n_images,
        "n_videos": n_videos,
        "n_text_only": n_text_only,
        "n_text_tokens": n_text_tokens,
        "n_media_tokens": n_media_tokens,
        "n_missing_text_tokens": n_missing_text_tokens,
        "n_missing_mm_inputs_meta": n_missing_mm_inputs_meta,
    }


def _log_dataset_loading_summary(timings, wall_time, total_samples, rank=None):
    """Print a visual summary of per-dataset loading times and data statistics."""
    if not timings:
        return

    BAR_WIDTH = 32
    BLOCKS = " ▏▎▍▌▋▊▉█"

    def _bar(value, max_value):
        if max_value <= 0:
            return " " * BAR_WIDTH
        ratio = min(value / max_value, 1.0)
        filled = ratio * BAR_WIDTH
        full = int(filled)
        frac = filled - full
        bar = "█" * full
        if frac > 0.05 and full < BAR_WIDTH:
            bar += BLOCKS[int(frac * 8)]
        return bar.ljust(BAR_WIDTH)

    # Check whether per-dataset media stats were collected
    has_stats = any("n_images" in t for t in timings.values())

    # Sort by total time descending (slowest first)
    ranked = sorted(timings.items(), key=lambda kv: kv[1]["total"], reverse=True)
    max_time = ranked[0][1]["total"] if ranked else 1.0
    max_name = max(len(name) for name in timings)
    max_name = max(max_name, 7)  # at least "Dataset"

    rank_str = f" (rank {rank})" if rank is not None else ""
    extra_cols = 30 if has_stats else 0
    sep = "─" * (max_name + BAR_WIDTH + 42 + extra_cols)

    if has_stats:
        header_stats = f"  {'Images':>8}  {'Videos':>8}  {'TextOnly':>8}"
        blank_stats = f"  {'':>8}  {'':>8}  {'':>8}"
    else:
        header_stats = blank_stats = ""

    lines = [
        "",
        f"  ┌{sep}┐",
        f"  │  DATASET LOADING SUMMARY{rank_str:<{len(sep) - 26}}│",
        f"  ├{sep}┤",
        f"  │  {'Dataset':<{max_name}}  {'Timeline':<{BAR_WIDTH}}  {'I/O':>6}  {'Conv':>6}  {'Total':>6}  {'Samples':>8}{header_stats}  │",
        f"  ├{sep}┤",
    ]

    for name, t in ranked:
        bar = _bar(t["total"], max_time)
        if has_stats:
            row_stats = f"  {t.get('n_images', 0):>8,}  {t.get('n_videos', 0):>8,}  {t.get('n_text_only', 0):>8,}"
        else:
            row_stats = ""
        lines.append(
            f"  │  {name:<{max_name}}  {bar}  {t['io']:>5.1f}s  {t['convert']:>5.1f}s  {t['total']:>5.1f}s  {t['n_samples']:>8,}{row_stats}  │"
        )

    sum_io = sum(t["io"] for t in timings.values())
    sum_cv = sum(t["convert"] for t in timings.values())
    sum_t = sum(t["total"] for t in timings.values())

    if has_stats:
        total_images = sum(t.get("n_images", 0) for t in timings.values())
        total_videos = sum(t.get("n_videos", 0) for t in timings.values())
        total_text_only = sum(t.get("n_text_only", 0) for t in timings.values())
        sum_stats = f"  {total_images:>8,}  {total_videos:>8,}  {total_text_only:>8,}"
    else:
        sum_stats = ""

    lines.extend(
        [
            f"  ├{sep}┤",
            f"  │  {'Sum (thread)':<{max_name}}  {'':>{BAR_WIDTH}}  {sum_io:>5.1f}s  {sum_cv:>5.1f}s  {sum_t:>5.1f}s  {total_samples:>8,}{sum_stats}  │",
            f"  │  {'Wall clock':<{max_name}}  {'':>{BAR_WIDTH}}  {'':>6}  {'':>6}  {wall_time:>5.1f}s  {'':>8}{blank_stats}  │",
            f"  │  {'Parallelism':<{max_name}}  {'':>{BAR_WIDTH}}  {'':>6}  {'':>6}  {sum_t / max(wall_time, 1e-6):>5.1f}x  {'':>8}{blank_stats}  │",
            f"  └{sep}┘",
            "",
        ]
    )

    # Per-rank totals are visible in the Sum row of the timing table above.
    # Global aggregation (all_reduce) is performed by the caller after the
    # distributed barrier, so we do not attempt it here.

    logger.info("\n".join(lines))


class _ExamplesWithStats(list):
    """list subclass that carries pre-computed dataset statistics.

    Attached by :func:`make_meta_dataset` so downstream code (e.g.
    ``_log_global_dataset_stats``) can read aggregated stats without
    re-scanning all examples.
    """

    __slots__ = ("stats",)


@dataclass
class MetaDatasetConfig:
    """Construction-time configuration for the meta (multi-source) VLM dataset."""

    builds_with_data_parallel_rank: ClassVar[bool] = True

    path_or_dataset: str = ""
    """Path to the meta JSON file that defines the datasets to load."""
    dataset_names: list[str] | None = None
    """Names of datasets to load from the meta file. ``None`` loads all."""
    split: str = "train"
    """Dataset split to load (passed through for API consistency)."""
    shard_data: bool = False
    """If ``True``, each rank loads only its ``1/world_size`` slice."""

    def build(self, *, rank: int | None = None, world_size: int | None = None) -> list[dict[str, object]]:
        """Build the meta VLM dataset from this config.

        Args:
            rank: Runtime data-parallel rank. Inferred from ``torch.distributed`` when ``None``.
            world_size: Runtime data-parallel world size. Inferred from ``torch.distributed`` when ``None``.

        Returns:
            Materialized multimodal examples from the selected source datasets.
        """
        return make_meta_dataset(
            path_or_dataset=self.path_or_dataset,
            dataset_names=self.dataset_names,
            split=self.split,
            shard_data=self.shard_data,
            rank=rank,
            world_size=world_size,
        )


def make_meta_dataset(
    path_or_dataset,
    dataset_names=None,
    split="train",
    shard_data=False,
    rank=None,
    world_size=None,
    **kwargs,
):
    """Load datasets defined in a meta JSON file and convert to Automodel conversation format.

    The meta JSON file maps dataset names to their configurations. Each configuration can have:
        - file_name (str): Path to the data file (JSON/JSONL). Relative paths are resolved
          against the meta file's directory.
        - columns (dict): Column name mapping (messages, images, videos).
        - tags (dict): Tag mapping (role_tag, content_tag, user_tag, assistant_tag, system_tag).
        - media_dir (str): Directory prefix for media files.
        - sample_ratio (float): Sampling ratio (0.0 to 1.0, default 1.0).

    When ``shard_data=True``, each rank loads only its ``1/world_size`` slice of
    every dataset file (interleaved: ``raw_data[rank::world_size]``).  This
    reduces per-rank memory and I/O.  The caller should use a local sampler
    (e.g. ``RandomSampler``) instead of ``DistributedSampler`` since data is
    already partitioned.

    Video frame sampling (fps, min_frames, max_frames) should be configured on
    the **processor** rather than here.  For example in YAML::

        processor:
          _target_: transformers.AutoProcessor.from_pretrained
          pretrained_model_name_or_path: ...
          fps: 1
          min_frames: 4
          max_frames: 128

    Example meta JSON::

        {
            "my_dataset": {
                "file_name": "data/train.jsonl",
                "columns": {"messages": "conversations"},
                "media_dir": "/data/media"
            }
        }

    Args:
        path_or_dataset (str): Path to the meta JSON file.
        dataset_names (list[str] | None): Which datasets to load. None means all.
        split (str): Unused, kept for API consistency.
        shard_data (bool): If True, each rank loads only its 1/world_size slice.
        rank (int | None): Data-parallel rank. Inferred from torch.distributed if None.
        world_size (int | None): Data-parallel world size. Inferred from torch.distributed if None.
        **kwargs: Additional arguments (unused).

    Returns:
        list[dict]: Combined list of examples in Automodel conversation format.
    """
    # Resolve sharding parameters
    if shard_data:
        if rank is None or world_size is None:
            import torch.distributed as dist

            if dist.is_initialized():
                rank = rank if rank is not None else dist.get_rank()
                world_size = world_size if world_size is not None else dist.get_world_size()
            else:
                logger.warning(
                    "shard_data=True but torch.distributed is not initialized. Loading full dataset on this process."
                )
                shard_data = False

    with open(path_or_dataset) as f:
        meta = json.load(f)

    meta_dir = os.path.dirname(os.path.abspath(path_or_dataset))

    if dataset_names is not None:
        missing = set(dataset_names) - set(meta.keys())
        if missing:
            raise ValueError(f"Dataset(s) not found in meta file: {missing}")
        selected = {name: meta[name] for name in dataset_names}
    else:
        selected = meta

    def _load_one_dataset(ds_name, ds_config):
        """Load and convert a single dataset. Returns (name, examples, timing_dict)."""
        t_start = time.monotonic()

        file_name = ds_config.get("file_name")
        if not file_name:
            raise ValueError(f"Dataset '{ds_name}' missing 'file_name' in meta config")

        if not os.path.isabs(file_name):
            file_name = os.path.join(meta_dir, file_name)

        columns = ds_config.get("columns", {})
        tags = ds_config.get("tags", {})
        media_dir = ds_config.get("media_dir")
        sample_ratio = ds_config.get("sample_ratio", 1.0)

        is_jsonl = file_name.endswith(".jsonl")
        do_shard = shard_data and world_size > 1

        t_io_start = time.monotonic()
        if is_jsonl and (do_shard or sample_ratio != 1.0):
            # Optimized path: only json.loads the lines this rank needs
            raw_data, total = _load_jsonl_for_rank(file_name, sample_ratio, rank, world_size)
            if do_shard:
                logger.info(
                    "Rank %d/%d: sharded dataset '%s' — %d/%d samples",
                    rank,
                    world_size,
                    ds_name,
                    len(raw_data),
                    total,
                )
        else:
            # JSON array files or no sharding needed — load everything
            raw_data = _load_json_or_jsonl(file_name)

            if sample_ratio < 1.0:
                n_samples = max(1, math.floor(len(raw_data) * sample_ratio))
                rng = random.Random(42)
                raw_data = rng.sample(raw_data, n_samples)
            elif sample_ratio > 1.0:
                n_full_copies = int(sample_ratio)
                frac = sample_ratio - n_full_copies
                n_extra = math.floor(len(raw_data) * frac)
                original = list(raw_data)
                raw_data = original * n_full_copies
                if n_extra > 0:
                    raw_data += random.Random(42).sample(original, n_extra)

            if do_shard:
                total = len(raw_data)
                per_rank = total // world_size
                if per_rank == 0:
                    logger.warning(
                        "Dataset '%s' has only %d samples but world_size=%d — "
                        "this dataset contributes 0 samples to every rank.",
                        ds_name,
                        total,
                        world_size,
                    )
                raw_data = raw_data[rank::world_size][:per_rank]
                logger.info(
                    "Rank %d/%d: sharded dataset '%s' — %d/%d samples",
                    rank,
                    world_size,
                    ds_name,
                    len(raw_data),
                    total,
                )
        t_io = time.monotonic() - t_io_start

        t_convert_start = time.monotonic()
        examples = [
            _convert_sharegpt_to_conversation(example, columns=columns, tags=tags, media_dir=media_dir)
            for example in raw_data
        ]
        t_convert = time.monotonic() - t_convert_start

        t_total = time.monotonic() - t_start
        stats = _collect_sample_stats(examples)

        # Emit one warning per dataset (not per sample) for missing precomputed fields
        n_missing_tt = stats.pop("n_missing_text_tokens", 0)
        n_missing_mm = stats.pop("n_missing_mm_inputs_meta", 0)
        n_total = len(examples)
        if n_missing_tt > 0:
            logger.warning(
                "Dataset '%s': %d/%d examples missing '_text_tokens' — "
                "falling back to chars//3 estimate. "
                "Run scripts/precompute_tokens.py to populate this field.",
                ds_name,
                n_missing_tt,
                n_total,
            )
        if n_missing_mm > 0:
            logger.warning(
                "Dataset '%s': %d/%d media-containing examples missing 'mm_inputs_meta' — "
                "falling back to 500 tokens per media item. "
                "Run scripts/precompute_tokens.py to populate this field.",
                ds_name,
                n_missing_mm,
                n_total,
            )

        timing = {"io": t_io, "convert": t_convert, "total": t_total, "n_samples": len(examples), **stats}
        return ds_name, examples, timing

    # Use a deterministic iteration order (sorted keys) so that the
    # concatenated result is identical across ranks and across runs
    # (important for StatefulDataLoader resume).
    ordered_names = sorted(selected.keys())
    num_workers = min(len(ordered_names), os.cpu_count() or 8, 32)
    all_examples = []
    all_timings = {}  # name -> timing dict

    t_load_start = time.monotonic()
    if num_workers <= 1:
        for ds_name in ordered_names:
            name, examples, timing = _load_one_dataset(ds_name, selected[ds_name])
            all_examples.extend(examples)
            all_timings[name] = timing
    else:
        logger.info("Loading %d datasets in parallel with %d workers", len(ordered_names), num_workers)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_map = {
                ds_name: executor.submit(_load_one_dataset, ds_name, selected[ds_name]) for ds_name in ordered_names
            }
            # Collect in sorted key order, not completion order
            for ds_name in ordered_names:
                try:
                    name, examples, timing = future_map[ds_name].result()
                    all_examples.extend(examples)
                    all_timings[name] = timing
                except Exception:
                    logger.exception("Failed to load dataset '%s'", ds_name)
                    raise
    t_load_total = time.monotonic() - t_load_start

    _lmdb_env_cache.clear()

    # ── Pretty summary ──────────────────────────────────────────────
    _log_dataset_loading_summary(all_timings, t_load_total, len(all_examples), rank)

    # Attach aggregated stats to the returned list so callers can do an
    # all_reduce without re-scanning all examples.
    result = _ExamplesWithStats(all_examples)
    result.stats = {
        k: sum(t.get(k, 0) for t in all_timings.values())
        for k in ("n_images", "n_videos", "n_text_only", "n_text_tokens", "n_media_tokens")
    }
    return result


def _resolve_processor_token_id(processor, attr_names, token_names):
    """Resolve a model-specific media token id from processor/config/tokenizer."""
    tokenizer = getattr(processor, "tokenizer", processor)
    unk_id = getattr(tokenizer, "unk_token_id", None)

    config = getattr(processor, "config", None)
    for source in (processor, config, tokenizer, getattr(tokenizer, "config", None)):
        if source is None:
            continue
        for attr in attr_names:
            value = getattr(source, attr, None)
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                try:
                    token_id = tokenizer.convert_tokens_to_ids(value)
                except Exception:
                    continue
                if isinstance(token_id, int) and token_id != unk_id:
                    return token_id
    for token in token_names:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            continue
        if isinstance(token_id, int) and token_id != unk_id:
            return token_id
    return None


def _grid_media_token_count(grid, merge_size: int) -> int:
    if grid is None:
        return 0
    grid_t = torch.as_tensor(grid)
    if grid_t.numel() == 0:
        return 0
    if grid_t.ndim == 1:
        grid_t = grid_t.view(1, -1)
    merge_len = int(merge_size) ** 2
    return int((grid_t.to(torch.long).prod(dim=-1) // merge_len).sum().item())


def _media_token_mismatch(input_ids, result, processor) -> str | None:
    """Return a mismatch description if media grids survived without tokens."""
    image_token_id = _resolve_processor_token_id(
        processor,
        ("image_token_id", "image_token_index", "image_token"),
        ("<|image_pad|>", "<image>", "<|image|>"),
    )
    video_token_id = _resolve_processor_token_id(
        processor,
        ("video_token_id", "video_token_index", "video_token"),
        ("<|video_pad|>", "<video>", "<|video|>"),
    )

    image_grid = result.get("image_grid_thw")
    if image_grid is not None and image_token_id is not None:
        image_merge_size = getattr(getattr(processor, "image_processor", None), "merge_size", 2)
        expected = _grid_media_token_count(image_grid, image_merge_size)
        actual = int((input_ids == image_token_id).sum().item())
        if actual != expected:
            return f"image tokens={actual}, expected={expected}"

    video_grid = result.get("video_grid_thw")
    if video_grid is not None and video_token_id is not None:
        video_merge_size = getattr(getattr(processor, "video_processor", None), "merge_size", 2)
        expected = _grid_media_token_count(video_grid, video_merge_size)
        actual = int((input_ids == video_token_id).sum().item())
        if actual != expected:
            return f"video tokens={actual}, expected={expected}"

    return None


@dataclass
class PreTokenizedDatasetWrapperConfig:
    """Construction-time configuration for :class:`PreTokenizedDatasetWrapper`."""

    max_length: int | None = None
    """Maximum token sequence length. Overlong samples are replaced or truncated."""
    max_retries: int = 10
    """Number of retry attempts when a sample fails to tokenize."""
    truncate: bool = False
    """If ``True``, truncate overlong samples instead of replacing them."""
    inject_fake_images: bool = True
    """If ``True``, inject a fake image into text-only samples for sharded training."""
    post_tokenize_hook: Callable[[dict[str, object]], dict[str, object]] | None = None
    """Optional declarative callback applied to each tokenizer result."""

    def build(
        self,
        *,
        dataset: torch.utils.data.Dataset | Sequence[dict[str, object]],
        processor: "ProcessorMixin",
    ) -> "PreTokenizedDatasetWrapper":
        """Build a :class:`PreTokenizedDatasetWrapper` from this config.

        Args:
            dataset: The raw dataset (conversations) to wrap.
            processor: HuggingFace processor for tokenization.
        """
        return PreTokenizedDatasetWrapper(
            dataset=dataset,
            processor=processor,
            max_length=self.max_length,
            max_retries=self.max_retries,
            truncate=self.truncate,
            post_tokenize_hook=self.post_tokenize_hook,
            inject_fake_images=self.inject_fake_images,
        )


class PreTokenizedDatasetWrapper(torch.utils.data.Dataset):
    """Dataset wrapper that tokenizes samples in ``__getitem__``.

    Instead of deferring ``apply_chat_template`` to the collate function, this
    wrapper performs tokenization per-sample so that:

    * The collate function only needs to pad and stack.
    * Overlong samples are detected **after** precise tokenization (including
      media-token expansion) and replaced with a different random sample.
    * Tokenization work is distributed across DataLoader workers.

    Each ``__getitem__`` call returns a dict with at least::

        {
            "input_ids":      (seq_len,),
            "attention_mask": (seq_len,),
            "labels":         (seq_len,),
        }

    Plus optional media tensors (``pixel_values``, ``image_grid_thw``,
    ``pixel_values_videos``, ``video_grid_thw``).
    """

    def __init__(
        self,
        dataset,
        processor,
        max_length=None,
        max_retries=10,
        truncate=False,
        post_tokenize_hook=None,
        inject_fake_images=True,
    ):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length
        self.truncate = truncate
        self.max_retries = max_retries
        self.post_tokenize_hook = post_tokenize_hook
        self.inject_fake_images = inject_fake_images
        # Compatibility attributes expected by build_dataloader
        self.preload_media = False

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        from nemo_automodel.components.datasets.vlm.collate_fns import (
            _extract_media_from_conversations,
            build_labels_from_template,
        )
        from nemo_automodel.components.datasets.vlm.fake_image import (
            _conversation_has_media,
            inject_fake_image_into_conversation,
            mask_fake_vision_tokens_single,
        )

        for attempt in range(self.max_retries):
            try:
                example = self.dataset[idx]
                # preserve_video_metadata=True: store _video_fps and
                # _frame_indices on each video item so we can fix timestamps.
                example = _preload_media(
                    example,
                    self.processor,
                    preserve_video_metadata=True,
                )
                conversation = example["conversation"]

                # Inject fake image into pure-text samples for FSDP/Zero3.
                injected_fake = self.inject_fake_images and not _conversation_has_media(conversation)
                if injected_fake:
                    conversation = inject_fake_image_into_conversation(conversation)

                # Ensure images are RGB
                for message in conversation:
                    content = message.get("content")
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and isinstance(item.get("image"), Image.Image):
                                item["image"] = item["image"].convert("RGB")

                # Render template text.
                text = self.processor.apply_chat_template(
                    [conversation],
                    tokenize=False,
                )
                if isinstance(text, list):
                    text = text[0]

                # Extract pre-loaded media for the processor.
                images, videos = _extract_media_from_conversations(
                    [conversation],
                )

                # Build video_metadata from preserved _video_fps / _frame_indices
                # so the processor inserts correct timestamps and skips re-sampling.
                video_metadata = _build_video_metadata(conversation)

                processor_kwargs = {
                    "text": [text],
                    "images": images,
                    "videos": videos,
                    "return_tensors": "pt",
                    "do_sample_frames": False,
                }
                if video_metadata:
                    processor_kwargs["video_metadata"] = [video_metadata]

                result = self.processor(**processor_kwargs)
                if self.post_tokenize_hook is not None:
                    result = self.post_tokenize_hook(result, self.processor)

                input_ids = result["input_ids"][0]  # (seq_len,)
                seq_len = input_ids.shape[0]

                # Precise length check — truncate or replace overlong samples
                if self.max_length is not None and seq_len > self.max_length:
                    if not self.truncate:
                        logger.warning(
                            "Sample %d: %d tokens > max_length %d, replacing.",
                            idx,
                            seq_len,
                            self.max_length,
                        )
                        idx = random.randint(0, len(self.dataset) - 1)
                        continue

                # Build labels BEFORE truncation so the full assistant text
                # can be matched against the full input_ids.
                labels = build_labels_from_template(
                    result["input_ids"],  # (1, seq_len)
                    [conversation],
                    self.processor,
                )[0]  # (seq_len,)

                # Now truncate if needed (after labels are built)
                if self.truncate and self.max_length is not None and seq_len > self.max_length:
                    ml = self.max_length
                    input_ids = input_ids[:ml]
                    labels = labels[:ml]
                    result = {
                        k: (
                            v[:, :ml]
                            if isinstance(v, torch.Tensor) and v.dim() == 2 and v.shape[1] == seq_len
                            else v[:ml]
                            if isinstance(v, torch.Tensor) and v.dim() == 1 and v.shape[0] == seq_len
                            else v
                        )
                        for k, v in result.items()
                    }
                    seq_len = ml

                mismatch = _media_token_mismatch(input_ids, result, self.processor)
                if mismatch is not None:
                    logger.warning(
                        "Sample %d: media token mismatch after tokenization/truncation (%s), replacing.",
                        idx,
                        mismatch,
                    )
                    idx = random.randint(0, len(self.dataset) - 1)
                    continue

                output = {
                    "input_ids": input_ids,
                    "attention_mask": result["attention_mask"][0],
                    "labels": labels,
                }

                # Copy media tensors (these don't have a sample batch dim)
                for key in (
                    "pixel_values",
                    "pixel_values_videos",
                    "image_grid_thw",
                    "video_grid_thw",
                    "second_per_grid_ts",
                    "image_position_ids",
                    "mm_token_type_ids",
                ):
                    if key in result and result[key] is not None:
                        output[key] = result[key]

                # Mask fake vision tokens so they don't affect attention.
                if injected_fake:
                    mask_fake_vision_tokens_single(output, self.processor)

                return output

            except Exception as e:
                logger.warning(
                    "Error processing sample %d (attempt %d/%d): %s",
                    idx,
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                idx = random.randint(0, len(self.dataset) - 1)

        raise RuntimeError(f"Failed to load a valid sample after {self.max_retries} retries")

    def robust_collate(self, collate_fn):
        """Wrap *collate_fn* so that on failure the entire batch is re-sampled."""
        from nemo_automodel.components.datasets.vlm.collate_fns import make_robust_collate

        return make_robust_collate(self, collate_fn, self.max_retries)


class RobustDatasetWrapper(torch.utils.data.Dataset):
    """Wrapper that catches ``__getitem__`` and collate errors, substituting random replacement samples.

    This handles failures such as corrupted files, missing media, bad data,
    or processor errors (e.g. multimodal token mismatch from truncation)
    without crashing the entire training run.
    """

    def __init__(self, dataset, max_retries: int = 10):
        self.dataset = dataset
        self.max_retries = max_retries
        self.preload_media = False
        self.processor = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        from nemo_automodel.components.datasets.vlm.fake_image import (
            _conversation_has_media,
            inject_fake_image_into_conversation,
        )

        for _ in range(self.max_retries):
            try:
                example = self.dataset[idx]
                if self.preload_media:
                    example = _preload_media(example, self.processor)
                    # Inject fake image into pure-text samples for FSDP/Zero3.
                    conversation = example.get("conversation")
                    if conversation and not _conversation_has_media(conversation):
                        example["conversation"] = inject_fake_image_into_conversation(conversation)
                        example["_injected_fake"] = True
                return example
            except Exception as e:
                logger.warning(f"Error loading sample {idx}: {e}. Retrying with a different sample.")
                idx = random.randint(0, len(self.dataset) - 1)
        raise RuntimeError(f"Failed to load a valid sample after {self.max_retries} retries")

    def robust_collate(self, collate_fn):
        """Wrap a collate_fn so that on failure the entire batch is re-sampled and retried."""
        from nemo_automodel.components.datasets.vlm.collate_fns import make_robust_collate

        return make_robust_collate(self, collate_fn, self.max_retries)

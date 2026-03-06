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

import io
import json
import logging
import math
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch.utils.data
from datasets import load_dataset
from PIL import Image

from nemo_automodel.components.datasets.vlm.utils import json2token

logger = logging.getLogger(__name__)

try:
    import lmdb

    HAVE_LMDB = True
except ImportError:
    HAVE_LMDB = False


_lmdb_env_cache: dict[str, "lmdb.Environment"] = {}


def _resolve_lmdb_image(path):
    """Read an image from an LMDB database.

    Paths use the format ``<lmdb_dir>::<key>``, e.g.
    ``/data/my_db.lmdb::0000000087``.

    Returns:
        PIL.Image.Image: The decoded image.
    """
    if not HAVE_LMDB:
        raise ImportError("lmdb package is required to load LMDB images: pip install lmdb")

    lmdb_path, key = path.split("::", 1)
    if lmdb_path not in _lmdb_env_cache:
        _lmdb_env_cache[lmdb_path] = lmdb.open(lmdb_path, readonly=True, lock=False)
    env = _lmdb_env_cache[lmdb_path]
    with env.begin() as txn:
        data = txn.get(key.encode())
    if data is None:
        raise KeyError(f"Key '{key}' not found in LMDB database '{lmdb_path}'")
    return Image.open(io.BytesIO(data)).convert("RGB")


def _read_video_frames(video_path, processor=None, frame_indices=None, return_metadata=False):
    """Read and sample video frames from a video file using decord.

    If *frame_indices* is provided (e.g. from a dataset annotation), those
    exact frame numbers are used.  Otherwise, frame sampling uses the same
    ``smart_nframes`` + ``linspace`` strategy as ``qwen_vl_utils`` to ensure
    that preloaded frames are identical to those produced by the processor's
    own video pipeline.

    Args:
        video_path: Path to the video file.
        processor: HuggingFace processor whose ``video_processor`` supplies
            default fps / max_frames / min_frames.
        frame_indices: Explicit list of 0-based frame indices to extract.
        return_metadata: If True, return ``(frames, video_fps, used_indices)``
            so callers can preserve timing information for timestamp calculation.

    Returns:
        list[PIL.Image.Image]: Sampled video frames as RGB PIL Images.
        If *return_metadata* is True, returns
        ``(frames, video_fps, used_indices)`` instead.
    """
    import decord
    import torch as _torch

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    if total_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")

    video_fps = vr.get_avg_fps()

    # Determine temporal_patch_size for even-frame alignment (default 2)
    temporal_patch_size = 2
    if processor is not None and hasattr(processor, "video_processor"):
        temporal_patch_size = getattr(processor.video_processor, "temporal_patch_size", 2)

    if frame_indices is not None:
        # Use explicitly specified frame indices, clamp to valid range
        indices = [min(i, total_frames - 1) for i in frame_indices]
        # Pad to temporal_patch_size alignment
        remainder = len(indices) % temporal_patch_size
        if remainder != 0:
            indices.extend([indices[-1]] * (temporal_patch_size - remainder))
    else:
        # Get frame sampling config from processor
        target_fps = None
        max_frames = None
        min_frames = 4
        if processor is not None and hasattr(processor, "video_processor"):
            vp = processor.video_processor
            target_fps = getattr(vp, "fps", None)
            max_frames = getattr(vp, "max_frames", None)
            min_frames = getattr(vp, "min_frames", min_frames)

        # ---------------------------------------------------------------
        # Calculate target frame count using the same algorithm as
        # qwen_vl_utils.smart_nframes:
        #   nframes = total_frames / video_fps * target_fps
        #   clamped to [min_frames, min(max_frames, total_frames)]
        #   rounded UP to temporal_patch_size alignment
        # ---------------------------------------------------------------
        if target_fps is not None and video_fps > 0:
            nframes = total_frames / video_fps * target_fps
        else:
            nframes = float(total_frames)

        nframes = max(nframes, min_frames)
        if max_frames is not None:
            nframes = min(nframes, max_frames)
        nframes = min(nframes, total_frames)

        # Round UP to temporal_patch_size boundary (matches sampler,
        # HF video processor, and LLaMA-Factory).  When nframes exceeds
        # total_frames after rounding, linspace naturally repeats the
        # last frame indices which is the standard padding approach.
        nframes = int(nframes)
        remainder = nframes % temporal_patch_size
        if remainder != 0:
            nframes += temporal_patch_size - remainder
        nframes = max(nframes, temporal_patch_size)

        # Uniformly sample frame indices (matching qwen_vl_utils linspace).
        indices = _torch.linspace(0, total_frames - 1, nframes).round().long().tolist()

    frames = vr.get_batch(indices).asnumpy()
    pil_frames = [Image.fromarray(f).convert("RGB") for f in frames]
    if return_metadata:
        return pil_frames, video_fps, indices
    return pil_frames


def _preload_media(example, processor=None, preserve_video_metadata=False):
    """Pre-load image and video files in a conversation example.

    Images are loaded as PIL RGB Images.
    Videos are decoded into lists of PIL RGB Images (sampled frames).

    When *preserve_video_metadata* is ``True``, the original video fps and
    the sampled frame indices are stored on each video content item as
    ``_video_fps`` and ``_frame_indices``.  This allows downstream code
    (e.g. :func:`_build_video_metadata`) to construct ``VideoMetadata``
    for the processor so it inserts correct timestamps.
    """
    conversation = example.get("conversation")
    if not conversation:
        return example
    for message in conversation:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            media_type = item.get("type")
            if media_type == "image":
                img = item.get("image")
                if isinstance(img, str):
                    if "::" in img:
                        item["image"] = _resolve_lmdb_image(img)
                    else:
                        item["image"] = Image.open(img).convert("RGB")
                elif isinstance(img, Image.Image):
                    item["image"] = img.convert("RGB")
            elif media_type == "video":
                vid = item.get("video")
                if isinstance(vid, str):
                    if preserve_video_metadata:
                        frames, fps, indices = _read_video_frames(
                            vid, processor,
                            frame_indices=item.get("frame_indices"),
                            return_metadata=True,
                        )
                        item["video"] = frames
                        item["_video_fps"] = fps
                        item["_frame_indices"] = indices
                    else:
                        item["video"] = _read_video_frames(
                            vid, processor,
                            frame_indices=item.get("frame_indices"),
                        )
    return example


def _build_video_metadata(conversation):
    """Build a list of ``VideoMetadata`` from preserved ``_video_fps`` / ``_frame_indices``.

    ``_preload_media(preserve_video_metadata=True)`` stores these on each
    video content item.  Passing the resulting metadata to the processor
    ensures correct timestamps and prevents double frame-sampling.

    Returns an empty list if no video metadata is found.
    """
    from transformers.video_utils import VideoMetadata

    metadata_list = []
    for msg in conversation:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "video":
                continue
            fps = item.get("_video_fps")
            indices = item.get("_frame_indices")
            video = item.get("video")
            if fps is not None and indices is not None and video is not None:
                metadata_list.append(VideoMetadata(
                    total_num_frames=len(video),
                    fps=fps,
                    frames_indices=list(indices),
                ))
    return metadata_list


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


def make_cord_v2_dataset(
    path_or_dataset="naver-clova-ix/cord-v2",
    split="train",
    **kwargs,
):
    """Load and preprocess the CORD-V2 dataset for image-to-text fine-tuning."""
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


def make_medpix_dataset(path_or_dataset="medpix-dataset/medpix-dataset", split="train", **kwargs):
    """Load and preprocess the MedPix dataset for image-to-text fine-tuning."""
    dataset = load_dataset(path_or_dataset, split=split)

    def format(example):
        return {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example["image_id"]},
                        {"type": "text", "text": example["question"]},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": example["answer"]}]},
            ],
        }

    return [format(example) for example in dataset]


def make_cv17_dataset(path_or_dataset="ysdede/commonvoice_17_tr_fixed", split="train", **kwargs):
    """Load and preprocess the CommonVoice 17 dataset for audio-to-text fine-tuning."""
    dataset = load_dataset(path_or_dataset, split=split)
    all_columns = dataset.column_names
    columns_to_remove = [col for col in all_columns if col not in ["audio", "transcription"]]
    dataset = dataset.remove_columns(columns_to_remove)

    def format(example):
        return {
            "conversation": [
                {"role": "user", "content": "<|audio_1|>Transcribe the Turkish audio clip."},
                {"role": "assistant", "content": example["transcription"]},
            ],
            "audio": (example["audio"]["array"], example["audio"]["sampling_rate"]),
        }

    ret = [format(example) for example in dataset]
    return ret


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
        tags (dict): Tag mapping with keys 'role_tag', 'content_tag', 'user_tag', 'assistant_tag'.
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
        else:
            continue

        if role == "assistant":
            conversation.append({
                "role": "assistant",
                "content": [{"type": "text", "text": content_text}],
            })
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
                    file_path, total, world_size,
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
                file_path, n_samples, sample_ratio, world_size,
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
            row_stats = (
                f"  {t.get('n_images', 0):>8,}"
                f"  {t.get('n_videos', 0):>8,}"
                f"  {t.get('n_text_only', 0):>8,}"
            )
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
        total_text_tokens = sum(t.get("n_text_tokens", 0) for t in timings.values())
        total_media_tokens = sum(t.get("n_media_tokens", 0) for t in timings.values())
        sum_stats = f"  {total_images:>8,}  {total_videos:>8,}  {total_text_only:>8,}"
    else:
        sum_stats = ""

    lines.extend([
        f"  ├{sep}┤",
        f"  │  {'Sum (thread)':<{max_name}}  {'':>{BAR_WIDTH}}  {sum_io:>5.1f}s  {sum_cv:>5.1f}s  {sum_t:>5.1f}s  {total_samples:>8,}{sum_stats}  │",
        f"  │  {'Wall clock':<{max_name}}  {'':>{BAR_WIDTH}}  {'':>6}  {'':>6}  {wall_time:>5.1f}s  {'':>8}{blank_stats}  │",
        f"  │  {'Parallelism':<{max_name}}  {'':>{BAR_WIDTH}}  {'':>6}  {'':>6}  {sum_t / max(wall_time, 1e-6):>5.1f}x  {'':>8}{blank_stats}  │",
        f"  └{sep}┘",
        "",
    ])

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
        - tags (dict): Tag mapping (role_tag, content_tag, user_tag, assistant_tag).
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
                    "shard_data=True but torch.distributed is not initialized. "
                    "Loading full dataset on this process."
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
                    rank, world_size, ds_name, len(raw_data), total,
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
                        ds_name, total, world_size,
                    )
                raw_data = raw_data[rank::world_size][:per_rank]
                logger.info(
                    "Rank %d/%d: sharded dataset '%s' — %d/%d samples",
                    rank, world_size, ds_name, len(raw_data), total,
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
                ds_name, n_missing_tt, n_total,
            )
        if n_missing_mm > 0:
            logger.warning(
                "Dataset '%s': %d/%d media-containing examples missing 'mm_inputs_meta' — "
                "falling back to 500 tokens per media item. "
                "Run scripts/precompute_tokens.py to populate this field.",
                ds_name, n_missing_mm, n_total,
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
                ds_name: executor.submit(_load_one_dataset, ds_name, selected[ds_name])
                for ds_name in ordered_names
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

    def __init__(self, dataset, processor, max_length=None, max_retries=10):
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length
        self.max_retries = max_retries
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
                    example, self.processor, preserve_video_metadata=True,
                )
                conversation = example["conversation"]

                # Inject fake image into pure-text samples for FSDP/Zero3.
                injected_fake = not _conversation_has_media(conversation)
                if injected_fake:
                    conversation = inject_fake_image_into_conversation(conversation)

                # Ensure images are RGB
                for message in conversation:
                    content = message.get("content")
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and isinstance(
                                item.get("image"), Image.Image
                            ):
                                item["image"] = item["image"].convert("RGB")

                # Render template text.
                text = self.processor.apply_chat_template(
                    [conversation], tokenize=False,
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

                input_ids = result["input_ids"][0]  # (seq_len,)
                seq_len = input_ids.shape[0]

                # Precise length check — replace overlong samples
                if self.max_length is not None and seq_len > self.max_length:
                    logger.warning(
                        "Sample %d: %d tokens > max_length %d, replacing.",
                        idx, seq_len, self.max_length,
                    )
                    idx = random.randint(0, len(self.dataset) - 1)
                    continue

                # Build labels using template markers
                labels = build_labels_from_template(
                    result["input_ids"],  # (1, seq_len)
                    [conversation],
                    self.processor,
                )[0]  # (seq_len,)

                output = {
                    "input_ids": input_ids,
                    "attention_mask": result["attention_mask"][0],
                    "labels": labels,
                }

                # Copy media tensors (these don't have a sample batch dim)
                for key in ("pixel_values", "pixel_values_videos",
                            "image_grid_thw", "video_grid_thw",
                            "second_per_grid_ts"):
                    if key in result and result[key] is not None:
                        output[key] = result[key]

                # Mask fake vision tokens so they don't affect attention.
                if injected_fake:
                    mask_fake_vision_tokens_single(output, self.processor)

                return output

            except Exception as e:
                logger.warning(
                    "Error processing sample %d (attempt %d/%d): %s",
                    idx, attempt + 1, self.max_retries, e,
                )
                idx = random.randint(0, len(self.dataset) - 1)

        raise RuntimeError(
            f"Failed to load a valid sample after {self.max_retries} retries"
        )

    def robust_collate(self, collate_fn):
        """Wrap *collate_fn* so that on failure the entire batch is re-sampled."""
        dataset = self

        def wrapper(examples):
            last_error = None
            for attempt in range(dataset.max_retries):
                try:
                    return collate_fn(examples)
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Collate failed (attempt {attempt + 1}/{dataset.max_retries}): {e}. "
                        "Re-sampling batch."
                    )
                    examples = [
                        dataset[random.randint(0, len(dataset) - 1)]
                        for _ in range(len(examples))
                    ]
            raise RuntimeError(
                f"Collate failed after {dataset.max_retries} retries. Last error: {last_error}"
            )

        return wrapper


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
        dataset = self

        def wrapper(examples):
            last_error = None
            for attempt in range(dataset.max_retries):
                try:
                    return collate_fn(examples)
                except Exception as e:
                    last_error = e
                    logger.warning(
                        f"Collate failed (attempt {attempt + 1}/{dataset.max_retries}): {e}. "
                        "Re-sampling batch."
                    )
                    examples = [
                        dataset[random.randint(0, len(dataset) - 1)] for _ in range(len(examples))
                    ]
            raise RuntimeError(
                f"Collate failed after {dataset.max_retries} retries. Last error: {last_error}"
            )

        return wrapper

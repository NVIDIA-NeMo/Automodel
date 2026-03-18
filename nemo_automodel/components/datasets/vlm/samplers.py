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

import logging
import math
import time

import torch
from torch.utils.data import Sampler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight reimplementations of the smart_resize helpers from HF
# transformers (qwen2_vl / qwen3_vl).  Kept local so this module has no
# dependency on a specific model package.
# ---------------------------------------------------------------------------


def _smart_resize_image(height, width, factor=28, min_pixels=56 * 56, max_pixels=14 * 14 * 4 * 1280):
    """Compute the resized (height, width) for an image, matching
    ``transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize``.
    """
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _smart_resize_video(
    num_frames, height, width, temporal_factor=2, factor=32, min_pixels=128 * 128, max_pixels=16 * 16 * 2 * 2 * 2 * 6144
):
    """Compute the resized (height, width) for a video, matching
    ``transformers.models.qwen3_vl.video_processing_qwen3_vl.smart_resize``.
    """
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = math.ceil(num_frames / temporal_factor) * temporal_factor
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class LengthGroupedSampler(Sampler):
    """Sampler that groups samples by total token count for balanced
    distributed training.

    With ``shard_data=True`` each rank owns a different subset of data.
    This sampler sorts every rank's indices by **total tokens**
    (``text_tokens + media_tokens``, descending).  All ranks share the
    same ``seed + epoch`` so position *N* on every rank corresponds to a
    sample of similar length, keeping cross-rank padding minimal.

    Per-epoch randomness is achieved by rotating the sorted order by a
    deterministic random offset (same on every rank).

    Args:
        dataset: The dataset to sample from.
        seed: Base random seed (same value on every rank).
        processor: Optional HuggingFace processor (e.g. ``Qwen2VLProcessor``).
            Used to read ``image_processor`` / ``video_processor`` attributes
            for accurate media token estimation via ``smart_resize``.
    """

    def __init__(self, dataset, seed=42, processor=None, max_length=None, batch_size=1):
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0
        self.max_length = max_length
        self.batch_size = max(1, batch_size)

        # Extract processor config for accurate media token estimation
        self._image_cfg = self._extract_image_config(processor) if processor is not None else None
        self._video_cfg = self._extract_video_config(processor) if processor is not None else None

        if self._image_cfg is not None:
            logger.info("LengthGroupedSampler image config: %s", self._image_cfg)
        if self._video_cfg is not None:
            logger.info("LengthGroupedSampler video config: %s", self._video_cfg)

        # Compute text and media tokens separately for two-level sorting
        self.text_lengths, self.media_lengths = self._compute_or_load_lengths(dataset)

        # Total token count per sample (text + media) — used for sorting
        self.lengths = [t + m for t, m in zip(self.text_lengths, self.media_lengths)]

        # Pre-filter overlong samples using estimated lengths.
        # PreTokenizedDatasetWrapper still acts as a safety-net retry for the
        # rare case where the heuristic underestimates the true tokenized length.
        all_indices = range(len(dataset))
        if max_length is not None:
            # Use 1.2x headroom: the estimated length is a heuristic, so only
            # drop samples that are clearly overlong.  Borderline samples are
            # left to PreTokenizedDatasetWrapper's precise tokenize-and-retry.
            filter_threshold = max_length + 512
            kept = [i for i in all_indices if self.lengths[i] <= filter_threshold]
            n_dropped = len(dataset) - len(kept)
            if n_dropped:
                logger.info(
                    "LengthGroupedSampler: pre-filtered %d/%d samples with "
                    "estimated length > %.0f tokens (1.2 * max_length %d).",
                    n_dropped,
                    len(dataset),
                    filter_threshold,
                    max_length,
                )
        else:
            kept = list(all_indices)

        # Sort by total tokens (descending), then shuffle within small
        # buckets each epoch to add randomness while preserving grouping.
        self.sorted_indices = sorted(
            kept,
            key=lambda i: self.lengths[i],
            reverse=True,
        )

        # Cross-rank count alignment: with shard_data=True each rank owns a
        # different slice of the corpus, so different ranks may filter out
        # different numbers of overlong samples.  An imbalanced sampler length
        # causes distributed deadlock.  All-reduce MIN so every rank uses the
        # same number of steps (we drop the tail — the shortest samples).
        if max_length is not None:
            import torch.distributed as dist

            if dist.is_initialized():
                count = torch.tensor(len(self.sorted_indices), dtype=torch.long).cuda()
                dist.all_reduce(count, op=dist.ReduceOp.MIN)
                min_count = count.item()
                if min_count < len(self.sorted_indices):
                    logger.info(
                        "LengthGroupedSampler: truncating from %d to %d samples "
                        "to align with the rank that filtered the most.",
                        len(self.sorted_indices),
                        min_count,
                    )
                    self.sorted_indices = self.sorted_indices[:min_count]

    # ------------------------------------------------------------------
    # Fast length computation with disk caching
    # ------------------------------------------------------------------

    @staticmethod
    def _get_raw_samples(dataset):
        """Unwrap dataset wrappers to get the underlying list for direct access."""
        raw = dataset
        while hasattr(raw, "dataset"):
            raw = raw.dataset
        if isinstance(raw, list):
            return raw
        return None

    def _compute_or_load_lengths(self, dataset):
        """Compute token lengths with direct list access for speed."""
        # Access underlying list directly, bypassing wrapper __getitem__ overhead
        raw_samples = self._get_raw_samples(dataset)
        if raw_samples is None:
            raw_samples = [dataset[i] for i in range(len(dataset))]

        n = len(raw_samples)

        # Compute lengths with progress logging
        logger.info("Estimating token lengths for %d samples...", n)
        t0 = time.monotonic()
        text_lengths = [0] * n
        media_lengths = [0] * n

        for i, example in enumerate(raw_samples):
            text_lengths[i], media_lengths[i] = self._estimate_tokens(example)
            if (i + 1) % 100_000 == 0 or i == n - 1:
                elapsed = time.monotonic() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                logger.info(
                    "  %d/%d samples (%.1fs elapsed, %.0f samples/s)",
                    i + 1,
                    n,
                    elapsed,
                    rate,
                )

        elapsed = time.monotonic() - t0
        logger.info("Token length estimation done in %.1fs (%.0f samples/s)", elapsed, n / max(elapsed, 1e-6))

        return text_lengths, media_lengths

    # ------------------------------------------------------------------
    # Processor config extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_image_config(processor):
        ip = getattr(processor, "image_processor", None)
        if ip is None:
            return None
        patch_size = getattr(ip, "patch_size", 14)
        merge_size = getattr(ip, "merge_size", 2)
        # Qwen2VL/Qwen3VL store min/max_pixels as direct attributes;
        # fall back to ip.size dict with both Qwen-style and HF-style keys.
        size = getattr(ip, "size", {}) or {}
        min_pixels = getattr(ip, "min_pixels", None) or size.get("min_pixels") or size.get("shortest_edge") or 56 * 56
        max_pixels = (
            getattr(ip, "max_pixels", None) or size.get("max_pixels") or size.get("longest_edge") or 14 * 14 * 4 * 1280
        )
        return {
            "patch_size": patch_size,
            "merge_size": merge_size,
            "factor": patch_size * merge_size,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        }

    @staticmethod
    def _extract_video_config(processor):
        vp = getattr(processor, "video_processor", None)
        if vp is None:
            return None
        patch_size = getattr(vp, "patch_size", 16)
        merge_size = getattr(vp, "merge_size", 2)
        temporal_patch_size = getattr(vp, "temporal_patch_size", 2)
        # Qwen2VL/Qwen3VL store min/max_pixels as direct attributes;
        # fall back to vp.size dict with both Qwen-style and HF-style keys.
        size = getattr(vp, "size", {}) or {}
        min_pixels = getattr(vp, "min_pixels", None) or size.get("min_pixels") or size.get("shortest_edge") or 128 * 128
        max_pixels = (
            getattr(vp, "max_pixels", None)
            or size.get("max_pixels")
            or size.get("longest_edge")
            or 16 * 16 * 2 * 2 * 2 * 6144
        )
        fps = getattr(vp, "fps", 2.0)
        min_frames = getattr(vp, "min_frames", 4)
        max_frames = getattr(vp, "max_frames", 768)
        return {
            "patch_size": patch_size,
            "merge_size": merge_size,
            "temporal_patch_size": temporal_patch_size,
            "factor": patch_size * merge_size,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
            "fps": fps,
            "min_frames": min_frames,
            "max_frames": max_frames,
        }

    # ------------------------------------------------------------------
    # Media token estimation (accurate, using smart_resize)
    # ------------------------------------------------------------------

    def _estimate_image_tokens(self, img_meta):
        """Estimate token count for one image from its ``[height, width]`` metadata."""
        cfg = self._image_cfg
        height, width = int(img_meta[0]), int(img_meta[1])
        resized_h, resized_w = _smart_resize_image(
            height,
            width,
            factor=cfg["factor"],
            min_pixels=cfg["min_pixels"],
            max_pixels=cfg["max_pixels"],
        )
        merge_length = cfg["merge_size"] ** 2
        return (resized_h // cfg["patch_size"]) * (resized_w // cfg["patch_size"]) // merge_length

    def _estimate_video_tokens(self, vid_meta):
        """Estimate token count for one video from its
        ``[total_frames, height, width, fps, duration]`` metadata.
        """
        cfg = self._video_cfg
        total_frames = int(vid_meta[0])
        height = int(vid_meta[1])
        width = int(vid_meta[2])
        fps = float(vid_meta[3])
        duration = float(vid_meta[4])

        if total_frames == 0 and fps > 0:
            total_frames = int(duration * fps)

        # Compute sampled frame count (mirrors HF video processor logic)
        if fps > 0:
            nframes = max(1, int(total_frames / fps * cfg["fps"]))
        else:
            nframes = max(1, int(duration * cfg["fps"]))

        nframes = min(total_frames, cfg["max_frames"], nframes)
        nframes = max(cfg["min_frames"], nframes)

        tp = cfg["temporal_patch_size"]
        if nframes % tp != 0:
            nframes = ((nframes + tp - 1) // tp) * tp

        resized_h, resized_w = _smart_resize_video(
            nframes,
            height,
            width,
            temporal_factor=tp,
            factor=cfg["factor"],
            min_pixels=cfg["min_pixels"],
            max_pixels=cfg["max_pixels"],
        )
        grid_t = nframes // tp
        merge_length = cfg["merge_size"] ** 2
        return grid_t * (resized_h // cfg["patch_size"]) * (resized_w // cfg["patch_size"]) // merge_length

    # ------------------------------------------------------------------
    # Length estimation
    # ------------------------------------------------------------------

    def _estimate_tokens(self, example):
        """Return ``(text_tokens, media_tokens)`` for one example.

        Uses pre-computed ``_text_tokens`` / ``_media_tokens`` when available
        (written by ``scripts/precompute_tokens.py``).  Otherwise falls back
        to heuristic estimation.
        """
        # --- text tokens ---
        precomputed_text = example.get("_text_tokens")
        if precomputed_text is not None:
            text_tokens = int(precomputed_text)
        else:
            # Fallback: heuristic ~1 token per 3 chars
            total_chars = 0
            for msg in example.get("conversation", []):
                content = msg.get("content", [])
                if isinstance(content, str):
                    total_chars += len(content)
                elif isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":
                            total_chars += len(item.get("text", ""))
            text_tokens = total_chars // 3

        # --- media tokens (always computed from config, never cached) ---
        media_count = 0
        for msg in example.get("conversation", []):
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if item.get("type") in ("image", "video"):
                        media_count += 1

        mm_meta = example.get("mm_inputs_meta")
        if mm_meta is not None and (self._image_cfg is not None or self._video_cfg is not None):
            media_tokens = 0
            images_meta = mm_meta.get("images_meta")
            if images_meta and self._image_cfg is not None:
                for img_meta in images_meta:
                    if img_meta is not None:
                        media_tokens += self._estimate_image_tokens(img_meta)

            videos_meta = mm_meta.get("videos_meta")
            if videos_meta and self._video_cfg is not None:
                for vid_meta in videos_meta:
                    if vid_meta is not None:
                        media_tokens += self._estimate_video_tokens(vid_meta)
        else:
            media_tokens = media_count * 500

        return text_tokens, media_tokens

    # ------------------------------------------------------------------
    # Sampler protocol
    # ------------------------------------------------------------------

    def set_epoch(self, epoch):
        """Set the epoch for deterministic shuffling (standard PyTorch pattern)."""
        self.epoch = epoch

    def __iter__(self):
        # Deterministic generator seeded identically on every rank.
        # All ranks share the same seed + epoch → same chunk permutation →
        # chunk K on every rank corresponds to similar total tokens
        # (because each rank's sorted_indices is ordered by length and
        # chunks are contiguous slices of that sorted order).
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Chunk sorted_indices into groups of batch_size so that samples
        # within a chunk have similar lengths (they are adjacent in the
        # sorted order).  Then shuffle at the chunk level to add
        # per-epoch randomness while preserving intra-batch length
        # similarity and cross-rank alignment.
        bs = self.batch_size
        chunks = [self.sorted_indices[i : i + bs] for i in range(0, len(self.sorted_indices), bs)]
        chunk_perm = torch.randperm(len(chunks), generator=g)
        indices = []
        for ci in chunk_perm:
            indices.extend(chunks[ci])

        return iter(indices)

    def __len__(self):
        return len(self.sorted_indices)

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

"""Neat packing for VLM (vision-language model) pre-tokenized datasets.

Packing is split into two phases:

1. **Plan** (instant) — scan raw dataset for estimated token lengths,
   run ``greedy_knapsack`` to assign samples to bins.  No tokenization,
   no media loading.
2. **Materialize** (lazy, in ``__getitem__``) — when the DataLoader
   requests pack *k*, load + tokenize + shift + concat the samples
   assigned to bin *k*.  Runs in DataLoader worker processes, fully
   parallel.

This keeps the packing setup O(N) and lightweight, while the expensive
tokenization + media loading is distributed across ``num_workers``.
"""

import inspect
import logging
import time
from typing import Callable

import torch
import torch.utils.data

from nemo_automodel.components.datasets.llm.neat_packing import (
    greedy_knapsack,
)
from nemo_automodel.components.datasets.vlm.samplers import (
    LengthGroupedSampler,
    _smart_resize_image,
    _smart_resize_video,
)

logger = logging.getLogger(__name__)

MEDIA_KEYS = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts")


# ---------------------------------------------------------------------------
# Visual-token-balanced greedy knapsack
# ---------------------------------------------------------------------------

def greedy_knapsack_vt_balanced(
    lengths: list[int],
    max_length: int,
    visual_tokens: list[int],
) -> list[list[int]]:
    """Pack samples with standard FFD, then interleave bins by VT for balance.

    Uses the standard greedy knapsack (FFD) for optimal packing efficiency,
    then reorders bins so that consecutive packs have similar visual token
    counts.  This ensures data-parallel ranks in the same training step
    process packs with comparable VIT workload, reducing straggler effects.

    Args:
        lengths: Total token length (text + media) per sample.
        max_length: Maximum capacity per pack.
        visual_tokens: Number of media tokens per sample.

    Returns:
        A list of bins, where each bin is a list of sample indices.
    """
    t0 = time.perf_counter()

    # Phase 1: Standard FFD packing (same as greedy_knapsack)
    bins = greedy_knapsack(lengths, max_length)

    t1 = time.perf_counter()
    logger.info(
        "  VT-balanced knapsack: FFD packed %d samples -> %d bins in %.1fs",
        len(lengths), len(bins), t1 - t0,
    )

    # Phase 2: Sort bins by VT sum so consecutive packs have similar VT.
    # With DistributedSampler (sequential partitioning), ranks in the same
    # training step get consecutive packs.  Sorting ensures those packs
    # have similar VIT workload, reducing straggler effects.
    vt_sums = [sum(visual_tokens[i] for i in b) for b in bins]
    sorted_idx = sorted(range(len(bins)), key=lambda i: vt_sums[i])
    result = [bins[i] for i in sorted_idx]

    # Log VT balance statistics
    if result:
        import statistics
        vt_sums_sorted = [vt_sums[i] for i in sorted_idx]
        vt_mean = statistics.mean(vt_sums_sorted)
        vt_std = statistics.stdev(vt_sums_sorted) if len(vt_sums_sorted) > 1 else 0
        vt_min = min(vt_sums_sorted)
        vt_max = max(vt_sums_sorted)
        logger.info(
            "  VT balance: %d packs | VT per pack: mean=%.0f, std=%.0f, min=%d, max=%d, ratio=%.1fx",
            len(result), vt_mean, vt_std, vt_min, vt_max,
            vt_max / max(vt_min, 1),
        )
        # Log consecutive-pair balance (simulates DP=2 scenario)
        if len(vt_sums_sorted) > 1:
            pair_diffs = [abs(vt_sums_sorted[i] - vt_sums_sorted[i+1])
                          for i in range(0, len(vt_sums_sorted) - 1, 2)]
            logger.info(
                "  VT consecutive-pair diff: mean=%.0f, max=%d (lower=better DP balance)",
                statistics.mean(pair_diffs), max(pair_diffs),
            )

    return result


# ---------------------------------------------------------------------------
# Length estimation (no tokenization, no media loading)
# ---------------------------------------------------------------------------

def _estimate_image_tokens(img_meta, image_cfg: dict) -> int:
    """Estimate token count for one image from its ``[height, width]`` metadata."""
    height, width = int(img_meta[0]), int(img_meta[1])
    resized_h, resized_w = _smart_resize_image(
        height, width,
        factor=image_cfg["factor"],
        min_pixels=image_cfg["min_pixels"],
        max_pixels=image_cfg["max_pixels"],
    )
    merge_length = image_cfg["merge_size"] ** 2
    return (resized_h // image_cfg["patch_size"]) * (resized_w // image_cfg["patch_size"]) // merge_length


def _estimate_video_tokens(vid_meta, video_cfg: dict) -> int:
    """Estimate token count for one video from its
    ``[total_frames, height, width, fps, duration]`` metadata.
    """
    total_frames = int(vid_meta[0])
    height = int(vid_meta[1])
    width = int(vid_meta[2])
    fps = float(vid_meta[3])
    duration = float(vid_meta[4])

    if total_frames == 0 and fps > 0:
        total_frames = int(duration * fps)

    if fps > 0:
        nframes = max(1, int(total_frames / fps * video_cfg["fps"]))
    else:
        nframes = max(1, int(duration * video_cfg["fps"]))

    nframes = min(total_frames, video_cfg["max_frames"], nframes)
    nframes = max(video_cfg["min_frames"], nframes)

    tp = video_cfg["temporal_patch_size"]
    if nframes % tp != 0:
        nframes = ((nframes + tp - 1) // tp) * tp

    resized_h, resized_w = _smart_resize_video(
        nframes, height, width,
        temporal_factor=tp,
        factor=video_cfg["factor"],
        min_pixels=video_cfg["min_pixels"],
        max_pixels=video_cfg["max_pixels"],
    )
    grid_t = nframes // tp
    merge_length = video_cfg["merge_size"] ** 2
    return grid_t * (resized_h // video_cfg["patch_size"]) * (resized_w // video_cfg["patch_size"]) // merge_length


def _estimate_sample_length(
    example: dict,
    image_cfg: dict | None = None,
    video_cfg: dict | None = None,
    return_media_tokens: bool = False,
) -> int | tuple[int, int]:
    """Estimate token count from raw conversation without tokenization.

    Uses pre-computed ``_text_tokens`` (from ``precompute_tokens.py``) when
    available, otherwise falls back to ``chars // 3``.  Media tokens are
    estimated via ``smart_resize`` when processor configs are provided,
    otherwise falls back to 500 per media item.

    Args:
        return_media_tokens: If True, return ``(total_tokens, media_tokens)``
            instead of just ``total_tokens``.
    """
    # Text tokens
    precomputed = example.get("_text_tokens")
    if precomputed is not None:
        text_tokens = int(precomputed)
    else:
        total_chars = 0
        for msg in example.get("conversation", []):
            content = msg.get("content", [])
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        total_chars += len(item.get("text", ""))
        text_tokens = total_chars // 3

    # Media tokens
    media_count = 0
    for msg in example.get("conversation", []):
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") in ("image", "video"):
                    media_count += 1

    mm_meta = example.get("mm_inputs_meta")
    if mm_meta is not None and (image_cfg is not None or video_cfg is not None):
        media_tokens = 0
        images_meta = mm_meta.get("images_meta")
        if images_meta and image_cfg is not None:
            for img_meta in images_meta:
                if img_meta is not None:
                    media_tokens += _estimate_image_tokens(img_meta, image_cfg)

        videos_meta = mm_meta.get("videos_meta")
        if videos_meta and video_cfg is not None:
            for vid_meta in videos_meta:
                if vid_meta is not None:
                    media_tokens += _estimate_video_tokens(vid_meta, video_cfg)
    else:
        media_tokens = media_count * 500

    total = text_tokens + media_tokens
    if return_media_tokens:
        return total, media_tokens
    return total


# ---------------------------------------------------------------------------
# Per-sample helpers (called inside __getitem__, in worker processes)
# ---------------------------------------------------------------------------

def _compute_mrope_position_ids(
    sample: dict,
    get_rope_index: Callable,
) -> torch.Tensor | None:
    """Compute mRoPE 3D position IDs for a single sample.

    Returns ``[3, seq_len]`` or ``None`` if not applicable.
    """
    try:
        sig = inspect.signature(get_rope_index)
    except (ValueError, TypeError):
        return None

    if "input_ids" not in sig.parameters:
        return None

    input_ids = sample["input_ids"]
    if isinstance(input_ids, torch.Tensor) and input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    all_kwargs = {
        "input_ids": input_ids,
        "image_grid_thw": sample.get("image_grid_thw"),
        "video_grid_thw": sample.get("video_grid_thw"),
        "second_per_grid_ts": sample.get("second_per_grid_ts"),
        "attention_mask": sample.get("attention_mask"),
    }
    if all_kwargs["attention_mask"] is not None:
        am = all_kwargs["attention_mask"]
        if isinstance(am, torch.Tensor) and am.ndim == 1:
            all_kwargs["attention_mask"] = am.unsqueeze(0)

    kwargs = {k: v for k, v in all_kwargs.items() if k in sig.parameters}

    with torch.no_grad():
        position_ids, _ = get_rope_index(**kwargs)
        if position_ids.ndim == 3:
            position_ids = position_ids[:, 0, :]
        return position_ids


def _shift_sample(sample: dict, has_mrope: bool = False) -> dict:
    """Apply per-sample autoregressive shift before concatenation."""
    out = {}
    out["input_ids"] = sample["input_ids"][:-1]
    out["labels"] = sample["labels"][1:]
    out["attention_mask"] = sample["attention_mask"][:-1]

    if has_mrope and "position_ids" in sample and sample["position_ids"] is not None:
        out["position_ids"] = sample["position_ids"][:, :-1]

    for key in MEDIA_KEYS:
        if key in sample and sample[key] is not None:
            out[key] = sample[key]
    return out


def _build_packed_vlm_sample(
    samples: list[dict],
    pack_size: int,
    padding_idx: int,
    has_mrope: bool = False,
) -> dict:
    """Concatenate multiple shifted VLM samples into one packed sample."""
    all_input_ids: list[int] = []
    all_labels: list[int] = []
    all_attention_mask: list[int] = []
    all_position_ids_1d: list[int] = []
    mrope_position_ids_list: list[torch.Tensor] = []

    pixel_values_list: list[torch.Tensor] = []
    image_grid_thw_list: list[torch.Tensor] = []
    pixel_values_videos_list: list[torch.Tensor] = []
    video_grid_thw_list: list[torch.Tensor] = []
    second_per_grid_ts_list: list[torch.Tensor] = []
    n_images = 0
    n_videos = 0

    for seq_idx, sample in enumerate(samples, start=1):
        ids = sample["input_ids"]
        labs = sample["labels"]
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if isinstance(labs, torch.Tensor):
            labs = labs.tolist()

        seq_len = len(ids)
        all_input_ids.extend(ids)
        all_labels.extend(labs)
        all_attention_mask.extend([seq_idx] * seq_len)

        if has_mrope and "position_ids" in sample:
            mrope_position_ids_list.append(sample["position_ids"])
        else:
            all_position_ids_1d.extend(range(seq_len))

        if "pixel_values" in sample and sample["pixel_values"] is not None:
            pixel_values_list.append(sample["pixel_values"])
        if "image_grid_thw" in sample and sample["image_grid_thw"] is not None:
            n_images += sample["image_grid_thw"].shape[0]
            image_grid_thw_list.append(sample["image_grid_thw"])
        if "pixel_values_videos" in sample and sample["pixel_values_videos"] is not None:
            pixel_values_videos_list.append(sample["pixel_values_videos"])
        if "video_grid_thw" in sample and sample["video_grid_thw"] is not None:
            n_videos += sample["video_grid_thw"].shape[0]
            video_grid_thw_list.append(sample["video_grid_thw"])
        if "second_per_grid_ts" in sample and sample["second_per_grid_ts"] is not None:
            second_per_grid_ts_list.append(sample["second_per_grid_ts"])

    # No padding here — collater pads to batch-max for efficiency.
    packed = {
        "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
        "labels": torch.tensor(all_labels, dtype=torch.long),
        "attention_mask": torch.tensor(all_attention_mask, dtype=torch.long),
        "n_images": n_images,
        "n_videos": n_videos,
    }

    if has_mrope and mrope_position_ids_list:
        packed["position_ids"] = torch.cat(mrope_position_ids_list, dim=1)
    else:
        packed["position_ids"] = torch.tensor(all_position_ids_1d, dtype=torch.long)

    packed["pixel_values"] = torch.cat(pixel_values_list, dim=0) if pixel_values_list else None
    packed["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0) if image_grid_thw_list else None
    packed["pixel_values_videos"] = torch.cat(pixel_values_videos_list, dim=0) if pixel_values_videos_list else None
    packed["video_grid_thw"] = torch.cat(video_grid_thw_list, dim=0) if video_grid_thw_list else None
    packed["second_per_grid_ts"] = torch.cat(second_per_grid_ts_list, dim=0) if second_per_grid_ts_list else None

    return packed


# ---------------------------------------------------------------------------
# PackedDatasetWrapper — lazy packing via __getitem__
# ---------------------------------------------------------------------------

class PackedDatasetWrapper(torch.utils.data.Dataset):
    """A Dataset that materializes packs lazily in ``__getitem__``.

    The constructor only stores bin assignments (which sample indices go
    into each pack).  The actual tokenization, media loading, shift, and
    concatenation happen when a pack is requested — inside DataLoader
    worker processes, fully parallel.

    Args:
        inner_dataset: The ``PreTokenizedDatasetWrapper`` that tokenizes
            individual samples.
        bins: List of bins from ``greedy_knapsack``, where each bin is a
            list of sample indices into ``inner_dataset``.
        pack_size: Target packed sequence length (after shift).
        padding_idx: Token ID for padding.
        get_rope_index: Optional ``model.get_rope_index`` for mRoPE.
        max_retries: Max retries when a sample fails to tokenize.
    """

    def __init__(
        self,
        inner_dataset,
        bins: list[list[int]],
        pack_size: int,
        padding_idx: int = 0,
        get_rope_index: Callable | None = None,
        max_retries: int = 10,
    ):
        self.inner = inner_dataset
        self.bins = bins
        self.pack_size = pack_size
        self.padding_idx = padding_idx
        self.get_rope_index = get_rope_index
        self.has_mrope = get_rope_index is not None
        self.max_retries = max_retries

    def __len__(self):
        return len(self.bins)

    def __getitem__(self, pack_idx: int) -> dict:
        """Materialize one pack: tokenize + shift + concat all samples in the bin."""
        bin_indices = self.bins[pack_idx]
        shifted_samples: list[dict] = []

        for sample_idx in bin_indices:
            sample = self.inner[sample_idx]  # tokenize + load media

            if self.has_mrope and self.get_rope_index is not None:
                mrope_pos = _compute_mrope_position_ids(sample, self.get_rope_index)
                if mrope_pos is not None:
                    sample["position_ids"] = mrope_pos

            shifted = _shift_sample(sample, has_mrope=self.has_mrope)
            seq_len = shifted["input_ids"].shape[0]

            # If real length exceeds pack_size after shift, skip this sample
            if seq_len > self.pack_size:
                logger.warning(
                    "Pack %d: sample %d has %d tokens (> pack_size %d), skipping.",
                    pack_idx, sample_idx, seq_len, self.pack_size,
                )
                continue

            shifted_samples.append(shifted)

        # Truncate if total exceeds pack_size (estimation was wrong)
        total = 0
        kept: list[dict] = []
        for s in shifted_samples:
            slen = s["input_ids"].shape[0]
            if total + slen <= self.pack_size:
                kept.append(s)
                total += slen
            else:
                logger.debug(
                    "Pack %d: dropping overflow sample (%d tokens, %d/%d used).",
                    pack_idx, slen, total, self.pack_size,
                )

        if not kept:
            # Fallback: return a padding-only pack
            kept = [{"input_ids": torch.tensor([], dtype=torch.long),
                     "labels": torch.tensor([], dtype=torch.long)}]

        return _build_packed_vlm_sample(kept, self.pack_size, self.padding_idx, has_mrope=self.has_mrope)

    def robust_collate(self, collate_fn):
        """Wrap collate_fn with retry logic, delegating to inner dataset."""
        if hasattr(self.inner, "robust_collate"):
            return self.inner.robust_collate(collate_fn)
        return collate_fn


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def neat_pack_dataset_vlm(
    dataset,
    pack_size: int,
    padding_idx: int = 0,
    drop_long_samples: bool = False,
    max_packs: int | None = None,
    get_rope_index: Callable | None = None,
    ds_raw=None,
    packing_ratio: float = 1.0,
    processor=None,
    balance_media_tokens: bool = True,
) -> PackedDatasetWrapper:
    """Create a lazily-packed VLM dataset.

    1. Estimates token lengths from ``ds_raw`` (no tokenization).
    2. Runs knapsack to assign samples to bins.  When
       ``balance_media_tokens=True`` (default), uses a two-phase
       algorithm that balances visual token counts across packs,
       reducing VIT compute/memory imbalance and straggler effects.
    3. Returns a ``PackedDatasetWrapper`` whose ``__getitem__`` tokenizes
       and builds packs on-the-fly in DataLoader workers.

    Args:
        dataset: ``PreTokenizedDatasetWrapper`` for per-sample tokenization.
        pack_size: Target packed sequence length (after shift).
        padding_idx: Token ID for padding.
        drop_long_samples: Drop samples whose estimated length exceeds
            ``pack_size``.
        max_packs: Optional cap on number of packs.
        get_rope_index: Optional ``model.get_rope_index`` for mRoPE.
        ds_raw: Raw dataset (conversations) for fast length estimation.
            Falls back to ``len(dataset)`` if not provided.
        packing_ratio: Fill ratio for knapsack bins (default 1.0).
            E.g. ``0.9`` means knapsack only fills bins to ``pack_size * 0.9``,
            leaving 10% headroom to absorb estimation errors.  This reduces
            overflow drops at ``__getitem__`` time.  The actual ``pack_size``
            is still used as the hard limit.
        processor: Optional HuggingFace processor (e.g. ``Qwen2VLProcessor``).
            Used to extract ``image_processor`` / ``video_processor`` configs
            for accurate media token estimation via ``smart_resize``.
        balance_media_tokens: If True (default), use VT-balanced knapsack
            that distributes visual tokens evenly across packs.  Falls back
            to standard knapsack if no media tokens are detected.

    Returns:
        A ``PackedDatasetWrapper`` (torch Dataset).
    """
    # Extract processor configs for accurate media token estimation
    image_cfg = LengthGroupedSampler._extract_image_config(processor) if processor is not None else None
    video_cfg = LengthGroupedSampler._extract_video_config(processor) if processor is not None else None
    if image_cfg is not None or video_cfg is not None:
        logger.info("Neat packing VLM: using processor configs for media token estimation.")
    else:
        logger.warning(
            "Neat packing VLM: no processor provided — media tokens will use "
            "fallback estimate (500/item). Pass processor= for accurate packing."
        )

    # Knapsack bin capacity: leave headroom for estimation inaccuracy
    knapsack_capacity = int(pack_size * packing_ratio)

    # ── Stage 1: estimate lengths (+ media tokens) ───────────────
    estimated_media_tokens: list[int] = []

    if ds_raw is not None:
        N = len(ds_raw)
        logger.info(
            "Neat packing VLM: estimating lengths for %d samples "
            "(pack_size=%d, packing_ratio=%.2f, knapsack_capacity=%d)...",
            N, pack_size, packing_ratio, knapsack_capacity,
        )

        estimated_lengths: list[int] = []
        valid_indices: list[int] = []
        dropped_count = 0
        log_interval = max(1, N // 10)  # log every 10%
        t0 = time.perf_counter()

        for i in range(N):
            est_len, media_toks = _estimate_sample_length(
                ds_raw[i], image_cfg=image_cfg, video_cfg=video_cfg, return_media_tokens=True,
            )
            est_len -= 1  # -1 for shift
            if est_len > pack_size:
                if drop_long_samples:
                    dropped_count += 1
                    continue
                # Keep it — real length might be shorter; __getitem__ will handle overflow
            est_len = min(est_len, knapsack_capacity)
            estimated_lengths.append(est_len)
            estimated_media_tokens.append(media_toks)
            valid_indices.append(i)

            if (i + 1) % log_interval == 0 or (i + 1) == N:
                elapsed = time.perf_counter() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(
                    "  Length estimation: %d/%d (%.0f%%) | %.0f samples/s | dropped %d",
                    i + 1, N, 100.0 * (i + 1) / N, rate, dropped_count,
                )

        est_elapsed = time.perf_counter() - t0
        n_visual = sum(1 for vt in estimated_media_tokens if vt > 0)
        logger.info(
            "  Length estimation done in %.1fs (%d valid, %d dropped, %d visual).",
            est_elapsed, len(valid_indices), dropped_count, n_visual,
        )
    else:
        # No raw dataset — use dataset length, assume uniform distribution
        N = len(dataset)
        logger.info("Neat packing VLM: no ds_raw, using uniform estimate for %d samples.", N)
        estimated_lengths = [knapsack_capacity // 2] * N
        estimated_media_tokens = [0] * N
        valid_indices = list(range(N))

    # ── Stage 2: knapsack ────────────────────────────────────────
    has_visual = any(vt > 0 for vt in estimated_media_tokens)
    use_vt_balanced = balance_media_tokens and has_visual

    if use_vt_balanced:
        logger.info(
            "Neat packing VLM: running VT-balanced knapsack on %d samples...",
            len(estimated_lengths),
        )
        t1 = time.perf_counter()
        bins_local = greedy_knapsack_vt_balanced(
            estimated_lengths, knapsack_capacity, estimated_media_tokens,
        )
        knapsack_elapsed = time.perf_counter() - t1
        logger.info("  VT-balanced knapsack done in %.1fs.", knapsack_elapsed)
    else:
        if balance_media_tokens and not has_visual:
            logger.info("Neat packing VLM: no visual samples detected, using standard knapsack.")
        logger.info("Neat packing VLM: running greedy knapsack on %d samples...", len(estimated_lengths))
        t1 = time.perf_counter()
        bins_local = greedy_knapsack(estimated_lengths, knapsack_capacity)
        knapsack_elapsed = time.perf_counter() - t1
        logger.info("  Greedy knapsack done in %.1fs.", knapsack_elapsed)

    # Convert local bin indices back to real dataset indices
    bins = [[valid_indices[j] for j in bin_local] for bin_local in bins_local]

    if max_packs is not None:
        bins = bins[:max_packs]

    # ── Packing statistics ──────────────────────────────────────
    total_est_tokens = sum(estimated_lengths)
    n_packs = len(bins)
    utilization = 100.0 * total_est_tokens / (n_packs * pack_size) if n_packs else 0

    samples_per_pack = [len(b) for b in bins]
    avg_samples = sum(samples_per_pack) / n_packs if n_packs else 0
    min_samples = min(samples_per_pack) if samples_per_pack else 0
    max_samples = max(samples_per_pack) if samples_per_pack else 0

    est_arr = estimated_lengths
    avg_len = sum(est_arr) / len(est_arr) if est_arr else 0
    min_len = min(est_arr) if est_arr else 0
    max_len = max(est_arr) if est_arr else 0

    logger.info(
        "Neat packing VLM: %d samples -> %d packs "
        "(estimated utilization: %.1f%%)",
        len(valid_indices), n_packs, utilization,
    )
    logger.info(
        "  Samples per pack: avg=%.1f, min=%d, max=%d",
        avg_samples, min_samples, max_samples,
    )
    logger.info(
        "  Estimated token lengths: avg=%.0f, min=%d, max=%d",
        avg_len, min_len, max_len,
    )

    # ── Return lazy dataset ──────────────────────────────────────
    return PackedDatasetWrapper(
        inner_dataset=dataset,
        bins=bins,
        pack_size=pack_size,
        padding_idx=padding_idx,
        get_rope_index=get_rope_index,
    )

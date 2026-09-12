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

"""Single authoritative utility for VLM media token count estimation.

Offline packing (``neat_packing_vlm``) and length-grouped sampling
(``samplers``) need per-sample token lengths *before* tokenization and media
loading.  Text tokens come from precomputed counts or a chars-based heuristic;
media tokens are resolved here, from the HuggingFace processor, through three
tiers (first available wins):

1. **Processor API** — ``processor._get_num_multimodal_tokens(image_sizes=...)``
   (transformers >= 4.53).  The processor's own authoritative count, computed
   from image sizes without loading pixels.
2. **Probe** — run the real processor once per unique image size on a
   synthetic image and count the expansion in ``input_ids``.  Costs one real
   preprocess per unique size during planning; pixel values are never stored.
3. **Qwen geometry fast path** — a local replica of the Qwen ``smart_resize``
   + patch/merge math.  Only correct for Qwen-geometry ViTs
   (qwen2_vl/qwen2_5_vl/qwen3_vl-style processors); kept as a cheap fallback
   when the processor exposes neither tier 1 nor a probe-able interface.

Videos always use the Qwen geometry math: the sampled-frame count depends on
fps/duration metadata that the tier-1 API does not accept, and probing would
require decoding video frames.

Estimating media tokens with model-specific math that has silently drifted
from the processor's real expansion causes whole documents to be dropped at
pack materialization time, so tier 3 is covered by a parity test against the
real processor output
(``tests/unit_tests/datasets/vlm/test_media_token_estimation.py``).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import ProcessorMixin

logger = logging.getLogger(__name__)

DEFAULT_TOKENS_PER_MEDIA_ITEM = 500
"""Fallback media token estimate when no processor-derived count is available."""


# ---------------------------------------------------------------------------
# Qwen geometry fast path (tier 3)
#
# Local reimplementations of the smart_resize helpers from HF transformers
# (qwen2_vl / qwen3_vl).  Kept local so this module has no dependency on a
# specific model package.  Only correct for Qwen-geometry ViTs; guarded
# against drift by the parity test named in the module docstring.
# ---------------------------------------------------------------------------


def _smart_resize_image(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280,
) -> tuple[int, int]:
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
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 32,
    min_pixels: int = 128 * 128,
    max_pixels: int = 16 * 16 * 2 * 2 * 2 * 6144,
) -> tuple[int, int]:
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


@dataclass(frozen=True)
class _ImageGeometry:
    """Qwen-geometry image processor settings used by the tier-3 fast path."""

    patch_size: int
    merge_size: int
    factor: int
    min_pixels: int
    max_pixels: int


@dataclass(frozen=True)
class _VideoGeometry:
    """Qwen-geometry video processor settings used for video estimation."""

    patch_size: int
    merge_size: int
    temporal_patch_size: int
    factor: int
    min_pixels: int
    max_pixels: int
    fps: float
    min_frames: int
    max_frames: int


def _extract_image_geometry(processor: "ProcessorMixin | None") -> _ImageGeometry | None:
    """Read Qwen-geometry settings from ``processor.image_processor``, if present."""
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return None
    # `or`-defaults throughout: transformers 5.x processors expose several of
    # these attributes with value None (e.g. min_pixels on default-constructed
    # Qwen2VLImageProcessor), so getattr defaults alone are not enough.
    patch_size = getattr(ip, "patch_size", None) or 14
    merge_size = getattr(ip, "merge_size", None) or 2
    # Qwen2VL/Qwen3VL store min/max_pixels as direct attributes;
    # fall back to ip.size (dict or SizeDict) with both Qwen-style and HF-style keys.
    size = getattr(ip, "size", None) or {}
    min_pixels = getattr(ip, "min_pixels", None) or size.get("min_pixels") or size.get("shortest_edge") or 56 * 56
    max_pixels = (
        getattr(ip, "max_pixels", None) or size.get("max_pixels") or size.get("longest_edge") or 14 * 14 * 4 * 1280
    )
    return _ImageGeometry(
        patch_size=patch_size,
        merge_size=merge_size,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )


def _extract_video_geometry(processor: "ProcessorMixin | None") -> _VideoGeometry | None:
    """Read Qwen-geometry settings from ``processor.video_processor``, if present."""
    vp = getattr(processor, "video_processor", None)
    if vp is None:
        return None
    # `or`-defaults throughout: transformers 5.x processors expose several of
    # these attributes with value None (e.g. fps on Qwen2VLVideoProcessor).
    patch_size = getattr(vp, "patch_size", None) or 16
    merge_size = getattr(vp, "merge_size", None) or 2
    temporal_patch_size = getattr(vp, "temporal_patch_size", None) or 2
    # Qwen2VL/Qwen3VL store min/max_pixels as direct attributes;
    # fall back to vp.size (dict or SizeDict) with both Qwen-style and HF-style keys.
    size = getattr(vp, "size", None) or {}
    min_pixels = getattr(vp, "min_pixels", None) or size.get("min_pixels") or size.get("shortest_edge") or 128 * 128
    max_pixels = (
        getattr(vp, "max_pixels", None)
        or size.get("max_pixels")
        or size.get("longest_edge")
        or 16 * 16 * 2 * 2 * 2 * 6144
    )
    return _VideoGeometry(
        patch_size=patch_size,
        merge_size=merge_size,
        temporal_patch_size=temporal_patch_size,
        factor=patch_size * merge_size,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        fps=getattr(vp, "fps", None) or 2.0,
        min_frames=getattr(vp, "min_frames", None) or 4,
        max_frames=getattr(vp, "max_frames", None) or 768,
    )


def _sequence_length(input_ids: object) -> int:
    """Length of the first sequence in a tokenizer/processor ``input_ids`` output.

    Args:
        input_ids: Token ids as either a flat sequence of length ``[seq]``, a
            batched nested sequence of shape ``[batch, seq]``, or a tensor of
            shape ``[seq]`` / ``[batch, seq]``.  Only the last axis is read, so
            batched inputs must hold a single sequence for the count to be
            meaningful.

    Returns:
        The number of token ids along the sequence axis.
    """
    if hasattr(input_ids, "shape"):  # torch tensor / numpy array
        return int(input_ids.shape[-1])
    if input_ids and isinstance(input_ids[0], (list, tuple)):
        return len(input_ids[0])
    return len(input_ids)


class MediaTokenEstimator:
    """Resolves per-media token counts from the HuggingFace processor.

    One instance is created per planning pass (packing plan or sampler init)
    and queried per sample through :meth:`estimate_media_tokens`.  Image counts
    are cached per unique ``(height, width)``, so the probe tier runs the real
    processor at most once per unique image size.

    Args:
        processor: HuggingFace processor (e.g. ``Qwen2VLProcessor``), or
            ``None``.  Without a processor every media item falls back to
            ``DEFAULT_TOKENS_PER_MEDIA_ITEM``.
    """

    def __init__(self, processor: "ProcessorMixin | None") -> None:
        self.processor = processor
        self._image_geometry = _extract_image_geometry(processor)
        self._video_geometry = _extract_video_geometry(processor)
        self._image_token_cache: dict[tuple[int, int], int] = {}
        self._processor_api_failed = False
        self._probe_failed = False
        self._probe_logged = False

    @property
    def can_estimate(self) -> bool:
        """Whether a processor is attached (media counts are better than the flat default)."""
        return self.processor is not None

    def estimate_media_tokens(
        self,
        images_meta: Sequence[Sequence[int] | None] | None = None,
        videos_meta: Sequence[Sequence[float] | None] | None = None,
    ) -> int:
        """Estimate the total media token count for one sample.

        Args:
            images_meta: Per-image ``[height, width]`` metadata; entries may be
                ``None`` for images whose size is unknown.
            videos_meta: Per-video ``[total_frames, height, width, fps,
                duration]`` metadata; entries may be ``None``.

        Returns:
            Sum of the estimated token counts over all media items.  Items
            with ``None`` metadata are skipped.
        """
        total = 0
        for image_meta in images_meta or ():
            if image_meta is not None:
                total += self._image_tokens(int(image_meta[0]), int(image_meta[1]))
        for video_meta in videos_meta or ():
            if video_meta is not None:
                total += self._video_tokens(
                    total_frames=int(video_meta[0]),
                    height=int(video_meta[1]),
                    width=int(video_meta[2]),
                    fps=float(video_meta[3]),
                    duration=float(video_meta[4]),
                )
        return total

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------

    def _image_tokens(self, height: int, width: int) -> int:
        """Token count for one image of size ``(height, width)``.

        Resolution order: processor API (tier 1), probe (tier 2), Qwen
        geometry (tier 3), then ``DEFAULT_TOKENS_PER_MEDIA_ITEM``.
        """
        key = (height, width)
        cached = self._image_token_cache.get(key)
        if cached is not None:
            return cached
        count = self._image_tokens_from_processor_api(height, width)
        if count is None:
            count = self._probe_image_tokens(height, width)
        if count is None:
            count = self._image_tokens_from_geometry(height, width)
        if count is None:
            count = DEFAULT_TOKENS_PER_MEDIA_ITEM
        self._image_token_cache[key] = count
        return count

    def _image_tokens_from_processor_api(self, height: int, width: int) -> int | None:
        """Tier 1: ask the processor itself (transformers >= 4.53)."""
        get_counts = getattr(self.processor, "_get_num_multimodal_tokens", None)
        if get_counts is None or self._processor_api_failed:
            return None
        try:
            counts = get_counts(image_sizes=[(height, width)]).num_image_tokens
            return int(counts[0])
        except Exception:
            logger.warning(
                "processor._get_num_multimodal_tokens failed for %s; falling back to probe/geometry estimation.",
                type(self.processor).__name__,
                exc_info=True,
            )
            self._processor_api_failed = True
            return None

    def _probe_image_tokens(self, height: int, width: int) -> int | None:
        """Tier 2: run the real processor on a synthetic image and count the expansion."""
        processor = self.processor
        tokenizer = getattr(processor, "tokenizer", None)
        image_token = getattr(processor, "image_token", None)
        if self._probe_failed or not callable(processor) or tokenizer is None or not image_token:
            return None
        try:
            from PIL import Image

            if not self._probe_logged:
                logger.info(
                    "%s exposes no _get_num_multimodal_tokens; probing it once per unique image size "
                    "to resolve media token counts.",
                    type(processor).__name__,
                )
                self._probe_logged = True
            image = Image.new("RGB", (width, height))
            expanded = _sequence_length(processor(text=image_token, images=[image])["input_ids"])
            base = _sequence_length(tokenizer(image_token)["input_ids"])
            # The single placeholder token was replaced by `count` tokens.
            count = expanded - base + 1
            return count if count > 0 else None
        except Exception:
            logger.warning(
                "Media token probe failed for %s; falling back to geometry estimation.",
                type(processor).__name__,
                exc_info=True,
            )
            self._probe_failed = True
            return None

    def _image_tokens_from_geometry(self, height: int, width: int) -> int | None:
        """Tier 3: Qwen-geometry smart_resize + patch/merge math."""
        geometry = self._image_geometry
        if geometry is None:
            return None
        resized_h, resized_w = _smart_resize_image(
            height,
            width,
            factor=geometry.factor,
            min_pixels=geometry.min_pixels,
            max_pixels=geometry.max_pixels,
        )
        merge_length = geometry.merge_size**2
        return (resized_h // geometry.patch_size) * (resized_w // geometry.patch_size) // merge_length

    # ------------------------------------------------------------------
    # Videos
    # ------------------------------------------------------------------

    def _video_tokens(
        self,
        *,
        total_frames: int,
        height: int,
        width: int,
        fps: float,
        duration: float,
    ) -> int:
        """Token count for one video from its decode metadata.

        Always uses the Qwen geometry math (mirrors the HF video processor's
        frame sampling + smart_resize); falls back to
        ``DEFAULT_TOKENS_PER_MEDIA_ITEM`` when the processor exposes no video
        settings.

        Args:
            total_frames: Number of frames in the source video (0 if unknown).
            height: Source frame height in pixels.
            width: Source frame width in pixels.
            fps: Source frames per second (0 if unknown).
            duration: Source duration in seconds.
        """
        geometry = self._video_geometry
        if geometry is None:
            return DEFAULT_TOKENS_PER_MEDIA_ITEM

        if total_frames == 0 and fps > 0:
            total_frames = int(duration * fps)

        # Compute sampled frame count (mirrors HF video processor logic)
        if fps > 0:
            nframes = max(1, int(total_frames / fps * geometry.fps))
        else:
            nframes = max(1, int(duration * geometry.fps))

        nframes = min(total_frames, geometry.max_frames, nframes)
        nframes = max(geometry.min_frames, nframes)

        tp = geometry.temporal_patch_size
        if nframes % tp != 0:
            nframes = ((nframes + tp - 1) // tp) * tp

        resized_h, resized_w = _smart_resize_video(
            nframes,
            height,
            width,
            temporal_factor=tp,
            factor=geometry.factor,
            min_pixels=geometry.min_pixels,
            max_pixels=geometry.max_pixels,
        )
        grid_t = nframes // tp
        merge_length = geometry.merge_size**2
        return grid_t * (resized_h // geometry.patch_size) * (resized_w // geometry.patch_size) // merge_length

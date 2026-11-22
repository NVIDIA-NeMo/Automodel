# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native image and video preprocessing for Muse Spark Open."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature


def _least_distortion_fit(
    ideal_height: float,
    ideal_width: float,
    image_height: int,
    image_width: int,
    minimum: int,
    maximum: int,
) -> tuple[int, int]:
    """Choose an integer grid near the ideal size with minimum aspect error."""
    candidates = set(
        itertools.product(
            (math.floor(ideal_height), math.ceil(ideal_height)),
            (math.floor(ideal_width), math.ceil(ideal_width)),
        )
    )
    candidates = {
        (height, width)
        for height, width in candidates
        if height > 0 and width > 0 and minimum <= height * width <= maximum
    }
    if not candidates:
        raise ValueError("No rounded Muse Spark Open resize target satisfies the token budget")
    aspect = image_height / image_width
    return min(candidates, key=lambda item: abs(item[0] / item[1] - aspect))


def _least_distance_fit(
    ideal_height: float,
    ideal_width: float,
    minimum: int,
    maximum: int,
) -> tuple[int, int]:
    """Find the closest positive integer grid inside a token-count interval."""
    best: tuple[float, int, int] | None = None
    for height in range(1, maximum + 1):
        lower = math.ceil(minimum / height)
        upper = math.floor(maximum / height)
        if lower > upper:
            continue
        width = min(max(round(ideal_width), lower), upper)
        distance = (height - ideal_height) ** 2 + (width - ideal_width) ** 2
        if best is None or distance < best[0]:
            best = (distance, height, width)
    if best is None:
        raise ValueError("No Muse Spark Open resize target satisfies the token budget")
    return best[1], best[2]


def find_resize_target(
    image_height: int,
    image_width: int,
    token_patch_size: int,
    minimum_tokens: int,
    maximum_tokens: int,
) -> tuple[int, int, int]:
    """Return an aspect-preserving aligned canvas and visual-token count.

    Args:
        image_height: Source image height in pixels.
        image_width: Source image width in pixels.
        token_patch_size: Pixel extent represented by one post-shuffle token.
        minimum_tokens: Minimum number of visual tokens.
        maximum_tokens: Maximum number of visual tokens.

    Returns:
        ``(height, width, tokens)`` with height and width aligned to
        ``token_patch_size``.
    """
    ideal_height = image_height / token_patch_size
    ideal_width = image_width / token_patch_size
    ideal_count = ideal_height * ideal_width
    aspect = ideal_width / ideal_height
    if ideal_count > maximum_tokens:
        ideal_height = math.sqrt(maximum_tokens / aspect)
        ideal_width = ideal_height * aspect
    elif ideal_count < minimum_tokens:
        ideal_height = math.sqrt(minimum_tokens / aspect)
        ideal_width = ideal_height * aspect
    try:
        patches_height, patches_width = _least_distortion_fit(
            ideal_height,
            ideal_width,
            image_height,
            image_width,
            minimum_tokens,
            maximum_tokens,
        )
    except ValueError:
        patches_height, patches_width = _least_distance_fit(
            ideal_height,
            ideal_width,
            minimum_tokens,
            maximum_tokens,
        )
    return (
        patches_height * token_patch_size,
        patches_width * token_patch_size,
        patches_height * patches_width,
    )


def _image_to_uint8(image: Any) -> torch.Tensor:
    """Convert one PIL/NumPy/Tensor image to contiguous uint8 ``[3,H,W]``."""
    if isinstance(image, Image.Image):
        array = np.array(image.convert("RGB"), copy=True)
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()
    if isinstance(image, np.ndarray):
        value = torch.from_numpy(np.array(image, copy=True))
    else:
        value = torch.as_tensor(image).detach().cpu()
    if value.ndim != 3:
        raise ValueError(f"Muse Spark Open expected an image with rank 3, got {tuple(value.shape)}")
    if value.shape[0] in (1, 3, 4):
        value = value[:3]
    else:
        value = value[..., :3].permute(2, 0, 1)
    if value.dtype != torch.uint8:
        if value.is_floating_point() and value.numel() and value.max().item() <= 1.0:
            value = value * 255
        value = value.clamp(0, 255).to(torch.uint8)
    if value.shape[0] == 1:
        value = value.expand(3, -1, -1)
    return value.contiguous()


def _resize_and_normalize(
    image: Any,
    target_height: int,
    target_width: int,
    image_mean: tuple[float, float, float],
    image_std: tuple[float, float, float],
) -> torch.Tensor:
    """Lanczos-resize one image and return float32 normalized ``[3,H,W]``."""
    value = _image_to_uint8(image)
    pil_image = Image.fromarray(value.permute(1, 2, 0).numpy())
    pil_image = pil_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    value = torch.from_numpy(np.array(pil_image, copy=True)).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(image_mean).reshape(3, 1, 1)
    std = torch.tensor(image_std).reshape(3, 1, 1)
    return (value.clamp_(0.0, 1.0) - mean) / std


def _patchify(clip: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, int, int]:
    """Flatten ``[temporal,3,H,W]`` into row-major temporal patch vectors."""
    temporal, channels, height, width = clip.shape
    if height % patch_size or width % patch_size:
        raise ValueError("Muse Spark Open preprocessed pixels must align to patch_size")
    grid_height, grid_width = height // patch_size, width // patch_size
    patches = (
        clip.reshape(temporal, channels, grid_height, patch_size, grid_width, patch_size)
        .permute(2, 4, 0, 1, 3, 5)
        .reshape(grid_height * grid_width, temporal * channels * patch_size * patch_size)
    )
    return patches, grid_height, grid_width


@dataclass(frozen=True)
class MuseSparkVideoFeatures:
    """Preprocessed flattened video patches and their placeholder metadata."""

    patches: torch.Tensor
    grid_thw: tuple[int, int, int]
    tokens_per_clip: int
    timestamps: tuple[float, ...]


class MuseSparkImageProcessor(BaseImageProcessor):
    """Resize, normalize, and flatten Muse Spark Open image/video patches."""

    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(
        self,
        *,
        patch_size: int = 14,
        patch_temporal: int = 2,
        downsample_factor: int = 2,
        min_image_tokens: int = 1,
        max_image_tokens: int = 4096,
        max_video_tokens: int = 4096,
        max_tokens_per_frame: int = 144,
        num_frames_per_video: int = 96,
        sampling_fps: float | None = 2.0,
        frame_factor: int = 2,
        image_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
        image_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
        **kwargs: Any,
    ) -> None:
        """Store the checkpoint preprocessing contract."""
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.patch_temporal = patch_temporal
        self.downsample_factor = downsample_factor
        self.min_image_tokens = min_image_tokens
        self.max_image_tokens = max_image_tokens
        self.max_video_tokens = max_video_tokens
        self.max_tokens_per_frame = max_tokens_per_frame
        self.num_frames_per_video = num_frames_per_video
        self.sampling_fps = sampling_fps
        self.frame_factor = frame_factor
        self.image_mean = tuple(image_mean)
        self.image_std = tuple(image_std)

    @property
    def token_patch_size(self) -> int:
        """Return the pixel side represented by one post-shuffle token."""
        return self.patch_size * self.downsample_factor

    def preprocess(
        self,
        images: Any,
        *,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> BatchFeature:
        """Preprocess a flat image list into patch and grid tensors.

        Args:
            images: One image or a flat list of PIL/NumPy/Tensor images.
            return_tensors: Optional Transformers tensor type.
            **kwargs: Unused standard image-processor overrides.

        Returns:
            ``pixel_values`` has layout ``[sum(h*w),3*t*p*p]`` and
            ``image_grid_thw`` has layout ``[images,3]``.
        """
        del kwargs
        flat_images = list(images) if isinstance(images, (list, tuple)) else [images]
        patches: list[torch.Tensor] = []
        grids: list[tuple[int, int, int]] = []
        token_counts: list[int] = []
        for image in flat_images:
            value = _image_to_uint8(image)
            height, width = value.shape[-2:]
            target_height, target_width, token_count = find_resize_target(
                height,
                width,
                self.token_patch_size,
                self.min_image_tokens,
                self.max_image_tokens,
            )
            value = _resize_and_normalize(
                value,
                target_height,
                target_width,
                self.image_mean,
                self.image_std,
            )
            clip = value.unsqueeze(0).expand(self.patch_temporal, -1, -1, -1)
            image_patches, grid_height, grid_width = _patchify(clip, self.patch_size)
            patches.append(image_patches)
            grids.append((1, grid_height, grid_width))
            token_counts.append(token_count)
        data = {
            "pixel_values": torch.cat(patches) if patches else torch.empty(0),
            "image_grid_thw": torch.tensor(grids, dtype=torch.long),
            "image_token_counts": torch.tensor(token_counts, dtype=torch.long),
        }
        return BatchFeature(data=data, tensor_type=return_tensors)

    def preprocess_video(self, video: Any) -> MuseSparkVideoFeatures:
        """Preprocess one video into flattened temporal clips.

        Args:
            video: A ``[frames,channels,height,width]`` tensor, a sequence of
                frames, or ``(frames, metadata)`` where metadata may contain
                ``fps`` and ``frames_indices``.

        Returns:
            Flattened patches, one ``[clips,h,w]`` grid, tokens per clip, and
            sampled timestamps.
        """
        metadata: dict[str, Any] = {}
        if isinstance(video, tuple) and len(video) == 2 and isinstance(video[1], dict):
            frames, metadata = video
        else:
            frames = video
        if isinstance(frames, torch.Tensor) and frames.ndim == 4:
            frame_list = list(frames)
        else:
            frame_list = list(frames)
        if not frame_list:
            raise ValueError("Muse Spark Open video input has no frames")
        fps = float(metadata.get("fps", 1.0) or 1.0)
        source_indices = list(metadata.get("frames_indices", range(len(frame_list))))
        factor = max(self.frame_factor, self.patch_temporal)
        if self.sampling_fps and fps > 0:
            requested = int(len(frame_list) * self.sampling_fps / fps)
        else:
            requested = len(frame_list)
        requested = min(max(requested, factor), self.num_frames_per_video, len(frame_list))
        requested = max(factor, requested // factor * factor)
        if len(frame_list) > requested:
            selected = np.linspace(0, len(frame_list) - 1, requested, dtype=int).tolist()
            frame_list = [frame_list[index] for index in selected]
            source_indices = [source_indices[index] for index in selected]
        while len(frame_list) < factor:
            frame_list.append(frame_list[-1])
            source_indices.append(source_indices[-1])
        aligned = len(frame_list) // factor * factor
        frame_list, source_indices = frame_list[:aligned], source_indices[:aligned]

        first = _image_to_uint8(frame_list[0])
        target_height, target_width, tokens_per_clip = find_resize_target(
            first.shape[-2],
            first.shape[-1],
            self.token_patch_size,
            self.min_image_tokens,
            min(self.max_tokens_per_frame, self.max_video_tokens),
        )
        normalized = [
            _resize_and_normalize(frame, target_height, target_width, self.image_mean, self.image_std)
            for frame in frame_list
        ]
        clips = []
        for offset in range(0, len(normalized), self.patch_temporal):
            patch, grid_height, grid_width = _patchify(
                torch.stack(normalized[offset : offset + self.patch_temporal]),
                self.patch_size,
            )
            clips.append(patch)
        timestamps = tuple(float(index) / fps for index in source_indices[:: self.patch_temporal])
        return MuseSparkVideoFeatures(
            patches=torch.cat(clips),
            grid_thw=(len(clips), grid_height, grid_width),
            tokens_per_clip=tokens_per_clip,
            timestamps=timestamps,
        )

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        """Return the visual placeholder count for an unprocessed image."""
        values = images_kwargs or {}
        return find_resize_target(
            height,
            width,
            values.get("token_patch_size", self.token_patch_size),
            values.get("min_image_tokens", self.min_image_tokens),
            values.get("max_image_tokens", self.max_image_tokens),
        )[2]


__all__ = [
    "MuseSparkImageProcessor",
    "MuseSparkVideoFeatures",
    "find_resize_target",
]

# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from collections.abc import Callable, MutableMapping
from contextlib import contextmanager
from typing import Any

import torch

VLM_PP_MEDIA_KEY = "_vlm_pp_media_chunks"

_VLM_MEDIA_KEYS = (
    "pixel_values",
    "patch_pixel_values",
    "num_patches",
    "patch_newline_mask",
    "image_grid_hws",
    "image_grid_thw",
    "image_sizes",
    "image_position_ids",
    "n_images_per_sample",
    "pixel_values_videos",
    "video_grid_thw",
    "n_videos_per_sample",
)


def chunk_vlm_media(
    pixel_values: torch.Tensor,
    image_grid: torch.Tensor,
    batch_size: int,
    n_microbatches: int,
    n_images_per_sample: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split VLM pixel values and media metadata into PP microbatch chunks.

    Handles four layouts:
    1. ``[N, C, H, W]`` with ``N == batch_size`` -- one full image per sample.
    2. ``[N, max_patches, D]`` with ``N == batch_size`` -- padded patches per image.
    3. Flat patches ``[total_patches, D]`` with per-sample media counts from
       ``n_images_per_sample``.
    4. Flat patches with ``n_images == batch_size`` -- legacy one-image-per-sample.
    """
    n_images = image_grid.shape[0]
    pixel_values_chunks: list[torch.Tensor] = []
    image_grid_chunks: list[torch.Tensor] = []

    if pixel_values.shape[0] == batch_size and pixel_values.dim() in (3, 4):
        # 4D full-image tensors and 3D padded-patch tensors are indexed by sample.
        pixel_values_chunks = list(pixel_values.chunk(n_microbatches, dim=0))
        image_grid_chunks = list(image_grid.chunk(n_microbatches, dim=0))
    elif pixel_values.dim() == 3 and n_images_per_sample is not None:
        # Multi-image padded-patch layout: split by image counts per sample.
        cumsum_images = torch.cumsum(n_images_per_sample, dim=0)
        # Match torch.chunk-style uneven splits so trailing samples are not dropped.
        samples_per_mb = -(-batch_size // n_microbatches)
        for mb_idx in range(n_microbatches):
            s_start = mb_idx * samples_per_mb
            s_end = min(s_start + samples_per_mb, batch_size)
            img_start = 0 if s_start == 0 else int(cumsum_images[s_start - 1].item())
            img_end = int(cumsum_images[s_end - 1].item()) if s_end > 0 else 0
            pixel_values_chunks.append(pixel_values[img_start:img_end])
            image_grid_chunks.append(image_grid[img_start:img_end])
    elif n_images_per_sample is not None:
        # General flat-patch layout: map samples -> media entries -> patches.
        patch_counts = image_grid.prod(dim=1)
        cumsum_patches = torch.cumsum(patch_counts, dim=0)
        cumsum_images = torch.cumsum(n_images_per_sample, dim=0)

        samples_per_mb = -(-batch_size // n_microbatches)
        for mb_idx in range(n_microbatches):
            s_start = mb_idx * samples_per_mb
            s_end = min(s_start + samples_per_mb, batch_size)

            img_start = 0 if s_start == 0 else cumsum_images[s_start - 1].item()
            img_end = cumsum_images[s_end - 1].item() if s_end > 0 else 0

            image_grid_chunks.append(image_grid[img_start:img_end])

            patch_start = 0 if img_start == 0 else cumsum_patches[img_start - 1].item()
            patch_end = cumsum_patches[img_end - 1].item() if img_end > 0 else 0
            pixel_values_chunks.append(pixel_values[int(patch_start) : int(patch_end)])
    elif n_images == batch_size:
        # Legacy: exactly one image per sample.
        patch_counts = image_grid.prod(dim=1)
        cumsum = torch.cumsum(patch_counts, dim=0)

        images_per_mb = -(-batch_size // n_microbatches)
        for mb_idx in range(n_microbatches):
            img_start = mb_idx * images_per_mb
            img_end = min(img_start + images_per_mb, n_images)

            image_grid_chunks.append(image_grid[img_start:img_end])

            patch_start = 0 if img_start == 0 else cumsum[img_start - 1].item()
            patch_end = cumsum[img_end - 1].item() if img_end > 0 else 0
            pixel_values_chunks.append(pixel_values[int(patch_start) : int(patch_end)])
    else:
        raise ValueError(
            "VLM PP chunking cannot align pixel_values with the batch: "
            f"pixel_values.shape={tuple(pixel_values.shape)}, "
            f"image_grid.shape={tuple(image_grid.shape)}, "
            f"n_images={n_images}, batch_size={batch_size}, "
            f"n_images_per_sample={'set' if n_images_per_sample is not None else 'None'}. "
            "Either ensure pixel_values has shape [batch_size, ...] (one media tensor per "
            "sample) or pass n_images_per_sample so the chunker can map images to samples."
        )

    return pixel_values_chunks, image_grid_chunks


def chunk_step3_media(
    pixel_values: torch.Tensor,
    *,
    batch_size: int,
    n_microbatches: int,
    num_patches: torch.Tensor | None = None,
    patch_pixel_values: torch.Tensor | None = None,
    patch_newline_mask: torch.Tensor | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Chunk image tensors with per-sample patch counts for PP microbatches.

    Step3 processors emit one full image per sample in ``pixel_values`` and a
    flat list of optional crop patches in ``patch_pixel_values``. ``num_patches``
    maps samples to the flat patch tensor. Processors may also emit
    ``pixel_values`` itself as a flat patch tensor using the same mapping.
    """
    if num_patches is None:
        num_patches = torch.zeros(batch_size, dtype=torch.long, device=pixel_values.device)
    else:
        num_patches = num_patches.to(dtype=torch.long).view(-1)
        if num_patches.numel() != batch_size:
            raise ValueError(
                f"num_patches must have length batch_size={batch_size}, got shape={tuple(num_patches.shape)}."
            )

    flat_pixel_values = pixel_values.shape[0] != batch_size
    if flat_pixel_values and int(num_patches.sum().item()) != pixel_values.shape[0]:
        raise ValueError(
            "VLM PP chunking cannot align pixel_values with num_patches: "
            f"pixel_values.shape={tuple(pixel_values.shape)}, sum(num_patches)={int(num_patches.sum().item())}."
        )

    samples_per_mb = -(-batch_size // n_microbatches)
    cumsum_patches = torch.cumsum(num_patches.cpu(), dim=0)

    result: dict[str, list[torch.Tensor]] = {
        "pixel_values": [],
        "num_patches": [],
    }
    if patch_pixel_values is not None:
        result["patch_pixel_values"] = []
    if patch_newline_mask is not None:
        result["patch_newline_mask"] = []

    for mb_idx in range(n_microbatches):
        sample_start = mb_idx * samples_per_mb
        sample_end = min(sample_start + samples_per_mb, batch_size)
        patch_start = 0 if sample_start == 0 else int(cumsum_patches[sample_start - 1].item())
        patch_end = int(cumsum_patches[sample_end - 1].item()) if sample_end > 0 else patch_start
        pixel_start, pixel_end = (patch_start, patch_end) if flat_pixel_values else (sample_start, sample_end)
        result["pixel_values"].append(pixel_values[pixel_start:pixel_end])
        result["num_patches"].append(num_patches[sample_start:sample_end])
        if patch_pixel_values is not None:
            result["patch_pixel_values"].append(patch_pixel_values[patch_start:patch_end])
        if patch_newline_mask is not None:
            result["patch_newline_mask"].append(patch_newline_mask[patch_start:patch_end])

    return result


def _select_image_grid(
    image_grid_hws: torch.Tensor | None,
    image_grid_thw: torch.Tensor | None,
    image_sizes: torch.Tensor | None,
    image_position_ids: torch.Tensor | None,
) -> torch.Tensor | None:
    if image_grid_hws is not None:
        return image_grid_hws
    if image_grid_thw is not None:
        return image_grid_thw
    if image_sizes is not None:
        return image_sizes
    return image_position_ids


def prepare_vlm_media_for_pp(
    batch: MutableMapping[str, Any],
    *,
    batch_size: int,
    n_microbatches: int,
) -> MutableMapping[str, Any]:
    """Move VLM media tensors into pre-chunked PP media storage on the batch.

    This is intended to run from VLM collate/dataloader code when PP is enabled.
    The returned batch no longer carries raw media tensors that PyTorch PP would
    chunk by row incorrectly; instead it carries ``VLM_PP_MEDIA_KEY`` with
    per-microbatch media chunks.
    """
    if n_microbatches < 1:
        raise ValueError(f"n_microbatches must be >= 1, got {n_microbatches}")

    if not any(key in batch for key in _VLM_MEDIA_KEYS):
        return batch

    pixel_values = batch.pop("pixel_values", None)
    patch_pixel_values = batch.pop("patch_pixel_values", None)
    num_patches = batch.pop("num_patches", None)
    patch_newline_mask = batch.pop("patch_newline_mask", None)
    image_grid_hws = batch.pop("image_grid_hws", None)
    image_grid_thw = batch.pop("image_grid_thw", None)
    image_sizes = batch.pop("image_sizes", None)
    image_position_ids = batch.pop("image_position_ids", None)
    n_images_per_sample = batch.pop("n_images_per_sample", None)
    pixel_values_videos = batch.pop("pixel_values_videos", None)
    video_grid_thw = batch.pop("video_grid_thw", None)
    n_videos_per_sample = batch.pop("n_videos_per_sample", None)

    image_grid = _select_image_grid(image_grid_hws, image_grid_thw, image_sizes, image_position_ids)
    pp_media: dict[str, list[torch.Tensor]] = {}

    if pixel_values is not None and image_grid is None:
        step3_media = chunk_step3_media(
            pixel_values,
            batch_size=batch_size,
            n_microbatches=n_microbatches,
            num_patches=num_patches,
            patch_pixel_values=patch_pixel_values,
            patch_newline_mask=patch_newline_mask,
        )
        pp_media.update(step3_media)

    if pixel_values_videos is not None and video_grid_thw is None:
        raise ValueError("VLM PP media prep requires video_grid_thw with pixel_values_videos.")

    if pixel_values is not None and image_grid is not None:
        pixel_values_chunks, image_grid_chunks = chunk_vlm_media(
            pixel_values,
            image_grid,
            batch_size=batch_size,
            n_microbatches=n_microbatches,
            n_images_per_sample=n_images_per_sample,
        )
        pp_media["pixel_values"] = pixel_values_chunks
        pp_media["image_grid_hws"] = image_grid_chunks

    if pixel_values_videos is not None and video_grid_thw is not None:
        pixel_values_videos_chunks, video_grid_thw_chunks = chunk_vlm_media(
            pixel_values_videos,
            video_grid_thw,
            batch_size=batch_size,
            n_microbatches=n_microbatches,
            n_images_per_sample=n_videos_per_sample,
        )
        pp_media["pixel_values_videos"] = pixel_values_videos_chunks
        pp_media["video_grid_thw"] = video_grid_thw_chunks

    if pp_media:
        batch[VLM_PP_MEDIA_KEY] = pp_media

    return batch


def wrap_vlm_collate_for_pp(
    collate_fn: Callable[[Any], MutableMapping[str, Any]],
    *,
    n_microbatches: int,
) -> Callable[[Any], MutableMapping[str, Any]]:
    """Wrap a VLM collate function so it prepares media tensors for PP."""

    def wrapper(examples):
        batch = collate_fn(examples)
        if not isinstance(batch, MutableMapping):
            return batch
        if not any(key in batch for key in _VLM_MEDIA_KEYS):
            return batch
        if "input_ids" not in batch:
            raise ValueError("VLM PP media prep requires input_ids to infer the local batch size.")
        return prepare_vlm_media_for_pp(
            batch,
            batch_size=batch["input_ids"].shape[0],
            n_microbatches=n_microbatches,
        )

    return wrapper


@contextmanager
def stage_vlm_media_for_pp(pp: Any, model_parts: list[torch.nn.Module], batch: MutableMapping[str, Any]):
    """Attach dataloader-prepared VLM media chunks to PP stage 0 for one schedule call."""
    pp_media = batch.pop(VLM_PP_MEDIA_KEY, None)
    stage0_model = model_parts[0] if pp_media and getattr(pp.info, "has_first_stage", False) else None
    staged = False

    if stage0_model is not None:
        if "pixel_values" in pp_media:
            stage0_model._vlm_pixel_values_chunks = pp_media["pixel_values"]
            stage0_model._vlm_image_grid_hws_chunks = pp_media.get("image_grid_hws")
            stage0_model._vlm_num_patches_chunks = pp_media.get("num_patches")
            stage0_model._vlm_patch_pixel_values_chunks = pp_media.get("patch_pixel_values")
            stage0_model._vlm_patch_newline_mask_chunks = pp_media.get("patch_newline_mask")
            staged = True
        if "pixel_values_videos" in pp_media:
            stage0_model._vlm_pixel_values_videos_chunks = pp_media["pixel_values_videos"]
            stage0_model._vlm_video_grid_thw_chunks = pp_media.get("video_grid_thw")
            staged = True
        if staged:
            stage0_model._vlm_chunk_idx = 0

    try:
        yield
    finally:
        if staged and stage0_model is not None:
            stage0_model._vlm_pixel_values_chunks = None
            stage0_model._vlm_image_grid_hws_chunks = None
            stage0_model._vlm_num_patches_chunks = None
            stage0_model._vlm_patch_pixel_values_chunks = None
            stage0_model._vlm_patch_newline_mask_chunks = None
            stage0_model._vlm_pixel_values_videos_chunks = None
            stage0_model._vlm_video_grid_thw_chunks = None
            stage0_model._vlm_chunk_idx = None


__all__ = [
    "VLM_PP_MEDIA_KEY",
    "chunk_vlm_media",
    "prepare_vlm_media_for_pp",
    "stage_vlm_media_for_pp",
    "wrap_vlm_collate_for_pp",
]

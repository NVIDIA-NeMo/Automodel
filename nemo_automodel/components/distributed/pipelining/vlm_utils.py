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

from collections.abc import Iterator, MutableMapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

_VLM_MEDIA_KEYS = (
    "pixel_values",
    "image_grid_hws",
    "image_grid_thw",
    "image_sizes",
    "image_position_ids",
    "n_images_per_sample",
)


def chunk_vlm_media(
    pixel_values: torch.Tensor,
    image_grid: torch.Tensor,
    batch_size: int,
    n_microbatches: int,
    n_images_per_sample: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split VLM pixel values and image metadata into PP microbatch chunks.

    Handles four layouts:
    1. ``[N, C, H, W]`` with ``N == batch_size`` -- one full image per sample.
    2. ``[N, max_patches, D]`` with ``N == batch_size`` -- padded patches per image.
    3. Flat patches ``[total_patches, D]`` with per-sample image counts from
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
        samples_per_mb = batch_size // n_microbatches
        for mb_idx in range(n_microbatches):
            s_start = mb_idx * samples_per_mb
            s_end = min(s_start + samples_per_mb, batch_size)
            img_start = 0 if s_start == 0 else int(cumsum_images[s_start - 1].item())
            img_end = int(cumsum_images[s_end - 1].item()) if s_end > 0 else 0
            pixel_values_chunks.append(pixel_values[img_start:img_end])
            image_grid_chunks.append(image_grid[img_start:img_end])
    elif n_images_per_sample is not None:
        # General flat-patch layout: map samples -> images -> patches.
        patch_counts = image_grid.prod(dim=1)
        cumsum_patches = torch.cumsum(patch_counts, dim=0)
        cumsum_images = torch.cumsum(n_images_per_sample, dim=0)

        samples_per_mb = batch_size // n_microbatches
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

        images_per_mb = batch_size // n_microbatches
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


def _get_pp_n_microbatches(pp: Any) -> int:
    schedule = getattr(getattr(pp, "info", None), "schedule", None)
    n_microbatches = getattr(schedule, "_n_microbatches", None)
    if n_microbatches is None:
        n_microbatches = getattr(schedule, "n_microbatches", None)
    if n_microbatches is None:
        raise RuntimeError("Unable to determine PP schedule n_microbatches for VLM media chunking.")
    return int(n_microbatches)


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


@contextmanager
def stage_vlm_media_for_pp(
    pp: Any,
    model_parts: Sequence[nn.Module],
    batch: MutableMapping[str, Any],
    input_ids: torch.Tensor,
) -> Iterator[MutableMapping[str, Any]]:
    """Stage VLM media tensors for the first pipeline stage during a PP schedule call.

    PyTorch PP chunks normal tensor args by batch rows. VLM media tensors may be
    flat patch streams or per-image tensors, so Automodel splits them by sample
    ownership and stores per-microbatch chunks on the first stage model for the
    VLM-aware forward path to consume.
    """
    if not any(key in batch for key in _VLM_MEDIA_KEYS):
        yield batch
        return

    pixel_values = batch.pop("pixel_values", None)
    image_grid_hws = batch.pop("image_grid_hws", None)
    image_grid_thw = batch.pop("image_grid_thw", None)
    image_sizes = batch.pop("image_sizes", None)
    image_position_ids = batch.pop("image_position_ids", None)
    n_images_per_sample = batch.pop("n_images_per_sample", None)

    stage0_model: nn.Module | None = None
    staged = False
    image_grid = _select_image_grid(image_grid_hws, image_grid_thw, image_sizes, image_position_ids)

    if pixel_values is not None and image_grid is None:
        raise ValueError(
            "VLM PP staging requires media metadata with pixel_values. Expected one of "
            "image_grid_hws, image_grid_thw, image_sizes, or image_position_ids."
        )

    if getattr(pp.info, "has_first_stage", False) and pixel_values is not None and image_grid is not None:
        stage0_model = model_parts[0]
        pixel_values_chunks, image_grid_chunks = chunk_vlm_media(
            pixel_values,
            image_grid,
            batch_size=input_ids.shape[0],
            n_microbatches=_get_pp_n_microbatches(pp),
            n_images_per_sample=n_images_per_sample,
        )

        stage0_model._vlm_pixel_values_chunks = pixel_values_chunks
        stage0_model._vlm_image_grid_hws_chunks = image_grid_chunks
        stage0_model._vlm_chunk_idx = 0
        staged = True

    try:
        yield batch
    finally:
        if staged and stage0_model is not None:
            stage0_model._vlm_pixel_values_chunks = None
            stage0_model._vlm_image_grid_hws_chunks = None
            stage0_model._vlm_chunk_idx = None


__all__ = ["chunk_vlm_media", "stage_vlm_media_for_pp"]

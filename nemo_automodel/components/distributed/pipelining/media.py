# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Consume explicitly staged pipeline media in microbatch order."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VlmMediaChunk:
    """One microbatch's views of staged image and video tensors.

    Absent or empty staged tensors become None; every nonempty field aliases
    its staged tensor unchanged.

    Attributes:
        pixel_values: Image pixels in processor layout with images or patches
            on axis 0: complete images [images, channels, height, width],
            padded patches [images, max_patches, features], flat patch rows
            [patches, patch_features], or patch-major
            [patches, channels, patch_height, patch_width].
        image_grid: Per-image metadata with images on axis 0: patch grids
            [images, 2] (height, width) or [images, 3] (time, height, width),
            original sizes [images, 2], or per-patch positions
            [images, max_patches, 2].
        pixel_values_videos: Video pixels with patches on axis 0, e.g. flat
            patch rows [patches, patch_features] in video order.
        video_grid: Tensor of shape [videos, 3] storing each video's
            (time, height, width) patch grid.
        patch_pixel_values: Crop patches of shape [patches, channels, height,
            width], concatenated in image order.
        num_patches: Tensor of shape [batch] with each sample's crop-patch
            count (Step3 stages one image per sample); it indexes
            patch_pixel_values and patch_newline_mask.
        patch_newline_mask: Tensor of shape [patches] marking patch-row
            newline positions.
    """

    pixel_values: torch.Tensor | None
    image_grid: torch.Tensor | None
    pixel_values_videos: torch.Tensor | None
    video_grid: torch.Tensor | None
    patch_pixel_values: torch.Tensor | None = None
    num_patches: torch.Tensor | None = None
    patch_newline_mask: torch.Tensor | None = None


def next_vlm_media_chunk(module: torch.nn.Module) -> VlmMediaChunk | None:
    """Advance staged media once, including for a text-only microbatch.

    Args:
        module: Pipeline stage carrying image/video chunk lists. Pixel tensors
            retain their image/patch-major layouts; metadata is [media, coordinates].
            Optional patch pixels retain [patches, channels, height, width], patch
            counts [images], and newline masks [patches] for patch-based consumers.

    Returns:
        The next chunk with original nonempty tensor views, or None when no media
        is staged. Advances the shared cursor once without inspecting token values.

    Raises:
        ValueError: If a media or grid list is exhausted before the schedule.
    """
    images = getattr(module, "_vlm_pixel_values_chunks", None)
    videos = getattr(module, "_vlm_pixel_values_videos_chunks", None)
    if images is None and videos is None:
        return None
    index = getattr(module, "_vlm_chunk_idx", 0) or 0

    def take(chunks: list[torch.Tensor | None] | None, name: str) -> torch.Tensor | None:
        """Read one optional image/patch-major tensor without moving the cursor.

        Args:
            chunks: Optional per-microbatch tensors whose image/patch axis is
                first and whose trailing axes are unchanged; entries may be None.
            name: Media field name for an exhausted-list diagnostic.

        Returns:
            The next nonempty tensor, aliasing its original processor layout,
            or None for absent/empty media.
        """
        if chunks is None:
            return None
        if index < 0 or index >= len(chunks):
            raise ValueError(f"PP {name} chunks exhausted at microbatch {index}; staged {len(chunks)} chunks")
        value = chunks[index]
        return value if value is not None and value.shape[0] else None

    result = VlmMediaChunk(
        take(images, "image"),
        take(getattr(module, "_vlm_image_grid_hws_chunks", None), "image metadata"),
        take(videos, "video"),
        take(getattr(module, "_vlm_video_grid_thw_chunks", None), "video metadata"),
        take(getattr(module, "_vlm_patch_pixel_values_chunks", None), "patch pixels"),
        take(getattr(module, "_vlm_num_patches_chunks", None), "patch counts"),
        take(getattr(module, "_vlm_patch_newline_mask_chunks", None), "patch newline metadata"),
    )
    module._vlm_chunk_idx = index + 1
    return result

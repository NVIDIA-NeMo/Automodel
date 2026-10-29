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

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VlmMediaChunk:
    """One microbatch's views of staged image tensors.

    Pixels retain their processor layout, with images or patches on axis zero.
    Grids have shape [images, coordinates]. Empty media
    chunks become None; nonempty tensors alias the staged inputs unchanged.
    """

    pixel_values: torch.Tensor | None
    image_grid: torch.Tensor | None


def next_vlm_media_chunk(module: torch.nn.Module) -> VlmMediaChunk | None:
    """Advance staged media once, including for a text-only microbatch.

    Args:
        module: Pipeline stage carrying image chunk lists. Pixel tensors retain
            their image/patch-major layouts; metadata is [images, coordinates].

    Returns:
        The next chunk with original nonempty tensor views, or None when no media
        is staged. Advances the shared cursor once without inspecting token values.

    Raises:
        ValueError: If a media or grid list is exhausted before the schedule.
    """
    images = getattr(module, "_vlm_pixel_values_chunks", None)
    if images is None:
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
    )
    module._vlm_chunk_idx = index + 1
    return result


def pad_image_tensors(values: Sequence[torch.Tensor]) -> torch.Tensor:
    """Zero-pad processor images to a common spatial size without resizing.

    Args:
        values: Image tensors [images, channels, height, width], with matching
            channel counts. Original unpadded sizes travel in `image_sizes`.

    Returns:
        Concatenated images [total_images, channels, max_height, max_width].
        Pixels stay at their original top-left coordinates; inputs are unchanged.
    """
    if not values or any(value.ndim != 4 for value in values):
        raise ValueError("Image padding requires nonempty [images, channels, height, width] tensors")
    height = max(value.shape[-2] for value in values)
    width = max(value.shape[-1] for value in values)
    return torch.cat(
        [torch.nn.functional.pad(value, (0, width - value.shape[-1], 0, height - value.shape[-2])) for value in values],
        dim=0,
    )

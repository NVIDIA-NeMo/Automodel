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

"""VLM-specific Engine subclass.

Adds two VLM-only intercepts via the ``_pre_cp_hook`` and
``_pre_pp_schedule_hook`` subclass methods defined by :class:`Engine`:

1. **CP multimodal pre-embed** — when context-parallelism is active and the
   model exposes ``prepare_model_inputs_for_cp``, run a forward with
   ``_pre_embed_only=True`` over the multimodal inputs and replace those
   keys with the pre-embedded outputs. The rest of the pipeline (CP
   shaping, attention) then operates on already-embedded tokens.
2. **PP media-tensor chunking** — ``pixel_values`` and ``image_grid`` have
   non-standard structures that can't be naively chunked along dim 0 by the
   PP scheduler. Pre-chunk via :func:`chunk_vlm_media` and stash the chunks
   on the stage-0 model for its forward to consume per microbatch.

``chunk_vlm_media`` is colocated here (rather than imported from the recipe)
because it's only used by the VLM Engine and consumers that want to fork it.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from nemo_automodel.components.utils.model_utils import VLM_INPUT_KEYS
from nemo_automodel.engine import Engine

logger = logging.getLogger(__name__)


# ── PP media-tensor chunking ─────────────────────────────────────────


def chunk_vlm_media(
    pixel_values: torch.Tensor,
    image_grid: torch.Tensor,
    batch_size: int,
    n_microbatches: int,
    n_images_per_sample: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split VLM ``pixel_values`` and ``image_grid`` into PP microbatch chunks.

    Handles four layouts:
      1. ``[N, C, H, W]`` with ``N == batch_size`` — one full image per sample.
      2. ``[N, max_patches, D]`` with ``N == batch_size`` — Gemma4 style.
      3. Flat patches ``[total_patches, D]`` with per-sample image counts from
         ``n_images_per_sample`` (general case, works for packed sequences).
      4. Flat patches with ``n_images == batch_size`` — legacy 1-image-per-sample.
    """
    n_images = image_grid.shape[0]
    pixel_values_chunks: list[torch.Tensor] = []
    image_grid_chunks: list[torch.Tensor] = []

    if pixel_values.shape[0] == batch_size and pixel_values.dim() in (3, 4):
        # Layouts 1 and 2 — indexed by sample along dim 0.
        pixel_values_chunks = list(pixel_values.chunk(n_microbatches, dim=0))
        image_grid_chunks = list(image_grid.chunk(n_microbatches, dim=0))
    elif pixel_values.dim() == 3 and n_images_per_sample is not None:
        # Gemma4 multi-image: ``[N_total_images, max_patches, patch_dim]``.
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
        # General case: per-sample image counts; works for packed sequences.
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
        # Legacy: exactly 1 image per sample.
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
        pixel_values_chunks.append(pixel_values)
        image_grid_chunks.append(image_grid)
        for _ in range(n_microbatches - 1):
            pixel_values_chunks.append(pixel_values[:0])
            image_grid_chunks.append(image_grid[:0])
        logger.warning(
            "VLM chunking: n_images=%d != batch_size=%d, giving all images to first microbatch",
            n_images, batch_size,
        )

    return pixel_values_chunks, image_grid_chunks


# ── VLMEngine ────────────────────────────────────────────────────────


class VLMEngine(Engine):
    """Engine for vision-language models.

    All overrides live in two hook methods. The base
    :meth:`Engine.forward_backward` body is unchanged.
    """

    def _pre_cp_hook(self, mb: dict) -> dict:
        """Run multimodal pre-embed when CP is active and the model supports it."""
        if self.pp_enabled or self.mesh is None or self.mesh.device_mesh is None:
            return mb

        dm = self.mesh.device_mesh
        cp_active = "cp" in getattr(dm, "mesh_dim_names", ()) and dm["cp"].size() > 1
        if not cp_active:
            return mb

        model = self.parts[0]
        if not hasattr(model, "prepare_model_inputs_for_cp"):
            return mb

        mm_kwargs = {k: mb[k] for k in VLM_INPUT_KEYS if mb.get(k) is not None}
        if not mm_kwargs:
            return mb

        with torch.no_grad():
            prepared = model(_pre_embed_only=True, **mm_kwargs)

        for k in VLM_INPUT_KEYS:
            mb.pop(k, None)
        mb.update(prepared)
        return mb

    def _pre_pp_schedule_hook(self, mb: dict, *, pp: Any, input_ids: torch.Tensor) -> dict:
        """Pre-chunk ``pixel_values`` and ``image_grid`` along PP; stash on stage0."""
        if not pp.info.has_first_stage:
            return mb

        pixel_values = mb.pop("pixel_values", None)
        image_grid_hws = mb.pop("image_grid_hws", None)
        image_grid_thw = mb.pop("image_grid_thw", None)
        image_sizes = mb.pop("image_sizes", None)
        image_position_ids = mb.pop("image_position_ids", None)
        n_images_per_sample = mb.pop("n_images_per_sample", None)

        image_grid = image_grid_hws if image_grid_hws is not None else image_grid_thw
        if image_grid is None and image_sizes is not None:
            image_grid = image_sizes
        if image_grid is None and image_position_ids is not None:
            image_grid = image_position_ids

        if pixel_values is None or image_grid is None:
            # No media — restore the popped keys for downstream forward.
            for k, v in (
                ("pixel_values", pixel_values),
                ("image_grid_hws", image_grid_hws),
                ("image_grid_thw", image_grid_thw),
                ("image_sizes", image_sizes),
                ("image_position_ids", image_position_ids),
                ("n_images_per_sample", n_images_per_sample),
            ):
                if v is not None:
                    mb[k] = v
            return mb

        stage0_model = self.parts[0]
        n_microbatches = pp._info.schedule._n_microbatches
        batch_size = input_ids.shape[0]

        pixel_values_chunks, image_grid_chunks = chunk_vlm_media(
            pixel_values, image_grid, batch_size, n_microbatches,
            n_images_per_sample=n_images_per_sample,
        )

        # Stash chunks on the stage-0 model for its forward to consume.
        stage0_model._vlm_pixel_values_chunks = pixel_values_chunks
        stage0_model._vlm_image_grid_hws_chunks = image_grid_chunks
        stage0_model._vlm_chunk_idx = 0

        return mb


__all__ = ["VLMEngine", "chunk_vlm_media"]

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

from nemo_automodel.shared.pipeline import PP_MEDIA_INDEX_KEY

VLM_PP_MEDIA_KEY = "_vlm_pp_media_chunks"

#: Attribute on the stage-0 module holding ``{media name: [one tensor per microbatch]}``.
PP_MEDIA_CHUNKS_ATTR = "_pp_media_chunks"

_VLM_MEDIA_KEYS = (
    "pixel_values",
    "patch_pixel_values",
    "num_patches",
    "patch_newline_mask",
    "image_grid_hws",
    "image_grid_thw",
    "grid_thws",
    "image_flags",
    "image_sizes",
    "image_position_ids",
    "n_images_per_sample",
    "pixel_values_videos",
    "video_grid_thw",
    "second_per_grid_ts",
    "n_videos_per_sample",
)


def _microbatch_sample_bounds(batch_size: int, n_microbatches: int) -> list[tuple[int, int]]:
    """Return the ``[start, end)`` sample range owned by every PP microbatch.

    The boundaries must match how ``torch.distributed.pipelining`` actually
    splits the batch, because the pipeline schedule splits ``input_ids`` and the
    ``pp_media_index`` companion tensor itself. torch splits with
    ``torch.tensor_split``, which gives the first ``batch_size % n_microbatches``
    microbatches one extra sample rather than filling ceil-sized microbatches
    and leaving a short tail. Using ceil-sized bounds here would put samples
    from two different media chunks into one microbatch whenever the batch size
    is not divisible by the microbatch count, and the media lookup -- which
    reads the first sample's index -- would then silently pair those samples
    with another microbatch's images.

    Args:
        batch_size: Number of samples in the batch.
        n_microbatches: Number of PP microbatches the batch is split into.

    Returns:
        One ``(start, end)`` sample range per microbatch, in microbatch order.
    """
    base, remainder = divmod(batch_size, n_microbatches)
    bounds: list[tuple[int, int]] = []
    start = 0
    for mb_idx in range(n_microbatches):
        end = start + base + (1 if mb_idx < remainder else 0)
        bounds.append((start, end))
        start = end
    return bounds


def build_pp_media_index(batch_size: int, n_microbatches: int) -> torch.Tensor:
    """Build the per-sample microbatch index that addresses staged media chunks.

    Args:
        batch_size: Number of samples in the batch.
        n_microbatches: Number of PP microbatches the batch is split into.

    Returns:
        int64 CPU tensor of shape [batch] whose entry ``i`` is the microbatch
        index that owns sample ``i``, matching the chunk order produced by
        :func:`chunk_vlm_media` and :func:`chunk_patch_mapped_media`.
    """
    index = torch.zeros(batch_size, dtype=torch.int64)
    for mb_idx, (start, end) in enumerate(_microbatch_sample_bounds(batch_size, n_microbatches)):
        index[start:end] = mb_idx
    return index


def _chunk_rows_like(tensor: torch.Tensor, name: str, reference_chunks: list[torch.Tensor]) -> list[torch.Tensor]:
    """Split a row-aligned metadata tensor with the row boundaries of another chunk list.

    Args:
        tensor: Tensor whose axis 0 is aligned row-for-row with the concatenation
            of ``reference_chunks``, e.g. one row per image.
        name: Batch key of ``tensor``, used for error messages.
        reference_chunks: Already-chunked tensors defining the row boundaries.

    Returns:
        One slice of ``tensor`` per entry of ``reference_chunks``.

    Raises:
        ValueError: If ``tensor`` has a different number of rows than the
            concatenated ``reference_chunks``.
    """
    total_rows = sum(chunk.shape[0] for chunk in reference_chunks)
    if tensor.shape[0] != total_rows:
        raise ValueError(
            f"VLM PP chunking cannot align '{name}' with the chunked media: "
            f"{name}.shape={tuple(tensor.shape)} has {tensor.shape[0]} rows but the media chunks "
            f"cover {total_rows} rows. '{name}' must carry exactly one row per media entry."
        )
    chunks: list[torch.Tensor] = []
    start = 0
    for reference in reference_chunks:
        end = start + reference.shape[0]
        chunks.append(tensor[start:end])
        start = end
    return chunks


def chunk_vlm_media(
    pixel_values: torch.Tensor,
    image_grid: torch.Tensor,
    batch_size: int,
    n_microbatches: int,
    n_images_per_sample: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split VLM pixel values and media metadata into PP microbatch chunks.

    Handles four layouts:
    1. ``[batch, channels, height, width]`` -- one full image per sample.
    2. ``[batch, max_patches, dim]`` -- padded patches per image.
    3. Flat patches ``[total_patches, dim]`` with per-sample media counts from
       ``n_images_per_sample``.
    4. Flat patches with ``n_images == batch_size`` -- legacy one-image-per-sample.

    Args:
        pixel_values: Media tensor in one of the four layouts above; axis 0 is
            either the sample axis or the flattened patch axis.
        image_grid: Tensor of shape [n_media, grid_dims] holding one grid row per
            media entry, e.g. ``[t, h, w]`` or ``[h, w]``.
        batch_size: Number of samples in the batch.
        n_microbatches: Number of PP microbatches to split the batch into.
        n_images_per_sample: Optional tensor of shape [batch] whose entry ``i`` is
            the number of media entries belonging to sample ``i``.

    Returns:
        ``(pixel_values_chunks, image_grid_chunks)``: two lists of exactly
        ``n_microbatches`` tensors, sliced along axis 0 and ordered by microbatch.
        Microbatches whose samples carry no media get an empty (zero-row) chunk so
        that chunks stay addressable by microbatch index.

    Raises:
        ValueError: If ``pixel_values`` cannot be aligned with the batch.
    """
    n_images = image_grid.shape[0]
    pixel_values_chunks: list[torch.Tensor] = []
    image_grid_chunks: list[torch.Tensor] = []

    bounds = _microbatch_sample_bounds(batch_size, n_microbatches)

    if pixel_values.shape[0] == batch_size and pixel_values.dim() in (3, 4):
        # 4D full-image tensors and 3D padded-patch tensors are indexed by sample.
        for s_start, s_end in bounds:
            pixel_values_chunks.append(pixel_values[s_start:s_end])
            image_grid_chunks.append(image_grid[s_start:s_end])
    elif pixel_values.dim() == 3 and n_images_per_sample is not None:
        # Multi-image padded-patch layout: split by image counts per sample.
        cumsum_images = torch.cumsum(n_images_per_sample, dim=0)
        for s_start, s_end in bounds:
            img_start = 0 if s_start == 0 else int(cumsum_images[s_start - 1].item())
            img_end = int(cumsum_images[s_end - 1].item()) if s_end > 0 else 0
            pixel_values_chunks.append(pixel_values[img_start:img_end])
            image_grid_chunks.append(image_grid[img_start:img_end])
    elif n_images_per_sample is not None:
        # General flat-patch layout: map samples -> media entries -> patches.
        patch_counts = image_grid.prod(dim=1)
        cumsum_patches = torch.cumsum(patch_counts, dim=0)
        cumsum_images = torch.cumsum(n_images_per_sample, dim=0)

        for s_start, s_end in bounds:
            img_start = 0 if s_start == 0 else int(cumsum_images[s_start - 1].item())
            img_end = int(cumsum_images[s_end - 1].item()) if s_end > 0 else 0

            image_grid_chunks.append(image_grid[img_start:img_end])

            patch_start = 0 if img_start == 0 else int(cumsum_patches[img_start - 1].item())
            patch_end = int(cumsum_patches[img_end - 1].item()) if img_end > 0 else 0
            pixel_values_chunks.append(pixel_values[patch_start:patch_end])
    elif n_images == batch_size:
        # Legacy: exactly one image per sample.
        patch_counts = image_grid.prod(dim=1)
        cumsum = torch.cumsum(patch_counts, dim=0)

        for img_start, img_end in bounds:
            image_grid_chunks.append(image_grid[img_start:img_end])

            patch_start = 0 if img_start == 0 else int(cumsum[img_start - 1].item())
            patch_end = int(cumsum[img_end - 1].item()) if img_end > 0 else 0
            pixel_values_chunks.append(pixel_values[patch_start:patch_end])
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


def chunk_patch_mapped_media(
    pixel_values: torch.Tensor,
    *,
    batch_size: int,
    n_microbatches: int,
    num_patches: torch.Tensor | None = None,
    patch_pixel_values: torch.Tensor | None = None,
    patch_newline_mask: torch.Tensor | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Chunk media that carries per-sample patch counts instead of a grid.

    Selected on batch structure, not on model identity: it handles any batch
    whose ``pixel_values`` arrives without grid metadata and instead maps
    samples to media rows through ``num_patches``. Either one full image per
    sample plus a flat crop-patch tensor in ``patch_pixel_values``, or
    ``pixel_values`` itself as a flat patch tensor using the same mapping.
    Step3-style processors are the current producers of this layout.

    Args:
        pixel_values: Either ``[batch, ...]`` (one image tensor per sample) or a
            row-flattened ``[total_patches, ...]`` tensor whose patches are
            concatenated along axis 0 in sample order, where
            ``total_patches == sum(num_patches)``.
        batch_size: Number of samples in the batch.
        n_microbatches: Number of PP microbatches to split the batch into.
        num_patches: Tensor of shape ``[batch]``; entry ``i`` is the patch count
            for sample ``i``. Defaults to all-zeros when the processor emits no
            crop patches.
        patch_pixel_values: Optional row-flattened crop patches of shape
            ``[total_patches, ...]`` indexed by ``num_patches``.
        patch_newline_mask: Optional tensor of shape ``[total_patches]`` marking
            patch-row newline positions, indexed by ``num_patches``.

    Returns:
        dict[str, list[torch.Tensor]]: Per-microbatch slices keyed by
        ``pixel_values`` and ``num_patches`` (plus ``patch_pixel_values`` and
        ``patch_newline_mask`` when provided). ``pixel_values`` is sliced along
        the patch axis for the flat layout and along the sample axis otherwise.
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

    bounds = _microbatch_sample_bounds(batch_size, n_microbatches)
    cumsum_patches = torch.cumsum(num_patches.cpu(), dim=0)

    result: dict[str, list[torch.Tensor]] = {
        "pixel_values": [],
        "num_patches": [],
    }
    if patch_pixel_values is not None:
        result["patch_pixel_values"] = []
    if patch_newline_mask is not None:
        result["patch_newline_mask"] = []

    for sample_start, sample_end in bounds:
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
    Media tensors are indexed by media entry (image, video, patch), not by sample,
    so ``torch.distributed.pipelining`` would split them along the wrong axis. They
    are therefore removed from the batch and stored pre-chunked under
    ``VLM_PP_MEDIA_KEY``. The batch instead gains ``PP_MEDIA_INDEX_KEY``, an int64
    tensor of shape [batch] that torch splits in lockstep with ``input_ids``, so
    every forward can look up its own chunk regardless of execution order.

    Args:
        batch: Collated batch whose media tensors are moved into PP media storage.
            Mutated in place and also returned.
        batch_size: Number of samples in the batch.
        n_microbatches: Number of PP microbatches the batch is split into.

    Returns:
        The same mapping, without raw media tensors and with ``VLM_PP_MEDIA_KEY``
        plus ``PP_MEDIA_INDEX_KEY`` (shape [batch]) added when media was staged.

    Raises:
        ValueError: If ``n_microbatches`` is below 1, if videos are present without
            ``video_grid_thw``, or if a per-media metadata tensor cannot be aligned
            with the chunked media.
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
    grid_thws = batch.pop("grid_thws", None)
    image_flags = batch.pop("image_flags", None)
    image_sizes = batch.pop("image_sizes", None)
    image_position_ids = batch.pop("image_position_ids", None)
    n_images_per_sample = batch.pop("n_images_per_sample", None)
    pixel_values_videos = batch.pop("pixel_values_videos", None)
    video_grid_thw = batch.pop("video_grid_thw", None)
    second_per_grid_ts = batch.pop("second_per_grid_ts", None)
    n_videos_per_sample = batch.pop("n_videos_per_sample", None)

    image_grid = _select_image_grid(image_grid_hws, image_grid_thw, image_sizes, image_position_ids)
    pp_media: dict[str, list[torch.Tensor]] = {}

    if pixel_values is not None and image_grid is None:
        patch_mapped_media = chunk_patch_mapped_media(
            pixel_values,
            batch_size=batch_size,
            n_microbatches=n_microbatches,
            num_patches=num_patches,
            patch_pixel_values=patch_pixel_values,
            patch_newline_mask=patch_newline_mask,
        )
        pp_media.update(patch_mapped_media)

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

    # Metadata carrying one row per media entry rather than one row per sample.
    # Left in the batch it would be row-split along the media axis by torch.
    per_image_reference = pp_media.get("image_grid_hws", pp_media.get("pixel_values"))
    for name, tensor in (("grid_thws", grid_thws), ("image_flags", image_flags)):
        if tensor is None:
            continue
        if per_image_reference is None:
            raise ValueError(f"VLM PP media prep found '{name}' but no image tensors to align it with.")
        pp_media[name] = _chunk_rows_like(tensor, name, per_image_reference)
    if second_per_grid_ts is not None:
        video_reference = pp_media.get("video_grid_thw")
        if video_reference is None:
            raise ValueError("VLM PP media prep found 'second_per_grid_ts' but no video tensors to align it with.")
        pp_media["second_per_grid_ts"] = _chunk_rows_like(second_per_grid_ts, "second_per_grid_ts", video_reference)

    if pp_media:
        batch[VLM_PP_MEDIA_KEY] = pp_media
        batch[PP_MEDIA_INDEX_KEY] = build_pp_media_index(batch_size, n_microbatches)

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
    """Attach dataloader-prepared VLM media chunks to PP stage 0 for one schedule call.

    The chunks are exposed as a single ``_pp_media_chunks`` mapping of media name to
    one tensor per microbatch. Stage-0 forwards select their own entry with
    ``nemo_automodel.shared.pipeline.pp_media_chunk`` using the ``pp_media_index``
    kwarg the batch carries, so no cursor or other mutable state is involved and a
    probe forward run for runtime shape inference consumes nothing.

    Args:
        pp: Built ``AutoPipeline`` whose ``info.has_first_stage`` decides whether
            this rank owns stage 0.
        model_parts: Local pipeline stage modules; ``model_parts[0]`` is stage 0.
        batch: Batch carrying ``VLM_PP_MEDIA_KEY``; the key is always removed so
            raw media never reaches ``schedule.step()``.

    Yields:
        None, for the duration of one schedule call.
    """
    pp_media = batch.pop(VLM_PP_MEDIA_KEY, None)
    stage0_model = model_parts[0] if pp_media and getattr(pp.info, "has_first_stage", False) else None

    if stage0_model is None:
        yield
        return

    setattr(stage0_model, PP_MEDIA_CHUNKS_ATTR, dict(pp_media))
    try:
        yield
    finally:
        setattr(stage0_model, PP_MEDIA_CHUNKS_ATTR, None)


__all__ = [
    "PP_MEDIA_CHUNKS_ATTR",
    "VLM_PP_MEDIA_KEY",
    "build_pp_media_index",
    "chunk_patch_mapped_media",
    "chunk_vlm_media",
    "prepare_vlm_media_for_pp",
    "stage_vlm_media_for_pp",
    "wrap_vlm_collate_for_pp",
]

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

"""
Collate functions and dataloader builders for multiresolution diffusion training.

Supports both image and video pipelines via the FlowMatchingPipeline
expected batch format.
"""

import functools
import logging
from typing import Callable, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from .sampler import SequentialBucketSampler
from .text_to_image_dataset import TextToImageDataset
from .text_to_video_dataset import TextToVideoDataset, collate_optional_video_fields

logger = logging.getLogger(__name__)


def collate_fn_production(batch: List[Dict]) -> Dict:
    """Production collate function with verification."""
    # Verify all samples have same resolution
    resolutions = [tuple(item["crop_resolution"].tolist()) for item in batch]
    assert len(set(resolutions)) == 1, f"Mixed resolutions in batch: {set(resolutions)}"

    # Stack tensors
    latents = torch.stack([item["latent"] for item in batch])
    crop_resolutions = torch.stack([item["crop_resolution"] for item in batch])
    original_resolutions = torch.stack([item["original_resolution"] for item in batch])
    crop_offsets = torch.stack([item["crop_offset"] for item in batch])

    # Collect metadata
    prompts = [item["prompt"] for item in batch]
    image_paths = [item["image_path"] for item in batch]
    bucket_ids = [item["bucket_id"] for item in batch]
    aspect_ratios = [item["aspect_ratio"] for item in batch]

    output = {
        "latent": latents,
        "crop_resolution": crop_resolutions,
        "original_resolution": original_resolutions,
        "crop_offset": crop_offsets,
        "prompt": prompts,
        "image_path": image_paths,
        "bucket_id": bucket_ids,
        "aspect_ratio": aspect_ratios,
    }

    # Handle text encodings
    if "clip_hidden" in batch[0]:
        output["clip_hidden"] = torch.stack([item["clip_hidden"] for item in batch])
        output["pooled_prompt_embeds"] = torch.stack([item["pooled_prompt_embeds"] for item in batch])
        output["prompt_embeds"] = torch.stack([item["prompt_embeds"] for item in batch])
    else:
        output["clip_tokens"] = torch.stack([item["clip_tokens"] for item in batch])
        output["t5_tokens"] = torch.stack([item["t5_tokens"] for item in batch])

    return output


def collate_fn_text_to_image(batch: List[Dict]) -> Dict:
    """
    Text-to-image collate function that transforms multiresolution batch output
    to match FlowMatchingPipeline expected format.

    Args:
        batch: List of samples from TextToImageDataset

    Returns:
        Dict compatible with FlowMatchingPipeline.step()
    """
    # First, use the production collate to stack tensors
    production_batch = collate_fn_production(batch)

    # Keep latent as 4D [B, C, H, W] for image (not video)
    latent = production_batch["latent"]

    # Use "image_latents" key for 4D tensors
    image_batch = {
        "image_latents": latent,
        "data_type": "image",
        "metadata": {
            "prompts": production_batch.get("prompt", []),
            "image_paths": production_batch.get("image_path", []),
            "bucket_ids": production_batch.get("bucket_id", []),
            "aspect_ratios": production_batch.get("aspect_ratio", []),
            "crop_resolution": production_batch.get("crop_resolution"),
            "original_resolution": production_batch.get("original_resolution"),
            "crop_offset": production_batch.get("crop_offset"),
        },
    }

    # Handle text embeddings (pre-encoded vs tokenized)
    if "prompt_embeds" in production_batch:
        # Pre-encoded text embeddings
        image_batch["text_embeddings"] = production_batch["prompt_embeds"]
        image_batch["pooled_prompt_embeds"] = production_batch["pooled_prompt_embeds"]
        # Also include CLIP hidden for models that need it
        if "clip_hidden" in production_batch:
            image_batch["clip_hidden"] = production_batch["clip_hidden"]
    else:
        # Tokenized - need to encode during training (not supported yet)
        image_batch["t5_tokens"] = production_batch["t5_tokens"]
        image_batch["clip_tokens"] = production_batch["clip_tokens"]
        raise NotImplementedError(
            "On-the-fly text encoding not yet supported. Please use pre-encoded text embeddings in your dataset."
        )

    return image_batch


def _build_multiresolution_dataloader_core(
    *,
    dataset,
    collate_fn: Callable,
    batch_size: int,
    dp_rank: int,
    dp_world_size: int,
    base_resolution: Tuple[int, int] = (512, 512),
    drop_last: bool = True,
    shuffle: bool = True,
    dynamic_batch_size: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, SequentialBucketSampler]:
    """Internal helper: create sampler + DataLoader from dataset and collate fn."""
    sampler = SequentialBucketSampler(
        dataset,
        base_batch_size=batch_size,
        base_resolution=base_resolution,
        drop_last=drop_last,
        shuffle_buckets=shuffle,
        shuffle_within_bucket=shuffle,
        dynamic_batch_size=dynamic_batch_size,
        num_replicas=dp_world_size,
        rank=dp_rank,
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    return dataloader, sampler


def build_text_to_image_multiresolution_dataloader(
    *,
    # TextToImageDataset parameters
    cache_dir: str,
    train_text_encoder: bool = False,
    # Dataloader parameters
    batch_size: int = 1,
    dp_rank: int = 0,
    dp_world_size: int = 1,
    base_resolution: Tuple[int, int] = (256, 256),
    drop_last: bool = True,
    shuffle: bool = True,
    dynamic_batch_size: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, SequentialBucketSampler]:
    """
    Build a text-to-image multiresolution dataloader for TrainDiffusionRecipe.

    This wraps the existing TextToImageDataset and SequentialBucketSampler
    with a text-to-image collate function.

    Args:
        cache_dir: Directory containing preprocessed cache (metadata.json, shards, and resolution subdirs)
        train_text_encoder: If True, returns tokens instead of embeddings
        batch_size: Batch size per GPU
        dp_rank: Data parallel rank
        dp_world_size: Data parallel world size
        base_resolution: Base resolution for dynamic batch sizing
        drop_last: Drop incomplete batches
        shuffle: Shuffle data
        dynamic_batch_size: Scale batch size by resolution
        num_workers: DataLoader workers
        pin_memory: Pin memory for GPU transfer
        prefetch_factor: Prefetch batches per worker

    Returns:
        Tuple of (DataLoader, SequentialBucketSampler)
    """
    logger.info("Building text-to-image multiresolution dataloader:")
    logger.info(f"  cache_dir: {cache_dir}")
    logger.info(f"  train_text_encoder: {train_text_encoder}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  dp_rank: {dp_rank}, dp_world_size: {dp_world_size}")

    dataset = TextToImageDataset(
        cache_dir=cache_dir,
        train_text_encoder=train_text_encoder,
    )

    dataloader, sampler = _build_multiresolution_dataloader_core(
        dataset=dataset,
        collate_fn=collate_fn_text_to_image,
        batch_size=batch_size,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        base_resolution=base_resolution,
        drop_last=drop_last,
        shuffle=shuffle,
        dynamic_batch_size=dynamic_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    logger.info(f"  Dataset size: {len(dataset)}")
    logger.info(f"  Batches per epoch: {len(sampler)}")

    return dataloader, sampler


def collate_fn_video(batch: List[Dict], model_type: str = "wan") -> Dict:
    """
    Video-compatible collate function for multiresolution video training.

    Concatenates video_latents (5D) and text_embeddings (3D) along the batch dim,
    matching the format expected by FlowMatchingPipeline with SimpleAdapter.

    Args:
        batch: List of samples from TextToVideoDataset
        model_type: Model type for model-specific field handling

    Returns:
        Dict compatible with FlowMatchingPipeline.step()
    """
    # Verify all samples have the same bucket resolution
    resolutions = [tuple(item["bucket_resolution"].tolist()) for item in batch]
    assert len(set(resolutions)) == 1, f"Mixed resolutions in batch: {set(resolutions)}"

    video_latents = torch.cat([item["video_latents"] for item in batch], dim=0)
    text_embeddings = torch.cat([item["text_embeddings"] for item in batch], dim=0)

    result = {
        "video_latents": video_latents,
        "text_embeddings": text_embeddings,
        "data_type": "video",
    }

    # Collate model-specific optional fields
    collate_optional_video_fields(batch, result)

    return result


def build_video_multiresolution_dataloader(
    *,
    cache_dir: str,
    model_type: str = "wan",
    device: str = "cpu",
    batch_size: int = 1,
    dp_rank: int = 0,
    dp_world_size: int = 1,
    base_resolution: Tuple[int, int] = (512, 512),
    drop_last: bool = True,
    shuffle: bool = True,
    dynamic_batch_size: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
) -> Tuple[DataLoader, SequentialBucketSampler]:
    """
    Build a multiresolution video dataloader for TrainDiffusionRecipe.

    Uses TextToVideoDataset with SequentialBucketSampler for bucket-based
    multiresolution video training (e.g. Wan, Hunyuan).

    Args:
        cache_dir: Directory containing preprocessed cache (metadata.json + shards + WxH/*.meta)
        model_type: Model type ("wan", "hunyuan", etc.)
        device: Device to load tensors to
        batch_size: Batch size per GPU
        dp_rank: Data parallel rank
        dp_world_size: Data parallel world size
        base_resolution: Base resolution for dynamic batch sizing
        drop_last: Drop incomplete batches
        shuffle: Shuffle data
        dynamic_batch_size: Scale batch size by resolution
        num_workers: DataLoader workers
        pin_memory: Pin memory for GPU transfer
        prefetch_factor: Prefetch batches per worker

    Returns:
        Tuple of (DataLoader, SequentialBucketSampler)
    """
    logger.info("Building video multiresolution dataloader:")
    logger.info(f"  cache_dir: {cache_dir}")
    logger.info(f"  model_type: {model_type}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  dp_rank: {dp_rank}, dp_world_size: {dp_world_size}")

    dataset = TextToVideoDataset(
        cache_dir=cache_dir,
        model_type=model_type,
        device=device,
    )

    collate = functools.partial(collate_fn_video, model_type=model_type)

    dataloader, sampler = _build_multiresolution_dataloader_core(
        dataset=dataset,
        collate_fn=collate,
        batch_size=batch_size,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        base_resolution=base_resolution,
        drop_last=drop_last,
        shuffle=shuffle,
        dynamic_batch_size=dynamic_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    logger.info(f"  Dataset size: {len(dataset)}")
    logger.info(f"  Batches per epoch: {len(sampler)}")

    return dataloader, sampler

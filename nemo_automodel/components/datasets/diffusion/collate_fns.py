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
Flux-compatible collate function that wraps the multiresolution dataloader output
to match the FlowMatchingPipeline expected batch format.
"""

import logging
from typing import Dict, List, Tuple

from torch.utils.data import DataLoader

from .sampler import (
    SequentialBucketSampler,
    collate_fn_production,
)
from .text_to_image_dataset import TextToImageDataset

logger = logging.getLogger(__name__)


def collate_fn_flux(batch: List[Dict]) -> Dict:
    """
    Flux-compatible collate function that transforms multiresolution batch output
    to match FlowMatchingPipeline expected format.

    Args:
        batch: List of samples from TextToImageDataset

    Returns:
        Dict compatible with FlowMatchingPipeline.step()
    """
    # First, use the production collate to stack tensors
    production_batch = collate_fn_production(batch)

    # Keep latent as 4D [B, C, H, W] for Flux (image model, not video)
    latent = production_batch["latent"]

    # Use "image_latents" key for 4D tensors (FluxAdapter expects 4D)
    flux_batch = {
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
        flux_batch["text_embeddings"] = production_batch["prompt_embeds"]
        flux_batch["pooled_prompt_embeds"] = production_batch["pooled_prompt_embeds"]
        # Also include CLIP hidden for models that need it
        if "clip_hidden" in production_batch:
            flux_batch["clip_hidden"] = production_batch["clip_hidden"]
    else:
        # Tokenized - need to encode during training (not supported yet)
        flux_batch["t5_tokens"] = production_batch["t5_tokens"]
        flux_batch["clip_tokens"] = production_batch["clip_tokens"]
        raise NotImplementedError(
            "On-the-fly text encoding not yet supported. Please use pre-encoded text embeddings in your dataset."
        )

    return flux_batch


def build_flux_multiresolution_dataloader(
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
    Build a Flux-compatible multiresolution dataloader for TrainDiffusionRecipe.

    This wraps the existing TextToImageDataset and SequentialBucketSampler
    with a Flux-compatible collate function.

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
    logger.info("Building Flux multiresolution dataloader:")
    logger.info(f"  cache_dir: {cache_dir}")
    logger.info(f"  train_text_encoder: {train_text_encoder}")
    logger.info(f"  batch_size: {batch_size}")
    logger.info(f"  dp_rank: {dp_rank}, dp_world_size: {dp_world_size}")

    # Create dataset
    dataset = TextToImageDataset(
        cache_dir=cache_dir,
        train_text_encoder=train_text_encoder,
    )

    # Create sampler
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

    # Create dataloader with Flux-compatible collate
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_fn_flux,  # Use Flux-compatible collate
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    logger.info(f"  Dataset size: {len(dataset)}")
    logger.info(f"  Batches per epoch: {len(sampler)}")

    return dataloader, sampler

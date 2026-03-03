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

import logging
import math
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

from .text_to_image_dataset import TextToImageDataset

logger = logging.getLogger(__name__)


class SequentialBucketSampler(Sampler[List[int]]):
    """
    Production-grade Sampler that:
    1. Supports Distributed Data Parallel (DDP) - splits data across GPUs
    2. Deterministic shuffling via torch.Generator (resumable training)
    3. Lazy batch generation (saves RAM compared to pre-computing all batches)
    4. Guarantees equal batch counts across all ranks (prevents DDP deadlocks)

    - Processes all images in bucket A before moving to bucket B
    - Shuffles samples within each bucket (deterministically)
    - Drops incomplete batches at end of each bucket
    - Uses dynamic batch sizes based on resolution
    """

    def __init__(
        self,
        dataset: TextToImageDataset,
        base_batch_size: int = 32,
        base_resolution: Tuple[int, int] = (512, 512),
        drop_last: bool = True,
        shuffle_buckets: bool = True,
        shuffle_within_bucket: bool = True,
        dynamic_batch_size: bool = False,
        seed: int = 42,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        """
        Args:
            dataset: TextToImageDataset
            base_batch_size: Batch size (fixed if dynamic_batch_size=False,
                            or base for scaling if dynamic_batch_size=True)
            base_resolution: Reference resolution for batch size scaling
                            (only used if dynamic_batch_size=True)
            drop_last: Drop incomplete batches
            shuffle_buckets: Shuffle bucket order
            shuffle_within_bucket: Shuffle samples within each bucket
            dynamic_batch_size: If True, scale batch size based on resolution.
                               If False (default), use base_batch_size for all buckets.
            seed: Random seed for deterministic shuffling (resumable training)
            num_replicas: Number of distributed processes (auto-detected if None)
            rank: Rank of current process (auto-detected if None)
        """
        self.dataset = dataset
        self.base_batch_size = base_batch_size
        self.base_resolution = base_resolution
        self.drop_last = drop_last
        self.shuffle_buckets = shuffle_buckets
        self.shuffle_within_bucket = shuffle_within_bucket
        self.dynamic_batch_size = dynamic_batch_size
        self.seed = seed
        self.epoch = 0

        # Handle Distributed Training (DDP)
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        self.num_replicas = num_replicas
        self.rank = rank

        self.bucket_keys = dataset.sorted_bucket_keys
        self.bucket_groups = dataset.bucket_groups
        self.calculator = dataset.calculator

        # Pre-calculate total batches (same for all ranks)
        self._total_batches = self._calculate_total_batches()

        logger.info("\nSequentialBucketSampler created:")
        logger.info(f"  Total batches per rank: {self._total_batches}")
        logger.info(f"  Dynamic batch size: {dynamic_batch_size}")
        logger.info(
            f"  Base batch size: {base_batch_size}" + (f" @ {base_resolution}" if dynamic_batch_size else " (fixed)")
        )

    def _get_batch_size(self, resolution: Tuple[int, int]) -> int:
        """Get batch size for resolution (dynamic or fixed based on setting)."""
        if not self.dynamic_batch_size:
            return self.base_batch_size

        return self.calculator.get_dynamic_batch_size(
            resolution,
            self.base_batch_size,
            self.base_resolution,
        )

    def _calculate_total_batches(self) -> int:
        """
        Calculate total batches ensuring ALL ranks get the same count.
        We pad each bucket to be divisible by (num_replicas * batch_size).
        """
        count = 0
        for bucket_key in self.bucket_keys:
            total_indices = len(self.bucket_groups[bucket_key]["indices"])
            batch_size = self._get_batch_size(self.bucket_groups[bucket_key]["resolution"])

            # Pad to make divisible by num_replicas first
            padded_total = math.ceil(total_indices / self.num_replicas) * self.num_replicas
            per_rank_indices = padded_total // self.num_replicas

            if self.drop_last:
                count += per_rank_indices // batch_size
            else:
                count += (per_rank_indices + batch_size - 1) // batch_size

        return count

    def set_epoch(self, epoch: int):
        """Crucial for reproducibility and different shuffles per epoch."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[List[int]]:
        # Deterministic generator - SAME seed across all ranks
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # 1. Bucket Order Shuffling (deterministic, same across all ranks)
        current_bucket_keys = list(self.bucket_keys)
        if self.shuffle_buckets:
            perm = torch.randperm(len(current_bucket_keys), generator=g).tolist()
            current_bucket_keys = [current_bucket_keys[i] for i in perm]

        # 2. Iterate Buckets
        for key in current_bucket_keys:
            bucket = self.bucket_groups[key]
            indices = bucket["indices"].copy()
            resolution = bucket["resolution"]
            batch_size = self._get_batch_size(resolution)

            # 3. Deterministic Shuffle within bucket (same across all ranks)
            if self.shuffle_within_bucket:
                rand_indices = torch.randperm(len(indices), generator=g).tolist()
                indices = [indices[i] for i in rand_indices]

            # 4. Pad indices to ensure equal distribution across ranks
            total_size = math.ceil(len(indices) / self.num_replicas) * self.num_replicas
            padding_size = total_size - len(indices)
            if padding_size > 0:
                # Pad by repeating indices from the beginning
                indices = indices + indices[:padding_size]

            # 5. DDP Splitting: Subsample indices for this rank
            indices = indices[self.rank :: self.num_replicas]

            # 6. Yield Batches (Lazy Evaluation)
            for i in range(0, len(indices), batch_size):
                batch = indices[i : i + batch_size]

                if self.drop_last and len(batch) < batch_size:
                    continue

                if not batch:
                    continue

                yield batch

    def __len__(self) -> int:
        return self._total_batches

    def get_batch_info(self, batch_idx: int) -> Dict:
        """Get information about a specific batch.

        Note: With lazy evaluation, we don't pre-compute batches,
        so this returns bucket-level info for the estimated batch.
        """
        # Estimate which bucket this batch belongs to
        running_count = 0
        for bucket_key in self.bucket_keys:
            bucket = self.bucket_groups[bucket_key]
            total_indices = len(bucket["indices"])
            batch_size = self._get_batch_size(bucket["resolution"])

            padded_total = math.ceil(total_indices / self.num_replicas) * self.num_replicas
            per_rank_indices = padded_total // self.num_replicas

            if self.drop_last:
                num_batches = per_rank_indices // batch_size
            else:
                num_batches = (per_rank_indices + batch_size - 1) // batch_size

            if batch_idx < running_count + num_batches:
                return {
                    "bucket_key": bucket_key,
                    "resolution": bucket["resolution"],
                    "batch_size": batch_size,
                    "aspect_name": bucket["aspect_name"],
                }
            running_count += num_batches

        return {}


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


def build_multiresolution_dataloader(
    *,
    dataset: TextToImageDataset,
    base_batch_size: int,
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
    """
    Build production dataloader with sequential bucket iteration and distributed training support.

    Args:
        dataset: TextToImageDataset instance
        base_batch_size: Batch size (fixed, or base for scaling if dynamic_batch_size=True)
        dp_rank: Rank of current process in data parallel group
        dp_world_size: Total number of processes in data parallel group
        base_resolution: Reference resolution (only used if dynamic_batch_size=True)
        drop_last: Drop incomplete batches
        shuffle: Shuffle bucket order and samples within buckets each epoch
        dynamic_batch_size: If True, scale batch size based on resolution.
                           If False (default), use base_batch_size for all buckets.
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        prefetch_factor: How many batches to prefetch per worker

    Returns:
        Tuple of (DataLoader, SequentialBucketSampler) for production training
    """
    sampler = SequentialBucketSampler(
        dataset,
        base_batch_size=base_batch_size,
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
        collate_fn=collate_fn_production,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
    )

    return dataloader, sampler

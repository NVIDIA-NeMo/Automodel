# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for multiresolution dataloader components.

This module contains both CPU and GPU tests for:
- SequentialBucketSampler
- collate_fn_production
- build_multiresolution_dataloader

GPU tests are skipped when CUDA is not available.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest
import torch

from nemo_automodel.components.datasets.diffusion.sampler import (
    SequentialBucketSampler,
    build_multiresolution_dataloader,
    collate_fn_production,
)
from nemo_automodel.components.datasets.diffusion.text_to_image_dataset import (
    TextToImageDataset,
)


# ============================================================================
# Fixtures and Helpers
# ============================================================================


class MockCacheBuilder:
    """Helper class to create mock cache directories for testing."""

    def __init__(self, cache_dir: Path, num_samples: int = 10):
        self.cache_dir = cache_dir
        self.num_samples = num_samples
        self.metadata = []

    def create_sample(
        self,
        idx: int,
        crop_resolution: tuple = (512, 512),
        original_resolution: tuple = (1024, 768),
        aspect_ratio: float = 1.0,
    ) -> Dict:
        """Create a single sample cache file and metadata entry."""
        cache_file = self.cache_dir / f"sample_{idx:04d}.pt"

        # Create mock latent and text embeddings
        data = {
            "latent": torch.randn(16, crop_resolution[1] // 8, crop_resolution[0] // 8),
            "crop_offset": [0, 0],
            "prompt": f"Test prompt {idx}",
            "image_path": f"/fake/path/image_{idx}.jpg",
            "clip_hidden": torch.randn(1, 77, 768),
            "pooled_prompt_embeds": torch.randn(1, 768),
            "prompt_embeds": torch.randn(1, 256, 4096),
            "clip_tokens": torch.randint(0, 49408, (1, 77)),
            "t5_tokens": torch.randint(0, 32128, (1, 256)),
        }

        torch.save(data, cache_file)

        metadata_entry = {
            "cache_file": str(cache_file),
            "crop_resolution": list(crop_resolution),
            "original_resolution": list(original_resolution),
            "aspect_ratio": aspect_ratio,
            "bucket_id": idx % 5,
        }

        return metadata_entry

    def build_cache(
        self,
        resolutions: List[tuple] = None,
        aspect_ratios: List[float] = None,
    ):
        """Build the complete mock cache with metadata."""
        if resolutions is None:
            resolutions = [(512, 512)] * self.num_samples
        if aspect_ratios is None:
            aspect_ratios = [1.0] * self.num_samples

        self.metadata = []
        for idx in range(self.num_samples):
            res = resolutions[idx % len(resolutions)]
            ar = aspect_ratios[idx % len(aspect_ratios)]
            entry = self.create_sample(
                idx,
                crop_resolution=res,
                aspect_ratio=ar,
            )
            self.metadata.append(entry)

        # Create shard file
        shard_file = self.cache_dir / "metadata_shard_0000.json"
        with open(shard_file, "w") as f:
            json.dump(self.metadata, f)

        # Create main metadata file
        metadata_file = self.cache_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump({"shards": ["metadata_shard_0000.json"]}, f)

        return self.metadata


@pytest.fixture
def simple_dataset():
    """Create a simple dataset with uniform resolution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        builder = MockCacheBuilder(cache_dir, num_samples=20)
        builder.build_cache()
        dataset = TextToImageDataset(str(cache_dir))
        yield dataset


@pytest.fixture
def multi_resolution_dataset():
    """Create a dataset with multiple resolutions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        builder = MockCacheBuilder(cache_dir, num_samples=100)
        resolutions = [
            (512, 512),  # Square
            (512, 768),  # Portrait
            (768, 512),  # Landscape
        ]
        aspect_ratios = [1.0, 0.67, 1.5]
        builder.build_cache(resolutions=resolutions, aspect_ratios=aspect_ratios)
        dataset = TextToImageDataset(str(cache_dir))
        yield dataset


@pytest.fixture
def large_dataset():
    """Create a larger dataset for stress testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        builder = MockCacheBuilder(cache_dir, num_samples=500)
        resolutions = [
            (256, 256),
            (512, 512),
            (512, 768),
            (768, 512),
            (1024, 1024),
        ]
        aspect_ratios = [1.0, 1.0, 0.67, 1.5, 1.0]
        builder.build_cache(resolutions=resolutions, aspect_ratios=aspect_ratios)
        dataset = TextToImageDataset(str(cache_dir))
        yield dataset


# ============================================================================
# CPU Tests - SequentialBucketSampler
# ============================================================================


class TestSequentialBucketSamplerCPU:
    """CPU tests for SequentialBucketSampler."""

    def test_sampler_init_basic(self, simple_dataset):
        """Test basic sampler initialization."""
        sampler = SequentialBucketSampler(
            simple_dataset,
            base_batch_size=4,
            num_replicas=1,
            rank=0,
        )

        assert sampler.base_batch_size == 4
        assert sampler.num_replicas == 1
        assert sampler.rank == 0
        assert sampler.epoch == 0

    def test_sampler_len(self, simple_dataset):
        """Test sampler __len__ returns correct batch count."""
        sampler = SequentialBucketSampler(
            simple_dataset,
            base_batch_size=4,
            num_replicas=1,
            rank=0,
        )

        length = len(sampler)
        assert length > 0
        assert isinstance(length, int)

    def test_sampler_iter_yields_batches(self, simple_dataset):
        """Test sampler iteration yields batches of indices."""
        sampler = SequentialBucketSampler(
            simple_dataset,
            base_batch_size=4,
            num_replicas=1,
            rank=0,
        )

        batches = list(sampler)

        assert len(batches) > 0
        for batch in batches:
            assert isinstance(batch, list)
            assert all(isinstance(idx, int) for idx in batch)

    def test_sampler_batch_size_respected(self, simple_dataset):
        """Test batches have correct size (except possibly last)."""
        batch_size = 4
        sampler = SequentialBucketSampler(
            simple_dataset,
            base_batch_size=batch_size,
            num_replicas=1,
            rank=0,
            drop_last=True,
        )

        batches = list(sampler)

        for batch in batches:
            assert len(batch) == batch_size

    def test_sampler_drop_last_false(self, simple_dataset):
        """Test sampler with drop_last=False includes all samples."""
        sampler = SequentialBucketSampler(
            simple_dataset,
            base_batch_size=4,
            num_replicas=1,
            rank=0,
            drop_last=False,
        )

        batches = list(sampler)
        all_indices = [idx for batch in batches for idx in batch]

        # Should include indices for all or most samples
        assert len(all_indices) >= len(simple_dataset) - 1

    def test_sampler_set_epoch(self, simple_dataset):
        """Test set_epoch changes sampler state."""
        sampler = SequentialBucketSampler(
            simple_dataset,
            base_batch_size=4,
            num_replicas=1,
            rank=0,
        )

        assert sampler.epoch == 0
        sampler.set_epoch(5)
        assert sampler.epoch == 5

    def test_sampler_deterministic_shuffling(self, simple_dataset):
        """Test same seed produces same batch order."""
        sampler1 = SequentialBucketSampler(
            simple_dataset,
            base_batch_size=4,
            num_replicas=1,
            rank=0,
            seed=42,
        )

        sampler2 = SequentialBucketSampler(
            simple_dataset,
            base_batch_size=4,
            num_replicas=1,
            rank=0,
            seed=42,
        )

        batches1 = list(sampler1)
        batches2 = list(sampler2)

        assert batches1 == batches2

    def test_sampler_different_seeds_different_order(self, simple_dataset):
        """Test different seeds produce different batch orders."""
        sampler1 = SequentialBucketSampler(
            simple_dataset,
            base_batch_size=4,
            num_replicas=1,
            rank=0,
            seed=42,
            shuffle_within_bucket=True,
        )

        sampler2 = SequentialBucketSampler(
            simple_dataset,
            base_batch_size=4,
            num_replicas=1,
            rank=0,
            seed=123,
            shuffle_within_bucket=True,
        )

        batches1 = list(sampler1)
        batches2 = list(sampler2)

        # With different seeds, at least some batches should differ
        # (might be same if dataset is very small or uniform)
        if len(batches1) > 1:
            assert batches1 != batches2 or len(simple_dataset) <= 4

    def test_sampler_no_shuffle(self, simple_dataset):
        """Test sampler without shuffling."""
        sampler = SequentialBucketSampler(
            simple_dataset,
            base_batch_size=4,
            num_replicas=1,
            rank=0,
            shuffle_buckets=False,
            shuffle_within_bucket=False,
        )

        batches = list(sampler)
        assert len(batches) > 0

    def test_sampler_different_order_across_epochs(self, large_dataset):
        """Test that bucket and element order differs across epochs."""
        sampler = SequentialBucketSampler(
            large_dataset,
            base_batch_size=8,
            num_replicas=1,
            rank=0,
            seed=42,
            shuffle_buckets=True,
            shuffle_within_bucket=True,
            drop_last=False,  # Must be False to ensure all samples are yielded each epoch
        )

        # Collect batches for epoch 0
        sampler.set_epoch(0)
        batches_epoch_0 = list(sampler)

        # Collect batches for epoch 1
        sampler.set_epoch(1)
        batches_epoch_1 = list(sampler)

        # Collect batches for epoch 2
        sampler.set_epoch(2)
        batches_epoch_2 = list(sampler)

        # All epochs should have the same number of batches
        assert len(batches_epoch_0) == len(batches_epoch_1) == len(batches_epoch_2)

        # The order should be different across epochs
        # Compare flattened index lists to check ordering
        indices_epoch_0 = [idx for batch in batches_epoch_0 for idx in batch]
        indices_epoch_1 = [idx for batch in batches_epoch_1 for idx in batch]
        indices_epoch_2 = [idx for batch in batches_epoch_2 for idx in batch]

        # At least one pair of epochs should have different ordering
        assert (
            indices_epoch_0 != indices_epoch_1
            or indices_epoch_1 != indices_epoch_2
            or indices_epoch_0 != indices_epoch_2
        ), "Expected different ordering across epochs when shuffling is enabled"

        # Verify all epochs cover the same samples (just in different order)
        assert set(indices_epoch_0) == set(indices_epoch_1) == set(indices_epoch_2)

    def test_sampler_dynamic_batch_size_disabled(self, multi_resolution_dataset):
        """Test sampler with dynamic_batch_size=False uses fixed batch size."""
        batch_size = 4
        sampler = SequentialBucketSampler(
            multi_resolution_dataset,
            base_batch_size=batch_size,
            dynamic_batch_size=False,
            num_replicas=1,
            rank=0,
            drop_last=True,
        )

        batches = list(sampler)

        for batch in batches:
            assert len(batch) == batch_size

    def test_sampler_dynamic_batch_size_enabled(self, multi_resolution_dataset):
        """Test sampler with dynamic_batch_size=True varies batch size."""
        sampler = SequentialBucketSampler(
            multi_resolution_dataset,
            base_batch_size=8,
            base_resolution=(512, 512),
            dynamic_batch_size=True,
            num_replicas=1,
            rank=0,
            drop_last=True,
        )

        batches = list(sampler)

        # Get batch sizes
        batch_sizes = [len(batch) for batch in batches]

        # With multiple resolutions, batch sizes should vary
        # (or at least be calculated dynamically)
        assert len(batches) > 0

    def test_sampler_get_batch_info(self, simple_dataset):
        """Test get_batch_info returns bucket information."""
        sampler = SequentialBucketSampler(
            simple_dataset,
            base_batch_size=4,
            num_replicas=1,
            rank=0,
        )

        info = sampler.get_batch_info(0)

        if info:  # May be empty if batch_idx is out of range
            assert "bucket_key" in info or info == {}
            if "bucket_key" in info:
                assert "resolution" in info
                assert "batch_size" in info


class TestSequentialBucketSamplerDistributedCPU:
    """CPU tests for SequentialBucketSampler distributed training support."""

    def test_multi_rank_same_batch_count(self, large_dataset):
        """Test all ranks get same number of batches."""
        world_size = 4
        batch_counts = []

        for rank in range(world_size):
            sampler = SequentialBucketSampler(
                large_dataset,
                base_batch_size=8,
                num_replicas=world_size,
                rank=rank,
            )
            batch_counts.append(len(sampler))

        # All ranks should have same batch count
        assert len(set(batch_counts)) == 1

    def test_multi_rank_different_samples(self, large_dataset):
        """Test different ranks get different samples."""
        world_size = 2

        sampler0 = SequentialBucketSampler(
            large_dataset,
            base_batch_size=8,
            num_replicas=world_size,
            rank=0,
            seed=42,
        )

        sampler1 = SequentialBucketSampler(
            large_dataset,
            base_batch_size=8,
            num_replicas=world_size,
            rank=1,
            seed=42,
        )

        batches0 = list(sampler0)
        batches1 = list(sampler1)

        # Ranks should have different samples (due to DDP splitting)
        all_indices0 = set(idx for batch in batches0 for idx in batch)
        all_indices1 = set(idx for batch in batches1 for idx in batch)

        # There might be some overlap due to padding, but mostly different
        # The intersection should be much smaller than either set
        assert len(all_indices0 & all_indices1) < len(all_indices0)

    def test_single_rank_equivalent(self, simple_dataset):
        """Test single rank (world_size=1) processes all data."""
        sampler = SequentialBucketSampler(
            simple_dataset,
            base_batch_size=4,
            num_replicas=1,
            rank=0,
        )

        batches = list(sampler)
        all_indices = [idx for batch in batches for idx in batch]

        # Should cover most of the dataset
        assert len(set(all_indices)) >= len(simple_dataset) * 0.5


# ============================================================================
# CPU Tests - collate_fn_production
# ============================================================================


class TestCollateFnProductionCPU:
    """CPU tests for collate_fn_production."""

    def test_collate_basic(self, simple_dataset):
        """Test basic collation of batch items."""
        # Get some samples
        items = [simple_dataset[i] for i in range(4)]

        batch = collate_fn_production(items)

        assert isinstance(batch, dict)
        assert "latent" in batch
        assert "crop_resolution" in batch

    def test_collate_stacks_tensors(self, simple_dataset):
        """Test collate function stacks tensors correctly."""
        items = [simple_dataset[i] for i in range(4)]

        batch = collate_fn_production(items)

        assert batch["latent"].shape[0] == 4
        assert batch["crop_resolution"].shape[0] == 4
        assert batch["original_resolution"].shape[0] == 4
        assert batch["crop_offset"].shape[0] == 4

    def test_collate_preserves_metadata(self, simple_dataset):
        """Test collate function preserves metadata lists."""
        items = [simple_dataset[i] for i in range(4)]

        batch = collate_fn_production(items)

        assert len(batch["prompt"]) == 4
        assert len(batch["image_path"]) == 4
        assert len(batch["bucket_id"]) == 4
        assert len(batch["aspect_ratio"]) == 4

    def test_collate_handles_embeddings(self, simple_dataset):
        """Test collate with embedding mode."""
        items = [simple_dataset[i] for i in range(4)]

        batch = collate_fn_production(items)

        if "clip_hidden" in items[0]:
            assert batch["clip_hidden"].shape[0] == 4
            assert batch["pooled_prompt_embeds"].shape[0] == 4
            assert batch["prompt_embeds"].shape[0] == 4

    def test_collate_same_resolution_required(self, multi_resolution_dataset):
        """Test collate requires same resolution in batch."""
        # Get items with DIFFERENT resolutions (should fail)
        items = []
        res_set = set()
        for i in range(len(multi_resolution_dataset)):
            item = multi_resolution_dataset[i]
            res = tuple(item["crop_resolution"].tolist())
            if res not in res_set:
                items.append(item)
                res_set.add(res)
            if len(items) >= 2:
                break

        if len(items) >= 2:
            with pytest.raises(AssertionError, match="Mixed resolutions"):
                collate_fn_production(items)


# ============================================================================
# CPU Tests - build_multiresolution_dataloader
# ============================================================================


class TestBuildMultiresolutionDataloaderCPU:
    """CPU tests for build_multiresolution_dataloader."""

    def test_build_dataloader_returns_tuple(self, simple_dataset):
        """Test function returns dataloader and sampler."""
        dataloader, sampler = build_multiresolution_dataloader(
            dataset=simple_dataset,
            base_batch_size=4,
            dp_rank=0,
            dp_world_size=1,
            num_workers=0,
        )

        assert dataloader is not None
        assert sampler is not None
        assert isinstance(sampler, SequentialBucketSampler)

    def test_dataloader_iteration(self, simple_dataset):
        """Test dataloader can be iterated."""
        dataloader, sampler = build_multiresolution_dataloader(
            dataset=simple_dataset,
            base_batch_size=4,
            dp_rank=0,
            dp_world_size=1,
            num_workers=0,
        )

        batch_count = 0
        for batch in dataloader:
            assert isinstance(batch, dict)
            assert "latent" in batch
            batch_count += 1
            if batch_count >= 2:
                break

        assert batch_count > 0

    def test_dataloader_batch_content(self, simple_dataset):
        """Test dataloader batches have correct content."""
        dataloader, _ = build_multiresolution_dataloader(
            dataset=simple_dataset,
            base_batch_size=4,
            dp_rank=0,
            dp_world_size=1,
            num_workers=0,
        )

        for batch in dataloader:
            assert batch["latent"].dim() == 4  # [B, C, H, W]
            assert len(batch["prompt"]) == batch["latent"].shape[0]
            break

    def test_dataloader_with_shuffle(self, simple_dataset):
        """Test dataloader with shuffle enabled."""
        dataloader, _ = build_multiresolution_dataloader(
            dataset=simple_dataset,
            base_batch_size=4,
            dp_rank=0,
            dp_world_size=1,
            shuffle=True,
            num_workers=0,
        )

        for batch in dataloader:
            assert batch is not None
            break

    def test_dataloader_without_shuffle(self, simple_dataset):
        """Test dataloader with shuffle disabled."""
        dataloader, _ = build_multiresolution_dataloader(
            dataset=simple_dataset,
            base_batch_size=4,
            dp_rank=0,
            dp_world_size=1,
            shuffle=False,
            num_workers=0,
        )

        for batch in dataloader:
            assert batch is not None
            break

    def test_dataloader_with_dynamic_batch(self, multi_resolution_dataset):
        """Test dataloader with dynamic batch sizing."""
        dataloader, _ = build_multiresolution_dataloader(
            dataset=multi_resolution_dataset,
            base_batch_size=8,
            base_resolution=(512, 512),
            dp_rank=0,
            dp_world_size=1,
            dynamic_batch_size=True,
            num_workers=0,
        )

        for batch in dataloader:
            assert batch is not None
            break


# ============================================================================
# GPU Tests - SequentialBucketSampler
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestSequentialBucketSamplerGPU:
    """GPU tests for SequentialBucketSampler."""

    def test_sampler_with_gpu_tensors(self, simple_dataset):
        """Test sampler works with dataset that loads GPU tensors."""

        sampler = SequentialBucketSampler(
            simple_dataset,
            base_batch_size=4,
            num_replicas=1,
            rank=0,
        )

        batches = list(sampler)
        assert len(batches) > 0

        # Load a batch and move to GPU
        for batch_indices in batches[:1]:
            items = [simple_dataset[i] for i in batch_indices]
            for item in items:
                latent_gpu = item["latent"].cuda()
                assert latent_gpu.is_cuda
            break


# ============================================================================
# GPU Tests - collate_fn_production
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestCollateFnProductionGPU:
    """GPU tests for collate_fn_production."""

    def test_collate_gpu_tensors(self, simple_dataset):
        """Test collate with GPU tensors."""

        # Get samples and move to GPU
        items = []
        for i in range(4):
            item = simple_dataset[i]
            # Keep on CPU for collation (standard practice)
            items.append(item)

        batch = collate_fn_production(items)

        # Move batch to GPU
        batch_gpu = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        assert batch_gpu["latent"].is_cuda
        assert batch_gpu["crop_resolution"].is_cuda

    def test_collate_then_transfer_to_gpu(self, simple_dataset):
        """Test collating on CPU then transferring to GPU."""
        items = [simple_dataset[i] for i in range(4)]
        batch = collate_fn_production(items)

        device = torch.device("cuda:0")

        # Transfer tensors to GPU
        latent_gpu = batch["latent"].to(device)
        assert latent_gpu.device.type == "cuda"

        crop_res_gpu = batch["crop_resolution"].to(device)
        assert crop_res_gpu.device.type == "cuda"


# ============================================================================
# GPU Tests - build_multiresolution_dataloader
# ============================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestBuildMultiresolutionDataloaderGPU:
    """GPU tests for build_multiresolution_dataloader."""

    def test_dataloader_with_pin_memory(self, simple_dataset):
        """Test dataloader with pin_memory for faster GPU transfer."""

        dataloader, _ = build_multiresolution_dataloader(
            dataset=simple_dataset,
            base_batch_size=4,
            dp_rank=0,
            dp_world_size=1,
            pin_memory=True,
            num_workers=0,
        )

        for batch in dataloader:
            # Transfer to GPU (should be faster with pinned memory)
            latent_gpu = batch["latent"].cuda(non_blocking=True)
            assert latent_gpu.is_cuda
            break

    def test_dataloader_batch_to_gpu(self, simple_dataset):
        """Test full batch transfer to GPU."""
        dataloader, _ = build_multiresolution_dataloader(
            dataset=simple_dataset,
            base_batch_size=4,
            dp_rank=0,
            dp_world_size=1,
            num_workers=0,
        )

        device = torch.device("cuda:0")

        for batch in dataloader:
            # Transfer all tensors to GPU
            latent_gpu = batch["latent"].to(device)
            crop_res_gpu = batch["crop_resolution"].to(device)
            orig_res_gpu = batch["original_resolution"].to(device)
            crop_offset_gpu = batch["crop_offset"].to(device)

            assert latent_gpu.is_cuda
            assert crop_res_gpu.is_cuda
            assert orig_res_gpu.is_cuda
            assert crop_offset_gpu.is_cuda

            # Check embeddings if present
            if "clip_hidden" in batch:
                clip_hidden_gpu = batch["clip_hidden"].to(device)
                pooled_prompt_embeds_gpu = batch["pooled_prompt_embeds"].to(device)
                prompt_embeds_gpu = batch["prompt_embeds"].to(device)

                assert clip_hidden_gpu.is_cuda
                assert pooled_prompt_embeds_gpu.is_cuda
                assert prompt_embeds_gpu.is_cuda

            break

    def test_dataloader_gpu_memory_cleanup(self, simple_dataset):
        """Test GPU memory is properly cleaned up after iteration."""
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        dataloader, _ = build_multiresolution_dataloader(
            dataset=simple_dataset,
            base_batch_size=4,
            dp_rank=0,
            dp_world_size=1,
            num_workers=0,
        )

        device = torch.device("cuda:0")

        # Process some batches
        for i, batch in enumerate(dataloader):
            latent_gpu = batch["latent"].to(device)
            del latent_gpu
            if i >= 2:
                break

        # Cleanup
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()

        # Memory should be back to approximately initial level
        # (allowing some overhead for CUDA context)
        assert final_memory < initial_memory + 10 * 1024 * 1024  # 10MB tolerance

    def test_dataloader_multi_gpu_simulation(self, large_dataset):
        """Test dataloader configuration for multi-GPU training."""
        gpu_count = torch.cuda.device_count()

        # Create dataloaders for each GPU (simulated)
        dataloaders = []
        for rank in range(min(gpu_count, 2)):  # Use up to 2 GPUs for test
            dl, _ = build_multiresolution_dataloader(
                dataset=large_dataset,
                base_batch_size=8,
                dp_rank=rank,
                dp_world_size=min(gpu_count, 2),
                num_workers=0,
            )
            dataloaders.append(dl)

        # Verify each dataloader produces batches
        for rank, dl in enumerate(dataloaders):
            batch_count = 0
            for batch in dl:
                assert batch["latent"] is not None
                batch_count += 1
                if batch_count >= 2:
                    break
            assert batch_count > 0

    def test_gpu_operations_on_batch(self, simple_dataset):
        """Test performing GPU operations on loaded batch."""
        dataloader, _ = build_multiresolution_dataloader(
            dataset=simple_dataset,
            base_batch_size=4,
            dp_rank=0,
            dp_world_size=1,
            num_workers=0,
        )

        device = torch.device("cuda:0")

        for batch in dataloader:
            latent_gpu = batch["latent"].to(device)

            # Perform some operations
            latent_normalized = latent_gpu / latent_gpu.std()
            latent_mean = latent_gpu.mean(dim=(2, 3))

            assert latent_normalized.is_cuda
            assert latent_mean.is_cuda
            assert torch.isfinite(latent_normalized).all()
            assert torch.isfinite(latent_mean).all()

            break


# ============================================================================
# Integration Tests
# ============================================================================


class TestDataloaderIntegration:
    """Integration tests for full dataloader pipeline."""

    def test_full_epoch_iteration_cpu(self, simple_dataset):
        """Test iterating through a full epoch on CPU."""
        dataloader, sampler = build_multiresolution_dataloader(
            dataset=simple_dataset,
            base_batch_size=4,
            dp_rank=0,
            dp_world_size=1,
            num_workers=0,
        )

        batch_count = 0
        for batch in dataloader:
            assert batch["latent"] is not None
            batch_count += 1

        assert batch_count == len(sampler)

    def test_multiple_epochs_cpu(self, simple_dataset):
        """Test iterating through multiple epochs."""
        dataloader, sampler = build_multiresolution_dataloader(
            dataset=simple_dataset,
            base_batch_size=4,
            dp_rank=0,
            dp_world_size=1,
            num_workers=0,
        )

        for epoch in range(3):
            sampler.set_epoch(epoch)
            batch_count = 0
            for batch in dataloader:
                batch_count += 1
            assert batch_count > 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_full_epoch_iteration_gpu(self, simple_dataset):
        """Test iterating through a full epoch with GPU transfer."""
        dataloader, sampler = build_multiresolution_dataloader(
            dataset=simple_dataset,
            base_batch_size=4,
            dp_rank=0,
            dp_world_size=1,
            pin_memory=True,
            num_workers=0,
        )

        device = torch.device("cuda:0")
        batch_count = 0

        for batch in dataloader:
            latent_gpu = batch["latent"].to(device, non_blocking=True)
            assert latent_gpu.is_cuda
            batch_count += 1

        assert batch_count == len(sampler)

    def test_deterministic_across_ranks(self, large_dataset):
        """Test deterministic behavior across simulated distributed ranks."""
        world_size = 2
        seed = 42

        # Create samplers for two ranks
        sampler0 = SequentialBucketSampler(
            large_dataset,
            base_batch_size=8,
            num_replicas=world_size,
            rank=0,
            seed=seed,
        )

        sampler1 = SequentialBucketSampler(
            large_dataset,
            base_batch_size=8,
            num_replicas=world_size,
            rank=1,
            seed=seed,
        )

        # Same batch count
        assert len(sampler0) == len(sampler1)

        # Different actual samples
        batches0 = list(sampler0)
        batches1 = list(sampler1)

        all_indices0 = set(idx for batch in batches0 for idx in batch)
        all_indices1 = set(idx for batch in batches1 for idx in batch)

        # Combined should cover more of the dataset
        combined = all_indices0 | all_indices1
        assert len(combined) > len(all_indices0)

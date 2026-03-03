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

"""Unit tests for collate_fns.py: collate_fn_flux, build_flux_multiresolution_dataloader."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from nemo_automodel.components.datasets.diffusion.collate_fns import (
    build_flux_multiresolution_dataloader,
    collate_fn_flux,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_production_batch(
    batch_size=2,
    has_prompt_embeds=True,
    has_clip_hidden=False,
):
    """Create a dict matching collate_fn_production output."""
    result = {
        "latent": torch.randn(batch_size, 16, 64, 64),
        "prompt": [f"prompt {i}" for i in range(batch_size)],
        "image_path": [f"/path/img_{i}.jpg" for i in range(batch_size)],
        "bucket_id": list(range(batch_size)),
        "aspect_ratio": [1.0] * batch_size,
        "crop_resolution": torch.tensor([[512, 512]] * batch_size),
        "original_resolution": torch.tensor([[1024, 1024]] * batch_size),
        "crop_offset": torch.tensor([[0, 0]] * batch_size),
    }
    if has_prompt_embeds:
        result["prompt_embeds"] = torch.randn(batch_size, 256, 4096)
        result["pooled_prompt_embeds"] = torch.randn(batch_size, 768)
    if has_clip_hidden:
        result["clip_hidden"] = torch.randn(batch_size, 77, 768)
    return result


# =============================================================================
# TestCollateFnFlux
# =============================================================================


class TestCollateFnFlux:
    """Tests for collate_fn_flux."""

    def test_pre_encoded_embeddings(self):
        prod_batch = _make_production_batch(has_prompt_embeds=True)
        with patch(
            "nemo_automodel.components.datasets.diffusion.collate_fns.collate_fn_production",
            return_value=prod_batch,
        ):
            result = collate_fn_flux([{}, {}])  # Dummy batch items

        assert "image_latents" in result
        assert "text_embeddings" in result
        assert "pooled_prompt_embeds" in result
        assert result["data_type"] == "image"
        assert result["image_latents"].shape == prod_batch["latent"].shape

    def test_with_clip_hidden(self):
        prod_batch = _make_production_batch(has_prompt_embeds=True, has_clip_hidden=True)
        with patch(
            "nemo_automodel.components.datasets.diffusion.collate_fns.collate_fn_production",
            return_value=prod_batch,
        ):
            result = collate_fn_flux([{}, {}])

        assert "clip_hidden" in result

    def test_without_clip_hidden(self):
        prod_batch = _make_production_batch(has_prompt_embeds=True, has_clip_hidden=False)
        with patch(
            "nemo_automodel.components.datasets.diffusion.collate_fns.collate_fn_production",
            return_value=prod_batch,
        ):
            result = collate_fn_flux([{}, {}])

        assert "clip_hidden" not in result

    def test_tokenized_input_raises(self):
        prod_batch = _make_production_batch(has_prompt_embeds=False)
        prod_batch["t5_tokens"] = torch.randint(0, 32000, (2, 256))
        prod_batch["clip_tokens"] = torch.randint(0, 49408, (2, 77))
        with patch(
            "nemo_automodel.components.datasets.diffusion.collate_fns.collate_fn_production",
            return_value=prod_batch,
        ):
            with pytest.raises(NotImplementedError, match="On-the-fly text encoding"):
                collate_fn_flux([{}, {}])

    def test_metadata_fields(self):
        prod_batch = _make_production_batch(has_prompt_embeds=True)
        with patch(
            "nemo_automodel.components.datasets.diffusion.collate_fns.collate_fn_production",
            return_value=prod_batch,
        ):
            result = collate_fn_flux([{}, {}])

        meta = result["metadata"]
        assert "prompts" in meta
        assert "image_paths" in meta
        assert "bucket_ids" in meta
        assert "aspect_ratios" in meta
        assert "crop_resolution" in meta
        assert "original_resolution" in meta
        assert "crop_offset" in meta


# =============================================================================
# TestBuildFluxMultiresolutionDataloader
# =============================================================================


class MockCacheBuilder:
    """Helper to create mock cache directories for TextToImageDataset."""

    def __init__(self, cache_dir: Path, num_samples: int = 10):
        self.cache_dir = cache_dir
        self.num_samples = num_samples

    def build_cache(self, resolution=(512, 512)):
        metadata = []
        for idx in range(self.num_samples):
            cache_file = self.cache_dir / f"sample_{idx:04d}.pt"
            data = {
                "latent": torch.randn(16, resolution[1] // 8, resolution[0] // 8),
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
            metadata.append(
                {
                    "cache_file": str(cache_file),
                    "crop_resolution": list(resolution),
                    "original_resolution": [1024, 768],
                    "aspect_ratio": 1.0,
                    "bucket_id": idx % 5,
                }
            )

        shard_file = self.cache_dir / "metadata_shard_0000.json"
        with open(shard_file, "w") as f:
            json.dump(metadata, f)

        metadata_file = self.cache_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump({"shards": ["metadata_shard_0000.json"]}, f)

        return metadata


class TestBuildFluxMultiresolutionDataloader:
    """Tests for build_flux_multiresolution_dataloader."""

    def test_returns_dataloader_and_sampler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            builder = MockCacheBuilder(cache_dir, num_samples=10)
            builder.build_cache()

            dl, sampler = build_flux_multiresolution_dataloader(
                cache_dir=str(cache_dir),
                batch_size=2,
                dp_rank=0,
                dp_world_size=1,
                num_workers=0,
            )

            from nemo_automodel.components.datasets.diffusion.sampler import SequentialBucketSampler

            assert isinstance(sampler, SequentialBucketSampler)
            assert dl is not None

    def test_iteration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            builder = MockCacheBuilder(cache_dir, num_samples=10)
            builder.build_cache()

            dl, _ = build_flux_multiresolution_dataloader(
                cache_dir=str(cache_dir),
                batch_size=2,
                dp_rank=0,
                dp_world_size=1,
                num_workers=0,
            )

            batch = next(iter(dl))
            assert "image_latents" in batch
            assert "text_embeddings" in batch
            assert "pooled_prompt_embeds" in batch
            assert batch["data_type"] == "image"

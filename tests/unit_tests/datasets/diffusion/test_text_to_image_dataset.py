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

"""Unit tests for TextToImageDataset."""

import json
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest
import torch

from nemo_automodel.components.datasets.diffusion.text_to_image_dataset import (
    TextToImageDataset,
)


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
def simple_cache_dir():
    """Create a simple temporary cache directory with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        builder = MockCacheBuilder(cache_dir, num_samples=5)
        builder.build_cache()
        yield cache_dir


@pytest.fixture
def multi_resolution_cache_dir():
    """Create a cache directory with multiple resolutions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        builder = MockCacheBuilder(cache_dir, num_samples=20)
        resolutions = [
            (512, 512),  # Square
            (512, 768),  # Portrait
            (768, 512),  # Landscape
            (640, 640),  # Another square
        ]
        aspect_ratios = [1.0, 0.67, 1.5, 1.0]
        builder.build_cache(resolutions=resolutions, aspect_ratios=aspect_ratios)
        yield cache_dir


class TestTextToImageDatasetInit:
    """Tests for TextToImageDataset initialization."""

    def test_init_with_valid_cache(self, simple_cache_dir):
        """Test initialization with valid cache directory."""
        dataset = TextToImageDataset(str(simple_cache_dir))

        assert len(dataset) == 5
        assert hasattr(dataset, "metadata")
        assert hasattr(dataset, "bucket_groups")
        assert hasattr(dataset, "calculator")

    def test_init_missing_cache_dir(self):
        """Test initialization with non-existent cache directory raises error."""
        with pytest.raises(FileNotFoundError):
            TextToImageDataset("/nonexistent/path/to/cache")

    def test_init_missing_metadata(self):
        """Test initialization with missing metadata.json raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No metadata.json"):
                TextToImageDataset(tmpdir)

    def test_init_invalid_metadata_format(self):
        """Test initialization with invalid metadata format raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            # Create invalid metadata (list instead of dict with shards)
            metadata_file = cache_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump([{"invalid": "format"}], f)

            with pytest.raises(ValueError, match="Invalid metadata format"):
                TextToImageDataset(tmpdir)

    def test_init_train_text_encoder_false(self, simple_cache_dir):
        """Test initialization with train_text_encoder=False."""
        dataset = TextToImageDataset(str(simple_cache_dir), train_text_encoder=False)

        assert dataset.train_text_encoder is False

    def test_init_train_text_encoder_true(self, simple_cache_dir):
        """Test initialization with train_text_encoder=True."""
        dataset = TextToImageDataset(str(simple_cache_dir), train_text_encoder=True)

        assert dataset.train_text_encoder is True


class TestBucketGrouping:
    """Tests for bucket grouping functionality."""

    def test_bucket_groups_created(self, simple_cache_dir):
        """Test that bucket groups are created."""
        dataset = TextToImageDataset(str(simple_cache_dir))

        assert len(dataset.bucket_groups) > 0

    def test_all_samples_in_buckets(self, simple_cache_dir):
        """Test that all samples are assigned to buckets."""
        dataset = TextToImageDataset(str(simple_cache_dir))

        total_in_buckets = sum(len(group["indices"]) for group in dataset.bucket_groups.values())
        assert total_in_buckets == len(dataset)

    def test_bucket_group_structure(self, simple_cache_dir):
        """Test bucket group has required fields."""
        dataset = TextToImageDataset(str(simple_cache_dir))

        required_fields = {"indices", "aspect_name", "aspect_ratio", "resolution", "pixels"}
        for bucket_key, group in dataset.bucket_groups.items():
            assert required_fields.issubset(group.keys())

    def test_sorted_bucket_keys(self, multi_resolution_cache_dir):
        """Test bucket keys are sorted by pixel count."""
        dataset = TextToImageDataset(str(multi_resolution_cache_dir))

        pixels_list = [dataset.bucket_groups[key]["pixels"] for key in dataset.sorted_bucket_keys]
        assert pixels_list == sorted(pixels_list)

    def test_aspect_ratio_classification(self, multi_resolution_cache_dir):
        """Test aspect ratio to name classification."""
        dataset = TextToImageDataset(str(multi_resolution_cache_dir))

        # Test the internal method
        assert dataset._aspect_ratio_to_name(0.5) == "tall"
        assert dataset._aspect_ratio_to_name(0.8) == "tall"
        assert dataset._aspect_ratio_to_name(1.0) == "square"
        assert dataset._aspect_ratio_to_name(1.1) == "square"
        assert dataset._aspect_ratio_to_name(1.5) == "wide"
        assert dataset._aspect_ratio_to_name(2.0) == "wide"


class TestDatasetGetItem:
    """Tests for dataset __getitem__ method."""

    def test_getitem_returns_dict(self, simple_cache_dir):
        """Test __getitem__ returns a dictionary."""
        dataset = TextToImageDataset(str(simple_cache_dir))

        item = dataset[0]
        assert isinstance(item, dict)

    def test_getitem_has_required_fields_embeddings(self, simple_cache_dir):
        """Test __getitem__ returns required fields when train_text_encoder=False."""
        dataset = TextToImageDataset(str(simple_cache_dir), train_text_encoder=False)

        item = dataset[0]

        required_fields = {
            "latent",
            "crop_resolution",
            "original_resolution",
            "crop_offset",
            "prompt",
            "image_path",
            "bucket_id",
            "aspect_ratio",
            "clip_hidden",
            "pooled_prompt_embeds",
            "prompt_embeds",
        }
        assert required_fields.issubset(item.keys())

    def test_getitem_has_required_fields_tokens(self, simple_cache_dir):
        """Test __getitem__ returns token fields when train_text_encoder=True."""
        dataset = TextToImageDataset(str(simple_cache_dir), train_text_encoder=True)

        item = dataset[0]

        required_fields = {
            "latent",
            "crop_resolution",
            "original_resolution",
            "crop_offset",
            "prompt",
            "image_path",
            "bucket_id",
            "aspect_ratio",
            "clip_tokens",
            "t5_tokens",
        }
        assert required_fields.issubset(item.keys())

    def test_getitem_latent_is_tensor(self, simple_cache_dir):
        """Test latent is a torch tensor."""
        dataset = TextToImageDataset(str(simple_cache_dir))

        item = dataset[0]
        assert isinstance(item["latent"], torch.Tensor)

    def test_getitem_resolutions_are_tensors(self, simple_cache_dir):
        """Test resolutions are tensors."""
        dataset = TextToImageDataset(str(simple_cache_dir))

        item = dataset[0]
        assert isinstance(item["crop_resolution"], torch.Tensor)
        assert isinstance(item["original_resolution"], torch.Tensor)
        assert isinstance(item["crop_offset"], torch.Tensor)

    def test_getitem_index_range(self, simple_cache_dir):
        """Test __getitem__ works for all valid indices."""
        dataset = TextToImageDataset(str(simple_cache_dir))

        for idx in range(len(dataset)):
            item = dataset[idx]
            assert item is not None

    def test_getitem_embeddings_shapes(self, simple_cache_dir):
        """Test embedding shapes when train_text_encoder=False."""
        dataset = TextToImageDataset(str(simple_cache_dir), train_text_encoder=False)

        item = dataset[0]

        # CLIP hidden should be [77, 768]
        assert item["clip_hidden"].dim() == 2
        assert item["clip_hidden"].shape[0] == 77

        # Pooled prompt embeds should be [768]
        assert item["pooled_prompt_embeds"].dim() == 1

        # Prompt embeds should be [256, 4096]
        assert item["prompt_embeds"].dim() == 2

    def test_getitem_tokens_shapes(self, simple_cache_dir):
        """Test token shapes when train_text_encoder=True."""
        dataset = TextToImageDataset(str(simple_cache_dir), train_text_encoder=True)

        item = dataset[0]

        # CLIP tokens should be [77]
        assert item["clip_tokens"].dim() == 1
        assert item["clip_tokens"].shape[0] == 77

        # T5 tokens should be [256]
        assert item["t5_tokens"].dim() == 1


class TestGetBucketInfo:
    """Tests for get_bucket_info method."""

    def test_bucket_info_structure(self, multi_resolution_cache_dir):
        """Test get_bucket_info returns correct structure."""
        dataset = TextToImageDataset(str(multi_resolution_cache_dir))

        info = dataset.get_bucket_info()

        assert "total_buckets" in info
        assert "buckets" in info
        assert isinstance(info["total_buckets"], int)
        assert isinstance(info["buckets"], dict)

    def test_bucket_info_total_matches(self, multi_resolution_cache_dir):
        """Test total_buckets matches bucket_groups count."""
        dataset = TextToImageDataset(str(multi_resolution_cache_dir))

        info = dataset.get_bucket_info()

        assert info["total_buckets"] == len(dataset.bucket_groups)

    def test_bucket_info_sample_counts(self, multi_resolution_cache_dir):
        """Test bucket sample counts sum to total dataset size."""
        dataset = TextToImageDataset(str(multi_resolution_cache_dir))

        info = dataset.get_bucket_info()

        total_samples = sum(info["buckets"].values())
        assert total_samples == len(dataset)


class TestMultiShardLoading:
    """Tests for loading metadata from multiple shards."""

    def test_multiple_shards(self):
        """Test loading metadata from multiple shard files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)

            # Create samples for shard 1
            shard1_data = []
            for idx in range(3):
                cache_file = cache_dir / f"shard1_sample_{idx}.pt"
                data = {
                    "latent": torch.randn(16, 64, 64),
                    "crop_offset": [0, 0],
                    "prompt": f"Shard 1 prompt {idx}",
                    "image_path": f"/fake/shard1/image_{idx}.jpg",
                    "clip_hidden": torch.randn(1, 77, 768),
                    "clip_pooled": torch.randn(1, 768),
                    "t5_hidden": torch.randn(1, 256, 4096),
                }
                torch.save(data, cache_file)
                shard1_data.append(
                    {
                        "cache_file": str(cache_file),
                        "crop_resolution": [512, 512],
                        "original_resolution": [1024, 768],
                        "aspect_ratio": 1.0,
                        "bucket_id": idx,
                    }
                )

            # Create samples for shard 2
            shard2_data = []
            for idx in range(2):
                cache_file = cache_dir / f"shard2_sample_{idx}.pt"
                data = {
                    "latent": torch.randn(16, 64, 96),
                    "crop_offset": [0, 0],
                    "prompt": f"Shard 2 prompt {idx}",
                    "image_path": f"/fake/shard2/image_{idx}.jpg",
                    "clip_hidden": torch.randn(1, 77, 768),
                    "clip_pooled": torch.randn(1, 768),
                    "t5_hidden": torch.randn(1, 256, 4096),
                }
                torch.save(data, cache_file)
                shard2_data.append(
                    {
                        "cache_file": str(cache_file),
                        "crop_resolution": [512, 768],
                        "original_resolution": [1024, 1536],
                        "aspect_ratio": 0.67,
                        "bucket_id": idx + 3,
                    }
                )

            # Write shard files
            with open(cache_dir / "shard_0000.json", "w") as f:
                json.dump(shard1_data, f)
            with open(cache_dir / "shard_0001.json", "w") as f:
                json.dump(shard2_data, f)

            # Write main metadata
            with open(cache_dir / "metadata.json", "w") as f:
                json.dump({"shards": ["shard_0000.json", "shard_0001.json"]}, f)

            # Load dataset
            dataset = TextToImageDataset(str(cache_dir))

            assert len(dataset) == 5  # 3 + 2 samples

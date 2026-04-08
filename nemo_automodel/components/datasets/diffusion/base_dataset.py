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

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List

from torch.utils.data import Dataset

from .multi_tier_bucketing import MultiTierBucketCalculator

logger = logging.getLogger(__name__)


class BaseMultiresolutionDataset(Dataset, ABC):
    """Abstract base class for multiresolution datasets with bucket-based sampling."""

    def __init__(self, cache_dir: str, quantization: int = 64):
        """
        Args:
            cache_dir: Directory containing preprocessed cache (metadata.json + shards)
            quantization: Resolution quantization factor (64 for images, 8 for video)
        """
        self.cache_dir = Path(cache_dir)

        # Load metadata
        self.metadata = self._load_metadata()

        logger.info(f"Loaded dataset with {len(self.metadata)} samples")

        # Group by bucket
        self._group_by_bucket()

        # Initialize bucket calculator for dynamic batch sizes
        self.calculator = MultiTierBucketCalculator(quantization=quantization)

    def _load_metadata(self) -> List[Dict]:
        """Load metadata from cache directory.

        Expects metadata.json with "shards" key referencing shard files.
        """
        metadata_file = self.cache_dir / "metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(f"No metadata.json found in {self.cache_dir}")

        with open(metadata_file, "r") as f:
            data = json.load(f)

        if not isinstance(data, dict) or "shards" not in data:
            raise ValueError(f"Invalid metadata format in {metadata_file}. Expected dict with 'shards' key.")

        # Load all shard files
        metadata = []
        for shard_name in data["shards"]:
            shard_path = self.cache_dir / shard_name
            with open(shard_path, "r") as f:
                shard_data = json.load(f)
                metadata.extend(shard_data)

        return metadata

    def _aspect_ratio_to_name(self, aspect_ratio: float) -> str:
        """Convert aspect ratio to a descriptive name."""
        if aspect_ratio < 0.85:
            return "tall"
        elif aspect_ratio > 1.18:
            return "wide"
        else:
            return "square"

    def _group_by_bucket(self):
        """Group samples by bucket (aspect_ratio + resolution)."""
        self.bucket_groups = {}

        # Support both bucket_resolution (video) and crop_resolution (image) keys
        resolution_key = "bucket_resolution" if "bucket_resolution" in self.metadata[0] else "crop_resolution"

        for idx, item in enumerate(self.metadata):
            aspect_ratio = item.get("aspect_ratio", 1.0)
            aspect_name = self._aspect_ratio_to_name(aspect_ratio)
            resolution = tuple(item[resolution_key])
            bucket_key = (aspect_name, resolution)

            if bucket_key not in self.bucket_groups:
                self.bucket_groups[bucket_key] = {
                    "indices": [],
                    "aspect_name": aspect_name,
                    "aspect_ratio": aspect_ratio,
                    "resolution": resolution,
                    "pixels": resolution[0] * resolution[1],
                }

            self.bucket_groups[bucket_key]["indices"].append(idx)

        # Sort buckets by resolution (low to high for optimal memory usage)
        self.sorted_bucket_keys = sorted(self.bucket_groups.keys(), key=lambda k: self.bucket_groups[k]["pixels"])

        logger.info(f"\nDataset organized into {len(self.bucket_groups)} buckets:")
        for key in self.sorted_bucket_keys:
            bucket = self.bucket_groups[key]
            aspect_name, resolution = key
            logger.info(
                f"  {aspect_name:6s} {resolution[0]:4d}x{resolution[1]:4d}: {len(bucket['indices']):5d} samples"
            )

    def get_bucket_info(self) -> Dict:
        """Get bucket organization information."""
        return {
            "total_buckets": len(self.bucket_groups),
            "buckets": {f"{k[0]}/{k[1][0]}x{k[1][1]}": len(v["indices"]) for k, v in self.bucket_groups.items()},
        }

    def __len__(self) -> int:
        return len(self.metadata)

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict:
        """Load a single sample. Subclasses must implement."""
        ...

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

"""Unit tests for MultiTierBucketCalculator."""

import numpy as np
import pytest
from PIL import Image

from nemo_automodel.components.datasets.diffusion.multi_tier_bucketing import (
    MultiTierBucketCalculator,
)


class TestMultiTierBucketCalculatorInit:
    """Tests for MultiTierBucketCalculator initialization."""

    def test_default_initialization(self):
        """Test default initialization with 256x256 max pixels."""
        calc = MultiTierBucketCalculator()

        assert calc.quantization == 64
        assert calc.max_pixels == 256 * 256
        assert len(calc.buckets) > 0

    def test_custom_max_pixels(self):
        """Test initialization with custom max_pixels."""
        max_pixels = 512 * 512
        calc = MultiTierBucketCalculator(max_pixels=max_pixels)

        assert calc.max_pixels == max_pixels
        # All buckets should be within pixel budget
        for bucket in calc.buckets:
            assert bucket["pixels"] <= max_pixels

    def test_custom_quantization(self):
        """Test initialization with custom quantization."""
        calc = MultiTierBucketCalculator(quantization=32)

        assert calc.quantization == 32
        # All resolutions should be multiples of quantization
        for bucket in calc.buckets:
            width, height = bucket["resolution"]
            assert width % 32 == 0
            assert height % 32 == 0

    def test_from_preset_512p(self):
        """Test creation from 512p preset."""
        calc = MultiTierBucketCalculator.from_preset("512p")

        assert calc.max_pixels == 512 * 512
        for bucket in calc.buckets:
            assert bucket["pixels"] <= 512 * 512

    def test_from_preset_1024p(self):
        """Test creation from 1024p preset."""
        calc = MultiTierBucketCalculator.from_preset("1024p")

        assert calc.max_pixels == 1024 * 1024
        for bucket in calc.buckets:
            assert bucket["pixels"] <= 1024 * 1024

    def test_from_preset_invalid(self):
        """Test that invalid preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            MultiTierBucketCalculator.from_preset("invalid_preset")

    def test_all_presets_available(self):
        """Test all documented presets can be created."""
        expected_presets = ["256p", "512p", "768p", "1024p", "1536p"]
        for preset in expected_presets:
            calc = MultiTierBucketCalculator.from_preset(preset)
            assert calc is not None


class TestBucketGeneration:
    """Tests for bucket generation logic."""

    def test_buckets_have_required_fields(self):
        """Test that all buckets have required fields."""
        calc = MultiTierBucketCalculator()

        required_fields = {"id", "resolution", "aspect_ratio", "pixels"}
        for bucket in calc.buckets:
            assert required_fields.issubset(bucket.keys())

    def test_bucket_ids_are_sequential(self):
        """Test that bucket IDs are sequential starting from 0."""
        calc = MultiTierBucketCalculator()

        ids = [b["id"] for b in calc.buckets]
        assert ids == list(range(len(calc.buckets)))

    def test_bucket_pixels_match_resolution(self):
        """Test that bucket pixels equals width * height."""
        calc = MultiTierBucketCalculator()

        for bucket in calc.buckets:
            width, height = bucket["resolution"]
            assert bucket["pixels"] == width * height

    def test_bucket_aspect_ratio_matches_resolution(self):
        """Test that bucket aspect ratio matches resolution."""
        calc = MultiTierBucketCalculator()

        for bucket in calc.buckets:
            width, height = bucket["resolution"]
            expected_ar = width / height
            assert abs(bucket["aspect_ratio"] - expected_ar) < 1e-6

    def test_unique_resolutions(self):
        """Test that all bucket resolutions are unique."""
        calc = MultiTierBucketCalculator()

        resolutions = [b["resolution"] for b in calc.buckets]
        assert len(resolutions) == len(set(resolutions))

    def test_resolution_quantization(self):
        """Test all resolutions are multiples of quantization."""
        quantization = 64
        calc = MultiTierBucketCalculator(quantization=quantization)

        for bucket in calc.buckets:
            width, height = bucket["resolution"]
            assert width % quantization == 0, f"Width {width} not multiple of {quantization}"
            assert height % quantization == 0, f"Height {height} not multiple of {quantization}"

    def test_minimum_resolution(self):
        """Test all resolutions are at least quantization size."""
        quantization = 64
        calc = MultiTierBucketCalculator(quantization=quantization)

        for bucket in calc.buckets:
            width, height = bucket["resolution"]
            assert width >= quantization
            assert height >= quantization


class TestBucketLookup:
    """Tests for bucket lookup methods."""

    def test_get_bucket_for_square_image(self):
        """Test bucket selection for square images."""
        calc = MultiTierBucketCalculator(max_pixels=512 * 512)

        bucket = calc.get_bucket_for_image(512, 512)

        # Should return a bucket close to 1:1 aspect ratio
        assert abs(bucket["aspect_ratio"] - 1.0) < 0.1

    def test_get_bucket_for_wide_image(self):
        """Test bucket selection for wide/landscape images."""
        calc = MultiTierBucketCalculator(max_pixels=512 * 512)

        bucket = calc.get_bucket_for_image(1920, 1080)  # 16:9

        # Should return a bucket with AR > 1
        assert bucket["aspect_ratio"] > 1.0

    def test_get_bucket_for_tall_image(self):
        """Test bucket selection for tall/portrait images."""
        calc = MultiTierBucketCalculator(max_pixels=512 * 512)

        bucket = calc.get_bucket_for_image(1080, 1920)  # 9:16

        # Should return a bucket with AR < 1
        assert bucket["aspect_ratio"] < 1.0

    def test_get_bucket_by_resolution_existing(self):
        """Test get_bucket_by_resolution for existing resolution."""
        calc = MultiTierBucketCalculator()

        # Get a known bucket
        known_bucket = calc.buckets[0]
        resolution = known_bucket["resolution"]

        result = calc.get_bucket_by_resolution(*resolution)
        assert result == known_bucket

    def test_get_bucket_by_resolution_nonexisting(self):
        """Test get_bucket_by_resolution for non-existing resolution."""
        calc = MultiTierBucketCalculator()

        result = calc.get_bucket_by_resolution(12345, 67890)
        assert result is None

    def test_get_bucket_by_id(self):
        """Test get_bucket_by_id."""
        calc = MultiTierBucketCalculator()

        for i in range(len(calc.buckets)):
            bucket = calc.get_bucket_by_id(i)
            assert bucket["id"] == i

    def test_get_all_buckets(self):
        """Test get_all_buckets returns all buckets."""
        calc = MultiTierBucketCalculator()

        all_buckets = calc.get_all_buckets()
        assert len(all_buckets) == len(calc.buckets)
        assert all_buckets == calc.buckets


class TestResizeAndCrop:
    """Tests for resize_and_crop method."""

    def test_resize_crop_center_square(self):
        """Test center crop on square image to square target."""
        calc = MultiTierBucketCalculator()

        # Create a simple test image
        img = Image.new("RGB", (100, 100), color="red")

        cropped, offset = calc.resize_and_crop(img, 64, 64, crop_mode="center")

        assert cropped.size == (64, 64)
        assert isinstance(offset, tuple)
        assert len(offset) == 2

    def test_resize_crop_center_wide_to_square(self):
        """Test center crop on wide image to square target."""
        calc = MultiTierBucketCalculator()

        img = Image.new("RGB", (200, 100), color="blue")

        cropped, offset = calc.resize_and_crop(img, 64, 64, crop_mode="center")

        assert cropped.size == (64, 64)

    def test_resize_crop_center_tall_to_square(self):
        """Test center crop on tall image to square target."""
        calc = MultiTierBucketCalculator()

        img = Image.new("RGB", (100, 200), color="green")

        cropped, offset = calc.resize_and_crop(img, 64, 64, crop_mode="center")

        assert cropped.size == (64, 64)

    def test_resize_crop_random(self):
        """Test random crop produces correct size."""
        calc = MultiTierBucketCalculator()

        img = Image.new("RGB", (200, 200), color="white")

        cropped, offset = calc.resize_and_crop(img, 128, 128, crop_mode="random")

        assert cropped.size == (128, 128)

    def test_resize_crop_smart(self):
        """Test smart crop produces correct size."""
        calc = MultiTierBucketCalculator()

        img = Image.new("RGB", (200, 200), color="white")

        cropped, offset = calc.resize_and_crop(img, 128, 128, crop_mode="smart")

        assert cropped.size == (128, 128)

    def test_resize_crop_numpy_array_input(self):
        """Test resize_and_crop with numpy array input."""
        calc = MultiTierBucketCalculator()

        # Create a numpy array image
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)

        cropped, offset = calc.resize_and_crop(img_array, 64, 64, crop_mode="center")

        assert cropped.size == (64, 64)


class TestDynamicBatchSize:
    """Tests for dynamic batch size calculation."""

    def test_dynamic_batch_size_base_resolution(self):
        """Test dynamic batch size at base resolution."""
        calc = MultiTierBucketCalculator()

        base_res = (512, 512)
        base_batch = 32

        batch_size = calc.get_dynamic_batch_size(base_res, base_batch, base_res)

        assert batch_size == base_batch

    def test_dynamic_batch_size_higher_resolution(self):
        """Test dynamic batch size decreases for higher resolution."""
        calc = MultiTierBucketCalculator()

        base_res = (512, 512)
        high_res = (1024, 1024)  # 4x pixels
        base_batch = 32

        batch_size = calc.get_dynamic_batch_size(high_res, base_batch, base_res)

        # Should be approximately base_batch / 4
        assert batch_size < base_batch
        assert batch_size == 8  # 32 / 4

    def test_dynamic_batch_size_lower_resolution(self):
        """Test dynamic batch size increases for lower resolution."""
        calc = MultiTierBucketCalculator()

        base_res = (512, 512)
        low_res = (256, 256)  # 0.25x pixels
        base_batch = 32

        batch_size = calc.get_dynamic_batch_size(low_res, base_batch, base_res)

        # Should be approximately base_batch * 4
        assert batch_size > base_batch
        assert batch_size == 128  # 32 * 4

    def test_dynamic_batch_size_minimum(self):
        """Test dynamic batch size has minimum of 1."""
        calc = MultiTierBucketCalculator()

        base_res = (256, 256)
        huge_res = (4096, 4096)  # Very large resolution
        base_batch = 4

        batch_size = calc.get_dynamic_batch_size(huge_res, base_batch, base_res)

        assert batch_size >= 1


class TestAspectRatios:
    """Tests for aspect ratio handling."""

    def test_standard_aspect_ratios_defined(self):
        """Test that standard aspect ratios are defined."""
        assert len(MultiTierBucketCalculator.ASPECT_RATIOS) > 0

        # Should include common ratios
        ratios = [(w / h) for w, h in MultiTierBucketCalculator.ASPECT_RATIOS]

        # Check for square (1:1)
        assert any(abs(r - 1.0) < 0.01 for r in ratios)

        # Check for 16:9
        assert any(abs(r - 16 / 9) < 0.01 for r in ratios)

    def test_resolution_presets_defined(self):
        """Test that resolution presets are properly defined."""
        presets = MultiTierBucketCalculator.RESOLUTION_PRESETS

        assert "256p" in presets
        assert "512p" in presets
        assert "1024p" in presets

        # Verify values
        assert presets["256p"] == 256 * 256
        assert presets["512p"] == 512 * 512
        assert presets["1024p"] == 1024 * 1024

    def test_bucket_covers_aspect_ratio_range(self):
        """Test that buckets cover a range of aspect ratios."""
        calc = MultiTierBucketCalculator(max_pixels=512 * 512)

        aspect_ratios = [b["aspect_ratio"] for b in calc.buckets]

        # Should have both portrait (AR < 1) and landscape (AR > 1)
        has_portrait = any(ar < 0.9 for ar in aspect_ratios)
        has_landscape = any(ar > 1.1 for ar in aspect_ratios)

        assert has_portrait, "No portrait buckets found"
        assert has_landscape, "No landscape buckets found"

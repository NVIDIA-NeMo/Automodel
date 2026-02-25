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
from typing import Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


class MultiTierBucketCalculator:
    """
    Calculate resolution buckets constrained by a maximum pixel budget.
    Supports various aspect ratios, each scaled to fit within the pixel budget.
    """

    # Standard aspect ratios (width, height)
    ASPECT_RATIOS = [
        (1, 2),  # 0.500 - Ultra tall portrait
        (9, 16),  # 0.563 - Vertical video
        (2, 3),  # 0.667 - Classic portrait
        (3, 4),  # 0.750 - Portrait photo
        (4, 5),  # 0.800 - 8x10 photo
        (1, 1),  # 1.000 - Square
        (5, 4),  # 1.250 - Computer
        (4, 3),  # 1.333 - Classic TV
        (3, 2),  # 1.500 - 35mm film
        (16, 10),  # 1.600 - Widescreen computer
        (5, 3),  # 1.667 - European widescreen
        (16, 9),  # 1.778 - HD video
        (2, 1),  # 2.000 - Ultra wide
        (21, 9),  # 2.333 - Cinema
    ]

    # Common resolution presets (name -> max pixels)
    RESOLUTION_PRESETS = {
        "256p": 256 * 256,  # 65,536 pixels
        "512p": 512 * 512,  # 262,144 pixels
        "768p": 768 * 768,  # 589,824 pixels
        "1024p": 1024 * 1024,  # 1,048,576 pixels
        "1536p": 1536 * 1536,  # 2,359,296 pixels
    }

    def __init__(self, quantization: int = 64, max_pixels: Optional[int] = None, debug_mode: bool = False):
        """
        Args:
            quantization: Resolution must be multiple of this (64 for Flux)
            max_pixels: Maximum pixel count for buckets (default: 256*256 = 65536)
        """
        self.quantization = quantization
        self.max_pixels = max_pixels if max_pixels is not None else 256 * 256

        # Generate all buckets
        self.buckets = self._generate_all_buckets()

        # Create lookup structures
        self._build_lookup_structures()

        if debug_mode:
            logger.info(f"Generated {len(self.buckets)} resolution buckets (max {self.max_pixels} pixels):")
            self._print_bucket_summary()

    def _generate_all_buckets(self) -> List[Dict]:
        """Generate all unique resolution buckets within the pixel budget."""
        # First, calculate resolutions for all aspect ratios
        resolution_to_aspects: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

        for aspect_w, aspect_h in self.ASPECT_RATIOS:
            aspect_ratio = aspect_w / aspect_h

            # Calculate the maximum resolution for this aspect ratio
            # within the pixel budget
            resolution = self._calculate_max_resolution(aspect_ratio)

            if resolution is not None:
                if resolution not in resolution_to_aspects:
                    resolution_to_aspects[resolution] = []
                resolution_to_aspects[resolution].append((aspect_w, aspect_h))

        # Now create unique buckets, one per unique resolution
        buckets = []
        bucket_id = 0

        # Sort resolutions by aspect ratio (width/height) for consistent ordering
        sorted_resolutions = sorted(resolution_to_aspects.keys(), key=lambda r: r[0] / r[1])

        for resolution in sorted_resolutions:
            width, height = resolution
            # Use actual resolution aspect ratio
            aspect_ratio = width / height

            buckets.append(
                {
                    "id": bucket_id,
                    "resolution": resolution,
                    "aspect_ratio": aspect_ratio,
                    "pixels": width * height,
                }
            )
            bucket_id += 1

        return buckets

    def _calculate_max_resolution(
        self,
        aspect_ratio: float,
    ) -> Optional[Tuple[int, int]]:
        """
        Calculate the maximum resolution for an aspect ratio within the pixel budget.

        For a given aspect ratio r = w/h, and pixel budget P:
        w * h <= P
        w = r * h
        r * h * h <= P
        h <= sqrt(P / r)

        Then w = r * h
        """
        import math

        # Calculate maximum height
        max_height = math.sqrt(self.max_pixels / aspect_ratio)
        height = self._round_to_quantization(int(max_height))

        # Calculate corresponding width
        width = self._round_to_quantization(int(height * aspect_ratio))

        # Verify within budget (rounding might push us over)
        while width * height > self.max_pixels:
            # Reduce the larger dimension
            if width >= height:
                width -= self.quantization
            else:
                height -= self.quantization

        # Ensure minimum size
        if width < self.quantization or height < self.quantization:
            return None

        return (width, height)

    def _round_to_quantization(self, value: int) -> int:
        """Round value to nearest quantization multiple."""
        return round(value / self.quantization) * self.quantization

    def _build_lookup_structures(self):
        """Build efficient lookup structures."""
        # Group by resolution
        self.buckets_by_resolution = {}
        for bucket in self.buckets:
            res = bucket["resolution"]
            self.buckets_by_resolution[res] = bucket

    def _print_bucket_summary(self):
        """Print summary of generated buckets."""
        logger.info("\n=== Unique Resolution Buckets ===")
        for bucket in sorted(self.buckets, key=lambda x: x["aspect_ratio"]):
            w, h = bucket["resolution"]
            pixels_k = bucket["pixels"] / 1e3
            ar = bucket["aspect_ratio"]
            logger.info(f"  {w:4d}x{h:4d} (AR={ar:.3f}, {pixels_k:.1f}K pixels)")

    def get_bucket_for_image(self, image_width: int, image_height: int) -> Dict:
        """
        Get the best bucket for an image.

        Args:
            image_width: Original image width
            image_height: Original image height
            max_pixels: Override max pixels for this query (deprecated, use constructor)

        Returns:
            Bucket dictionary with resolution and metadata
        """
        image_aspect = image_width / image_height

        # Find best bucket by minimizing aspect ratio difference
        best_bucket = min(self.buckets, key=lambda b: abs(b["aspect_ratio"] - image_aspect))

        return best_bucket

    def get_bucket_by_resolution(self, width: int, height: int) -> Optional[Dict]:
        """Get bucket by exact resolution."""
        return self.buckets_by_resolution.get((width, height))

    def get_bucket_by_id(self, bucket_id: int) -> Dict:
        """Get bucket by ID."""
        return self.buckets[bucket_id]

    def get_all_buckets(self) -> List[Dict]:
        """Get all buckets."""
        return self.buckets

    def resize_and_crop(
        self,
        image,  # PIL Image or numpy array
        target_width: int,
        target_height: int,
        crop_mode: str = "center",
    ) -> Tuple:
        """
        Resize and crop image to target resolution.

        Args:
            image: PIL Image or numpy array
            target_width: Target width
            target_height: Target height
            crop_mode: 'center', 'random', or 'smart'

        Returns:
            (resized_image, crop_offset_x, crop_offset_y)
        """
        from PIL import Image

        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        orig_width, orig_height = image.size

        # Calculate scale to cover target area
        scale = max(target_width / orig_width, target_height / orig_height)

        # Resize
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        image = image.resize((new_width, new_height), Image.LANCZOS)

        # Calculate crop position
        if crop_mode == "center":
            left = (new_width - target_width) // 2
            top = (new_height - target_height) // 2
        elif crop_mode == "random":
            import random

            left = random.randint(0, max(0, new_width - target_width))
            top = random.randint(0, max(0, new_height - target_height))
        else:  # smart crop (focus on center of mass)
            # Simple smart crop: focus on center with slight randomness
            import random

            center_x = new_width // 2
            center_y = new_height // 2
            left = max(0, min(new_width - target_width, center_x - target_width // 2 + random.randint(-50, 50)))
            top = max(0, min(new_height - target_height, center_y - target_height // 2 + random.randint(-50, 50)))

        # Crop
        right = left + target_width
        bottom = top + target_height
        image = image.crop((left, top, right, bottom))

        return image, (left, top)

    def get_dynamic_batch_size(
        self,
        resolution: Tuple[int, int],
        base_batch_size: int = 32,
        base_resolution: Tuple[int, int] = (512, 512),
    ) -> int:
        """
        Calculate dynamic batch size based on resolution.
        Larger images get smaller batches to maintain GPU memory usage.

        Args:
            resolution: (width, height)
            base_batch_size: Batch size for base resolution
            base_resolution: Reference resolution

        Returns:
            Recommended batch size
        """
        width, height = resolution
        base_w, base_h = base_resolution

        # Calculate relative memory usage (approximately quadratic with resolution)
        resolution_ratio = (width * height) / (base_w * base_h)

        # Scale batch size inversely with resolution
        dynamic_batch_size = int(base_batch_size / resolution_ratio)

        # Ensure minimum batch size
        return max(1, dynamic_batch_size)

    @classmethod
    def from_preset(cls, preset: str, quantization: int = 64) -> "MultiTierBucketCalculator":
        """
        Create calculator from a named preset.

        Args:
            preset: One of '256p', '512p', '768p', '1024p', '1536p'
            quantization: Resolution quantization

        Returns:
            MultiTierBucketCalculator instance
        """
        if preset not in cls.RESOLUTION_PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {list(cls.RESOLUTION_PRESETS.keys())}")
        return cls(quantization=quantization, max_pixels=cls.RESOLUTION_PRESETS[preset])

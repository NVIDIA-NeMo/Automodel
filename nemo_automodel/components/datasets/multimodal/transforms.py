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
#
# Includes Apache-2.0 code adapted from ByteDance-Seed/Bagel. Upstream references:
#   https://github.com/bytedance-seed/BAGEL
#   data/transforms.py
# Upstream copyright: Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Class names, defaults, and arithmetic match the expected preprocessing
# output.

"""Image transforms for BAGEL's NaViT-style aspect-ratio-aware resize."""

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _require_torchvision():
    try:
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as F
    except ModuleNotFoundError as exc:
        if exc.name != "torchvision":
            raise
        raise ModuleNotFoundError(
            "torchvision is required to use BAGEL multimodal image transforms. "
            "Install the appropriate optional dependency before constructing these datasets."
        ) from exc
    return transforms, InterpolationMode, F


class MaxLongEdgeMinShortEdgeResize(torch.nn.Module):
    """Resize so longest/shortest edges stay within bounds and both edges are stride-divisible.

    Args:
        max_size: Maximum size for the longest edge.
        min_size: Minimum size for the shortest edge.
        stride: Value both edges must be divisible by (ViT patch size).
        max_pixels: Maximum total pixels for the full image.
        interpolation: Torchvision interpolation mode (default bicubic).
        antialias: Whether to apply antialiasing.
    """

    def __init__(
        self,
        max_size: int,
        min_size: int,
        stride: int,
        max_pixels: int,
        interpolation=None,
        antialias=True,
    ):
        super().__init__()
        if interpolation is None:
            _, interpolation_mode, _ = _require_torchvision()
            interpolation = interpolation_mode.BICUBIC
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.max_pixels = max_pixels
        self.interpolation = interpolation
        self.antialias = antialias

    def _make_divisible(self, value, stride):
        return max(stride, int(round(value / stride) * stride))

    def _apply_scale(self, width, height, scale):
        new_width = round(width * scale)
        new_height = round(height * scale)
        new_width = self._make_divisible(new_width, self.stride)
        new_height = self._make_divisible(new_height, self.stride)
        return new_width, new_height

    def forward(self, img, img_num=1):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size

        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = self._apply_scale(width, height, scale)

        if new_width * new_height > self.max_pixels / img_num:
            scale = self.max_pixels / img_num / (new_width * new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        _, _, functional = _require_torchvision()
        return functional.resize(img, (new_height, new_width), self.interpolation, antialias=self.antialias)


class ImageTransform:
    """Full BAGEL image transform: resize + to_tensor + normalize.

    Used for both ViT input (stride=14) and VAE input (stride=16, via
    separate instances). ``stride`` is exposed as an attribute so the
    dataset can compute patch counts without knowing the transform class.
    """

    def __init__(
        self,
        max_image_size,
        min_image_size,
        image_stride,
        max_pixels=14 * 14 * 9 * 1024,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    ):
        tv_transforms, _, _ = _require_torchvision()
        self.stride = image_stride

        self.resize_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size,
            min_size=min_image_size,
            stride=image_stride,
            max_pixels=max_pixels,
        )
        self.to_tensor_transform = tv_transforms.ToTensor()
        self.normalize_transform = tv_transforms.Normalize(mean=list(image_mean), std=list(image_std), inplace=True)

    def __call__(self, img, img_num=1):
        img = self.resize_transform(img, img_num=img_num)
        img = self.to_tensor_transform(img)
        img = self.normalize_transform(img)
        return img

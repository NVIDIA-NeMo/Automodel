# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Spatial image assembly shared by native and streaming VLM loaders."""

from collections.abc import Sequence

import torch


def pad_image_tensors(values: Sequence[torch.Tensor]) -> torch.Tensor:
    """Zero-pad processor images to a common spatial size without resizing.

    Args:
        values: Image tensors [images, channels, height, width], with matching
            channel counts. Original unpadded sizes travel in `image_sizes`.

    Returns:
        Concatenated images [total_images, channels, max_height, max_width].
        Pixels stay at their original top-left coordinates; inputs are unchanged.
    """
    if not values or any(value.ndim != 4 for value in values):
        raise ValueError("Image padding requires nonempty [images, channels, height, width] tensors")
    height = max(value.shape[-2] for value in values)
    width = max(value.shape[-1] for value in values)
    return torch.cat(
        [torch.nn.functional.pad(value, (0, width - value.shape[-1], 0, height - value.shape[-2])) for value in values],
        dim=0,
    )

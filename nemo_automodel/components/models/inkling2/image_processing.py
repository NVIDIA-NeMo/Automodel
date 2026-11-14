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

# The image preprocessing follows the HuggingFace Inkling 2 image processor (Apache-2.0).
# Copyright 2026 the HuggingFace Team. All rights reserved.
# Source: https://github.com/huggingface/transformers/pull/50682
# Revision: 109a7d19bf667168837f32929e00b791e195d14d
# Original file: src/transformers/models/inkling2/image_processing_inkling2.py

"""Native image preprocessing for the Inkling 2 vision tower.

Code provenance: the preprocessing follows ``Inkling2ImageProcessor`` in Hugging Face Transformers
(https://github.com/huggingface/transformers, commit 109a7d19bf667168837f32929e00b791e195d14d,
``src/transformers/models/inkling2/image_processing_inkling2.py``, Apache-2.0, Copyright 2026 the
HuggingFace Team). The RGB conversion is implemented here in full because the Transformers 5.18.0
backend converts PIL images before this processor could composite their alpha channel.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from transformers.image_processing_backends import TorchvisionBackend
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    ImageType,
    SizeDict,
    get_image_type,
    infer_channel_dimension_format,
)
from transformers.utils import TensorType

from nemo_automodel.shared.import_utils import safe_import

_, tvF = safe_import(
    "torchvision.transforms.v2.functional",
    msg="Inkling 2 image preprocessing requires torchvision.",
)


def _composite_on_checkerboard(image: torch.Tensor) -> torch.Tensor:
    """Composite an RGBA image onto the reference gray checkerboard.

    Args:
        image: Tensor of shape ``[4, height, width]`` with values on the 0-255 scale.

    Returns:
        uint8 tensor of shape ``[3, height, width]``.
    """
    height, width = image.shape[-2:]
    rows = torch.arange(height, device=image.device)[:, None] // 8
    columns = torch.arange(width, device=image.device)[None, :] // 8
    background = torch.where((rows + columns) % 2 == 0, 204, 153)
    rgba = image.to(torch.int32)
    alpha = rgba[3:4]
    return ((rgba[:3] * alpha + background * (255 - alpha) + 127) // 255).to(torch.uint8)


class Inkling2ImageProcessor(TorchvisionBackend):
    """Pad each image to whole ``size`` tiles and cut it into one tile per language-model token.

    Every tile covers ``spatial_merge_size x spatial_merge_size`` native vision patches. Images are
    not resized: an image of ``H x W`` pixels becomes ``ceil(H / 40) x ceil(W / 40)`` tokens with the
    released ``size`` of 40 pixels.
    """

    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    do_convert_rgb = True
    do_rescale = True
    do_normalize = True
    size = {"height": 40, "width": 40}
    model_input_names = ["pixel_values", "image_grid_thw"]

    def process_image(
        self,
        image: ImageInput,
        do_convert_rgb: bool | None = None,
        input_data_format: str | ChannelDimension | None = None,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Convert one PIL, NumPy or torch image to a channels-first tensor.

        With ``do_convert_rgb``, PIL images with transparency, and 4-channel arrays or tensors, are
        composited onto a checkerboard and 1-channel images are repeated to RGB.

        Args:
            image: A PIL image, or an array or tensor of shape ``[height, width]``,
                ``[channels, height, width]`` or ``[height, width, channels]`` on the 0-255 scale.
            do_convert_rgb: Whether to convert the image to three channels.
            input_data_format: Channel layout of an array or tensor input; inferred when ``None``.
            device: Optional device for the returned tensor.
            **kwargs: Other image-processor arguments, unused here.

        Returns:
            Tensor of shape ``[channels, height, width]``.
        """
        del kwargs
        image_type = get_image_type(image)
        if image_type == ImageType.PIL:
            if do_convert_rgb:
                image = image.convert("RGBA" if image.has_transparency_data else "RGB")
            tensor = tvF.pil_to_tensor(image)
            input_data_format = ChannelDimension.FIRST
        elif image_type == ImageType.NUMPY:
            tensor = torch.from_numpy(image)
        elif image_type == ImageType.TORCH:
            tensor = image
        else:
            raise ValueError(f"Unsupported input image type {image_type}")

        if tensor.ndim == 2:
            tensor = tensor[None]
        elif input_data_format is None:
            input_data_format = infer_channel_dimension_format(tensor, num_channels=(1, 3, 4))
        if input_data_format == ChannelDimension.LAST:
            tensor = tensor.permute(2, 0, 1)

        if do_convert_rgb:
            if tensor.shape[0] == 4:
                tensor = _composite_on_checkerboard(tensor)
            elif tensor.shape[0] == 1:
                tensor = tensor.expand(3, -1, -1)
        tensor = tensor.contiguous()
        return tensor.to(device) if device is not None else tensor

    def _preprocess(
        self,
        images: list[torch.Tensor],
        size: SizeDict,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        **kwargs: Any,
    ) -> BatchFeature:
        """Tile channels-first images.

        Args:
            images: Tensors of shape ``[channels, height, width]`` on the 0-255 scale.
            size: Tile height and width in pixels.
            do_rescale: Whether to multiply pixels by ``rescale_factor``.
            rescale_factor: Pixel rescaling multiplier.
            do_normalize: Whether to normalize channels.
            image_mean: Per-channel normalization means.
            image_std: Per-channel normalization standard deviations.
            return_tensors: Requested output tensor framework.
            **kwargs: Other image-processor arguments, unused here.

        Returns:
            A batch with ``pixel_values`` of shape ``[tokens, tile_height, tile_width, channels]``
            (tiles in raster order within each image, images back to back) and ``image_grid_thw``
            of shape ``[images, 3]`` holding each image's ``(1, rows, columns)`` in tiles.
        """
        del kwargs
        patches, image_grid_thw = [], []
        for image in images:
            channels, height, width = image.shape
            rows, columns = math.ceil(height / size.height), math.ceil(width / size.width)
            # Pad to whole tiles with -1 on the 0-255 scale, then cut one tile per image token.
            pad_size = SizeDict(height=rows * size.height, width=columns * size.width)
            image = self.pad([image.float()], pad_size=pad_size, fill_value=-1.0)[0]
            image = self.rescale_and_normalize(image, do_rescale, rescale_factor, do_normalize, image_mean, image_std)
            image = image.reshape(channels, rows, size.height, columns, size.width).permute(1, 3, 2, 4, 0)
            patches.append(image.flatten(0, 1))
            image_grid_thw.append([1, rows, columns])
        return BatchFeature(
            data={"pixel_values": torch.cat(patches), "image_grid_thw": torch.tensor(image_grid_thw)},
            tensor_type=return_tensors,
        )


__all__ = ["Inkling2ImageProcessor"]

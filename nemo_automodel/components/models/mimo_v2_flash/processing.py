# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""MiMo-V2 image preprocessing aligned with the official serving/training stack.

The stock ``Qwen2VLImageProcessor`` that ``AutoProcessor`` resolves for MiMo-V2
checkpoints resizes with PIL/torchvision bicubic (antialiased, uint8-round-trip)
and normalizes in 0-1 space. That is not what the model sees in production:
Xiaomi's own processor in the reference serving stack (sglang
``MiMoProcessor``, contributed as MiMo day-0 support and used for the RL
rollouts of MiMo-V2.6-*-RL) resizes with torch bilinear on 0-255 floats,
normalizes in 0-255 space, and flattens RGBA onto white instead of dropping
alpha. Grid geometry (``smart_resize`` targets, patch layout, token counts) is
identical for images whose short side is at least the patch factor; below that
the reference upscales the short side to the factor before rounding while HF
clamps each side independently, so a few tiny inputs (e.g. 16x192) get
different grids. For every other image only pixel values differ, so the
mismatch is a silent distribution shift of up to ~0.2 std per pixel.

Normalization statistics are read from the checkpoint's own
``preprocessor_config.json`` (via the stock image processor built by
``AutoProcessor``), never hard-coded: MiMo-V2.6 checkpoints declare ImageNet
values (Xiaomi commit "Align preprocessor config to training values"), while
MiMo-V2.5 declares CLIP values, and each is what that family trained with.
"""

import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    BaseImageProcessor,
    BatchFeature,
    ImagesKwargs,
    ProcessorMixin,
)

from nemo_automodel.components.models.mimo_v2_flash.config import MiMoV2Config, MiMoV2FlashConfig

logger = logging.getLogger(__name__)

ImageInput = Image.Image | str


class MimoV2ImageProcessorKwargs(ImagesKwargs, total=False):
    """Per-call keyword arguments understood by :class:`MimoV2ImageProcessor`.

    ``ProcessorMixin`` routes a flat kwarg into ``images_kwargs`` only when the
    key is declared in the image processor's ``valid_kwargs`` TypedDict;
    undeclared keys are dropped with a one-time warning. Declaring the pixel
    bounds here keeps per-call overrides such as ``processor(...,
    max_pixels=...)`` working, mirroring ``Qwen2VLImageProcessor``.
    """

    min_pixels: int
    max_pixels: int


def smart_resize(height: int, width: int, *, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Compute resize targets with the sglang ``MiMoProcessor`` semantics.

    Same algorithm as HF ``smart_resize`` except: a short side below ``factor``
    is upscaled to ``factor`` first (that branch also skips the aspect-ratio
    guard), and degenerate shrink results are rejected instead of clamped.

    Args:
        height: Source image height in pixels.
        width: Source image width in pixels.
        factor: Both output sides are rounded to a multiple of this.
        min_pixels: Lower bound on output area in pixels.
        max_pixels: Upper bound on output area in pixels.

    Returns:
        ``(resized_height, resized_width)`` in pixels.
    """
    if min(height, width) < factor:
        scale = factor / min(height, width)
        height = int(round(height * scale))
        width = int(round(width * scale))
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    if h_bar < factor or w_bar < factor:
        raise ValueError(
            f"image {height}x{width} cannot fit max_pixels={max_pixels}: resize target "
            f"({h_bar}, {w_bar}) is smaller than factor={factor}"
        )
    return h_bar, w_bar


def _to_rgb(pil_image: Image.Image) -> Image.Image:
    """Convert to RGB with the sglang semantics: RGBA is flattened onto white.

    EXIF orientation is deliberately not applied, matching the reference
    serving stack so training and serving see identical pixels.
    """
    if pil_image.mode == "RGBA":
        background = Image.new("RGB", pil_image.size, (255, 255, 255))
        background.paste(pil_image, mask=pil_image.split()[3])
        return background
    return pil_image.convert("RGB")


class MimoV2ImageProcessor(BaseImageProcessor):
    """Drop-in image processor for MiMo-V2 checkpoints matching sglang pixels.

    Implements the pixel path of the reference ``MiMoProcessor``: load as RGB
    (RGBA flattened onto white), cast to 0-255 float, resize with
    ``F.interpolate(mode="bilinear", align_corners=False)``, standardize with
    mean/std expressed in 0-255 scale, duplicate the frame along the temporal
    axis, and emit patches in the Qwen2VL merge-block order. Geometry
    (``smart_resize`` targets and patch order) matches ``Qwen2VLImageProcessor``
    except for the sub-``factor`` short-side branch described in the module
    docstring; only resampling and normalization arithmetic differ otherwise.

    Inheriting ``BaseImageProcessor`` provides the save/reload contract
    (``save_pretrained``/``from_pretrained``) that ``ProcessorMixin`` relies on
    when a checkpoint is written, and is required for registration with
    ``AutoImageProcessor`` (see :func:`_register_with_auto_image_processor`) so
    a saved processor directory reloads through ``AutoProcessor.from_pretrained``.
    Only JSON-serializable constructor arguments may be stored as instance
    attributes so ``to_dict()`` stays serializable.
    """

    model_input_names = ["pixel_values", "image_grid_thw"]
    valid_kwargs = MimoV2ImageProcessorKwargs

    def __init__(
        self,
        *,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        merge_size: int = 2,
        min_pixels: int = 8192,
        max_pixels: int = 8388608,
        image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        **kwargs,
    ) -> None:
        # Extra kwargs (e.g. ``image_processor_type`` on reload) are kept by the
        # base class so a save -> load round trip preserves them.
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_mean = image_mean
        self.image_std = image_std

    @property
    def size(self) -> dict[str, int]:
        """Pixel bounds in the HF ``size`` mapping shape used by token estimators."""
        return {"shortest_edge": self.min_pixels, "longest_edge": self.max_pixels}

    def fetch_images(self, image_url_or_urls: ImageInput | list) -> Image.Image | list:
        """Normalize raw image inputs to PIL images.

        Required by ``ProcessorMixin.prepare_inputs_layout`` (transformers >= 5.18)
        before the images reach :meth:`__call__`. Mirrors
        ``ImageProcessingMixin.fetch_images`` for the supported input types;
        RGB conversion stays in :meth:`__call__`.
        """
        if isinstance(image_url_or_urls, (list, tuple)):
            return [self.fetch_images(item) for item in image_url_or_urls]
        if isinstance(image_url_or_urls, str):
            return Image.open(image_url_or_urls)
        if isinstance(image_url_or_urls, Image.Image):
            return image_url_or_urls
        raise TypeError(f"expected a PIL image or a local path, got {type(image_url_or_urls).__name__}")

    def get_number_of_image_patches(self, height: int, width: int, images_kwargs: dict | None = None) -> int:
        """Number of 16x16 patches for one image, before the merge_size**2 reduction.

        Called by ``Qwen2_5_VLProcessor`` to expand ``<|image_pad|>`` placeholders.

        Args:
            height: Source image height in pixels.
            width: Source image width in pixels.
            images_kwargs: Optional per-call ``min_pixels``/``max_pixels`` overrides.

        Returns:
            ``grid_h * grid_w`` for the resized image.
        """
        min_pixels, max_pixels = self._pixel_bounds(images_kwargs)
        h_bar, w_bar = smart_resize(
            height, width, factor=self.patch_size * self.merge_size, min_pixels=min_pixels, max_pixels=max_pixels
        )
        return (h_bar // self.patch_size) * (w_bar // self.patch_size)

    def _pixel_bounds(self, images_kwargs: dict | None) -> tuple[int, int]:
        images_kwargs = images_kwargs or {}
        return (
            int(images_kwargs.get("min_pixels") or self.min_pixels),
            int(images_kwargs.get("max_pixels") or self.max_pixels),
        )

    def _load(self, image: ImageInput) -> torch.Tensor:
        """Load one image as a float32 tensor of shape [channels, height, width], 0-255 scale."""
        if isinstance(image, str):
            image = Image.open(image)
        if not isinstance(image, Image.Image):
            raise TypeError(f"expected a PIL image or a local path, got {type(image).__name__}")
        array = np.asarray(_to_rgb(image))
        return torch.from_numpy(array.copy()).permute(2, 0, 1).float()

    def _process_one(self, image: ImageInput, *, min_pixels: int, max_pixels: int) -> tuple[torch.Tensor, list[int]]:
        """Run one image through the sglang pixel path.

        Args:
            image: PIL image or local path.

        Returns:
            Tuple of ``patches`` of shape [grid_t*grid_h*grid_w,
            channels*temporal_patch_size*patch_size**2] in the Qwen2VL
            merge-block order, and ``[grid_t, grid_h, grid_w]``.
        """
        tensor = self._load(image)
        _, height, width = tensor.shape
        h_bar, w_bar = smart_resize(
            height, width, factor=self.patch_size * self.merge_size, min_pixels=min_pixels, max_pixels=max_pixels
        )
        resized = F.interpolate(tensor.unsqueeze(0), size=(h_bar, w_bar), mode="bilinear", align_corners=False).squeeze(
            0
        )
        # 0-255-space statistics, matching the reference stack. Computed per
        # call so the instance stays free of non-serializable cached tensors.
        mean255 = torch.tensor(self.image_mean, dtype=torch.float32).view(-1, 1, 1) * 255.0
        std255 = torch.tensor(self.image_std, dtype=torch.float32).view(-1, 1, 1) * 255.0
        standardized = (resized - mean255) / std255
        frames = standardized.unsqueeze(0).repeat(self.temporal_patch_size, 1, 1, 1)

        grid_t = 1
        grid_h, grid_w = h_bar // self.patch_size, w_bar // self.patch_size
        merge = self.merge_size
        patches = frames.contiguous().view(
            grid_t,
            self.temporal_patch_size,
            3,
            grid_h // merge,
            merge,
            self.patch_size,
            grid_w // merge,
            merge,
            self.patch_size,
        )
        patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
        patches = patches.view(grid_t * grid_h * grid_w, 3 * self.temporal_patch_size * self.patch_size**2)
        return patches, [grid_t, grid_h, grid_w]

    def __call__(
        self,
        images: ImageInput | list[ImageInput] | list[list[ImageInput]],
        *,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        return_tensors: str | None = "pt",
        **kwargs,
    ) -> BatchFeature:
        """Preprocess images into MiMo-V2 vision-tower inputs.

        Args:
            images: One PIL image / local path, a flat list of them, or a
                per-sample nested list (``[[img1], [img2], ...]``) as produced
                by ``ProcessorMixin.apply_chat_template`` on batched
                conversations. Nested input is flattened one level, preserving
                order.
            min_pixels: Per-call override of the instance pixel bound.
            max_pixels: Per-call override of the instance pixel bound.
            return_tensors: Kept for ``Qwen2VLImageProcessor`` interface
                compatibility; outputs are always torch tensors.
            **kwargs: Ignored, for the same interface compatibility.

        Returns:
            ``BatchFeature`` with ``pixel_values`` of shape [total_patches,
            channels*temporal_patch_size*patch_size**2] (float32) and
            ``image_grid_thw`` of shape [num_images, 3] (int64, ``[t, h, w]``
            per image in patch units).
        """
        if isinstance(images, (Image.Image, str)):
            images = [images]
        flat_images = []
        for image in images:
            if isinstance(image, (list, tuple)):
                flat_images.extend(image)
            else:
                flat_images.append(image)
        min_pixels = min_pixels or self.min_pixels
        max_pixels = max_pixels or self.max_pixels
        patches_list, grids = [], []
        for image in flat_images:
            patches, grid = self._process_one(image, min_pixels=min_pixels, max_pixels=max_pixels)
            patches_list.append(patches)
            grids.append(grid)
        return BatchFeature(
            data={
                "pixel_values": torch.cat(patches_list, dim=0),
                "image_grid_thw": torch.tensor(grids, dtype=torch.long),
            }
        )


def _register_with_auto_image_processor() -> None:
    """Register the class so a saved processor directory reloads through HF auto classes.

    ``ProcessorMixin.save_pretrained`` writes
    ``"image_processor_type": "MimoV2ImageProcessor"`` into the checkpoint's
    ``processor_config.json``; without this mapping ``AutoProcessor.from_pretrained``
    on that directory raises ``Unrecognized image processor``.
    """
    for config_cls in (MiMoV2FlashConfig, MiMoV2Config):
        AutoImageProcessor.register(
            config_cls,
            image_processor_classes={"torchvision": MimoV2ImageProcessor, "pil": MimoV2ImageProcessor},
            exist_ok=True,
        )


_register_with_auto_image_processor()


def build_mimo_v2_processor(pretrained_model_name_or_path: str, **kwargs: Any) -> ProcessorMixin:
    """Build the MiMo-V2 chat processor with sglang-aligned image preprocessing.

    Loads the checkpoint's stock processor (tokenizer, chat template, video
    and audio handling stay exactly as released) and replaces only the image
    processor with :class:`MimoV2ImageProcessor`, initialized from the
    checkpoint's own ``preprocessor_config.json`` values.

    Args:
        pretrained_model_name_or_path: HF model id or local checkpoint path.
        **kwargs: Forwarded to ``AutoProcessor.from_pretrained``, e.g.
            ``trust_remote_code`` or ``max_pixels`` (recipes use the latter to
            cap image token counts).

    Returns:
        The stock MiMo-V2 processor whose ``image_processor`` is a
        :class:`MimoV2ImageProcessor`.
    """
    processor = AutoProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs)
    ip = processor.image_processor
    processor.image_processor = MimoV2ImageProcessor(
        patch_size=ip.patch_size,
        temporal_patch_size=ip.temporal_patch_size,
        merge_size=ip.merge_size,
        min_pixels=ip.size["shortest_edge"],
        max_pixels=ip.size["longest_edge"],
        image_mean=tuple(ip.image_mean),
        image_std=tuple(ip.image_std),
    )
    logger.info(
        "MiMo-V2 processor: image preprocessing switched to the sglang-aligned path "
        "(min_pixels=%d, max_pixels=%d, mean=%s, std=%s)",
        ip.size["shortest_edge"],
        ip.size["longest_edge"],
        ip.image_mean,
        ip.image_std,
    )
    return processor

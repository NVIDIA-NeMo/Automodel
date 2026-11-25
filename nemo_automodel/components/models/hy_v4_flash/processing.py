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

"""Image-text processor of Hy4-flash.

The released chat template renders each image as ``<image_start><image_pad><image_end>``; this processor expands the
single pad into ``(h / merge) * (w / merge + 1)`` pads (one extra slot per merged row for the newline feature) and
produces the flat patches the vision tower consumes. Preprocessing follows the transformers reference
(``HYV4VLImageProcessor``): ``smart_resize`` to multiples of ``patch_size * merge``, bicubic resize on the uint8 tensor,
rescale by 1/255, normalize with mean/std 0.5, row-major patches. The released ``processor_config`` names LANCZOS as
the resampling filter, but the reference (and the original implementation) always resize bicubically; so does this code.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Required, TypeAlias, TypedDict

import numpy as np
import torch
from transformers import AutoTokenizer, BatchFeature, PreTrainedTokenizerFast, ProcessorMixin
from transformers.utils import cached_file

from nemo_automodel.components.models.hy_v4_flash.config import HYV4FlashConfig
from nemo_automodel.shared.import_utils import safe_import

_, Image = safe_import("PIL.Image")
_, _tv_transforms = safe_import("torchvision.transforms.v2")

ImageSource: TypeAlias = "Image.Image | str | Path | bytes"
TokenizerOption: TypeAlias = "bool | int | str | list[str] | None"


class ImageUrl(TypedDict):
    """``{"url": ...}`` form of an image reference."""

    url: str


class ChatContentBlock(TypedDict, total=False):
    """One text or image block of a chat message (the Hugging Face multimodal chat format)."""

    type: Required[Literal["text", "image", "image_url"]]
    text: str
    image: ImageSource
    path: str
    image_url: str | ImageUrl


class ChatMessage(TypedDict, total=False):
    """One chat message with string content or an ordered list of content blocks."""

    role: Required[Literal["system", "user", "assistant"]]
    content: Required[str | list[ChatContentBlock]]
    reasoning_content: str
    reasoning: str


IMAGE_PAD_TOKEN = "<｜hy_image▁pad:opensource｜>"
TEXT, IMAGE = 0, 1


@dataclass(frozen=True)
class HYV4FlashImageSettings:
    """Preprocessing constants of the released ``processor_config.json``.

    Attributes:
        patch_size: Spatial patch size of the vision tower.
        merge_size: Spatial merge factor of the projector.
        min_pixels: Lower bound of the resized area.
        max_pixels: Upper bound of the resized area.
        mean: Per-channel normalization mean.
        std: Per-channel normalization standard deviation.
    """

    patch_size: int = 16
    merge_size: int = 2
    min_pixels: int = 4096
    max_pixels: int = 4194304
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)


class ImageProcessorConfig(TypedDict):
    """The fields of the ``image_processor`` block of the released ``processor_config.json`` that are consumed."""

    patch_size: int
    merge_size: int
    min_pixels: int
    max_pixels: int
    image_mean: list[float]
    image_std: list[float]


class ProcessorConfig(TypedDict):
    """``processor_config.json`` as this processor writes it."""

    processor_class: str
    image_token_id: int
    image_processor: ImageProcessorConfig


def image_settings_from_processor_config(block: ImageProcessorConfig) -> HYV4FlashImageSettings:
    """Preprocessing constants of a checkpoint's ``processor_config.json`` ``image_processor`` block."""
    mean, std = block["image_mean"], block["image_std"]
    return HYV4FlashImageSettings(
        patch_size=block["patch_size"],
        merge_size=block["merge_size"],
        min_pixels=block["min_pixels"],
        max_pixels=block["max_pixels"],
        mean=(mean[0], mean[1], mean[2]),
        std=(std[0], std[1], std[2]),
    )


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Round both sides to multiples of ``factor`` while keeping the area inside ``[min_pixels, max_pixels]``."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def image_grid(height: int, width: int, settings: HYV4FlashImageSettings) -> tuple[int, int]:
    """Patch grid ``(grid_h, grid_w)`` of an image of the given pixel size."""
    resized_h, resized_w = smart_resize(
        height, width, settings.patch_size * settings.merge_size, settings.min_pixels, settings.max_pixels
    )
    return resized_h // settings.patch_size, resized_w // settings.patch_size


def num_image_tokens(grid_h: int, grid_w: int, merge: int) -> int:
    """Placeholder count of one image: ``(h / merge) * (w / merge + 1)`` (one newline slot per merged row)."""
    return (grid_h // merge) * (grid_w // merge + 1)


def preprocess_image(image: Image.Image, settings: HYV4FlashImageSettings) -> tuple[torch.Tensor, tuple[int, int]]:
    """Resize, normalize and patchify one image.

    Args:
        image: PIL image (converted to RGB).
        settings: Preprocessing constants.

    Returns:
        ``(patches, (grid_h, grid_w))`` with ``patches`` of shape [grid_h * grid_w, 3 * patch_size * patch_size] in
        row-major patch order.
    """
    rgb = np.asarray(image.convert("RGB"))
    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1)
    grid_h, grid_w = image_grid(tensor.shape[1], tensor.shape[2], settings)
    size = [grid_h * settings.patch_size, grid_w * settings.patch_size]
    resized = _tv_transforms.functional.resize(
        tensor, size, interpolation=_tv_transforms.InterpolationMode.BICUBIC, antialias=True
    )
    scaled = resized.float() * (1.0 / 255.0)
    mean = torch.tensor(settings.mean).view(3, 1, 1)
    std = torch.tensor(settings.std).view(3, 1, 1)
    normalized = (scaled - mean) / std
    patch = settings.patch_size
    patches = normalized.reshape(3, grid_h, patch, grid_w, patch).permute(1, 3, 0, 2, 4)
    return patches.reshape(grid_h * grid_w, 3 * patch * patch), (grid_h, grid_w)


def _open_image(value: Image.Image | str | Path | bytes) -> Image.Image:
    """Decode an image source (PIL image, path or encoded bytes)."""
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, (bytes, bytearray)):
        import io

        return Image.open(io.BytesIO(value))
    return Image.open(value)


def training_reasoning_effort(messages: Sequence[ChatMessage]) -> Literal["no_think", "high"]:
    """Reasoning mode of a complete (training) conversation, by the rule of the official fine-tuning scripts.

    The released chat template defaults to ``reasoning_effort='high'``, which is its inference default. The official
    SFT code instead treats a sample without a reasoning field as fast-thinking data and renders it with
    ``'no_think'``; only samples whose assistant turns carry ``reasoning_content`` (or ``reasoning``) use ``'high'``.
    Rendering answer-only data in ``'high'`` mode would train (and score) answers after an empty think block in the
    mode where the model expects to reason first.

    Args:
        messages: Chat messages of one conversation.

    Returns:
        ``"high"`` when an assistant message has non-empty reasoning, else ``"no_think"``.
    """
    for message in messages:
        if message.get("role") == "assistant" and (message.get("reasoning_content") or message.get("reasoning")):
            return "high"
    return "no_think"


class ProcessorOptions(TypedDict, total=False):
    """Tokenization options forwarded by the collator through ``processor_kwargs``."""

    padding: bool | str
    truncation: bool
    max_length: int


class HYV4FlashProcessor(ProcessorMixin):
    """Expand image placeholders and build vision inputs for the Hy4-flash chat format.

    Args:
        tokenizer: Fast tokenizer of the checkpoint (carries the chat template and the image tokens).
        image_token_id: Id of the image pad token (``config.image_token_id``).
        settings: Image preprocessing constants.
    """

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        image_token_id: int = 120120,
        settings: HYV4FlashImageSettings | None = None,
    ) -> None:
        self.image_token_id = image_token_id
        self.image_token = IMAGE_PAD_TOKEN
        self.settings = settings or HYV4FlashImageSettings()
        known = tokenizer.convert_tokens_to_ids(IMAGE_PAD_TOKEN)
        if known is not None and known != tokenizer.unk_token_id and known != image_token_id:
            raise ValueError(f"tokenizer image pad id {known} disagrees with image_token_id {image_token_id}")
        super().__init__(tokenizer=tokenizer, chat_template=tokenizer.chat_template)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, **kwargs: TokenizerOption
    ) -> HYV4FlashProcessor:
        """Load the tokenizer, the image token id and the image constants of a checkpoint."""
        loading = {
            key: kwargs[key]
            for key in ("cache_dir", "revision", "token", "trust_remote_code", "local_files_only")
            if key in kwargs
        }
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **loading)
        config = HYV4FlashConfig.from_pretrained(pretrained_model_name_or_path, **loading)
        hub = {key: value for key, value in loading.items() if key != "trust_remote_code"}
        with open(cached_file(pretrained_model_name_or_path, "processor_config.json", **hub)) as handle:
            settings = image_settings_from_processor_config(json.load(handle)["image_processor"])
        return cls(tokenizer=tokenizer, image_token_id=config.image_token_id, settings=settings)

    def to_dict(self) -> ProcessorConfig:
        """``processor_config.json`` content: the ``image_processor`` block that :meth:`from_pretrained` reads."""
        settings = self.settings
        return ProcessorConfig(
            processor_class=type(self).__name__,
            image_token_id=self.image_token_id,
            image_processor=ImageProcessorConfig(
                patch_size=settings.patch_size,
                merge_size=settings.merge_size,
                min_pixels=settings.min_pixels,
                max_pixels=settings.max_pixels,
                image_mean=list(settings.mean),
                image_std=list(settings.std),
            ),
        )

    # ------------------------------------------------------------------ tokenization
    def __call__(
        self,
        text: str | list[str],
        images: Image.Image | list[Image.Image] | list[list[Image.Image]] | None = None,
        *,
        return_tensors: str | None = None,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs: TokenizerOption,
    ) -> BatchFeature:
        """Tokenize rendered prompts, expanding every image pad.

        Args:
            text: Rendered prompt(s), each containing one image pad token per image.
            images: Images grouped per prompt (or one prompt's flat list).
            return_tensors: ``"pt"`` or ``None`` (ragged lists).
            padding: ``False``, ``True`` / ``"longest"`` or ``"max_length"`` (right padding).
            truncation: Whether ``max_length`` removes trailing tokens; cutting through an image raises.
            max_length: Token limit after image expansion.
            **kwargs: Extra tokenizer arguments.

        Returns:
            ``BatchFeature`` with ``input_ids`` / ``attention_mask`` / ``mm_token_type_ids`` of shape [batch, sequence]
            plus ``pixel_values`` [patches, 3 * patch_size**2] and ``image_grid_thw`` [images, 3] when images are kept.
        """
        texts = [text] if isinstance(text, str) else list(text)
        if images is None:
            groups = [[] for _ in texts]
        elif isinstance(images, list) and images and isinstance(images[0], list):
            groups = images
        elif len(texts) == 1:
            groups = [images if isinstance(images, list) else [images]]
        else:
            groups = [[image] for image in images]
        if len(groups) != len(texts):
            raise ValueError("The number of image groups must equal the number of prompts")
        if padding == "max_length" and max_length is None:
            raise ValueError("padding='max_length' requires max_length")
        kwargs.setdefault("add_special_tokens", False)
        tokenized = self.tokenizer(texts, padding=False, **kwargs)
        merge = self.settings.merge_size
        rows, types_rows, patches_all, grids_all = [], [], [], []
        for raw_ids, group in zip(tokenized["input_ids"], groups):
            if raw_ids.count(self.image_token_id) != len(group):
                raise ValueError("The number of image pad tokens must equal the number of supplied images")
            ids, types, spans = [], [], []
            pending = iter(group)
            for token in raw_ids:
                if token != self.image_token_id:
                    ids.append(token)
                    types.append(TEXT)
                    continue
                patches, (grid_h, grid_w) = preprocess_image(_open_image(next(pending)), self.settings)
                count = num_image_tokens(grid_h, grid_w, merge)
                spans.append((len(ids), len(ids) + count, patches, (1, grid_h, grid_w)))
                ids.extend([self.image_token_id] * count)
                types.extend([IMAGE] * count)
            if truncation and max_length is not None:
                if any(start < max_length < end for start, end, _, _ in spans):
                    raise ValueError("max_length truncates an image span")
                ids, types = ids[:max_length], types[:max_length]
            for start, _, patches, grid in spans:
                if start < len(ids):
                    patches_all.append(patches)
                    grids_all.append(grid)
            rows.append(ids)
            types_rows.append(types)
        longest = max(map(len, rows))
        target = max_length if padding == "max_length" else longest
        if target < longest:
            raise ValueError("A sequence exceeds the padding length; enable truncation or increase max_length")
        masks = [[1] * len(row) for row in rows]
        if padding:
            pad_id = self.tokenizer.pad_token_id
            for ids, types, mask in zip(rows, types_rows, masks):
                extra = target - len(ids)
                ids.extend([pad_id] * extra)
                types.extend([TEXT] * extra)
                mask.extend([0] * extra)
        elif return_tensors == "pt" and len({len(row) for row in rows}) != 1:
            raise ValueError("Variable-length tensor batches require padding")
        data = {"input_ids": rows, "attention_mask": masks, "mm_token_type_ids": types_rows}
        if patches_all:
            data["pixel_values"] = torch.cat(patches_all)
            data["image_grid_thw"] = torch.tensor(grids_all, dtype=torch.long)
        return BatchFeature(data=data, tensor_type=return_tensors)

    # ------------------------------------------------------------------ chat rendering
    def _split_images(self, messages: Sequence[ChatMessage]) -> tuple[list[ChatMessage], list[ImageSource]]:
        """Replace image payloads with bare ``{"type": "image"}`` blocks and collect the sources in order."""
        sources, rendered = [], []
        for message in messages:
            content = message.get("content")
            if isinstance(content, list):
                blocks = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") in ("image", "image_url"):
                        value = block.get("image", block.get("path", block.get("image_url")))
                        if isinstance(value, dict) and "url" in value:
                            value = value["url"]
                        if value is None:
                            raise ValueError("image content blocks require an image, path or image_url")
                        sources.append(value)
                        blocks.append({"type": "image"})
                    else:
                        blocks.append(block)
                message = {**message, "content": blocks}
            rendered.append(message)
        return rendered, sources

    def apply_chat_template(
        self,
        conversation: Sequence[ChatMessage] | Sequence[Sequence[ChatMessage]],
        *,
        tokenize: bool = False,
        return_dict: bool = False,
        return_tensors: str | None = None,
        add_generation_prompt: bool = False,
        processor_kwargs: ProcessorOptions | None = None,
        **kwargs: TokenizerOption,
    ) -> str | list[str] | torch.Tensor | BatchFeature:
        """Render conversations with the checkpoint's jinja template and, optionally, tokenize them.

        Args:
            conversation: Chat messages (or a batch of them) with string or text / image content blocks.
            tokenize: Whether to tokenize (expanding images) instead of returning the rendered text.
            return_dict: Whether to return the whole ``BatchFeature``.
            return_tensors: ``"pt"`` or ``None``.
            add_generation_prompt: Append the assistant header.
            processor_kwargs: Padding / truncation arguments for :meth:`__call__`.
            **kwargs: Extra jinja template variables. Without ``reasoning_effort``, a complete conversation
                (``add_generation_prompt=False``, the training case) is rendered in the mode
                :func:`training_reasoning_effort` picks; a generation prompt keeps the template's own default.

        Returns:
            Rendered text(s), ``input_ids``, or the full batch.
        """
        batched = bool(conversation) and isinstance(conversation[0], (list, tuple))
        conversations = conversation if batched else [conversation]
        texts, images = [], []
        for sample in conversations:
            messages, sources = self._split_images(sample)
            template_kwargs = dict(kwargs)
            if "reasoning_effort" not in template_kwargs and not add_generation_prompt:
                template_kwargs["reasoning_effort"] = training_reasoning_effort(messages)
            texts.append(
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_generation_prompt, **template_kwargs
                )
            )
            images.append(sources)
        if not tokenize:
            return texts if batched else texts[0]
        result = self(text=texts, images=images, return_tensors=return_tensors, **(processor_kwargs or {}))
        return result if return_dict else result["input_ids"]

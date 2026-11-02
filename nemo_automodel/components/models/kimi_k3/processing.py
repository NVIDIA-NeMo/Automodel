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

# Copyright 2025-2026 The Moonshot AI Team. All rights reserved.
# Image preprocessing is adapted from the Kimi K3 checkpoint implementation;
# those portions are distributed under the Kimi K3 License in KIMI_K3_LICENSE.

"""Local K3 image processing and the standard Hugging Face processor contract."""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import TypeAlias, TypedDict

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.processing_utils import ProcessorMixin
from transformers.utils import TensorType

from nemo_automodel.shared.import_utils import safe_import, safe_import_from

from .encoding import is_batched_conversation, normalize_xtml_tool_result_messages

_, Image = safe_import("PIL.Image")
_, TikTokenTokenizer = safe_import_from("nemo_automodel.components.models.kimi_k3.tokenization", "TikTokenTokenizer")

_ImageSource: TypeAlias = "Image.Image | str | Path | bytes"
_Conversation: TypeAlias = list[dict[str, object]]


class _BackgroundConfig(TypedDict, total=False):
    pattern: str
    chessboard_square_size: int
    chessboard_square_on_top_left: bool
    chessboard_white_value: int
    chessboard_gray_value: int


class _MediaConfig(TypedDict, total=False):
    patch_size: int
    merge_kernel_size: int
    in_patch_limit: int
    patch_limit_on_one_side: int
    fixed_output_tokens: int | None
    image_mean: list[float]
    image_std: list[float]
    transparent_bg_config: _BackgroundConfig | None
    transparent_bg_fill_stage: str


def _load_image(source: _ImageSource) -> Image.Image:
    """Load an image without discarding alpha before the configured resize stage."""
    if isinstance(source, Image.Image):
        return source
    if isinstance(source, str) and source.startswith("data:"):
        source = base64.b64decode(source.split(",", 1)[1])
    if isinstance(source, bytes):
        source = io.BytesIO(source)
    with Image.open(source) as image:
        return image.copy()


class KimiK3ImageProcessor(BaseImageProcessor):
    """Checkpoint-compatible RGB patches, including resize and alpha compositing.

    Args:
        media_proc_cfg: Image settings from the checkpoint's preprocessor_config.json.
        **kwargs: Hugging Face image-processor metadata.
    """

    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, media_proc_cfg: _MediaConfig | None = None, **kwargs: object) -> None:
        kwargs.pop("auto_map", None)
        super().__init__(**kwargs)
        self.media_proc_cfg = {
            "patch_size": 14,
            "merge_kernel_size": 2,
            "in_patch_limit": 65536,
            "patch_limit_on_one_side": 512,
            "fixed_output_tokens": None,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "transparent_bg_config": None,
            "transparent_bg_fill_stage": "before_resize",
            **(media_proc_cfg or {}),
        }
        cfg = self.media_proc_cfg
        if cfg["fixed_output_tokens"] is not None:
            raise ValueError("K3 image patch merging does not support fixed_output_tokens")
        for key in ("patch_size", "merge_kernel_size", "in_patch_limit", "patch_limit_on_one_side"):
            if cfg[key] <= 0:
                raise ValueError(f"K3 {key} must be positive")
        if cfg["patch_limit_on_one_side"] % cfg["merge_kernel_size"]:
            raise ValueError("K3 patch_limit_on_one_side must be divisible by merge_kernel_size")
        if cfg["transparent_bg_fill_stage"] not in ("before_resize", "after_resize"):
            raise ValueError("K3 transparent_bg_fill_stage must be before_resize or after_resize")
        if len(cfg["image_mean"]) != 3 or len(cfg["image_std"]) != 3 or min(cfg["image_std"]) <= 0:
            raise ValueError("K3 normalization requires three means and three positive standard deviations")

    @property
    def merge_size(self) -> int:
        """Expose spatial merging to the native media-token validator."""
        return self.media_proc_cfg["merge_kernel_size"]

    def _composite(self, image: Image.Image) -> Image.Image:
        """Apply the checkpoint's background to transparent PIL images."""
        cfg = self.media_proc_cfg["transparent_bg_config"]
        if cfg is None or image.mode == "RGB" or ("A" not in image.getbands() and "transparency" not in image.info):
            return image.convert("RGB")
        pixels = np.asarray(image.convert("RGBA"))
        height, width = pixels.shape[:2]
        pattern = cfg.get("pattern", "black")
        if pattern in ("white", "black", "gray"):
            background = np.full((height, width, 3), {"white": 255, "black": 0, "gray": 128}[pattern], dtype=np.uint8)
        elif pattern == "chessboard":
            size = cfg.get("chessboard_square_size", 16)
            if size <= 0:
                raise ValueError("K3 chessboard_square_size must be positive")
            rows, columns = np.indices((height, width))
            gray = (rows // size + columns // size) % 2 == int(cfg.get("chessboard_square_on_top_left", True))
            values = np.where(gray, cfg.get("chessboard_gray_value", 200), cfg.get("chessboard_white_value", 255))
            background = np.repeat(values[:, :, None], 3, axis=2).astype(np.uint8)
        else:
            raise ValueError(f"Unsupported K3 transparent background: {pattern}")
        alpha = pixels[:, :, 3:4].astype(np.float32) / 255.0
        return Image.fromarray((alpha * pixels[:, :, :3] + (1 - alpha) * background).astype(np.uint8))

    def preprocess(
        self,
        images: _ImageSource | list[_ImageSource],
        return_tensors: str | TensorType | None = None,
        **kwargs: object,
    ) -> BatchFeature:
        """Resize, pad, normalize and flatten images in input order.

        Args:
            images: PIL images, local paths, raw bytes, or data URLs.
            return_tensors: Hugging Face output tensor format.
            **kwargs: Hugging Face preprocessing options; K3 uses checkpoint settings.

        Returns:
            pixel_values: FP32 tensor of shape [patches, 3, patch_size, patch_size],
                with images and their patches in row-major order.
            image_grid_thw: Integer tensor of shape [images, 3], storing [1, height, width]
                in patch units. Empty input produces no media fields.
        """
        kwargs.pop("do_sample_frames", None)
        if kwargs:
            raise TypeError(f"Unsupported K3 image preprocessing options: {sorted(kwargs)}")
        images = images if isinstance(images, list) else [images]
        cfg = self.media_proc_cfg
        patch, merge = cfg["patch_size"], cfg["merge_kernel_size"]
        pixels, grids = [], []
        for source in images:
            image = _load_image(source)
            width, height = image.size
            scale = min(
                1.0,
                math.sqrt(cfg["in_patch_limit"] / (max(1.0, width // patch) * max(1.0, height // patch))),
                cfg["patch_limit_on_one_side"] * patch / width,
                cfg["patch_limit_on_one_side"] * patch / height,
            )
            new_width, new_height = max(1, int(width * scale)), max(1, int(height * scale))
            factor = patch * merge
            pad_width, pad_height = (-new_width) % factor, (-new_height) % factor
            if cfg["transparent_bg_fill_stage"] == "before_resize":
                image = self._composite(image)
            image = image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
            if cfg["transparent_bg_fill_stage"] == "after_resize":
                image = self._composite(image)
            array = np.pad(np.asarray(image), ((0, pad_height), (0, pad_width), (0, 0)))
            array = (array / 255.0).astype(np.float32)
            array -= np.asarray(cfg["image_mean"])
            array *= 1.0 / np.asarray(cfg["image_std"])
            grid_h, grid_w = array.shape[0] // patch, array.shape[1] // patch
            patches = array.reshape(grid_h, patch, grid_w, patch, 3).transpose(0, 2, 4, 1, 3)
            pixels.append(torch.from_numpy(patches.reshape(-1, 3, patch, patch)))
            grids.append([1, grid_h, grid_w])
        data = {"pixel_values": torch.cat(pixels), "image_grid_thw": torch.tensor(grids)} if pixels else {}
        return BatchFeature(data=data, tensor_type=return_tensors)


class KimiK3Processor(ProcessorMixin):
    """Local image/text processor returning expanded tokens before native packing.

    Args:
        image_processor: Local checkpoint-compatible image processor.
        tokenizer: Existing native K3 tokenizer and XTML chat encoder.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"
    image_token = "<|media_pad|>"
    image_placeholder = "<|kimi_image_placeholder|>"

    def __init__(self, image_processor: KimiK3ImageProcessor, tokenizer: TikTokenTokenizer) -> None:
        # Match the checkpoint processor's template metadata. The tokenizer's
        # synthetic generation marker supervises the full XTML assistant turn;
        # VLM SFT uses the existing response-only template-derived labels.
        super().__init__(image_processor, tokenizer, chat_template=None)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs: object) -> KimiK3Processor:
        """Load local implementations using checkpoint data, without executing checkpoint Python."""
        kwargs["trust_remote_code"] = False
        return cls(
            KimiK3ImageProcessor.from_pretrained(pretrained_model_name_or_path, **kwargs),
            TikTokenTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs),
        )

    @property
    def model_input_names(self) -> list[str]:
        """Return text and media fields consumed by the native model and pipeline collator."""
        return ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "n_images_per_sample"]

    def _image_prompts(self, images: list[Image.Image], media: BatchFeature) -> list[str]:
        """Build expanded placeholders from original resolutions and processed grids.

        Args:
            images: Loaded PIL images in document order.
            media: image_grid_thw tensor [images, 3] and pixel_values tensor
                [patches, 3, patch_size, patch_size] from the local image processor.

        Returns:
            One fully expanded textual image span per image.
        """
        merge = self.image_processor.media_proc_cfg["merge_kernel_size"]
        grids = media["image_grid_thw"].tolist() if images else []
        return [
            f"<|media_begin|>image {image.width}x{image.height}<|media_content|>"
            + self.image_token * ((height // merge) * (width // merge))
            + "<|media_end|>"
            for image, (_, height, width) in zip(images, grids, strict=True)
        ]

    def _finish(
        self,
        encoded: list[dict[str, list[int]]],
        media: BatchFeature,
        image_counts: list[int],
        *,
        padding: bool | str,
        return_tensors: str | TensorType | None,
        max_length: int | None,
    ) -> BatchFeature:
        """Pad text and attach ordered media without shifting labels.

        Args:
            encoded: Token and optional assistant-mask lists, one per document.
            media: FP32 pixel_values [patches, 3, patch_size, patch_size] and
                integer image_grid_thw [images, 3].
            image_counts: Number of images owned by each document.
            padding: Tokenizer padding policy.
            return_tensors: Hugging Face output format.
            max_length: Target length for max_length padding.

        Returns:
            BatchFeature with input_ids/attention_mask/optional assistant_masks
            [batch, sequence], plus unchanged media tensors and image counts [batch].
        """
        masks = [row.pop("assistant_masks", None) for row in encoded]
        merge = self.image_processor.media_proc_cfg["merge_kernel_size"]
        grids = media["image_grid_thw"].tolist() if sum(image_counts) else []
        begin_id, content_id, end_id = self.tokenizer.convert_tokens_to_ids(
            ["<|media_begin|>", "<|media_content|>", "<|media_end|>"]
        )
        cursor = 0
        for row, mask, count in zip(encoded, masks, image_counts, strict=True):
            expected = [(height // merge) * (width // merge) for _, height, width in grids[cursor : cursor + count]]
            ids = row["input_ids"]
            starts = [i for i, token in enumerate(ids) if token == begin_id]
            contents = [i for i, token in enumerate(ids) if token == content_id]
            ends = [i for i, token in enumerate(ids) if token == end_id]
            if ids.count(self.image_token_id) != sum(expected) or not (
                len(starts) == len(contents) == len(ends) == count
            ):
                raise ValueError("K3 image token count mismatch; truncation must retain every complete image span")
            previous_end = -1
            for start, content, end, length in zip(starts, contents, ends, expected, strict=True):
                if not (
                    previous_end < start < content < end
                    and end - content - 1 == length
                    and all(token == self.image_token_id for token in ids[content + 1 : end])
                ):
                    raise ValueError("K3 requires ordered, complete image spans without text gaps")
                if mask is not None:
                    mask[start : end + 1] = [0] * (end + 1 - start)
                previous_end = end
            cursor += count
        padded = self.tokenizer.pad(encoded, padding=padding, max_length=max_length, return_tensors=None)
        if masks[0] is not None:
            padded_masks = []
            for mask, ids in zip(masks, padded["input_ids"], strict=True):
                zeros = [0] * (len(ids) - len(mask))
                padded_masks.append(zeros + mask if self.tokenizer.padding_side == "left" else mask + zeros)
            padded["assistant_masks"] = padded_masks
        result = BatchFeature(data=dict(padded), tensor_type=return_tensors)
        result.update(media)
        if sum(image_counts):
            result["n_images_per_sample"] = torch.tensor(image_counts, dtype=torch.long)
        return result

    def __call__(
        self,
        text: str | list[str],
        images: _ImageSource | list[_ImageSource] | None = None,
        *,
        videos: object | None = None,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs: object,
    ) -> BatchFeature:
        """Process standard text/images inputs with expansion before truncation.

        Args:
            text: Rendered prompts containing K3 image placeholders.
            images: Images flattened in prompt order.
            videos: Unsupported; supplying videos raises ValueError.
            padding: Tokenizer padding policy.
            truncation: Whether to truncate text to max_length; removing image tokens raises.
            max_length: Maximum token length after image expansion.
            return_tensors: Hugging Face output tensor format.
            **kwargs: Tokenizer options. do_sample_frames is unused for image inputs.

        Returns:
            The text/media tensor layouts documented by _finish.
        """
        if videos is not None:
            raise ValueError("K3 processor currently supports images and text, not video")
        kwargs.pop("do_sample_frames", None)
        texts = [text] if isinstance(text, str) else text
        if not texts:
            raise ValueError("K3 processor requires at least one text prompt")
        sources = [] if images is None else images if isinstance(images, list) else [images]
        loaded = [_load_image(source) for source in sources]
        counts = [prompt.count(self.image_placeholder) for prompt in texts]
        if sum(counts) != len(loaded):
            raise ValueError("K3 image placeholder count must match the number of images")
        media = self.image_processor(loaded, return_tensors="pt")
        prompts = iter(self._image_prompts(loaded, media))
        expanded = []
        for prompt in texts:
            parts = prompt.split(self.image_placeholder)
            expanded.append("".join(part + next(prompts) for part in parts[:-1]) + parts[-1])
        encoded = self.tokenizer(expanded, truncation=truncation, max_length=max_length, **kwargs)
        rows = [{key: value[index] for key, value in encoded.items()} for index in range(len(texts))]
        return self._finish(rows, media, counts, padding=padding, return_tensors=return_tensors, max_length=max_length)

    def apply_chat_template(
        self,
        conversation: _Conversation | list[_Conversation],
        *,
        tokenize: bool = False,
        return_dict: bool = False,
        return_tensors: str | TensorType | None = None,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        add_generation_prompt: bool = False,
        return_assistant_tokens_mask: bool = False,
        processor_kwargs: dict[str, object] | None = None,
        **kwargs: object,
    ) -> str | list[str] | list[int] | list[list[int]] | torch.Tensor | BatchFeature:
        """Render native XTML chats and optionally process images and assistant masks.

        Args:
            conversation: One chat or a batch, with loaded images in content items.
            tokenize: Return tokens and processed images instead of rendered text.
            return_dict: Include masks and media with the token IDs.
            return_tensors: Hugging Face output tensor format.
            padding: Tokenizer padding policy.
            truncation: Truncate after image expansion; incomplete image spans are rejected.
            max_length: Token limit or max_length padding target.
            add_generation_prompt: Append the native assistant generation prefix.
            return_assistant_tokens_mask: Mark assistant tokens before padding.
            processor_kwargs: Image preprocessing options supplied by the VLM data pipeline.
            **kwargs: Native K3 chat-rendering options, including tools and thinking settings.

        Returns:
            Rendered text, token IDs, or BatchFeature with the layouts documented by _finish.
            Assistant masks have shape [batch, sequence] and never supervise image tokens.
        """
        if not tokenize:
            return self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs
            )
        options = dict(processor_kwargs or {})
        padding = options.pop("padding", padding)
        truncation = options.pop("truncation", truncation)
        max_length = options.pop("max_length", max_length)
        batched = is_batched_conversation(conversation)
        conversations = conversation if batched else [conversation]
        if not conversation:
            raise ValueError("K3 processor requires a nonempty conversation")
        # The native encoder re-sorts tool results into assistant tool_calls order,
        # so collect images from that same order; otherwise reordered image-bearing
        # tool replies would render their text in call order but keep the pixels
        # in arrival order. The helper is idempotent, so the encoder's own pass
        # leaves these messages unchanged.
        conversations = [normalize_xtml_tool_result_messages(messages) for messages in conversations]
        loaded, counts = [], []
        for messages in conversations:
            sources = []
            for message in messages:
                # Tool-call-only assistant turns carry no content; the native
                # encoder renders their tool calls separately.
                content = message.get("content")
                if isinstance(content, list):
                    for item in content:
                        if item["type"] == "image":
                            sources.append(item["image"])
                        elif item["type"] in ("video", "audio", "image_url"):
                            raise ValueError(
                                "K3 chat images must be loaded before processing; video/audio are unsupported"
                            )
            loaded.extend(_load_image(source) for source in sources)
            counts.append(len(sources))
        media = self.image_processor(loaded, return_tensors="pt", **options)
        prompts = self._image_prompts(loaded, media)
        rows, cursor = [], 0
        for messages, count in zip(conversations, counts, strict=True):
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                image_prompts=prompts[cursor : cursor + count],
                add_generation_prompt=add_generation_prompt,
                return_assistant_tokens_mask=return_assistant_tokens_mask,
                truncation=truncation,
                max_length=max_length,
                **kwargs,
            )
            rows.append(dict(encoded))
            cursor += count
        result = self._finish(
            rows, media, counts, padding=padding, return_tensors=return_tensors, max_length=max_length
        )
        if return_dict:
            return result
        ids = result["input_ids"]
        return ids if batched or return_tensors is not None else ids[0]

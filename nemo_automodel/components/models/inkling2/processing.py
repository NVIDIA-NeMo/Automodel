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

"""Released Inkling2 image/chat preprocessing for the shared VLM collator."""

import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, PreTrainedTokenizerBase, ProcessorMixin
from transformers.feature_extraction_utils import BatchFeature

_IMAGE_BODY = "<|unused_200054|>"
_CHAT_TEMPLATE = """{%- for message in messages -%}
{%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}
{%- for item in message['content'] -%}
{{- '<|message_' + role + '|><|content_' + item['type'] + '|>' -}}
{%- if item['type'] == 'image' -%}
{{- '<|unused_200054|>' * item['num_tokens'] + '<|end_message|>' -}}
{%- elif message['role'] == 'assistant' -%}
{% generation %}{{- item['text'] + '<|end_message|>' -}}{% endgeneration %}
{%- else -%}
{{- item['text'] + '<|end_message|>' -}}
{%- endif -%}
{%- endfor -%}
{%- if message['role'] == 'assistant' -%}
{% generation %}{{- '<|content_model_end_sampling|>' -}}{% endgeneration %}
{%- endif -%}
{%- endfor -%}"""


def image_patches(image: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare one unresized image using the released 40-pixel patch policy.

    RGBA is composited onto an 8-pixel 204/153 checkerboard. Padding is -1 in
    pixel space. Normalization uses the release's FP32 CLIP constants, then
    casts to BF16 and duplicates each spatial patch along the temporal axis.

    Returns:
        BF16 patches [rows*columns, 2, 40, 40, 3] and int64 grid [1, 3].
        Patches own storage; the source image is unchanged.
    """
    if image.mode == "RGBA":
        rgba = np.asarray(image, dtype=np.uint8).astype(np.int32)
        yy, xx = np.indices(rgba.shape[:2])
        background = np.where((yy // 8 + xx // 8) % 2 == 0, 204, 153)[..., None]
        alpha = rgba[..., 3:4]
        rgb = ((rgba[..., :3] * alpha + background * (255 - alpha) + 127) // 255).astype(np.uint8)
    else:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    rows, columns = (height + 39) // 40, (width + 39) // 40
    padded = np.full((rows * 40, columns * 40, 3), -1, dtype=np.float32)
    padded[:height, :width] = rgb
    normalized = (
        padded / np.float32(255) - np.array((0.48145466, 0.4578275, 0.40821073), dtype=np.float32)
    ) / np.array((0.26862954, 0.2613026, 0.2757771), dtype=np.float32)
    patches = (
        torch.from_numpy(normalized.reshape(rows, 40, columns, 40, 3).transpose(0, 2, 1, 3, 4).copy())
        .reshape(rows * columns, 40, 40, 3)
        .to(torch.bfloat16)
    )
    return patches[:, None].expand(-1, 2, -1, -1, -1).contiguous(), torch.tensor([[1, rows, columns]])


class Inkling2Processor(ProcessorMixin):
    """Image/text processor with explicit image-body ownership and assistant masks."""

    attributes = ["tokenizer"]
    tokenizer_class = "PreTrainedTokenizerBase"

    def __init__(self, tokenizer: PreTrainedTokenizerBase, chat_template: str | None = None) -> None:
        super().__init__(tokenizer=tokenizer, chat_template=chat_template or _CHAT_TEMPLATE)
        self.tokenizer.chat_template = self.chat_template
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = "<|content_model_end_sampling|>"
        self.tokenizer.eos_token = "<|content_model_end_sampling|>"
        self.image_token_id = 200005
        self.image_body_id = self.tokenizer.convert_tokens_to_ids(_IMAGE_BODY)
        if self.image_body_id != 200054 or self.tokenizer.convert_tokens_to_ids("<|content_image|>") != 200005:
            raise ValueError("Inkling2 requires the released tokenizer's reserved image IDs")

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike,
        cache_dir: str | os.PathLike | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        **kwargs: Any,
    ) -> "Inkling2Processor":
        """Load the released tokenizer and verify its image preprocessing contract."""
        path = Path(pretrained_model_name_or_path) / "image_preprocessing.json"
        if path.is_file():
            config = json.loads(path.read_text())
            expected = {
                "patch_size": 40,
                "temporal_patch_size": 2,
                "include_extra_column": False,
                "stretch_to_padded_grid": False,
                "overlay_rgba_images_on_checkerboard": True,
                "rescale_image_frac": None,
                "rescale_image_max_upscaled_long_edge": None,
            }
            if any(config.get(key) != value for key, value in expected.items()):
                raise ValueError("Unsupported Inkling2 image preprocessing settings")
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            **kwargs,
        )
        if not isinstance(tokenizer, PreTrainedTokenizerBase):
            raise TypeError("Inkling2 requires a Hugging Face tokenizer")
        return cls(tokenizer)

    # HF annotates this method as returning str even when return_dict=True.
    # Our SFT contract also keeps template controls keyword-only.
    def apply_chat_template(  # ty: ignore[invalid-method-override]
        self,
        conversation: list[dict[str, Any]] | list[list[dict[str, Any]]],
        *,
        tokenize: bool = False,
        return_dict: bool = False,
        return_tensors: str | None = None,
        return_assistant_tokens_mask: bool = False,
        processor_kwargs: dict[str, Any] | None = None,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> BatchFeature | torch.Tensor | str | list[str]:
        """Render image/text conversations through the released control-token grammar.

        Args:
            conversation: One conversation or a batch of HF-style role/content records.
                Images must be PIL objects; text content is a string or text record.
            tokenize: Produce token IDs and patches when true, rendered strings otherwise.
            return_dict: Return token fields plus media tensors when true.
            return_tensors: Tensor framework; training supports only 'pt'.
            return_assistant_tokens_mask: Include the template's assistant supervision mask.
            processor_kwargs: HF tokenizer padding, truncation and maximum-length options.
            add_generation_prompt: Accepted for the HF interface; this renderer adds no prefix.
            kwargs: Additional HuggingFace tokenizer template options.

        Returns:
            HF-compatible token output. Token fields are [batch, sequence]; image patches
            are BF16 [patches, 2, 40, 40, 3], grids int64 [images, 3]. The binary
            mm_token_type_ids marks image bodies only, excluding image headers.
            Images cut by truncation are rejected instead of silently misaligning
            features, including an image whose last body token is the final token: the
            training collator drops that position when it shifts inputs against labels.
        """
        del add_generation_prompt
        batched = bool(conversation) and isinstance(conversation[0], list)
        conversations = cast(list[list[dict[str, Any]]], conversation if batched else [conversation])
        prepared, all_patches, all_grids, counts = [], [], [], []
        for messages in conversations:
            output, count = [], 0
            for message in messages:
                role = message["role"]
                if role not in ("user", "assistant", "system"):
                    raise ValueError("Inkling2 SFT currently supports user, assistant and system roles")
                content = message["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                parts = []
                for item in content:
                    if item["type"] == "text":
                        if _IMAGE_BODY in item["text"]:
                            raise ValueError("Text cannot contain the reserved Inkling2 image-body placeholder")
                        parts.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image" and role != "assistant":
                        image = item.get("image")
                        if not isinstance(image, Image.Image):
                            raise TypeError("Inkling2 images must be decoded PIL images")
                        patches, grid = image_patches(image)
                        all_patches.append(patches)
                        all_grids.append(grid)
                        count += patches.shape[0]
                        parts.append({"type": "image", "num_tokens": patches.shape[0]})
                    else:
                        raise ValueError("Inkling2 SFT supports input images and text only")
                output.append({"role": role, "content": parts})
            prepared.append(output)
            counts.append(count)
        if tokenize and return_tensors != "pt":
            raise ValueError("Inkling2 tokenized image/text batches require return_tensors='pt'")
        result = self.tokenizer.apply_chat_template(
            prepared if batched else prepared[0],
            tokenize=tokenize,
            return_dict=tokenize,
            return_tensors=return_tensors,
            return_assistant_tokens_mask=return_assistant_tokens_mask,
            **(processor_kwargs or {}),
            **kwargs,
        )
        if not tokenize:
            return result
        ids = result["input_ids"]
        image_mask = ids == self.image_body_id
        # The shared collator drops the final position when it shifts inputs against
        # labels, so every image body has to end before it.
        if image_mask[..., :-1].sum(-1).tolist() != counts:
            raise ValueError("Inkling2 truncation removed image tokens; filter overlong images or increase max_length")
        result["input_ids"] = ids.masked_fill(image_mask, self.image_token_id)
        result["mm_token_type_ids"] = image_mask.long()
        if all_patches:
            result["pixel_values"] = torch.cat(all_patches)
            result["image_grid_thw"] = torch.cat(all_grids)
        if return_assistant_tokens_mask:
            result["assistant_masks"] = torch.as_tensor(result["assistant_masks"], dtype=torch.long)
        return BatchFeature(dict(result)) if return_dict else result["input_ids"]

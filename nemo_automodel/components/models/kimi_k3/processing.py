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


"""K3 image preprocessing for the native VLM packing pipeline."""

from typing import Protocol, cast

import torch
from transformers import PreTrainedTokenizerBase, ProcessorMixin


class _KimiProcessor(Protocol):
    tokenizer: PreTrainedTokenizerBase

    def apply_chat_template(
        self, conversation: list[dict[str, object]], *, add_generation_prompt: bool, tokenize: bool
    ) -> str: ...

    def __call__(
        self, *, text: str, medias: list[dict[str, object]], return_tensors: str
    ) -> dict[str, torch.Tensor]: ...


def tokenize_conversation(conversation: list[dict[str, object]], processor: ProcessorMixin) -> dict[str, torch.Tensor]:
    """Expand image placeholders before the native packer shifts each document.

    Args:
        conversation: One chat conversation with loaded PIL images in its content items.
        processor: HF K3 processor accepting its native medias argument.

    Returns:
        Unshifted input_ids and attention_mask of shape [1, sequence],
        with each image placeholder repeated for its spatially merged features.
        Image samples additionally contain pixel_values of shape
        [patches, channels, patch_height, patch_width] and image_grid_thw
        of shape [images, 3], ordered as temporal, height, width. The native
        wrapper constructs labels before packing; this function never shifts them.
    """
    kimi_processor = cast(_KimiProcessor, processor)
    medias = []
    for message in conversation:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    medias.append({"type": "image", "image": item["image"]})
    text = kimi_processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
    result = kimi_processor(text=text, medias=medias, return_tensors="pt")
    grid = result.pop("grid_thws", None)
    if grid is None:
        return result
    ids = result["input_ids"]
    positions = (ids[0] == kimi_processor.tokenizer.convert_tokens_to_ids("<|media_pad|>")).nonzero().flatten()
    if len(positions) != len(grid):
        raise ValueError("K3 requires exactly one unexpanded placeholder per image grid.")
    if not bool((grid[:, 0] == 1).all()):
        raise ValueError("K3 packed image preprocessing requires image grids with temporal size 1.")
    repeats = torch.ones_like(ids[0])
    repeats[positions] = (grid[:, 1] // 2) * (grid[:, 2] // 2)
    if not bool((repeats > 0).all()):
        raise ValueError("K3 image grids must produce at least one merged image token.")
    for key in ("input_ids", "attention_mask"):
        result[key] = torch.repeat_interleave(result[key], repeats, dim=1)
    result["image_grid_thw"] = grid
    return result

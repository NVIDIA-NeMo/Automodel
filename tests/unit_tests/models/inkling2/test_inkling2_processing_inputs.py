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

"""Unit tests for the native Inkling 2 image processor, processor and VLM collation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from nemo_automodel.components.datasets.vlm.collate_fns import default_collate_fn
from nemo_automodel.components.models.inkling2.image_processing import Inkling2ImageProcessor
from nemo_automodel.components.models.inkling2.processing import Inkling2Processor

pytestmark = pytest.mark.cpu_only

IMAGE_TOKEN = "<|unused_200054|>"
END_OF_SAMPLING = "<|content_model_end_sampling|>"
SPECIAL_TOKENS = [
    "<|message_user|>",
    "<|message_model|>",
    "<|content_text|>",
    "<|content_image|>",
    "<|end_message|>",
    END_OF_SAMPLING,
    IMAGE_TOKEN,
]
WORDS = ["describe", "compare", "red", "blue", "tile"]
# Same token layout as the released chat template for user images/text and assistant text.
CHAT_TEMPLATE = (
    "{%- set role_token = {'user': '<|message_user|>', 'assistant': '<|message_model|>'} -%}"
    "{%- for message in messages -%}"
    "{%- for part in message.content -%}"
    "{%- if part.type == 'image' -%}"
    "{{- role_token[message.role] ~ '<|content_image|><|unused_200054|><|end_message|>' -}}"
    "{%- else -%}"
    "{{- role_token[message.role] ~ '<|content_text|>' ~ part.text ~ '<|end_message|>' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if message.role == 'assistant' -%}{{- '<|content_model_end_sampling|>' -}}{%- endif -%}"
    "{%- endfor -%}"
)
TILE = 4
PAD_VALUE = (-1.0 / 255 - 0.5) / 0.5


def _tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    vocab = {token: index for index, token in enumerate(["[UNK]", *SPECIAL_TOKENS, *WORDS])}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tokenizer


def _processor() -> Inkling2Processor:
    tokenizer = _tokenizer()
    tokenizer.eos_token = END_OF_SAMPLING
    tokenizer.pad_token = END_OF_SAMPLING
    return Inkling2Processor(
        Inkling2ImageProcessor(size={"height": TILE, "width": TILE}), tokenizer, chat_template=CHAT_TEMPLATE
    )


def _normalized(pixels: torch.Tensor) -> torch.Tensor:
    """Map uint8 pixels to the processor's [-1, 1] range.

    Args:
        pixels: uint8 tensor of any shape, e.g. ``[channels, height, width]``.

    Returns:
        float32 tensor of the same shape.
    """
    return (pixels.float() / 255 - 0.5) / 0.5


def test_image_processor_tiles_in_raster_order_and_pads_with_minus_one():
    image = torch.arange(3 * 6 * 9, dtype=torch.int64).reshape(3, 6, 9).remainder(256).to(torch.uint8)

    outputs = Inkling2ImageProcessor(size={"height": TILE, "width": TILE})(image, return_tensors="pt")

    pixel_values = outputs["pixel_values"]
    assert pixel_values.shape == (6, TILE, TILE, 3)
    assert outputs["image_grid_thw"].tolist() == [[1, 2, 3]]
    # Tile (row 1, column 2) holds pixels [4:6, 8:9]; the rest of it is padding.
    tile = pixel_values[1 * 3 + 2]
    torch.testing.assert_close(tile[:2, :1], _normalized(image[:, 4:6, 8:9].permute(1, 2, 0)))
    torch.testing.assert_close(tile[2:], torch.full((2, TILE, 3), PAD_VALUE))
    torch.testing.assert_close(tile[:, 1:], torch.full((TILE, TILE - 1, 3), PAD_VALUE))
    torch.testing.assert_close(pixel_values[0], _normalized(image[:, :4, :4].permute(1, 2, 0)))


def test_rgba_image_is_composited_on_a_checkerboard():
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[0, 0] = [255, 0, 0, 255]

    pixel_values = Inkling2ImageProcessor(size={"height": 16, "width": 16})(
        Image.fromarray(rgba, mode="RGBA"), return_tensors="pt"
    )["pixel_values"][0]

    torch.testing.assert_close(pixel_values[0, 0], torch.tensor([1.0, -1.0, -1.0]))
    torch.testing.assert_close(pixel_values[0, 1], _normalized(torch.full((3,), 204)))
    torch.testing.assert_close(pixel_values[0, 8], _normalized(torch.full((3,), 153)))
    torch.testing.assert_close(pixel_values[8, 8], _normalized(torch.full((3,), 204)))


def test_grayscale_image_becomes_three_equal_channels():
    gray = np.arange(64, dtype=np.uint8).reshape(8, 8)

    pixel_values = Inkling2ImageProcessor(size={"height": 8, "width": 8})(
        Image.fromarray(gray, mode="L"), return_tensors="pt"
    )["pixel_values"][0]

    expected = _normalized(torch.from_numpy(gray))[..., None].expand(8, 8, 3)
    torch.testing.assert_close(pixel_values, expected)


def test_processor_expands_one_placeholder_per_tile():
    processor = _processor()
    images = [torch.zeros(3, 6, 9, dtype=torch.uint8), torch.zeros(3, 4, 4, dtype=torch.uint8)]

    outputs = processor(text=[f"compare {IMAGE_TOKEN} {IMAGE_TOKEN}"], images=images, return_tensors="pt")

    assert outputs["image_grid_thw"].tolist() == [[1, 2, 3], [1, 1, 1]]
    assert outputs["pixel_values"].shape == (7, TILE, TILE, 3)
    assert (outputs["input_ids"] == processor.image_token_id).sum().item() == 7
    assert processor.model_input_names == ["pixel_values", "image_grid_thw", "input_ids", "attention_mask"]


def _write_snapshot(directory: Path, image_processor_type: str = "Inkling2ImageProcessor") -> None:
    _tokenizer().save_pretrained(directory)
    (directory / "chat_template.jinja").write_text(CHAT_TEMPLATE)
    processor_config = {
        "image_processor": {
            "image_processor_type": image_processor_type,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "size": {"height": TILE, "width": TILE},
        },
        "image_token": IMAGE_TOKEN,
        "processor_class": "Inkling2Processor",
    }
    (directory / "processor_config.json").write_text(json.dumps(processor_config))


def test_from_pretrained_reads_a_checkpoint_snapshot(tmp_path):
    _write_snapshot(tmp_path)

    processor = Inkling2Processor.from_pretrained(str(tmp_path))

    assert processor.image_processor.size.height == TILE
    assert processor.image_token_id == processor.tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    assert processor.chat_template == CHAT_TEMPLATE
    assert processor.tokenizer.eos_token == END_OF_SAMPLING
    assert processor.tokenizer.pad_token == END_OF_SAMPLING


def test_from_pretrained_rejects_another_image_processor(tmp_path):
    _write_snapshot(tmp_path, image_processor_type="Qwen2VLImageProcessor")

    with pytest.raises(ValueError, match="Qwen2VLImageProcessor"):
        Inkling2Processor.from_pretrained(str(tmp_path))


def test_default_collate_trains_answers_through_the_sampling_terminator():
    processor = _processor()
    tokenizer = processor.tokenizer
    examples = [
        {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image.new("RGB", (9, 6))},
                        {"type": "text", "text": "describe"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "red blue"}]},
            ]
        },
        {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": Image.new("RGB", (4, 4))},
                        {"type": "text", "text": "describe"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "tile"}]},
            ]
        },
    ]

    batch = default_collate_fn(examples, processor=processor, max_length=32)

    assert batch["image_grid_thw"].tolist() == [[1, 2, 3], [1, 1, 1]]
    assert batch["pixel_values"].shape == (7, TILE, TILE, 3)
    assert (batch["input_ids"] == processor.image_token_id).sum().item() == batch["pixel_values"].shape[0]
    answers = [tokenizer.convert_ids_to_tokens(labels[labels != -100].tolist()) for labels in batch["labels"].unbind(0)]
    assert answers == [
        ["red", "blue", "<|end_message|>", END_OF_SAMPLING],
        ["tile", "<|end_message|>", END_OF_SAMPLING],
    ]

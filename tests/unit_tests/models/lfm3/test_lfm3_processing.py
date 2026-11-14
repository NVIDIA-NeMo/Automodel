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

"""LFM3-VL processor loading from a checkpoint whose processor config names remote-code classes."""

import json

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoProcessor, PreTrainedTokenizerFast

import nemo_automodel._transformers.registry  # noqa: F401  registers the LFM3-VL config and processor
from nemo_automodel.components.models.lfm3.processing import Lfm3VlProcessor

pytest.importorskip("torchvision", reason="The Qwen image and video processors need the vlm extra.")

MEDIA_TOKENS = {
    "image_token": "<|image|>",
    "video_token": "<|video|>",
    "vision_start_token": "<|vision_start|>",
    "vision_end_token": "<|vision_end|>",
}
CHAT_TEMPLATE = (
    "{% for message in messages %}{% for item in message['content'] %}"
    "{% if item['type'] == 'image' %}<|vision_start|><|image|><|vision_end|>{% else %}{{ item['text'] }}{% endif %}"
    "{% endfor %}{% endfor %}"
)


@pytest.fixture
def checkpoint(tmp_path):
    """A checkpoint directory laid out like LiquidAI/LFM3-VL-2B: remote-code processor names, no code files."""
    vocab = {token: index for index, token in enumerate(["<unk>", "describe", "this", *MEDIA_TOKENS.values()])}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<unk>", extra_special_tokens=MEDIA_TOKENS
    ).save_pretrained(tmp_path)
    pixels = {
        "image_mean": [0.5] * 3,
        "image_std": [0.5] * 3,
        "patch_size": 16,
        "merge_size": 2,
        "temporal_patch_size": 2,
    }
    processor_config = {
        "processor_class": "Lfm3VlProcessor",
        "auto_map": {"AutoProcessor": "processing_lfm3_vl.Lfm3VlProcessor"},
        "image_processor": {
            **pixels,
            "image_processor_type": "Lfm3VlImageProcessor",
            "auto_map": {"AutoImageProcessor": "image_processing_lfm3_vl.Lfm3VlImageProcessor"},
            "size": {"shortest_edge": 65536, "longest_edge": 16777216},
        },
        "video_processor": {
            **pixels,
            "video_processor_type": "Lfm3VlVideoProcessor",
            "auto_map": {"AutoVideoProcessor": "video_processing_lfm3_vl.Lfm3VlVideoProcessor"},
            "size": {"shortest_edge": 4096, "longest_edge": 25165824},
        },
    }
    (tmp_path / "processor_config.json").write_text(json.dumps(processor_config))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "lfm3_vl"}))
    (tmp_path / "chat_template.jinja").write_text(CHAT_TEMPLATE)
    return tmp_path


def test_auto_processor_builds_native_components_without_remote_code(checkpoint):
    processor = AutoProcessor.from_pretrained(checkpoint)

    assert type(processor) is Lfm3VlProcessor
    assert type(processor.image_processor).__name__ == "Qwen2VLImageProcessor"
    assert type(processor.video_processor).__name__ == "Qwen3VLVideoProcessor"
    assert list(processor.image_processor.image_mean) == [0.5] * 3 and processor.image_processor.patch_size == 16
    assert processor.video_processor.size == {"shortest_edge": 4096, "longest_edge": 25165824}


def test_image_prompt_gets_one_placeholder_per_merged_patch_block(checkpoint):
    processor = AutoProcessor.from_pretrained(checkpoint)
    image = np.random.default_rng(0).integers(0, 256, (320, 480, 3), dtype=np.uint8)
    conversation = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "describe this"}]}
    ]

    batch = processor.apply_chat_template(conversation, tokenize=True, return_dict=True, return_tensors="pt")

    frames, height, width = batch["image_grid_thw"][0].tolist()
    assert (frames, height, width) == (1, 20, 30)
    assert batch["pixel_values"].shape == (height * width, 3 * 2 * 16 * 16)
    assert int((batch["input_ids"] == processor.image_token_id).sum()) == height * width // 4
    assert set(batch) == {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    assert "mm_token_type_ids" not in processor.model_input_names


def test_saved_processor_reloads_with_the_same_settings(checkpoint, tmp_path_factory):
    saved = tmp_path_factory.mktemp("saved")
    processor = AutoProcessor.from_pretrained(checkpoint)

    processor.save_pretrained(saved)
    reloaded = Lfm3VlProcessor.from_pretrained(saved)

    assert json.loads((saved / "processor_config.json").read_text())["image_processor"]["image_processor_type"] == (
        "Qwen2VLImageProcessor"
    )
    assert reloaded.image_processor.to_dict() == processor.image_processor.to_dict()
    assert reloaded.chat_template == processor.chat_template

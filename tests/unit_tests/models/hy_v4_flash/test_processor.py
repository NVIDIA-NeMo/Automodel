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

"""The processor's checkpoint round trip: ``save_pretrained`` then ``AutoProcessor.from_pretrained``."""

from pathlib import Path

import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoProcessor, PreTrainedTokenizerFast

import nemo_automodel._transformers.registry  # noqa: F401  (registers the processor with AutoProcessor)
from nemo_automodel.components.models.hy_v4_flash.processing import (
    IMAGE_PAD_TOKEN,
    HYV4FlashImageSettings,
    HYV4FlashProcessor,
)
from tests.unit_tests.models.hy_v4_flash.conftest import tiny_config

IMAGE_TOKEN_ID = 2
TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"


def _tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"<unk>": 0, "<pad>": 1, IMAGE_PAD_TOKEN: IMAGE_TOKEN_ID, "describe": 3, "this": 4}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="<unk>", pad_token="<pad>")
    fast.chat_template = TEMPLATE
    return fast


def test_save_and_reload_keep_settings_template_and_encoding(tmp_path: Path) -> None:
    settings = HYV4FlashImageSettings(min_pixels=1024, max_pixels=16384, mean=(0.1, 0.2, 0.3), std=(0.4, 0.5, 0.6))
    processor = HYV4FlashProcessor(_tokenizer(), image_token_id=IMAGE_TOKEN_ID, settings=settings)
    tiny_config(image_token_id=IMAGE_TOKEN_ID).save_pretrained(tmp_path)
    processor.save_pretrained(tmp_path)

    loaded = AutoProcessor.from_pretrained(tmp_path, local_files_only=True)
    assert type(loaded) is HYV4FlashProcessor
    assert loaded.settings == settings and loaded.settings != HYV4FlashImageSettings()
    assert loaded.image_token_id == IMAGE_TOKEN_ID and loaded.chat_template == TEMPLATE

    generator = torch.Generator().manual_seed(0)
    pixels = torch.randint(0, 256, (72, 104, 3), generator=generator, dtype=torch.uint8)
    image = Image.fromarray(pixels.numpy())
    text = f"describe {IMAGE_PAD_TOKEN} this"
    before = processor(text, images=[image], return_tensors="pt")
    after = loaded(text, images=[image], return_tensors="pt")
    assert before.keys() == after.keys() and "pixel_values" in before
    for key in before:
        assert torch.equal(before[key], after[key]), key

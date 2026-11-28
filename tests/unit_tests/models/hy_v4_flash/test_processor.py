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

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoProcessor, PreTrainedTokenizerFast

import nemo_automodel._transformers.registry  # noqa: F401  (registers the processor with AutoProcessor)
from nemo_automodel.components.datasets.vlm.collate_fns import default_collate_fn
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


# Assistant turns as the released template lays them out: an empty think pair in ``no_think`` mode, the reasoning
# inside the pair in ``high`` mode (the template's default).
CHAT_TEMPLATE = (
    "{% for m in messages %}{% if m['role'] == 'assistant' %}<a> <think> "
    "{% if reasoning_effort != 'no_think' %}{{ m['reasoning_content'] or m['reasoning'] }} {% endif %}"
    "</think> {{ m['content'] }} <end> {% else %}<u> {{ m['content'] }} <end> {% endif %}{% endfor %}"
)
CHAT_WORDS = ("<u>", "<a>", "<think>", "</think>", "<end>", "u", "XSENTINELMARKERX", "what", "four", "because")


def _chat_processor() -> HYV4FlashProcessor:
    fast = _tokenizer()
    fast.add_tokens(list(CHAT_WORDS))
    fast.chat_template = CHAT_TEMPLATE
    return HYV4FlashProcessor(fast, image_token_id=IMAGE_TOKEN_ID)


def test_answer_only_sample_is_supervised_on_the_answer_and_the_end_token() -> None:
    processor = _chat_processor()
    sample = {"conversation": [{"role": "user", "content": "what"}, {"role": "assistant", "content": "four"}]}
    batch = default_collate_fn([sample], processor)
    supervised = batch["labels"][batch["labels"] != -100]
    assert processor.tokenizer.convert_ids_to_tokens(supervised.tolist()) == ["four", "<end>"]


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning"])
def test_sample_with_reasoning_is_rejected_instead_of_training_on_no_token(field: str) -> None:
    turn = {"role": "assistant", "content": "four", field: "because"}
    sample = {"conversation": [{"role": "user", "content": "what"}, turn]}
    with pytest.raises(ValueError, match="assistant reasoning are not supported"):
        default_collate_fn([sample], _chat_processor())


def test_saved_processor_reloads_from_its_own_directory_with_its_image_token_id(tmp_path: Path) -> None:
    # No model config next to it, or one that names another id: the persisted id of the processor wins.
    HYV4FlashProcessor(_tokenizer(), image_token_id=IMAGE_TOKEN_ID).save_pretrained(tmp_path)
    assert HYV4FlashProcessor.from_pretrained(tmp_path, local_files_only=True).image_token_id == IMAGE_TOKEN_ID
    tiny_config(image_token_id=IMAGE_TOKEN_ID + 1).save_pretrained(tmp_path)
    assert AutoProcessor.from_pretrained(tmp_path, local_files_only=True).image_token_id == IMAGE_TOKEN_ID


@pytest.mark.parametrize("block", ["image_url", "url"])
def test_image_data_uri_loads_like_the_same_pil_image(block: str) -> None:
    import base64
    import io

    pixels = torch.randint(0, 256, (64, 64, 3), generator=torch.Generator().manual_seed(1), dtype=torch.uint8)
    image = Image.fromarray(pixels.numpy())
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    source = {"type": "image_url", "image_url": {"url": uri}} if block == "image_url" else {"type": "image", "url": uri}
    tokenizer = _tokenizer()
    tokenizer.chat_template = (
        "{% for m in messages %}{% for b in m['content'] %}"
        "{% if b['type'] == 'image' %}" + IMAGE_PAD_TOKEN + " {% else %}{{ b['text'] }}{% endif %}"
        "{% endfor %}{% endfor %}"
    )
    processor = HYV4FlashProcessor(tokenizer, image_token_id=IMAGE_TOKEN_ID)

    def encode(content: dict[str, object]) -> dict[str, torch.Tensor]:
        messages = [{"role": "user", "content": [content, {"type": "text", "text": "describe this"}]}]
        return processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")

    from_uri, from_pil = encode(source), encode({"type": "image", "image": image})
    assert from_uri.keys() == from_pil.keys() and "pixel_values" in from_uri
    for key in from_pil:
        assert torch.equal(from_uri[key], from_pil[key]), key

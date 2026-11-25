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

import os
from pathlib import Path

import pytest

from nemo_automodel.components.models.hy_v4_flash.processing import HYV4FlashProcessor

CKPT = os.environ.get("HY4_CKPT_DIR")
TEMPLATE = (
    Path(__file__).resolve().parents[4] / "examples/llm_finetune/hy_v4_flash/hy4_flash_training_chat_template.jinja"
)
END = "<｜hy_end:opensource｜>"
CONVERSATIONS = [
    [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
    [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
    ],
]


@pytest.mark.skipif(CKPT is None, reason="HY4_CKPT_DIR (tokenizer + released chat template) not provided")
@pytest.mark.parametrize("conversation", CONVERSATIONS)
def test_training_template_renders_like_the_released_one_and_marks_only_answers(
    conversation: list[dict[str, str]],
) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    released = tokenizer.apply_chat_template(conversation, tokenize=False, reasoning_effort="no_think")
    ours = tokenizer.apply_chat_template(conversation, tokenize=False, chat_template=TEMPLATE.read_text())
    assert ours == released
    encoded = tokenizer.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
        chat_template=TEMPLATE.read_text(),
    )
    supervised = [token for token, flag in zip(encoded["input_ids"], encoded["assistant_masks"]) if flag]
    expected = "".join(m["content"] + END for m in conversation if m["role"] == "assistant")
    assert tokenizer.decode(supervised) == expected


class _RecordingTokenizer:
    """Stands in for the checkpoint tokenizer: records the template variables of every render."""

    chat_template = "unused"
    unk_token_id = 0

    def __init__(self) -> None:
        self.calls = []

    def convert_tokens_to_ids(self, token: str) -> None:
        return None

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: str | bool) -> str:
        self.calls.append(kwargs)
        return "rendered"


def _processor(tokenizer: _RecordingTokenizer) -> HYV4FlashProcessor:
    processor = HYV4FlashProcessor.__new__(HYV4FlashProcessor)
    processor.tokenizer = tokenizer
    return processor


ANSWER_ONLY = [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]
WITH_REASONING = [
    {"role": "user", "content": "2+2?"},
    {"role": "assistant", "content": "4", "reasoning_content": "Two and two make four."},
]


def test_training_render_uses_the_official_sft_reasoning_mode() -> None:
    """Answer-only training data is fast-thinking data: the released template's ``high`` default must not apply."""
    tokenizer = _RecordingTokenizer()
    processor = _processor(tokenizer)
    processor.apply_chat_template(ANSWER_ONLY)
    processor.apply_chat_template(WITH_REASONING)
    processor.apply_chat_template([ANSWER_ONLY, WITH_REASONING])
    assert [call["reasoning_effort"] for call in tokenizer.calls] == ["no_think", "high", "no_think", "high"]


def test_explicit_reasoning_effort_and_generation_prompts_are_left_alone() -> None:
    tokenizer = _RecordingTokenizer()
    processor = _processor(tokenizer)
    processor.apply_chat_template(ANSWER_ONLY, reasoning_effort="high")
    processor.apply_chat_template(ANSWER_ONLY[:1], add_generation_prompt=True)
    assert tokenizer.calls[0]["reasoning_effort"] == "high"
    assert "reasoning_effort" not in tokenizer.calls[1]  # inference keeps the template's own default


def test_released_processor_config_gives_the_default_image_settings() -> None:
    """The consumed fields of ``processor_config.json`` at tencent/Hy4-flash revision eccd3123 give the defaults."""
    from nemo_automodel.components.models.hy_v4_flash.processing import (
        HYV4FlashImageSettings,
        image_settings_from_processor_config,
    )

    released = {
        "patch_size": 16,
        "merge_size": 2,
        "min_pixels": 4096,
        "max_pixels": 4194304,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    assert image_settings_from_processor_config(released) == HYV4FlashImageSettings()
    changed = image_settings_from_processor_config({**released, "max_pixels": 1048576, "image_mean": [0.1, 0.2, 0.3]})
    assert changed.max_pixels == 1048576 and changed.mean == (0.1, 0.2, 0.3)

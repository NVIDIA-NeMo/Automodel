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

from pathlib import Path

import pytest
from jinja2.exceptions import TemplateError
from transformers.utils.chat_template_utils import render_jinja_template

from nemo_automodel.components.models.hy_v4_flash.processing import HYV4FlashProcessor

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


START, MIDDLE = "<｜hy_start:opensource｜>", "<｜hy_middle:opensource｜>"
HEADER = START + "system" + MIDDLE + "{system}<｜reasoning_mode:opensource｜>reasoning_effort:no_think" + END
ANSWER = START + "assistant" + MIDDLE + "<think:opensource></think:opensource>"
# What the released chat template (revision eccd3123) renders for CONVERSATIONS with reasoning_effort="no_think".
RELEASED_RENDERINGS = [
    HEADER.format(system="") + START + "user" + MIDDLE + "Hi" + END + ANSWER + "Hello!" + END,
    HEADER.format(system="Be brief.")
    + (START + "user" + MIDDLE + "Hi" + END + ANSWER + "Hello!" + END)
    + (START + "user" + MIDDLE + "2+2?" + END + ANSWER + "4" + END),
]


@pytest.mark.parametrize(("conversation", "released"), zip(CONVERSATIONS, RELEASED_RENDERINGS))
def test_training_template_renders_like_the_released_one_and_marks_only_answers(
    conversation: list[dict[str, str]], released: str
) -> None:
    rendered, spans = render_jinja_template(
        conversations=[conversation], chat_template=TEMPLATE.read_text(), return_assistant_tokens_mask=True
    )
    assert rendered == [released]
    supervised = [rendered[0][start:end] for start, end in spans[0]]
    assert supervised == [m["content"] + END for m in conversation if m["role"] == "assistant"]


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
    processor.apply_chat_template([ANSWER_ONLY, ANSWER_ONLY])
    assert [call["reasoning_effort"] for call in tokenizer.calls] == ["no_think"] * 3


def test_reasoning_samples_are_rejected_by_the_processor_and_by_the_text_template() -> None:
    with pytest.raises(ValueError, match="assistant reasoning are not supported"):
        _processor(_RecordingTokenizer()).apply_chat_template(WITH_REASONING)
    with pytest.raises(TemplateError, match="assistant reasoning are not supported"):
        render_jinja_template(conversations=[WITH_REASONING], chat_template=TEMPLATE.read_text())


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

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The GLM-5.3-Flash training chat template supervises the end-of-turn token.

GLM ends an assistant turn with the next role token (``<|user|>``, ``<|observation|>`` or
``<|system|>``) and the released template renders nothing after the final turn, so the stop
token is never inside the assistant span that ``ChatDataset`` supervises. The training template
renders the same text as the released one, adds a terminator after the final turn, and marks
each assistant span, including its terminator, with ``{% generation %}``.

The ``released`` strings below are what the released ``zai-org/GLM-5.3-Flash`` template
(revision ``eb9eb208eb0d988989d07a6a12d0fdeb5f52574a``) renders for each conversation.
"""

from pathlib import Path

import pytest
from transformers.utils.chat_template_utils import render_jinja_template

from nemo_automodel.components.datasets.llm import formatting_utils

TEMPLATE_PATH = Path(__file__).parents[4] / "examples/llm_finetune/glm5_next/glm5_3_flash_training_chat_template.jinja"
PREFIX = "[gMASK]<sop><|system|>Reasoning Effort: Max"
TOOL_CALLS = [{"type": "function", "id": "c1", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]

CASES = {
    "plain_two_turn": (
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there"},
            {"role": "user", "content": "bye"},
            {"role": "assistant", "content": "see you"},
        ],
        PREFIX + "<|user|>hi<|assistant|><think></think>hello there<|user|>bye<|assistant|><think></think>see you",
        "<|user|>",
        ["</think>hello there<|user|>", "</think>see you<|user|>"],
    ),
    "reasoning": (
        [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4", "reasoning_content": "add them"},
        ],
        PREFIX + "<|user|>2+2?<|assistant|><think>add them</think>4",
        "<|user|>",
        ["add them</think>4<|user|>"],
    ),
    "tool_call": (
        [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "", "tool_calls": TOOL_CALLS},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
            {"role": "assistant", "content": "It is sunny."},
        ],
        PREFIX
        + "<|user|>weather?<|assistant|><think></think><tool_call>get_weather<arg_key>city</arg_key>"
        + "<arg_value>Paris</arg_value></tool_call><|observation|><tool_response>sunny</tool_response>"
        + "<|assistant|><think></think>It is sunny.",
        "<|user|>",
        [
            "</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call><|observation|>",
            "</think>It is sunny.<|user|>",
        ],
    ),
    "ends_with_tool_call": (
        [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "checking", "tool_calls": TOOL_CALLS},
        ],
        PREFIX
        + "<|user|>weather?<|assistant|><think></think>checking<tool_call>get_weather<arg_key>city</arg_key>"
        + "<arg_value>Paris</arg_value></tool_call>",
        "<|observation|>",
        [
            "</think>checking<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>"
            "<|observation|>"
        ],
    ),
    "system_after_assistant": (
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "system", "content": "new rule"},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ],
        PREFIX
        + "<|user|>hi<|assistant|><think></think>hello<|system|>new rule<|user|>ok<|assistant|><think></think>done",
        "<|user|>",
        ["</think>hello<|system|>", "</think>done<|user|>"],
    ),
}


def _render(messages, **kwargs):
    (rendered,), (spans,) = render_jinja_template(
        conversations=[messages],
        chat_template=TEMPLATE_PATH.read_text(),
        return_assistant_tokens_mask=True,
        **kwargs,
    )
    return rendered, [rendered[start:end] for start, end in spans]


def test_template_takes_the_generation_mask_path():
    template = TEMPLATE_PATH.read_text()
    assert formatting_utils.GENERATION_REGEX.search(template) is not None
    assert template.count("{%- generation -%}") == template.count("{%- endgeneration -%}") == 1


@pytest.mark.parametrize("name", list(CASES))
def test_render_matches_released_template_plus_final_terminator(name):
    messages, released, terminator, _ = CASES[name]
    rendered, _ = _render(messages)
    assert rendered == released + terminator


@pytest.mark.parametrize("name", list(CASES))
def test_supervised_spans_end_with_the_end_of_turn_token(name):
    messages, _, _, expected_spans = CASES[name]
    _, spans = _render(messages)
    assert spans == expected_spans
    # Role headers, the generation prompt's <think>, and user / system / tool text stay unsupervised.
    for span in spans:
        assert "<|assistant|>" not in span
        assert "<think>" not in span
        assert "<tool_response>" not in span


def test_generation_prompt_is_unchanged():
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b", "reasoning_content": "r"},
        {"role": "user", "content": "c"},
    ]
    rendered, spans = _render(messages, add_generation_prompt=True)
    assert rendered == PREFIX + "<|user|>a<|assistant|><think>r</think>b<|user|>c<|assistant|><think>"
    assert spans == ["r</think>b<|user|>"]

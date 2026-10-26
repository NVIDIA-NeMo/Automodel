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

"""The DeepSeek-V4.1-Flash training chat template.

The template is a training preamble followed by the released ``deepseek-ai/DeepSeek-V4.1-Flash``
``chat_template.jinja`` body (revision ``2cba9e42aa026125f3ed06c6d98c1db82f7ca027``), unchanged
except for a ``{% generation %}`` block around every assistant turn. The preamble defaults to chat
mode when the caller passes no mode and then rejects reasoning that chat mode would drop. The
expected strings below are what the released template renders for each conversation.
"""

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from jinja2.exceptions import TemplateError
from transformers.utils.chat_template_utils import render_jinja_template

from nemo_automodel.components.datasets.llm import formatting_utils

REPO_ROOT = Path(__file__).parents[4]
TEMPLATE_PATH = REPO_ROOT / "examples/llm_finetune/deepseek_v41/deepseek_v41_flash_training_chat_template.jinja"
TEMPLATE = TEMPLATE_PATH.read_text()
TULU3_TEMPLATE = (REPO_ROOT / "examples/llm_finetune/deepseek_v41/tulu3_chat_template.jinja").read_text()
YAML_PATH = REPO_ROOT / "examples/speculative/dspark/deepseek_v41_flash_dspark.yaml"
# sha256 of the released chat_template.jinja after its leading comment block.
RELEASED_BODY_SHA256 = "899908e9e015a11ebc6e7ed39a0c57b7e99d30d81362f2358a2f30b3c54ece93"  # pragma: allowlist secret
BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
EFFORT = "<｜System｜>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n"
TOOL_CALLS = [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
DSML_CALL = (
    '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n'
    '<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
)
SYSTEM_CHAT = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
]
TWO_TURN_REASONING = [
    {"role": "user", "content": "Q1"},
    {"role": "assistant", "content": "A1", "reasoning_content": "R1"},
    {"role": "user", "content": "Q2"},
    {"role": "assistant", "content": "A2", "reasoning_content": "R2"},
]
TOOL_TURNS = [
    {"role": "user", "content": "weather?"},
    {"role": "assistant", "content": "", "reasoning_content": "call it", "tool_calls": TOOL_CALLS},
    {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
    {"role": "assistant", "content": "It is sunny.", "reasoning_content": "done"},
]


def _render(messages, template=TEMPLATE, **kwargs):
    (rendered,), (spans,) = render_jinja_template(
        conversations=[messages],
        chat_template=template,
        return_assistant_tokens_mask=True,
        **kwargs,
    )
    return rendered, [rendered[start:end] for start, end in spans]


def test_example_recipe_uses_the_training_template():
    recipe_args = yaml.safe_load(YAML_PATH.read_text())["recipe_args"]
    assert REPO_ROOT / recipe_args["chat_template"] == TEMPLATE_PATH


def test_template_takes_the_generation_mask_path():
    assert formatting_utils.GENERATION_REGEX.search(TEMPLATE) is not None
    assert TEMPLATE.count("{%- generation -%}") == TEMPLATE.count("{%- endgeneration -%}") == 1


def test_body_is_the_released_template_plus_the_generation_block():
    body = TEMPLATE.split("{#- Released template body. -#}\n", 1)[1]
    body = body.replace("        {%- generation -%}\n", "").replace("        {%- endgeneration -%}\n", "")
    assert hashlib.sha256(body.encode()).hexdigest() == RELEASED_BODY_SHA256


def test_leading_system_message_uses_the_system_token():
    rendered, spans = _render(SYSTEM_CHAT, thinking_mode="chat")
    assert rendered == BOS + "<｜System｜>Be brief.<｜User｜>Hi<｜Assistant｜></think>Hello!" + EOS
    assert spans == ["Hello!" + EOS]


def test_chat_mode_supervises_answers_only():
    rendered, spans = _render(TWO_TURN_REASONING, thinking_mode="chat")
    assert rendered == (
        BOS + "<｜User｜>Q1<｜Assistant｜></think>A1" + EOS + "<｜User｜>Q2<｜Assistant｜></think>A2" + EOS
    )
    assert spans == ["A1" + EOS, "A2" + EOS]


def test_thinking_mode_supervises_kept_reasoning_but_not_the_generation_prompt():
    rendered, spans = _render(TWO_TURN_REASONING, thinking_mode="thinking")
    assert rendered == (
        BOS
        + EFFORT
        + "<｜User｜>Q1<｜Assistant｜></think>A1"
        + EOS
        + "<｜User｜>Q2<｜Assistant｜><think>R2</think>A2"
        + EOS
    )
    # drop_thinking removes the earlier turn's reasoning; <think> belongs to the generation prompt.
    assert spans == ["A1" + EOS, "R2</think>A2" + EOS]


def test_tool_calls_and_results():
    rendered, spans = _render(TOOL_TURNS, thinking_mode="thinking", tools=TOOLS)
    assert rendered.startswith(BOS + EFFORT + "\n\n## Tools\n\n")
    assert rendered.split("<｜User｜>", 1)[1] == (
        "weather?<｜Assistant｜><think>call it</think>"
        + DSML_CALL
        + EOS
        + "<｜User｜><tool_result>sunny</tool_result><｜Assistant｜><think>done</think>It is sunny."
        + EOS
    )
    assert spans == ["call it</think>" + DSML_CALL + EOS, "done</think>It is sunny." + EOS]


@pytest.mark.parametrize("arguments", [json.dumps(TOOL_CALLS[0]["function"]["arguments"]), ["Paris"], None])
def test_non_object_tool_call_arguments_are_rejected(arguments):
    # The released body would render any non-object as a single DSML parameter named "arguments".
    function = {**TOOL_CALLS[0]["function"], "arguments": arguments}
    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "tool_calls": [{**TOOL_CALLS[0], "function": function}]},
    ]
    with pytest.raises(TemplateError, match="NeMoAutoTokenizer"):
        _render(messages, thinking_mode="chat", tools=TOOLS)


def test_mode_defaults_to_chat():
    assert _render(SYSTEM_CHAT) == _render(SYSTEM_CHAT, thinking_mode="chat")


@pytest.mark.parametrize("field", ["reasoning_content", "thinking"])
def test_default_chat_mode_rejects_reasoning_it_would_drop(field):
    messages = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A", field: "R"}]
    with pytest.raises(TemplateError, match="would be dropped in the default chat mode"):
        _render(messages)


@pytest.mark.parametrize(("enable_thinking", "expected_mode"), [(True, "thinking"), (False, "chat")])
def test_enable_thinking_alias_selects_the_mode(enable_thinking, expected_mode):
    assert _render(TWO_TURN_REASONING, enable_thinking=enable_thinking) == _render(
        TWO_TURN_REASONING, thinking_mode=expected_mode
    )


@pytest.mark.parametrize(
    ("thinking_mode", "expected"),
    [
        ("chat", BOS + "<｜User｜>Hi<｜Assistant｜></think>"),
        ("thinking", BOS + EFFORT + "<｜User｜>Hi<｜Assistant｜><think>"),
    ],
)
def test_generation_prompt_is_unchanged(thinking_mode, expected):
    rendered, spans = _render(
        [{"role": "user", "content": "Hi"}], thinking_mode=thinking_mode, add_generation_prompt=True
    )
    assert rendered == expected
    assert spans == []


@pytest.mark.parametrize(
    ("messages", "add_generation_prompt"),
    [
        (SYSTEM_CHAT + [{"role": "user", "content": "Again"}, {"role": "assistant", "content": "Hi again."}], False),
        ([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}] + SYSTEM_CHAT, False),
        ([{"role": "user", "content": "Hi"}], True),
    ],
)
def test_chat_mode_matches_the_tulu3_sft_template(messages, add_generation_prompt):
    # The V4.1 SFT recipes use tulu3_chat_template.jinja; both must frame the roles it accepts the same way.
    tokens = {"bos_token": BOS, "eos_token": EOS, "add_generation_prompt": add_generation_prompt}
    assert _render(messages, **tokens) == _render(messages, template=TULU3_TEMPLATE, **tokens)


def test_invalid_thinking_mode_is_rejected():
    with pytest.raises(TemplateError, match="Invalid thinking_mode"):
        _render(SYSTEM_CHAT, thinking_mode="fast")

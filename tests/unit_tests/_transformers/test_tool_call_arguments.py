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

"""Shared JSON-string to dict conversion of tool-call arguments for chat templates."""

import copy
import pickle

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from nemo_automodel._transformers.tokenization.registry import TokenizerRegistry
from nemo_automodel._transformers.tokenization.tool_call_arguments import (
    DictToolArgumentsTokenizer,
    tool_call_arguments_as_dicts,
    wrap_tokenizer,
)


def _conversation(arguments):
    call = {"id": "c0", "type": "function", "function": {"name": "ls", "arguments": arguments}}
    return [
        {"role": "user", "content": "list files"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
    ]


def _arguments(conversation):
    return conversation[1]["tool_calls"][0]["function"]["arguments"]


def test_json_object_strings_become_dicts_and_the_input_is_not_mutated():
    conversation = _conversation('{"path": "/w", "n": 2}')
    before = copy.deepcopy(conversation)
    converted = tool_call_arguments_as_dicts(conversation)
    assert _arguments(converted) == {"path": "/w", "n": 2}
    assert conversation == before
    # Messages without tool calls are passed through as the same objects.
    assert converted[0] is conversation[0]


@pytest.mark.parametrize("arguments", ["[1, 2]", "not json", '"text"', {"path": "/w"}])
def test_other_arguments_are_left_as_is(arguments):
    conversation = _conversation(arguments)
    assert tool_call_arguments_as_dicts(conversation)[1] is conversation[1]


def test_batches_and_non_conversation_inputs():
    batch = [_conversation('{"a": 1}'), _conversation({"a": 1})]
    assert [_arguments(c) for c in tool_call_arguments_as_dicts(batch)] == [{"a": 1}, {"a": 1}]
    assert tool_call_arguments_as_dicts("plain text") == "plain text"
    assert tool_call_arguments_as_dicts([]) == []


class _RecordingTokenizer:
    def apply_chat_template(self, conversation, *args, **kwargs):
        return conversation, args, kwargs


def test_wrapped_tokenizer_survives_copies():
    tok = wrap_tokenizer(_RecordingTokenizer())
    for copied in (tok, pickle.loads(pickle.dumps(tok)), copy.deepcopy(tok)):  # noqa: S301
        conversation, args, kwargs = copied.apply_chat_template(_conversation('{"a": 1}'), "x", tokenize=False)
        assert _arguments(conversation) == {"a": 1}
        assert args == ("x",) and kwargs == {"tokenize": False}


@pytest.mark.parametrize("model_type", ["beam", "deepseek_v41", "nemotron_h"])
def test_registered_for_models_whose_templates_need_dict_arguments(model_type: str) -> None:
    assert TokenizerRegistry.get_custom_tokenizer_cls(model_type) is DictToolArgumentsTokenizer


def test_factory_wraps_apply_chat_template(tmp_path):
    tok = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>")))
    tok.chat_template = (
        "{% for m in messages %}{% for tc in m.tool_calls or [] %}"
        "{{ tc.function.arguments is mapping }}{% endfor %}{% endfor %}"
    )
    tok.save_pretrained(tmp_path)
    loaded = DictToolArgumentsTokenizer.from_pretrained(str(tmp_path))
    assert loaded.apply_chat_template(_conversation('{"a": 1}'), tokenize=False) == "True"

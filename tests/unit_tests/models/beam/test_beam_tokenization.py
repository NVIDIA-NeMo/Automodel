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

"""Beam's tokenizer renders JSON-string tool-call arguments as the dicts Beam's chat template requires."""

import copy
import json
import pickle

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel._transformers.tokenization.tool_call_arguments import DictToolArgumentsTokenizer
from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset

# Mirrors the guard in Beam's chat template: arguments are rendered per parameter and must be a mapping.
_TEMPLATE = (
    "{%- for m in messages -%}"
    "<{{ m.role }}>"
    "{%- if m.role == 'assistant' -%}"
    "{% generation %}{{ m.content }}"
    "{%- for tc in m.tool_calls or [] -%}"
    "{%- if tc.function.arguments is mapping -%}"
    "<call {{ tc.function.name }}>{% for k in tc.function.arguments %}<p {{ k }}>{{ tc.function.arguments[k] }}</p>{% endfor %}</call>"
    "{%- else -%}{{ raise_exception('function.arguments must be a dict') }}{%- endif -%}"
    "{%- endfor -%}"
    "</s>{% endgeneration %}"
    "{%- else -%}{{ m.content }}{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt %}<assistant>{% endif -%}"
)
_WORDS = ["<s>", "</s>", "<unk>", "<pad>", "<user>", "<assistant>", "<tool>", "<call", "</call>", "<p", "</p>", ">"]


def _save_tokenizer(path, model_type: str) -> str:
    vocab = {w: i for i, w in enumerate(_WORDS + [chr(c) for c in range(32, 127)])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend, bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>"
    )
    tok.chat_template = _TEMPLATE
    tok.save_pretrained(path)
    (path / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(path)


@pytest.fixture(scope="module")
def beam_dir(tmp_path_factory):
    return _save_tokenizer(tmp_path_factory.mktemp("beam_tok"), "beam")


def _conversation(arguments):
    return [
        {"role": "user", "content": "list files"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "ls", "arguments": arguments}}],
        },
        {"role": "tool", "tool_call_id": "c0", "content": "a.py"},
        {"role": "assistant", "content": "done"},
    ]


def test_string_arguments_render_like_dict_arguments(beam_dir):
    tok = DictToolArgumentsTokenizer.from_pretrained(beam_dir)
    as_dict = tok.apply_chat_template(_conversation({"path": "/w", "n": 2}), tokenize=False)
    as_str = tok.apply_chat_template(_conversation('{"path": "/w", "n": 2}'), tokenize=False)
    assert as_str == as_dict
    assert "<call ls><p path>/w</p><p n>2</p></call>" in as_str
    assert tok.apply_chat_template(_conversation('{"path": "/w", "n": 2}')) == tok.apply_chat_template(
        _conversation({"path": "/w", "n": 2})
    )


@pytest.mark.parametrize("arguments", ["[1, 2]", "not json", '"text"'])
def test_non_object_arguments_still_reach_the_template(beam_dir, arguments):
    tok = DictToolArgumentsTokenizer.from_pretrained(beam_dir)
    with pytest.raises(Exception, match="function.arguments must be a dict"):
        tok.apply_chat_template(_conversation(arguments), tokenize=False)


def test_input_conversation_is_not_mutated(beam_dir):
    tok = DictToolArgumentsTokenizer.from_pretrained(beam_dir)
    conversation = _conversation('{"path": "/w"}')
    before = copy.deepcopy(conversation)
    tok.apply_chat_template(conversation, tokenize=False)
    assert conversation == before


def test_batched_conversations(beam_dir):
    tok = DictToolArgumentsTokenizer.from_pretrained(beam_dir)
    rendered = tok.apply_chat_template([_conversation('{"a": 1}'), _conversation({"a": 1})], tokenize=False)
    assert rendered[0] == rendered[1]


def test_registry_wraps_beam_only(tmp_path, beam_dir):
    beam = NeMoAutoTokenizer.from_pretrained(beam_dir)
    assert "<p path>/w</p>" in beam.apply_chat_template(_conversation('{"path": "/w"}'), tokenize=False)

    other = NeMoAutoTokenizer.from_pretrained(_save_tokenizer(tmp_path / "llama_tok", "llama"))
    with pytest.raises(Exception, match="function.arguments must be a dict"):
        other.apply_chat_template(_conversation('{"path": "/w"}'), tokenize=False)


def test_copies_keep_the_wrapper_and_save_is_unchanged(tmp_path, beam_dir):
    tok = DictToolArgumentsTokenizer.from_pretrained(beam_dir)
    for copied in (pickle.loads(pickle.dumps(tok)), copy.deepcopy(tok)):  # noqa: S301
        assert "<p path>/w</p>" in copied.apply_chat_template(_conversation('{"path": "/w"}'), tokenize=False)

    tok.save_pretrained(tmp_path / "beam")
    NeMoAutoTokenizer.from_pretrained(beam_dir, tokenizer_backend="nemo_wrapped_auto").save_pretrained(
        tmp_path / "default"
    )
    for name in ("tokenizer_config.json", "tokenizer.json"):
        assert (tmp_path / "beam" / name).read_text() == (tmp_path / "default" / name).read_text()


def test_chat_dataset_accepts_string_and_dict_arguments(tmp_path, beam_dir):
    rows = [{"messages": _conversation('{"path": "/w"}')}, {"messages": _conversation({"path": "/w"})}]
    data = tmp_path / "train.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ds = ChatDataset(
        str(data), tokenizer=DictToolArgumentsTokenizer.from_pretrained(beam_dir), seq_length=None, padding=False
    )
    first, second = ds[0], ds[1]
    assert first["input_ids"] == second["input_ids"]
    assert first["labels"] == second["labels"]
    assert any(label != -100 for label in first["labels"])

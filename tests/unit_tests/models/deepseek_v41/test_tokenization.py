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

"""DeepSeek-V4.1 tool calls through ``NeMoAutoTokenizer`` and ``ChatDataset`` with the training template."""

import copy
import functools
import json
import pickle
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset

TEMPLATE = (
    Path(__file__).parents[4] / "examples/llm_finetune/deepseek_v41/deepseek_v41_flash_training_chat_template.jinja"
).read_text()
ARGUMENTS = {"city": "Paris", "days": 2, "metric": True}
DSML_CALL = (
    '<｜DSML｜ invoke name="get_weather">\n'
    '<｜DSML｜ parameter name="city" string="true">Paris</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="days" string="false">2</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="metric" string="false">true</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>"
)


def _save_tokenizer(path: Path, model_type: str) -> str:
    # One token per printable ASCII character; the template's special characters map to <unk>.
    specials = ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<unk>", "<pad>"]
    vocab = {w: i for i, w in enumerate(specials + [chr(c) for c in range(32, 127)])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token=specials[0],
        eos_token=specials[1],
        unk_token="<unk>",
        pad_token="<pad>",
    )
    tok.chat_template = TEMPLATE
    tok.save_pretrained(path)
    (path / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(path)


@pytest.fixture(scope="module")
def v41_dir(tmp_path_factory):
    return _save_tokenizer(tmp_path_factory.mktemp("v41_tok"), "deepseek_v41")


def _conversation(arguments):
    call = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": arguments}}
    return [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
    ]


def test_registry_parses_json_string_arguments_for_v41(v41_dir):
    tok = NeMoAutoTokenizer.from_pretrained(v41_dir)
    as_str = tok.apply_chat_template(_conversation(json.dumps(ARGUMENTS)), tokenize=False, thinking_mode="chat")
    as_dict = tok.apply_chat_template(_conversation(ARGUMENTS), tokenize=False, thinking_mode="chat")
    assert as_str == as_dict
    assert DSML_CALL in as_str


def test_wrapped_tokenizer_survives_pickle_and_deepcopy(v41_dir):
    tok = NeMoAutoTokenizer.from_pretrained(v41_dir)
    assert isinstance(tok.apply_chat_template, functools.partial)
    expected = tok.apply_chat_template(_conversation(ARGUMENTS), tokenize=False, thinking_mode="chat")
    for copied in (pickle.loads(pickle.dumps(tok)), copy.deepcopy(tok)):  # noqa: S301
        rendered = copied.apply_chat_template(
            _conversation(json.dumps(ARGUMENTS)), tokenize=False, thinking_mode="chat"
        )
        assert rendered == expected


def test_unwrapped_tokenizer_rejects_json_string_arguments(tmp_path):
    other = NeMoAutoTokenizer.from_pretrained(_save_tokenizer(tmp_path / "llama_tok", "llama"))
    with pytest.raises(Exception, match="NeMoAutoTokenizer"):
        other.apply_chat_template(_conversation(json.dumps(ARGUMENTS)), tokenize=False, thinking_mode="chat")


def test_chat_dataset_trains_string_and_dict_arguments_identically(tmp_path, v41_dir):
    rows = [{"messages": _conversation(json.dumps(ARGUMENTS))}, {"messages": _conversation(ARGUMENTS)}]
    data = tmp_path / "train.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    dataset = ChatDataset(
        str(data),
        tokenizer=NeMoAutoTokenizer.from_pretrained(v41_dir),
        chat_template_kwargs={"thinking_mode": "chat"},
    )
    first, second = dataset[0], dataset[1]
    assert first["input_ids"] == second["input_ids"]
    assert first["labels"] == second["labels"]
    # Each argument is its own supervised DSML parameter, not one parameter named "arguments".
    supervised = "".join(dataset.tokenizer.convert_ids_to_tokens([t for t in first["labels"] if t != -100]))
    assert 'parameter name="city"' in supervised and 'parameter name="metric"' in supervised
    assert 'parameter name="arguments"' not in supervised

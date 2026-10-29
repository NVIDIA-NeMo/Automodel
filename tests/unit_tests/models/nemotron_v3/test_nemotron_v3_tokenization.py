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

"""Tool-call arguments reach Nemotron chat templates as mappings."""

import copy
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from nemo_automodel import NeMoAutoTokenizer
from nemo_automodel.components.datasets.llm.chat_dataset import ChatDataset
from nemo_automodel.components.datasets.llm.formatting_utils import format_chat_template


@pytest.mark.parametrize("arguments", [{"command": "ls", "limit": 2}, '{"command": "ls", "limit": 2}'])
def test_chat_dataset_renders_tool_parameters(tmp_path: Path, arguments: str | dict[str, str | int]) -> None:
    # Exercise the mapping-only items filter used by Nemotron's released template.
    template = (
        "{% for message in messages %}<{{ message.role }}>"
        "{% if message.role == 'assistant' %}{% generation %}"
        "{% for call in message.tool_calls %}{{ call.function.name }}:"
        "{% for key, value in call.function.arguments|items %}{{ key }}={{ value }};{% endfor %}"
        "{% endfor %}<eos>{% endgeneration %}"
        "{% else %}{{ message.content }}{% endif %}{% endfor %}"
    )
    tokens = ["<unk>", "<pad>", "<eos>"] + [chr(i) for i in range(32, 127)]
    backend = Tokenizer(models.WordLevel({token: i for i, token in enumerate(tokens)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
    reference = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>", chat_template=template
    )
    tokenizer_dir = tmp_path / "tokenizer"
    reference.save_pretrained(tokenizer_dir)
    (tokenizer_dir / "config.json").write_text(json.dumps({"model_type": "nemotron_h"}))
    messages = [
        {"role": "user", "content": "List files."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "bash", "arguments": arguments}}],
        },
    ]
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps({"messages": messages}) + "\n")
    tokenizer = NeMoAutoTokenizer.from_pretrained(str(tokenizer_dir))
    dataset = ChatDataset(str(data), tokenizer=tokenizer)

    expected_messages = copy.deepcopy(messages)
    expected_messages[1]["tool_calls"][0]["function"]["arguments"] = {"command": "ls", "limit": 2}
    expected_text = "<user>List files.<assistant>bash:command=ls;limit=2;<eos>"
    assert tokenizer.apply_chat_template(messages, tokenize=False) == expected_text
    assert reference.apply_chat_template(expected_messages, tokenize=False) == expected_text
    expected = format_chat_template(reference, expected_messages, reference.eos_token_id, reference.pad_token_id)
    actual = dataset[0]
    assert actual == expected
    assert any(label != -100 for label in actual["labels"])
    assert dataset.dataset[0]["messages"] == messages

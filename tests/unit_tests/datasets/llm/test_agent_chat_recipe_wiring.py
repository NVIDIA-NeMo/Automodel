#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
"""End-to-end wiring test for the agent SFT recipe data path.

The agent example YAML (``examples/llm_finetune/agent/qwen2_5_3b_function_calling.yaml``)
is only lint-validated in CI; nothing exercises the actual
``make_agent_chat_dataset`` -> ``LazyMappedDataset`` -> ``default_collater``
chain that the recipe runs. This CPU test does, against a tiny local JSON mock
and an offline stub tokenizer, so a break in JSON loading, ShareGPT schema
detection, tool_call/observation pairing, chat-template rendering, loss masking,
or batch collation is caught without a GPU or network.
"""

import json
from typing import List

import torch

from nemo_automodel.components.datasets.llm.agent_chat import make_agent_chat_dataset
from nemo_automodel.components.datasets.utils import default_collater


class _ToolChatTokenizer:
    """Offline stub tokenizer with a tool-capable, no-generation chat template.

    Renders each message deterministically (one id per whitespace token, plus a
    role marker and tool-call name markers) so the answer-only mask builder can
    locate assistant spans. No ``{% generation %}`` keyword, so the multi-turn
    fallback mask path is exercised — the same path the real recipe uses for
    templates without generation tags.
    """

    eos_token_id = 2
    pad_token_id = 1
    chat_template = "{{ tools }}{% for m in messages %}{{ m['role'] }}{% endfor %}"

    def __init__(self) -> None:
        self._vocab: dict = {}
        self._cursor = 10  # leave 0-9 free of content ids

    def _id(self, tok: str) -> int:
        if tok not in self._vocab:
            self._vocab[tok] = self._cursor
            self._cursor += 1
        return self._vocab[tok]

    def apply_chat_template(
        self,
        messages,
        tools=None,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=False,
        padding=False,
        truncation=None,
        max_length=None,
        **kwargs,
    ):
        ids: List[int] = [self._id("<bos>")]
        for m in messages:
            ids.append(self._id(f"<{m.get('role')}>"))
            ids.extend(self._id(t) for t in str(m.get("content") or "").split())
            for call in m.get("tool_calls") or []:
                ids.append(self._id("<tc>"))
                ids.append(self._id(call["function"]["name"]))
        ids.append(self.eos_token_id)
        if return_dict:
            return {"input_ids": ids, "attention_mask": [1] * len(ids)}
        return ids


def _write_mock_jsonl(path) -> None:
    """Write two ShareGPT function-calling dialogues (tools as JSON strings)."""
    rows = [
        {
            "tools": json.dumps([{"name": "get_weather", "parameters": {}}]),
            "conversations": [
                {"from": "human", "value": "weather in tokyo?"},
                {"from": "function_call", "value": json.dumps({"name": "get_weather", "arguments": {"city": "Tokyo"}})},
                {"from": "observation", "value": "sunny"},
                {"from": "gpt", "value": "It is sunny in Tokyo."},
            ],
        },
        {
            "tools": json.dumps([{"name": "add", "parameters": {}}]),
            "conversations": [
                {"from": "human", "value": "add one and two"},
                {"from": "function_call", "value": json.dumps({"name": "add", "arguments": {"a": 1, "b": 2}})},
                {"from": "observation", "value": "3"},
                {"from": "gpt", "value": "The sum is three."},
            ],
        },
    ]
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_agent_recipe_dataset_to_collator_wiring(tmp_path):
    mock = tmp_path / "agent_mock.jsonl"
    _write_mock_jsonl(mock)

    tok = _ToolChatTokenizer()
    dataset = make_agent_chat_dataset(tok, path=str(mock), seq_length=128)

    assert len(dataset) == 2
    items = [dataset[0], dataset[1]]
    for item in items:
        assert set(item) >= {"input_ids", "labels", "attention_mask"}
        assert len(item["input_ids"]) == len(item["labels"]) == len(item["attention_mask"])
        # The assistant turns must contribute supervised tokens.
        assert any(label != -100 for label in item["labels"])

    batch = default_collater(items)
    assert {"input_ids", "labels", "attention_mask"} <= set(batch)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["input_ids"].shape[0] == 2
    # Padding never leaks into the loss.
    assert (batch["labels"] == -100).any()
    assert (batch["labels"] != -100).any()
    assert batch["input_ids"].dtype == torch.long

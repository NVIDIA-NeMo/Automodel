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

import json

import pytest

from nemo_automodel.components.datasets.llm import xlam


def test_json_load_if_str_roundtrip_and_passthrough():
    payload = {"a": 1, "b": "two"}
    assert xlam._json_load_if_str(json.dumps(payload)) == payload
    assert xlam._json_load_if_str(payload) is payload


def test_convert_tools_maps_schema_and_required_fields():
    raw_tools = [
        {
            "name": "weather",
            "description": "desc",
            "parameters": {
                "city": {"type": "str", "description": "city name"},
                "days": {"type": "int", "default": 0},
                "units": {"type": "bool", "enum": [True, False]},
            },
        },
        {"description": "missing name should be skipped"},
    ]

    converted = xlam._convert_tools(raw_tools)
    assert len(converted) == 1

    func_def = converted[0]["function"]
    assert func_def["name"] == "weather"
    assert func_def["description"] == "desc"

    schema = func_def["parameters"]
    assert schema["type"] == "object"
    props = schema["properties"]
    assert props["city"]["type"] == "string" and props["city"]["description"] == "city name"
    assert props["days"]["type"] == "integer" and props["days"]["default"] == 0
    assert props["units"]["type"] == "boolean" and props["units"]["enum"] == [True, False]
    assert set(schema["required"]) == {"city", "units"}


def test_convert_tool_calls_preserves_arguments_and_ids():
    raw_calls = [
        {"name": "foo", "arguments": '{"x":1}'},
        {"name": "bar", "arguments": "{}"},
        {"arguments": "{}"},  # skipped: missing name
    ]

    converted = xlam._convert_tool_calls(raw_calls, example_id=10)
    assert [c["function"]["name"] for c in converted] == ["foo", "bar"]
    assert [c["id"] for c in converted] == ["call_10_0", "call_10_1"]
    assert converted[0]["function"]["arguments"] == '{"x":1}'


def test_format_example_builds_chat_payload(monkeypatch):
    captured = {}

    def fake_format_chat_template(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(xlam, "format_chat_template", fake_format_chat_template)

    class Tok:
        eos_token_id = 7
        pad_token_id = 3

    tok = Tok()
    example = {
        "id": 42,
        "query": "hello",
        "tools": '[{"name":"calc","parameters":{"x":{"type":"int","description":"num"}}}]',
        "answers": '[{"name":"calc","arguments":"{\\"x\\":1}"}]',
    }

    result = xlam._format_example(
        example,
        tok,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        seq_length=8,
        padding=True,
        truncation="longest",
    )

    assert result == {"ok": True}
    assert captured["tokenizer"] is tok
    assert captured["eos_token_id"] == tok.eos_token_id
    assert captured["pad_token_id"] == tok.pad_token_id
    assert captured["seq_length"] == 8
    assert captured["padding"] is True
    assert captured["truncation"] == "longest"
    assert captured["answer_only_loss_mask"] is True

    tools = captured["tools"]
    assert tools[0]["function"]["name"] == "calc"
    assert tools[0]["function"]["parameters"]["properties"]["x"]["type"] == "integer"

    formatted = captured["formatted_text"]
    assert formatted[0] == {"role": "user", "content": "hello"}
    assistant = formatted[1]
    assert assistant["role"] == "assistant" and assistant["content"] == ""
    call = assistant["tool_calls"][0]
    assert call["id"] == "call_42_0"
    assert call["function"]["name"] == "calc"
    assert call["function"]["arguments"] == '{"x":1}'


def test_make_xlam_dataset_respects_limit_and_maps(monkeypatch):
    rows = [
        {"id": 0, "query": "q0", "answers": [], "tools": []},
        {"id": 1, "query": "q1", "answers": [], "tools": []},
    ]
    dataset_calls = {}

    class DummyDataset:
        def __init__(self, items):
            self.items = items
            self.map_calls = []

        def map(self, fn, batched=False, remove_columns=None):
            self.map_calls.append({"batched": batched, "remove_columns": remove_columns})
            return [fn(item) for item in self.items]

    dummy_ds = DummyDataset(rows)

    def fake_load_dataset(name, split):
        dataset_calls["name"] = name
        dataset_calls["split"] = split
        return dummy_ds

    monkeypatch.setattr(xlam, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(xlam, "_add_pad_token", lambda tok: 13)

    fmt_calls = []

    def fake_format_example(example, tokenizer, eos_token_id, pad_token_id, seq_length, padding, truncation):
        fmt_calls.append(
            {
                "example": example,
                "tokenizer": tokenizer,
                "eos": eos_token_id,
                "pad": pad_token_id,
                "seq_length": seq_length,
                "padding": padding,
                "truncation": truncation,
            }
        )
        return {"formatted": example["id"]}

    monkeypatch.setattr(xlam, "_format_example", fake_format_example)

    class Tok:
        eos_token_id = 5

    tok = Tok()

    result = xlam.make_xlam_dataset(
        tokenizer=tok,
        seq_length=16,
        padding=True,
        truncation="longest",
        limit_dataset_samples=2,
        split="train",
        dataset_name="dummy",
    )

    assert dataset_calls == {"name": "dummy", "split": "train[:2]"}
    assert result == [{"formatted": 0}, {"formatted": 1}]
    assert dummy_ds.map_calls[0]["batched"] is False
    assert dummy_ds.map_calls[0]["remove_columns"] == ["id", "query", "answers", "tools"]
    assert fmt_calls[0]["pad"] == 13 and fmt_calls[0]["eos"] == tok.eos_token_id


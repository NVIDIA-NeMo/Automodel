#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import json
from pathlib import Path

import pytest

import nemo_automodel.components.datasets.llm.tool_calling_chat_dataset as tcd


def test_is_hf_repo_id_and_as_iter_and_normalize():
    # _is_hf_repo_id basic behavior
    assert tcd._is_hf_repo_id("org/name") is True
    # local-like path should be False (Path exists check may vary, so use a name with no slash)
    assert tcd._is_hf_repo_id("localpath") is False

    # _as_iter yields strings and rejects non-strings
    assert list(tcd._as_iter("a")) == ["a"]
    assert list(tcd._as_iter(["a", "b"])) == ["a", "b"]
    with pytest.raises(ValueError):
        list(tcd._as_iter(["a", 1]))

    # _normalize_messages converts content to string and validates roles
    msgs = [
        {"role": "system", "content": 123},
        {"role": "user", "content": None},
        {"role": "assistant", "content": True},
    ]
    norm = tcd._normalize_messages(msgs)
    assert [m["role"] for m in norm] == ["system", "user", "assistant"]
    assert [m["content"] for m in norm] == ["123", "None", "True"]

    with pytest.raises(ValueError):
        tcd._normalize_messages([{ "role": "badrole", "content": "x" }])


def test_load_openai_messages_local_and_errors(tmp_path, monkeypatch):
    # Create local files: JSONL and JSON
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text("\n".join([
        json.dumps({"messages": [{"role": "user", "content": "u1"}]}),
        json.dumps({"messages": [{"role": "assistant", "content": "a1"}]}),
    ]), encoding="utf-8")

    json_file = tmp_path / "data.json"
    json_file.write_text(json.dumps([
        {"messages": [{"role": "user", "content": "u2"}]},
        {"messages": [{"role": "assistant", "content": "a2"}]},
    ]), encoding="utf-8")

    rows = tcd._load_openai_messages([str(jsonl), str(json_file)])
    assert len(rows) == 4
    assert rows[0]["messages"][0]["content"] == "u1"

    # Missing file
    with pytest.raises(FileNotFoundError):
        tcd._load_openai_messages([str(jsonl), str(json_file), str(tmp_path / "missing.json")])

    # No files
    with pytest.raises(RuntimeError):
        tcd._load_openai_messages([])

    # HF branch: force as repo-id and ensure delegated call is returned
    monkeypatch.setattr(tcd, "_is_hf_repo_id", lambda v: True)
    sentinel = object()
    monkeypatch.setattr(tcd, "load_dataset", lambda *a, **k: sentinel)
    assert tcd._load_openai_messages("org/name", split="train") is sentinel


def test_tool_calling_chat_dataset_happy_path_and_edge_cases(monkeypatch):
    # Stub tokenizer
    class Tok:
        eos_token_id = 1
        chat_template = "{{ default }}"

    tok = Tok()

    # Monkeypatch helpers used inside the module under test
    monkeypatch.setattr(tcd, "_has_chat_template", lambda _tok: True)
    monkeypatch.setattr(tcd, "_add_pad_token", lambda _tok: 3)

    calls = []

    def fake_format(tokenizer, normalized, eos_id, pad_id, **kwargs):
        calls.append({
            "normalized": normalized,
            "eos": eos_id,
            "pad": pad_id,
            "kwargs": kwargs,
        })
        return {"input_ids": [1, 2], "labels": [0, 1], "attention_mask": [1, 1]}

    monkeypatch.setattr(tcd, "format_chat_template", fake_format)

    # Two rows: one with valid tools list, one with invalid tools type that should be nulled
    dataset_rows = [
        {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
            "tools": [{"type": "function", "function": {"name": "t"}}],
        },
        {
            "messages": [
                {"role": "user", "content": 7},
                {"role": "assistant", "content": 8},
            ],
            "tools": {"not": "alist"},
        },
    ]

    monkeypatch.setattr(tcd, "_load_openai_messages", lambda *a, **k: dataset_rows)

    ds = tcd.ToolCallingChatDataset("ignored", tok, seq_length=16, start_of_turn_token="<|sot|>", chat_template="OVERRIDE")

    # init effects
    assert ds.pad_token_id == 3  # from _add_pad_token
    assert tok.chat_template == "OVERRIDE"
    assert len(ds) == 2

    item0 = ds[0]
    item1 = ds[1]
    assert item0["input_ids"] == [1, 2] and item1["attention_mask"] == [1, 1]

    # Verify calls captured the tools argument behavior
    assert calls[0]["kwargs"]["tools"] == dataset_rows[0]["tools"]
    assert calls[1]["kwargs"]["tools"] is None

    # Bad row: messages not a list → ValueError
    monkeypatch.setattr(tcd, "_load_openai_messages", lambda *a, **k: [{"messages": "oops"}])
    ds_bad = tcd.ToolCallingChatDataset("ignored", tok)
    with pytest.raises(ValueError):
        _ = ds_bad[0]


def test_tool_calling_chat_dataset_errors(monkeypatch):
    # No tokenizer
    with pytest.raises(ValueError):
        tcd.ToolCallingChatDataset("ignored", None)

    # Tokenizer provided but missing chat template support
    class Tok:
        eos_token_id = 1
        chat_template = None

    monkeypatch.setattr(tcd, "_has_chat_template", lambda _tok: False)
    with pytest.raises(ValueError):
        tcd.ToolCallingChatDataset("ignored", Tok())



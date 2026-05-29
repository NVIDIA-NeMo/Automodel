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

from nemo_automodel.components.datasets.llm import agent_chat


def test_json_load_if_str_roundtrip_and_passthrough():
    payload = {"a": 1, "b": "two"}
    assert agent_chat._json_load_if_str(json.dumps(payload)) == payload
    assert agent_chat._json_load_if_str(payload) is payload


def test_sharegpt_to_chatml_maps_all_roles():
    conversations = [
        {"from": "system", "value": "sys"},
        {"from": "human", "value": "hi"},
        {"from": "gpt", "value": "hello"},
        {"from": "function_call", "value": '{"name":"f","arguments":{}}'},
        {"from": "observation", "value": "obs"},
    ]
    out = agent_chat._sharegpt_to_chatml(conversations)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool_call", "tool_response"]
    assert out[1]["content"] == "hi"
    assert out[3]["content"] == '{"name":"f","arguments":{}}'


def test_sharegpt_to_chatml_rejects_unknown_role():
    with pytest.raises(ValueError, match="Unsupported sharegpt role"):
        agent_chat._sharegpt_to_chatml([{"from": "narrator", "value": "x"}])


def test_convert_messages_collapses_parallel_tool_calls_and_pairs_responses():
    messages = [
        {"role": "user", "content": "weather in BJ and SH?"},
        {"role": "tool_call", "content": '{"name":"aqi","arguments":{"city":"BJ"}}'},
        {"role": "tool_call", "content": '{"name":"aqi","arguments":{"city":"SH"}}'},
        {"role": "tool_response", "content": "BJ=10"},
        {"role": "tool_response", "content": "SH=72"},
        {"role": "assistant", "content": "BJ good, SH mild."},
    ]
    out = agent_chat._convert_messages(messages, example_id=42)

    # user / assistant(2 tool_calls) / tool / tool / assistant
    assert [m["role"] for m in out] == ["user", "assistant", "tool", "tool", "assistant"]

    assistant_call = out[1]
    assert assistant_call["content"] == ""
    assert len(assistant_call["tool_calls"]) == 2
    assert [c["function"]["name"] for c in assistant_call["tool_calls"]] == ["aqi", "aqi"]
    assert [c["id"] for c in assistant_call["tool_calls"]] == ["call_42_0", "call_42_1"]
    # arguments stay as JSON strings when input was a dict
    assert assistant_call["tool_calls"][0]["function"]["arguments"] == '{"city": "BJ"}'

    # tool_response pairs in order with the prior tool_call ids
    assert out[2]["tool_call_id"] == "call_42_0" and out[2]["content"] == "BJ=10"
    assert out[3]["tool_call_id"] == "call_42_1" and out[3]["content"] == "SH=72"

    assert out[4]["role"] == "assistant" and out[4]["content"] == "BJ good, SH mild."


def test_convert_messages_passes_string_arguments_through_unchanged():
    raw_args = '{"city":"BJ"}'
    messages = [
        {"role": "user", "content": "?"},
        {"role": "tool_call", "content": json.dumps({"name": "aqi", "arguments": raw_args})},
    ]
    out = agent_chat._convert_messages(messages)
    assert out[1]["tool_calls"][0]["function"]["arguments"] == raw_args


def test_convert_messages_merges_tool_calls_into_prior_assistant_turn():
    # Datasets like Swift's agent traces emit an assistant "think/text" turn
    # immediately followed by tool_call turns. Logically they are one
    # assistant message; emitting two consecutive assistant turns would
    # diverge from what the model produces at inference and may render as
    # two separate `<|im_start|>assistant` blocks under some chat templates.
    messages = [
        {"role": "user", "content": "click button"},
        {"role": "assistant", "content": "<think>I should click at (1, 2).</think>"},
        {"role": "tool_call", "content": '{"name":"click","arguments":{"x":1,"y":2}}'},
        {"role": "tool_response", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    out = agent_chat._convert_messages(messages, example_id="abc")

    # user / assistant(think + tool_calls) / tool / assistant — not 5 turns
    assert [m["role"] for m in out] == ["user", "assistant", "tool", "assistant"]

    merged = out[1]
    assert merged["content"] == "<think>I should click at (1, 2).</think>"
    assert len(merged["tool_calls"]) == 1
    assert merged["tool_calls"][0]["function"]["name"] == "click"
    assert merged["tool_calls"][0]["id"] == "call_abc_0"
    assert out[2]["tool_call_id"] == "call_abc_0"


def test_convert_messages_does_not_merge_when_prior_assistant_already_has_tool_calls():
    # Two distinct rounds of tool calls separated by a tool_response must
    # stay as two assistant turns; merging would conflate independent calls.
    messages = [
        {"role": "user", "content": "?"},
        {"role": "tool_call", "content": '{"name":"a","arguments":{}}'},
        {"role": "tool_response", "content": "ra"},
        {"role": "tool_call", "content": '{"name":"b","arguments":{}}'},
        {"role": "tool_response", "content": "rb"},
    ]
    out = agent_chat._convert_messages(messages)
    assert [m["role"] for m in out] == ["user", "assistant", "tool", "assistant", "tool"]
    assert out[1]["tool_calls"][0]["function"]["name"] == "a"
    assert out[3]["tool_calls"][0]["function"]["name"] == "b"


def test_convert_messages_rejects_unknown_role():
    with pytest.raises(ValueError, match="Unsupported role"):
        agent_chat._convert_messages([{"role": "narrator", "content": "x"}])


def test_convert_messages_requires_tool_call_name():
    with pytest.raises(ValueError, match="tool_call missing `name`"):
        agent_chat._convert_messages([{"role": "tool_call", "content": "{}"}])


def test_format_example_builds_chat_payload_for_chatml(monkeypatch):
    captured = {}

    def fake_format_chat_template(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(agent_chat, "format_chat_template", fake_format_chat_template)

    class Tok:
        eos_token_id = 7
        pad_token_id = 3

    tok = Tok()
    example = {
        "id": 5,
        "tools": '[{"type":"function","function":{"name":"aqi","parameters":{}}}]',
        "messages": [
            {"role": "user", "content": "weather?"},
            {"role": "tool_call", "content": '{"name":"aqi","arguments":{"city":"BJ"}}'},
            {"role": "tool_response", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ],
    }

    result = agent_chat._format_example(
        example,
        tok,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        seq_length=16,
        padding=False,
        truncation=False,
    )

    assert result == {"ok": True}
    assert captured["tokenizer"] is tok
    assert captured["seq_length"] == 16
    assert captured["answer_only_loss_mask"] is True

    tools = captured["tools"]
    assert tools[0]["function"]["name"] == "aqi"

    formatted = captured["formatted_text"]
    assert [m["role"] for m in formatted] == ["user", "assistant", "tool", "assistant"]
    assert formatted[1]["tool_calls"][0]["id"] == "call_5_0"
    assert formatted[2]["tool_call_id"] == "call_5_0"


def test_format_example_supports_sharegpt_input(monkeypatch):
    captured = {}

    def fake_format_chat_template(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(agent_chat, "format_chat_template", fake_format_chat_template)

    class Tok:
        eos_token_id = 0
        pad_token_id = 0

    example = {
        "tools": [],  # empty list should normalize to None
        "conversations": [
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": "hello"},
        ],
    }

    agent_chat._format_example(example, Tok(), 0, 0)

    assert captured["tools"] is None
    assert [m["role"] for m in captured["formatted_text"]] == ["user", "assistant"]


def test_format_example_rejects_missing_messages_and_conversations():
    class Tok:
        eos_token_id = 0
        pad_token_id = 0

    with pytest.raises(ValueError, match="missing both `messages` and `conversations`"):
        agent_chat._format_example({}, Tok(), 0, 0)


def test_format_example_rejects_non_list_tools():
    class Tok:
        eos_token_id = 0
        pad_token_id = 0

    example = {"tools": '{"not": "a list"}', "messages": [{"role": "user", "content": "x"}]}
    with pytest.raises(ValueError, match="`tools` must be a list"):
        agent_chat._format_example(example, Tok(), 0, 0)


def test_make_agent_chat_dataset_requires_exactly_one_source():
    class Tok:
        eos_token_id = 0
        pad_token_id = 0

    with pytest.raises(ValueError, match="Exactly one of"):
        agent_chat.make_agent_chat_dataset(Tok())
    with pytest.raises(ValueError, match="Exactly one of"):
        agent_chat.make_agent_chat_dataset(Tok(), dataset_name="foo", path="bar.json")


def test_make_agent_chat_dataset_loads_hub_split_with_limit(monkeypatch):
    rows = [
        {"id": 0, "messages": [{"role": "user", "content": "q0"}], "tools": []},
        {"id": 1, "messages": [{"role": "user", "content": "q1"}], "tools": []},
    ]
    captured_load = {}

    class DummyDataset:
        def __init__(self, items):
            self.items = items

        def __getitem__(self, idx):
            return self.items[idx]

        def __len__(self):
            return len(self.items)

    def fake_load_dataset(name_or_loader, split=None, data_files=None):
        captured_load["name"] = name_or_loader
        captured_load["split"] = split
        captured_load["data_files"] = data_files
        return DummyDataset(rows)

    monkeypatch.setattr(agent_chat, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(agent_chat, "_add_pad_token", lambda tok: 13)
    monkeypatch.setattr(agent_chat, "_format_example", lambda ex, *a, **kw: {"formatted": ex["id"]})

    class Tok:
        eos_token_id = 5

    ds = agent_chat.make_agent_chat_dataset(
        tokenizer=Tok(),
        dataset_name="dummy/agent",
        split="train",
        limit_dataset_samples=2,
    )

    assert captured_load["name"] == "dummy/agent"
    assert captured_load["split"] == "train[:2]"
    assert [ds[i] for i in range(len(ds))] == [{"formatted": 0}, {"formatted": 1}]


def test_convert_messages_orphan_tool_response_gets_synthetic_id():
    # tool_response that does not follow a tool_call group must fall back to a
    # synthetic tool_call_id rather than silently reusing IDs from an earlier
    # tool_call group.
    messages = [
        {"role": "tool_call", "content": '{"name":"f","arguments":{}}'},
        {"role": "tool_response", "content": "ok"},
        {"role": "assistant", "content": "done"},
        {"role": "tool_response", "content": "orphan"},
    ]
    out = agent_chat._convert_messages(messages, example_id=7)

    paired_id = out[1]["tool_call_id"]
    orphan_id = out[3]["tool_call_id"]
    assert paired_id == out[0]["tool_calls"][0]["id"]
    assert orphan_id != paired_id
    assert "response" in orphan_id


def test_convert_messages_tool_call_with_none_content_is_rejected():
    # A tool_call dict with ``content: None`` must surface as a clear
    # ValueError about the missing ``name`` rather than an AttributeError.
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "tool_call", "content": None},
    ]
    with pytest.raises(ValueError, match="tool_call missing `name`"):
        agent_chat._convert_messages(messages)


def test_format_example_accepts_empty_tools_string(monkeypatch):
    # ``tools: ""`` (used by some on-disk ShareGPT exports to mean "no tools")
    # must be normalized to ``None`` rather than crashing ``json.loads("")``.
    captured = {}

    def fake_format_chat_template(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(agent_chat, "format_chat_template", fake_format_chat_template)

    class Tok:
        eos_token_id = 5
        pad_token_id = 0

    agent_chat._format_example(
        {"tools": "", "messages": [{"role": "user", "content": "hi"}]},
        Tok(),
        eos_token_id=5,
        pad_token_id=0,
    )
    assert captured["tools"] is None


def test_mask_labels_to_last_turn_keeps_only_final_run():
    # Two supervised assistant runs separated by an ignored (user/tool) run.
    labels = [-100, 1, 2, -100, -100, 3, 4, -100]
    agent_chat._mask_labels_to_last_turn(labels)
    assert labels == [-100, -100, -100, -100, -100, 3, 4, -100]


def test_mask_labels_to_last_turn_single_run_unchanged():
    labels = [-100, -100, 5, 6, 7]
    agent_chat._mask_labels_to_last_turn(labels)
    assert labels == [-100, -100, 5, 6, 7]


def test_mask_labels_to_last_turn_no_supervised_tokens_is_noop():
    labels = [-100, -100, -100]
    agent_chat._mask_labels_to_last_turn(labels)
    assert labels == [-100, -100, -100]


def test_format_example_train_on_last_turn_only_masks_earlier_turns(monkeypatch):
    # ``labels`` carry two supervised assistant runs; with the flag set only
    # the final run survives. Without it, both runs stay supervised.
    def fake_format_chat_template(**kwargs):
        return {"input_ids": [10, 11, 12, 13, 14, 15], "labels": [-100, 1, -100, -100, 2, 3]}

    monkeypatch.setattr(agent_chat, "format_chat_template", fake_format_chat_template)

    class Tok:
        eos_token_id = 0
        pad_token_id = 0

    example = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x"}]}

    kept = agent_chat._format_example(example, Tok(), 0, 0)
    assert kept["labels"] == [-100, 1, -100, -100, 2, 3]

    masked = agent_chat._format_example(example, Tok(), 0, 0, train_on_last_turn_only=True)
    assert masked["labels"] == [-100, -100, -100, -100, 2, 3]


def test_convert_messages_preserves_assistant_reasoning_content():
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4", "reasoning_content": "add two and two"},
    ]
    out = agent_chat._convert_messages(messages)
    assert out[1]["content"] == "4"
    assert out[1]["reasoning_content"] == "add two and two"


def test_convert_messages_reasoning_survives_tool_call_merge():
    # An assistant reasoning turn immediately followed by a tool_call group
    # merges into one assistant message that keeps the reasoning trace.
    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "content": "", "reasoning_content": "I should call the weather tool"},
        {"role": "tool_call", "content": '{"name":"aqi","arguments":{"city":"BJ"}}'},
    ]
    out = agent_chat._convert_messages(messages, example_id=1)
    assert [m["role"] for m in out] == ["user", "assistant"]
    assert out[1]["reasoning_content"] == "I should call the weather tool"
    assert out[1]["tool_calls"][0]["function"]["name"] == "aqi"


def test_sharegpt_to_chatml_passes_reasoning_content_through():
    out = agent_chat._sharegpt_to_chatml([{"from": "gpt", "value": "hi", "reasoning_content": "be polite"}])
    assert out[0] == {"role": "assistant", "content": "hi", "reasoning_content": "be polite"}


def test_format_example_forwards_mask_reasoning_content(monkeypatch):
    captured = {}

    def fake_format_chat_template(**kwargs):
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(agent_chat, "format_chat_template", fake_format_chat_template)

    class Tok:
        eos_token_id = 0
        pad_token_id = 0

    example = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}

    agent_chat._format_example(example, Tok(), 0, 0)
    assert captured["mask_reasoning_content"] is False

    agent_chat._format_example(example, Tok(), 0, 0, mask_reasoning_content=True)
    assert captured["mask_reasoning_content"] is True

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

import pytest

import nemo_automodel.components.datasets.llm.chat_dataset as tcd
from nemo_automodel.components.datasets.llm.formatting_utils import _resolve_chat_template


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
    assert [m["content"] for m in norm] == ["123", "", "True"]

    with pytest.raises(ValueError):
        tcd._normalize_messages([{"role": "badrole", "content": "x"}])


def test_normalize_messages_supports_reasoning_and_tool_call_fields():
    msgs = [
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "think step",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "Seattle"}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": None},
    ]

    norm = tcd._normalize_messages(msgs)
    assert norm[0]["content"] == ""
    assert norm[0]["reasoning_content"] == "think step"
    assert norm[0]["tool_calls"][0]["function"]["arguments"] == '{"city": "Seattle"}'
    assert norm[1]["content"] == ""

    none_reasoning = tcd._normalize_messages([{"role": "assistant", "content": "", "reasoning_content": None}])
    assert none_reasoning[0]["reasoning_content"] == ""


def test_normalize_tool_calls_autofills_missing_id_and_type():
    msgs = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "fn_a", "arguments": {"x": 1}}},
                {"id": "", "type": "", "function": {"name": "fn_b", "arguments": "{}"}},
            ],
        }
    ]
    norm = tcd._normalize_messages(msgs)
    calls = norm[0]["tool_calls"]
    assert calls[0]["id"] == "call_0"
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["arguments"] == '{"x": 1}'
    assert calls[1]["id"] == "call_1"
    assert calls[1]["type"] == "function"


@pytest.mark.parametrize(
    ("message", "error_pattern"),
    [
        (
            [{"role": "assistant", "content": "", "reasoning_content": 1}],
            "reasoning_content",
        ),
        (
            [{"role": "assistant", "content": "", "tool_calls": "bad"}],
            "tool_calls",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": 123, "type": "function", "function": {"name": "fn", "arguments": "{}"}}],
                }
            ],
            "tool_calls\\[0\\]\\.id",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call_1", "type": 1, "function": {"name": "fn", "arguments": "{}"}}],
                }
            ],
            "tool_calls\\[0\\]\\.type",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"arguments": "{}"}}],
                }
            ],
            "function.name",
        ),
        (
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "fn"}}],
                }
            ],
            "function.arguments",
        ),
        (
            [{"role": "tool", "content": "result"}],
            "tool_call_id",
        ),
    ],
)
def test_normalize_messages_rejects_malformed_reasoning_and_tool_fields(message, error_pattern):
    with pytest.raises(ValueError, match=error_pattern):
        tcd._normalize_messages(message)


def test_load_openai_messages_local_and_errors(tmp_path, monkeypatch):
    # Create local files: JSONL and JSON
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps({"messages": [{"role": "user", "content": "u1"}]}),
                json.dumps({"messages": [{"role": "assistant", "content": "a1"}]}),
            ]
        ),
        encoding="utf-8",
    )

    json_file = tmp_path / "data.json"
    json_file.write_text(
        json.dumps(
            [
                {"messages": [{"role": "user", "content": "u2"}]},
                {"messages": [{"role": "assistant", "content": "a2"}]},
            ]
        ),
        encoding="utf-8",
    )

    rows = tcd._load_openai_messages([str(jsonl), str(json_file)])
    assert len(rows) == 4
    assert rows[0]["messages"][0]["content"] == "u1"

    # Missing file
    with pytest.raises(FileNotFoundError):
        tcd._load_openai_messages([str(jsonl), str(json_file), str(tmp_path / "missing.json")])

    # No files
    with pytest.raises(RuntimeError):
        tcd._load_openai_messages([])

    # HF branch: force as repo-id and ensure delegated call is returned.
    # Default shuffle_seed is None so no .shuffle() call is made.
    monkeypatch.setattr(tcd, "_is_hf_repo_id", lambda v: True)
    sentinel = object()
    monkeypatch.setattr(tcd, "load_dataset", lambda *a, **k: sentinel)
    assert tcd._load_openai_messages("org/name", split="train") is sentinel


def test_load_openai_messages_local_skip_invalid_samples(tmp_path):
    jsonl = tmp_path / "broken.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps({"messages": [{"role": "user", "content": "ok-1"}]}),
                '{"messages":[{"role":"user","content":"unterminated}]',
                json.dumps({"messages": [{"role": "assistant", "content": "ok-2"}]}),
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(json.JSONDecodeError):
        tcd._load_openai_messages(str(jsonl), skip_invalid_samples=False)

    rows = tcd._load_openai_messages(str(jsonl), skip_invalid_samples=True)
    assert len(rows) == 2
    assert rows[0]["messages"][0]["content"] == "ok-1"
    assert rows[1]["messages"][0]["content"] == "ok-2"


def test_load_openai_messages_hf_shuffle_and_slice(monkeypatch):
    """Verify that HF datasets are shuffled before slicing."""
    monkeypatch.setattr(tcd, "_is_hf_repo_id", lambda v: True)

    call_log = {}

    class _FakeDataset:
        def __init__(self, items):
            self._items = items

        def __len__(self):
            return len(self._items)

        def shuffle(self, seed=None):
            call_log["shuffle_seed"] = seed
            return self

        def select(self, indices):
            call_log["select_indices"] = list(indices)
            return _FakeDataset([self._items[i] for i in indices])

    fake_ds = _FakeDataset(list(range(100)))
    monkeypatch.setattr(tcd, "load_dataset", lambda *a, **k: fake_ds)

    # Default (shuffle_seed=None) — no shuffling
    result = tcd._load_openai_messages("org/name", split="train")
    assert "shuffle_seed" not in call_log
    assert result is fake_ds

    # With shuffle seed — shuffle then return
    call_log.clear()
    result = tcd._load_openai_messages("org/name", split="train", shuffle_seed=42)
    assert call_log["shuffle_seed"] == 42
    assert "select_indices" not in call_log

    # Split with slice — shuffle then select
    call_log.clear()
    result = tcd._load_openai_messages("org/name", split="train[10:20]", shuffle_seed=42)
    assert call_log["shuffle_seed"] == 42
    assert call_log["select_indices"] == list(range(10, 20))

    # Custom seed
    call_log.clear()
    tcd._load_openai_messages("org/name", split="train", shuffle_seed=123)
    assert call_log["shuffle_seed"] == 123


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
        calls.append(
            {
                "normalized": normalized,
                "eos": eos_id,
                "pad": pad_id,
                "kwargs": kwargs,
            }
        )
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

    ds = tcd.ChatDataset(
        "ignored",
        tok,
        seq_length=16,
        start_of_turn_token="<|sot|>",
        chat_template="OVERRIDE",
        mask_reasoning_content=True,
    )

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
    assert calls[0]["kwargs"]["mask_reasoning_content"] is True

    # Bad row: messages not a list → ValueError
    monkeypatch.setattr(tcd, "_load_openai_messages", lambda *a, **k: [{"messages": "oops"}])
    ds_bad = tcd.ChatDataset("ignored", tok)
    with pytest.raises(ValueError):
        _ = ds_bad[0]


def test_chat_dataset_skip_invalid_samples_does_not_filter_structured_bad_rows(monkeypatch):
    """skip_invalid_samples only affects JSONL parse errors, not invalid message rows after load."""

    class Tok:
        eos_token_id = 1
        chat_template = "{{ default }}"

    tok = Tok()
    monkeypatch.setattr(tcd, "_has_chat_template", lambda _tok: True)
    monkeypatch.setattr(tcd, "_add_pad_token", lambda _tok: 3)
    monkeypatch.setattr(
        tcd, "format_chat_template", lambda *a, **k: {"input_ids": [1], "labels": [1], "attention_mask": [1]}
    )

    dataset_rows = [
        {"messages": [{"role": "user", "content": "ok"}]},
        {"messages": "bad"},
        {"messages": [{"role": "assistant", "content": "ok2"}]},
    ]
    monkeypatch.setattr(tcd, "_load_openai_messages", lambda *a, **k: dataset_rows)

    ds = tcd.ChatDataset("ignored", tok, skip_invalid_samples=True)
    assert len(ds) == 3
    assert ds[0]["input_ids"] == [1]
    with pytest.raises(ValueError):
        _ = ds[1]
    assert ds[2]["attention_mask"] == [1]


def test_resolve_chat_template_none():
    assert _resolve_chat_template(None) is None


def test_resolve_chat_template_plain_text_file(tmp_path):
    template = "{% for msg in messages %}{{ msg.content }}{% endfor %}"
    f = tmp_path / "template.jinja"
    f.write_text(template, encoding="utf-8")
    assert _resolve_chat_template(str(f)) == template


def test_resolve_chat_template_json_file(tmp_path):
    template = "{% for msg in messages %}{{ msg.role }}: {{ msg.content }}{% endfor %}"
    f = tmp_path / "tokenizer_config.json"
    f.write_text(json.dumps({"chat_template": template, "other_key": 123}), encoding="utf-8")
    assert _resolve_chat_template(str(f)) == template


def test_resolve_chat_template_json_file_without_key(tmp_path):
    data = {"model_type": "llama", "vocab_size": 32000}
    f = tmp_path / "config.json"
    raw = json.dumps(data)
    f.write_text(raw, encoding="utf-8")
    assert _resolve_chat_template(str(f)) == raw


def test_resolve_chat_template_literal_string():
    template = "{% for msg in messages %}{{ msg.content }}{% endfor %}"
    assert _resolve_chat_template(template) == template


def test_resolve_chat_template_nonexistent_path():
    assert _resolve_chat_template("/no/such/file/template.jinja") == "/no/such/file/template.jinja"


def test_tool_calling_chat_dataset_errors(monkeypatch):
    # No tokenizer
    with pytest.raises(ValueError):
        tcd.ChatDataset("ignored", None)

    # Tokenizer provided but missing chat template support
    class Tok:
        eos_token_id = 1
        chat_template = None

    monkeypatch.setattr(tcd, "_has_chat_template", lambda _tok: False)
    with pytest.raises(ValueError):
        tcd.ChatDataset("ignored", Tok())


class TestParquetLoading:
    """Tests for local Parquet/directory loading in _load_openai_messages.

    Addresses review comment: the Parquet loading branch needs test coverage.
    """

    def test_load_parquet_directory(self, tmp_path):
        """Loading a directory containing .parquet files should work."""
        from datasets import Dataset

        data = [
            {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
            {"messages": [{"role": "user", "content": "bye"}, {"role": "assistant", "content": "goodbye"}]},
        ]
        ds = Dataset.from_list(data)
        ds.to_parquet(tmp_path / "data.parquet")

        result = tcd._load_openai_messages(str(tmp_path), split="train")
        assert len(result) == 2
        assert result[0]["messages"][0]["content"] == "hi"

    def test_load_parquet_with_split_slice(self, tmp_path):
        """Split slicing like 'train[:1]' should work on local Parquet datasets."""
        from datasets import Dataset

        data = [
            {"messages": [{"role": "user", "content": f"msg{i}"}, {"role": "assistant", "content": f"reply{i}"}]}
            for i in range(5)
        ]
        ds = Dataset.from_list(data)
        ds.to_parquet(tmp_path / "data.parquet")

        result = tcd._load_openai_messages(str(tmp_path), split="train[:2]")
        assert len(result) == 2

    def test_load_single_parquet_file(self, tmp_path):
        """Loading a single .parquet file path should work via data_files."""
        from datasets import Dataset

        data = [
            {"messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]},
            {"messages": [{"role": "user", "content": "ping"}, {"role": "assistant", "content": "pong"}]},
            {"messages": [{"role": "user", "content": "foo"}, {"role": "assistant", "content": "bar"}]},
        ]
        ds = Dataset.from_list(data)
        pq_file = tmp_path / "single.parquet"
        ds.to_parquet(pq_file)

        result = tcd._load_openai_messages(str(pq_file), split="train")
        assert len(result) == 3
        assert result[0]["messages"][0]["content"] == "hello"
        assert result[2]["messages"][1]["content"] == "bar"

    def test_directory_without_parquet_falls_through(self, tmp_path):
        """A directory without .parquet files should not be handled by the Parquet path."""
        (tmp_path / "data.jsonl").write_text('{"messages": [{"role": "user", "content": "hi"}]}\n')

        # Should fall through to JSON/JSONL loading or HF hub path, not the Parquet path
        result = tcd._load_openai_messages(str(tmp_path / "data.jsonl"), split="train")
        assert len(result) == 1

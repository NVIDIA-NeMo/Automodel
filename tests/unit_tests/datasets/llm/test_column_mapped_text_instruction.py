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

from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset import (
    ColumnMappedTextInstructionDataset,
    _str_is_hf_repo_id,
    make_iterable,
    _load_dataset,
    _check_all_values_equal_length,
    _has_chat_template,
)


def test_make_iterable_basic():
    # single string -> iterator with one element
    assert list(make_iterable("hello")) == ["hello"]

    # list of strings stays untouched
    assert list(make_iterable(["a", "b", "c"])) == ["a", "b", "c"]

    # invalid type should raise
    with pytest.raises(ValueError):
        list(make_iterable(123))  # type: ignore[arg-type]

def test_str_is_hf_repo_id():
    assert _str_is_hf_repo_id("allenai/c4") is True
    assert _str_is_hf_repo_id("some/local/path.json") is False
    assert _str_is_hf_repo_id("invalid_format") is False


def test_load_dataset_local_json(tmp_path: Path):
    data = [
        {"q": "How are you?", "a": "Fine."},
        {"q": "What is your name?", "a": "Bot."},
    ]
    file_path = tmp_path / "samples.json"
    with file_path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp)

    ds = _load_dataset(str(file_path))
    assert len(ds) == 2
    assert ds[0]["q"] == "How are you?"


class _DummyTokenizer:  # noqa: D401
    """Minimal tokenizer stub - only what's required for the dataset."""

    def __init__(self):
        self.pad_token = "<pad>"

    # Dummy special token ids expected by the dataset implementation
    eos_token_id = None  # End-of-sequence token (unused in tests)
    bos_token_id = None  # Beginning-of-sequence token (unused in tests)

    def __call__(self, text: str, add_special_tokens: bool = True):  # noqa: D401
        """Mimic the Hugging Face tokenizer ``__call__`` API.

        The real tokenizer would convert *text* into a list of integer token IDs.
        For the purpose of these unit tests we just assign a deterministic ID to
        each whitespace-separated token so that the returned structure matches
        what the dataset expects (a dict with an ``input_ids`` key).
        """

        # Very simple whitespace tokenisation - one integer per token.
        input_ids = list(range(len(text.split())))
        return {"input_ids": input_ids}


def test_column_mapped_dataset_basic_no_tokenizer(tmp_path: Path):
    samples = [
        {"query": "Why is the sky blue?", "response": "Rayleigh scattering."},
        {"query": "What is 2+2?", "response": "4."},
    ]
    jsonl_path = tmp_path / "toy.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for row in samples:
            fp.write(json.dumps(row) + "\n")

    column_mapping = {"question": "query", "answer": "response"}

    with pytest.raises(AssertionError):
        ds = ColumnMappedTextInstructionDataset(
            path_or_dataset_id=str(jsonl_path),
            column_mapping=column_mapping,
            tokenizer=None,
            answer_only_loss_mask=False,
        )

def test_column_mapped_dataset_bad_mapping(tmp_path: Path):
    samples = [
        {"query": "Why is the sky blue?", "response": "Rayleigh scattering."},
        {"query": "What is 2+2?", "response": "4."},
    ]
    jsonl_path = tmp_path / "toy.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for row in samples:
            fp.write(json.dumps(row) + "\n")

    column_mapping = {"question": "query", "answer": "response", "bad": "column"}

    with pytest.raises(AssertionError, match="Expected context to be in column_mapping"):
        ds = ColumnMappedTextInstructionDataset(
            path_or_dataset_id=str(jsonl_path),
            column_mapping=column_mapping,
            tokenizer=_DummyTokenizer(),
            answer_only_loss_mask=False,
        )

def test_column_mapped_dataset_basic(tmp_path: Path):
    samples = [
        {"query": "Why is the sky blue?", "response": "Rayleigh scattering."},
        {"query": "What is 2+2?", "response": "4."},
    ]
    jsonl_path = tmp_path / "toy.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for row in samples:
            fp.write(json.dumps(row) + "\n")

    column_mapping = {"question": "query", "answer": "response"}

    ds = ColumnMappedTextInstructionDataset(
        path_or_dataset_id=str(jsonl_path),
        column_mapping=column_mapping,
        tokenizer=_DummyTokenizer(),
        answer_only_loss_mask=False,
    )

    assert len(ds) == 2
    first = ds[0]
    assert set(first.keys()) == {"labels", "input_ids"}


def test_column_mapped_dataset_streaming(tmp_path: Path):
    """Verify behaviour when *streaming=True*.

    In streaming mode the dataset becomes an ``IterableDataset`` - length and
    random access are undefined, but iteration should lazily yield the mapped
    rows.  We check that these constraints are enforced and that the mapping
    logic still works.
    """

    import itertools

    samples = [
        {"q": "Who wrote Hamlet?", "a": "Shakespeare"},
        {"q": "Capital of France?", "a": "Paris"},
    ]

    jsonl_path = tmp_path / "toy_stream.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for row in samples:
            fp.write(json.dumps(row) + "\n")

    ds = ColumnMappedTextInstructionDataset(
        path_or_dataset_id=str(jsonl_path),
        column_mapping={"question": "q", "answer": "a"},
        tokenizer=_DummyTokenizer(),
        streaming=True,
        answer_only_loss_mask=False,
    )

    # __len__ and __getitem__ are not supported in streaming mode
    with pytest.raises(RuntimeError):
        _ = len(ds)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError):
        _ = ds[0]  # type: ignore[index]

    # But we can iterate and obtain the mapped columns
    first_two = list(itertools.islice(ds, 2))

class _TokenizerNoChat:  # noqa: D401 - minimal stub only
    """Tokenizer *without* chat‐template support."""

    def __call__(self, text: str):  # pragma: no cover - unused in these tests
        return {"input_ids": []}


class _TokenizerChat(_TokenizerNoChat):  # noqa: D401 - adds chat template features
    """Tokenizer *with* chat‐template support.

    Only the attributes consumed by ``_has_chat_template`` are implemented.
    """

    chat_template = "<dummy>"

    def apply_chat_template(self, messages):  # type: ignore[override]
        return []


class _TokenizerChatButNoCallable(_TokenizerNoChat):  # noqa: D401
    """Tokenizer with a *non‐callable* ``apply_chat_template`` attribute."""

    chat_template = "<dummy>"
    apply_chat_template = "not callable"  # type: ignore[assignment]

def test_has_chat_template_positive():
    """Tokenizer advertises *chat_template* and *apply_chat_template* is callable."""

    tok = _TokenizerChat()
    assert _has_chat_template(tok) is True


def test_has_chat_template_negative_missing_attrs():
    """Tokenizer lacks chat‐template attributes → returns *False*."""

    tok = _TokenizerNoChat()
    assert _has_chat_template(tok) is False


def test_has_chat_template_negative_not_callable():
    """``apply_chat_template`` exists but is *not* callable → returns *False*."""

    tok = _TokenizerChatButNoCallable()
    assert _has_chat_template(tok) is False

def test_check_all_values_equal_length_true():
    sample = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    assert _check_all_values_equal_length(sample) is True


def test_check_all_values_equal_length_false():
    sample = {"a": [1, 2, 3, 4], "b": [5, 6], "c": []}
    assert _check_all_values_equal_length(sample) is False
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

import itertools
import json
from pathlib import Path

import pytest

from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_iterable_dataset import (
    ColumnMappedTextInstructionIterableDataset,
)


class _DummyTokenizer:  # noqa: D401
    """Minimal tokenizer stub sufficient for dataset tokenization paths."""

    def __init__(self):
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self._counter = 3

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding=None,
        truncation=None,
        max_length=None,
    ):
        tokens = text.split()
        input_ids = list(range(self._counter, self._counter + len(tokens)))
        if add_special_tokens:
            input_ids = [self.bos_token_id] + input_ids + [self.eos_token_id]
        return {"input_ids": input_ids}


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")


def test_iterable_dataset_basic_iteration(tmp_path: Path):
    rows = [
        {"q": "Who wrote Hamlet?", "a": "Shakespeare"},
        {"q": "Capital of France?", "a": "Paris"},
    ]
    jsonl_path = tmp_path / "toy_stream.jsonl"
    _write_jsonl(jsonl_path, rows)

    ds = ColumnMappedTextInstructionIterableDataset(
        path_or_dataset_id=str(jsonl_path),
        column_mapping={"question": "q", "answer": "a"},
        tokenizer=_DummyTokenizer(),
        answer_only_loss_mask=False,
        repeat_on_exhaustion=False,
    )

    items = list(ds)
    assert len(items) == 2
    first = items[0]
    assert isinstance(first, dict)
    # Internal pad token ids helper is included; keys should at least contain these
    assert {"input_ids", "attention_mask", "labels"}.issubset(first.keys())


def test_iterable_dataset_limit_samples(tmp_path: Path):
    rows = [
        {"q": "1+1?", "a": "2"},
        {"q": "2+2?", "a": "4"},
    ]
    jsonl_path = tmp_path / "toy_stream_limit.jsonl"
    _write_jsonl(jsonl_path, rows)

    ds = ColumnMappedTextInstructionIterableDataset(
        path_or_dataset_id=str(jsonl_path),
        column_mapping={"question": "q", "answer": "a"},
        tokenizer=_DummyTokenizer(),
        answer_only_loss_mask=False,
        limit_dataset_samples=1,
        repeat_on_exhaustion=False,
    )

    items = list(ds)
    assert len(items) == 1


def test_iterable_dataset_repeat_on_exhaustion_true_is_repeatable(tmp_path: Path):
    rows = [
        {"q": "A?", "a": "a"},
        {"q": "B?", "a": "b"},
    ]
    jsonl_path = tmp_path / "toy_stream_repeat.jsonl"
    _write_jsonl(jsonl_path, rows)

    ds = ColumnMappedTextInstructionIterableDataset(
        path_or_dataset_id=str(jsonl_path),
        column_mapping={"question": "q", "answer": "a"},
        tokenizer=_DummyTokenizer(),
        answer_only_loss_mask=False,
        repeat_on_exhaustion=True,
    )

    # Should be able to take more items than the dataset size due to repeat
    taken = list(itertools.islice(ds, 5))
    assert len(taken) == 5


def test_iterable_dataset_bad_mapping_raises(tmp_path: Path):
    rows = [{"q": "Q?", "a": "A"}]
    jsonl_path = tmp_path / "toy_bad.jsonl"
    _write_jsonl(jsonl_path, rows)

    with pytest.raises(AssertionError):
        _ = ColumnMappedTextInstructionIterableDataset(
            path_or_dataset_id=str(jsonl_path),
            column_mapping={"question": "q", "answer": "a", "bad": "col"},
            tokenizer=_DummyTokenizer(),
        )


def test_iterable_dataset_requires_tokenizer(tmp_path: Path):
    rows = [{"q": "Q?", "a": "A"}]
    jsonl_path = tmp_path / "toy_tokenizer.jsonl"
    _write_jsonl(jsonl_path, rows)

    with pytest.raises(ValueError):
        _ = ColumnMappedTextInstructionIterableDataset(
            path_or_dataset_id=str(jsonl_path),
            column_mapping={"question": "q", "answer": "a"},
            tokenizer=None,  # type: ignore[arg-type]
        )



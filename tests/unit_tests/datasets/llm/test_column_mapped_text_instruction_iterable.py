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
from itertools import combinations, islice
from pathlib import Path

import pytest

from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_iterable_dataset import (
    ColumnMappedTextInstructionIterableDataset,
)
from nemo_automodel.components.datasets.loader import ParallelAwareDataloader


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
        # Advance counter so successive calls yield distinct id ranges
        self._counter += len(tokens) + (2 if add_special_tokens else 0)
        return {"input_ids": input_ids}


def _write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row) + "\n")


def test_iterable_dataset_shard_and_shuffle_smoke(monkeypatch, tmp_path: Path):
    class _StubHFIterable:
        def __init__(self, rows):
            self._rows = rows
            self._shard = None
            self._shuffled = False

        def __iter__(self):
            it = self._rows
            if self._shard is not None:
                n, idx = self._shard
                it = [r for i, r in enumerate(it) if i % n == idx]
            if self._shuffled:
                it = list(reversed(it))
            for r in it:
                yield r

        def shard(self, num_shards, index):
            self._shard = (num_shards, index)
            return self

        def shuffle(self, buffer_size, seed):
            self._shuffled = True
            return self

    rows = [
        {"q": "Q0?", "a": "A0"},
        {"q": "Q1?", "a": "A1"},
        {"q": "Q2?", "a": "A2"},
    ]

    def _fake_load_dataset(*args, **kwargs):
        return _StubHFIterable(rows)

    monkeypatch.setattr(
        "nemo_automodel.components.datasets.llm.column_mapped_text_instruction_iterable_dataset._load_dataset",
        _fake_load_dataset,
    )

    ds = (
        ColumnMappedTextInstructionIterableDataset(
            path_or_dataset_id="ignored.jsonl",
            column_mapping={"question": "q", "answer": "a"},
            tokenizer=_DummyTokenizer(),
            answer_only_loss_mask=False,
            repeat_on_exhaustion=True,
        )
        .shard(2, 1)
        .shuffle(buffer_size=2, seed=0)
    )

    first = next(iter(ds))
    assert {"input_ids", "attention_mask", "labels"}.issubset(first.keys())


@pytest.mark.parametrize("world_size", [2, 3])
def test_iterable_dataset_shard_splits_single_file_stream_by_example(tmp_path: Path, world_size: int):
    # Regression: a streamed JSONL file is one HF shard, and HF ``IterableDataset.shard``
    # splits by file, so rank 0 used to get every row and every other rank raised IndexError.
    rows = [{"q": f"Q{i}?", "a": f"A{i}"} for i in range(6)]
    jsonl_path = tmp_path / "single_file.jsonl"
    _write_jsonl(jsonl_path, rows)

    seen = []
    for rank in range(world_size):
        ds = ColumnMappedTextInstructionIterableDataset(
            path_or_dataset_id=str(jsonl_path),
            column_mapping={"question": "q", "answer": "a"},
            tokenizer=_DummyTokenizer(),
            answer_only_loss_mask=False,
            repeat_on_exhaustion=True,
        ).shard(world_size, rank)
        # Read one pass of the underlying stream; the wrapper itself repeats forever.
        seen.append([row["q"] for row in ds.dataset])

    assert all(len(rank_rows) == len(rows) // world_size for rank_rows in seen)
    assert sorted(q for rank_rows in seen for q in rank_rows) == sorted(row["q"] for row in rows)


class _RowIdTokenizer:
    """Tokenizer stub whose ids are the integer words of the text, so each sample names its row."""

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, add_special_tokens: bool = True, padding=None, truncation=None, max_length=None):
        return {"input_ids": [int(word) for word in text.split()]}


@pytest.mark.parametrize("file_sizes", [[16] * 6, [16] * 4, [16], [10, 22]])
def test_iterable_dataset_ranks_keep_disjoint_rows_across_repeats_and_restore(tmp_path: Path, file_sizes):
    # Regression: each rank repeats the stream on its own, so ranks reach a new epoch at different
    # steps. Every rank must own the same rows at every epoch, or two ranks train on the same rows.
    # Run the native loader over several repeats and a save/restore for each simulated rank.
    world_size = 3
    paths, row_ids = [], []
    for file_idx, size in enumerate(file_sizes):
        ids = range(1000 + len(row_ids), 1000 + len(row_ids) + size)
        path = tmp_path / f"part{file_idx}.jsonl"
        _write_jsonl(path, [{"q": str(row_id), "a": str(row_id)} for row_id in ids])
        paths.append(str(path))
        row_ids.extend(ids)

    def make_loader(rank: int) -> ParallelAwareDataloader:
        ds = ColumnMappedTextInstructionIterableDataset(
            path_or_dataset_id=paths,
            column_mapping={"question": "q", "answer": "a"},
            tokenizer=_RowIdTokenizer(),
            answer_only_loss_mask=False,
        )
        return ParallelAwareDataloader(
            ds,
            dp_rank=rank,
            dp_world_size=world_size,
            batch_size=1,
            collate_fn=lambda batch: batch[0]["input_ids"][0],
            shuffle=True,
            seed=4812,
            shuffle_buffer_size=7,
        )

    seen = []
    for rank in range(world_size):
        loader = make_loader(rank)
        rows = list(islice(loader, 32))
        restored = make_loader(rank)
        restored.load_state_dict(loader.state_dict())
        rows += islice(restored, 32)
        seen.append(set(rows))

    for rank_a, rank_b in combinations(range(world_size), 2):
        assert not seen[rank_a] & seen[rank_b], f"ranks {rank_a} and {rank_b} share rows"
    assert set().union(*seen) == set(row_ids)


def test_iterable_dataset_pad_token_fallback_with_eos(tmp_path: Path):
    class _TokNoPadWithEos:
        eos_token = "</s>"
        pad_token = None

    rows = [{"q": "Q?", "a": "A"}]
    jsonl_path = tmp_path / "toy_pad_eos.jsonl"
    _write_jsonl(jsonl_path, rows)

    tok = _TokNoPadWithEos()
    _ = ColumnMappedTextInstructionIterableDataset(
        path_or_dataset_id=str(jsonl_path),
        column_mapping={"question": "q", "answer": "a"},
        tokenizer=tok,
        answer_only_loss_mask=False,
        repeat_on_exhaustion=True,
    )
    assert tok.pad_token == tok.eos_token


def test_iterable_dataset_pad_token_fallback_without_eos(tmp_path: Path):
    class _TokNoPadNoEos:
        pad_token = None

    rows = [{"q": "Q?", "a": "A"}]
    jsonl_path = tmp_path / "toy_pad_noeos.jsonl"
    _write_jsonl(jsonl_path, rows)

    tok = _TokNoPadNoEos()
    _ = ColumnMappedTextInstructionIterableDataset(
        path_or_dataset_id=str(jsonl_path),
        column_mapping={"question": "q", "answer": "a"},
        tokenizer=tok,
        answer_only_loss_mask=False,
        repeat_on_exhaustion=True,
    )
    assert tok.pad_token == " "


def test_iterable_dataset_mapping_checks_missing_answer(tmp_path: Path):
    rows = [{"q": "Q?", "a": "A"}]
    jsonl_path = tmp_path / "toy_missing_answer.jsonl"
    _write_jsonl(jsonl_path, rows)

    with pytest.raises(AssertionError):
        _ = ColumnMappedTextInstructionIterableDataset(
            path_or_dataset_id=str(jsonl_path),
            column_mapping={"question": "q"},  # missing answer
            tokenizer=_DummyTokenizer(),
        )


def test_iterable_dataset_mapping_checks_two_keys_missing_both_context_and_question(tmp_path: Path):
    rows = [{"q": "Q?", "a": "A"}]
    jsonl_path = tmp_path / "toy_two_keys_invalid.jsonl"
    _write_jsonl(jsonl_path, rows)

    with pytest.raises(AssertionError, match="Expected context or question"):
        _ = ColumnMappedTextInstructionIterableDataset(
            path_or_dataset_id=str(jsonl_path),
            column_mapping={"answer": "a", "foo": "bar"},
            tokenizer=_DummyTokenizer(),
        )


def test_iterable_dataset_mapping_checks_invalid_num_columns(tmp_path: Path):
    rows = [{"q": "Q?", "a": "A"}]
    jsonl_path = tmp_path / "toy_invalid_cols.jsonl"
    _write_jsonl(jsonl_path, rows)

    with pytest.raises(ValueError, match="Expected 2 or 3 columns"):
        _ = ColumnMappedTextInstructionIterableDataset(
            path_or_dataset_id=str(jsonl_path),
            column_mapping={"answer": "a"},  # only 1 key
            tokenizer=_DummyTokenizer(),
        )


@pytest.mark.timeout(30)
def test_iterable_dataset_raises_when_every_row_is_filtered(tmp_path: Path):
    # The answer column does not exist in the data, so every row is skipped. Without a guard
    # the repeat-on-exhaustion loop spins forever without yielding.
    jsonl_path = tmp_path / "toy_no_answer_column.jsonl"
    _write_jsonl(jsonl_path, [{"q": "Q0?", "a": "A0"}, {"q": "Q1?", "a": "A1"}])

    ds = ColumnMappedTextInstructionIterableDataset(
        path_or_dataset_id=str(jsonl_path),
        column_mapping={"question": "q", "answer": "missing"},
        tokenizer=_DummyTokenizer(),
        answer_only_loss_mask=False,
    )

    with pytest.raises(ValueError, match="No valid samples"):
        next(iter(ds))

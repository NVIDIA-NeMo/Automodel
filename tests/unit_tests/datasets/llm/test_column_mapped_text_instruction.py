import json
from pathlib import Path

import pytest

from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset import (
    ColumnMappedTextInstructionDataset,
    _str_is_hf_repo_id,
    make_iterable,
    _load_dataset,
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
    """Minimal tokenizer stub – only what's required for the dataset."""

    def __init__(self):
        self.pad_token = "<pad>"


def test_column_mapped_dataset_basic(tmp_path: Path):
    samples = [
        {"question": "Why is the sky blue?", "answer": "Rayleigh scattering."},
        {"question": "What is 2+2?", "answer": "4."},
    ]
    jsonl_path = tmp_path / "toy.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fp:
        for row in samples:
            fp.write(json.dumps(row) + "\n")

    column_mapping = {"query": "question", "response": "answer"}

    ds = ColumnMappedTextInstructionDataset(
        path_or_dataset_id=str(jsonl_path),
        column_mapping=column_mapping,
        tokenizer=_DummyTokenizer(),
        answer_only_loss_mask=False,
    )

    assert len(ds) == 2
    first = ds[0]
    assert set(first.keys()) == {"query", "response"}
    assert first["query"] == "Why is the sky blue?"
    assert first["response"].startswith("Rayleigh") 
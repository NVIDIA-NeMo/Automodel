# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import os
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset import (
    ColumnMappedTextInstructionDataset,
)


def _write_jsonl(tmp_path: Path) -> Path:
    """Create a small JSONL dataset for testing."""
    rows = [
        {
            "context": "Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden statue of the Virgin Mary.",
            "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
            "answers": "Saint Bernadette Soubirous",
        },
        {
            "context": "Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised.",
            "question": "What is in front of the Notre Dame Main Building?",
            "answers": "a copper statue of Christ",
        },
        {
            "context": "Next to the Main Building is the Basilica of the Sacred Heart.",
            "question": "The Basilica of the Sacred heart at Notre Dame is beside to which structure?",
            "answers": "the Main Building",
        },
        {
            "context": "Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection.",
            "question": "What is the Grotto at Notre Dame?",
            "answers": "a Marian place of prayer and reflection",
        },
        {
            "context": "Atop the Main Building's gold dome is a golden statue of the Virgin Mary.",
            "question": "What sits on top of the Main Building at Notre Dame?",
            "answers": "a golden statue of the Virgin Mary",
        },
    ]
    p = tmp_path / "sample.jsonl"
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def _maybe_tokenizer_dir_candidates() -> list[Path]:
    """Return likely tokenizer directories present in CI test data mounts."""
    candidates: list[Path] = []
    # Known bundle with no chat template used elsewhere in the repo
    test_data_dir = os.environ.get("TEST_DATA_DIR")
    if test_data_dir:
        candidates.append(Path(test_data_dir) / "hf_mixtral_2l")
    # Explicit tokenizers used by existing unit tests
    base = Path("/home/TestData/akoumparouli/tokenizers/")
    names = [
        "gpt-oss-20b",
        "llama_3.2_1b",
        "qwen3_30b_a3b_instruct_2507",
    ]
    for n in names:
        candidates.append(base / n)
    return [p for p in candidates if p.exists()]


def _load_tokenizer(path: Path):
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return AutoTokenizer.from_pretrained(str(path))


def _first_sample(ds: ColumnMappedTextInstructionDataset):
    it = iter(ds)
    return next(it)


@pytest.mark.parametrize(
    "seq_length,padding,truncation",
    [
        (None, "do_not_pad", None),
        (16, "max_length", True),
        (16, "do_not_pad", True),
        (16, True, None),  # padding=True -> longest; with single example behaves like no-op pre-packaging
    ],
)
def test_dataset_non_chat_padding_truncation_options(tmp_path: Path, seq_length, padding, truncation):
    """Validate shapes and masking for non-chat tokenizers across padding/truncation options."""
    data_file = _write_jsonl(tmp_path)

    # Find a tokenizer without chat template
    for d in _maybe_tokenizer_dir_candidates():
        tok = _load_tokenizer(d)
        if getattr(tok, "chat_template", None) is None:
            break
    else:
        pytest.skip("No non-chat tokenizer available in test data mounts")

    column_mapping = {"context": "context", "question": "question", "answer": "answers"}

    ds = ColumnMappedTextInstructionDataset(
        path_or_dataset_id=str(data_file),
        column_mapping=column_mapping,
        tokenizer=tok,
        seq_length=seq_length,
        padding=padding,
        truncation=truncation,
        # answer_only_loss_mask default True
    )

    sample = _first_sample(ds)
    assert set(["input_ids", "labels", "attention_mask"]).issubset(sample.keys())
    assert len(sample["input_ids"]) == len(sample["labels"]) == len(sample["attention_mask"]) > 0

    if isinstance(seq_length, int):
        if truncation is True:
            assert len(sample["input_ids"]) == seq_length
            assert len(sample["labels"]) == seq_length
            # Trailing padding in labels must be masked
            assert sample["labels"][-1] == -100
            assert sample["attention_mask"][-1] in (0, 1)  # depending on pack length, end can be 0
        elif not truncation is True:
            assert len(sample["input_ids"]) != seq_length
            assert len(sample["labels"]) != seq_length

@pytest.mark.parametrize(
    "seq_length,padding,truncation",
    [
        (None, "do_not_pad", None),
        (128, "max_length", True),
        (16, "do_not_pad", True),
        (16, True, None),
    ],
)
def test_dataset_chat_padding_truncation_options(tmp_path: Path, seq_length, padding, truncation):
    """Validate shapes and masking for chat-template tokenizers across padding/truncation options."""
    data_file = _write_jsonl(tmp_path)

    # Find a tokenizer with chat template
    chat_tok = None
    for d in _maybe_tokenizer_dir_candidates():
        tok = _load_tokenizer(d)
        if getattr(tok, "chat_template", None) is not None and callable(getattr(tok, "apply_chat_template", None)):
            chat_tok = tok
            break
    if chat_tok is None:
        pytest.skip("No chat-template tokenizer available in test data mounts")

    # 3-column mapping
    column_mapping = {"context": "context", "question": "question", "answer": "answers"}

    ds = ColumnMappedTextInstructionDataset(
        path_or_dataset_id=str(data_file),
        column_mapping=column_mapping,
        tokenizer=chat_tok,
        seq_length=seq_length,
        padding=padding,
        truncation=truncation,
        start_of_turn_token="<|assistant|>",  # required when answer_only_loss_mask=True and chat template present
    )

    sample = _first_sample(ds)
    assert set(["input_ids", "labels", "attention_mask"]).issubset(sample.keys())
    assert len(sample["input_ids"]) == len(sample["labels"]) == len(sample["attention_mask"]) > 0

    if isinstance(seq_length, int):
        if truncation is True or padding == "max_length":
            assert len(sample["input_ids"]) == seq_length
            assert len(sample["labels"]) == seq_length
        elif not truncation is True:
            assert sample["labels"][-1] != -100


def test_dataset_two_column_mapping_non_chat(tmp_path: Path):
    """Ensure 2-column mapping (context+answer) works with non-chat tokenizer."""
    data_file = _write_jsonl(tmp_path)

    # Choose a non-chat tokenizer
    for d in _maybe_tokenizer_dir_candidates():
        tok = _load_tokenizer(d)
        if getattr(tok, "chat_template", None) is None:
            break
    else:
        pytest.skip("No non-chat tokenizer available in test data mounts")

    # Use only context and answers columns
    column_mapping = {"context": "context", "answer": "answers"}

    ds = ColumnMappedTextInstructionDataset(
        path_or_dataset_id=str(data_file),
        column_mapping=column_mapping,
        tokenizer=tok,
        seq_length=32,
        padding="max_length",
        truncation=True,
    )

    sample = _first_sample(ds)
    assert len(sample["input_ids"]) == 32
    assert len(sample["labels"]) == 32
    assert len(sample["attention_mask"]) == 32


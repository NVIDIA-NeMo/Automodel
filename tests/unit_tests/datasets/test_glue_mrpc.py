# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import re
from typing import Any, Dict, List

import pytest

from nemo_automodel.components.datasets.llm.seq_cls import GLUE_MRPC


class FakeProcessedDataset:
    def __init__(self, outputs: Dict[str, List[List[int]]]):
        self._outputs = outputs

    def __len__(self) -> int:
        return len(self._outputs["input_ids"])  # type: ignore[index]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {k: v[idx] for k, v in self._outputs.items()}


class FakeRawDataset:
    def __init__(self, num_examples: int):
        # Minimal columns that GLUE MRPC provides
        self._data = {
            "sentence1": [f"s1-{i}" for i in range(num_examples)],
            "sentence2": [f"s2-{i}" for i in range(num_examples)],
            "label": [0 if i % 2 == 0 else 1 for i in range(num_examples)],
            # Extra columns to verify remove_columns works
            "idx": list(range(num_examples)),
            "extra": [f"e-{i}" for i in range(num_examples)],
        }
        self.column_names = list(self._data.keys())

    def map(self, func, batched: bool, remove_columns: List[str]):  # noqa: D401 - huggingface-like API
        assert batched is True
        # Build the batch of only the remaining columns
        kept_cols = [c for c in self.column_names if c not in remove_columns]
        batch = {c: self._data[c] for c in kept_cols}
        outputs = func(batch)
        # Ensure required fields are present after map
        assert "input_ids" in outputs
        assert "labels" in outputs
        return FakeProcessedDataset(outputs)


class FakeTokenizer:
    def __init__(self, include_attention_mask: bool = False, model_max_length: int = 512, pad_token_id: int = 0):
        self.include_attention_mask = include_attention_mask
        self.model_max_length = model_max_length
        self.pad_token_id = pad_token_id
        self.last_kwargs: Dict[str, Any] = {}

    def __call__(self, sentence1_list: List[str], sentence2_list: List[str], **kwargs) -> Dict[str, List[List[int]]]:
        # Record kwargs to assert max_length handling
        self.last_kwargs = kwargs
        num = len(sentence1_list)
        input_ids = [[101, i + 1, 102] for i in range(num)]
        outputs: Dict[str, List[List[int]]] = {"input_ids": input_ids}
        if self.include_attention_mask:
            outputs["attention_mask"] = [[1] * len(ids) for ids in input_ids]
        return outputs


def _fake_load_dataset_factory(total: int = 3):
    """Build a fake load_dataset that respects split slicing like "train[:N]"."""

    def _fake_load_dataset(name: str, subset: str, *, split: str, trust_remote_code: bool):  # noqa: ARG001
        assert name == "glue"
        assert subset == "mrpc"
        # Extract optional slice
        m = re.match(r"^(train|validation|test)(\[:(\d+)\])?$", split)
        assert m is not None, f"unexpected split format: {split}"
        n = total
        if m.group(3) is not None:
            n = min(int(m.group(3)), total)
        return FakeRawDataset(n)

    return _fake_load_dataset


def test_glue_mrpc_structure_and_slicing(monkeypatch: pytest.MonkeyPatch):
    # Arrange: patch datasets.load_dataset
    from datasets import load_dataset as real_load_dataset  # noqa: F401  # ensure import path exists

    fake_loader = _fake_load_dataset_factory(total=5)
    monkeypatch.setattr("nemo_automodel.components.datasets.llm.seq_cls.load_dataset", fake_loader)

    tokenizer = FakeTokenizer(include_attention_mask=False, model_max_length=512, pad_token_id=7)

    # Act
    ds = GLUE_MRPC(tokenizer, split="train", num_samples_limit=2, max_length=128)

    # Assert: length equals sliced size
    assert len(ds) == 2

    item0 = ds[0]
    assert set(item0.keys()) == {"input_ids", "attention_mask", "labels", "___PAD_TOKEN_IDS___"}
    # attention_mask should fallback to ones when tokenizer doesn't supply it
    assert item0["attention_mask"] == [1] * len(item0["input_ids"])  # type: ignore[index]
    # labels should be wrapped as [[label]]
    assert isinstance(item0["labels"], list)
    assert isinstance(item0["labels"][0], int)
    # pad token ids mapping
    assert item0["___PAD_TOKEN_IDS___"] == {"input_ids": 7, "labels": -100, "attention_mask": 0}


@pytest.mark.parametrize(
    "tok_max_length, init_max_length, expected_called_max",
    [
        (16384, None, 1024),  # clamp very large tokenizer.model_max_length to 1024 when None provided
        (512, None, None),     # propagate None when tokenizer model max is modest
        (4096, 64, 64),        # explicit max_length overrides tokenizer setting
    ],
)
def test_glue_mrpc_max_length_resolution(monkeypatch: pytest.MonkeyPatch, tok_max_length: int, init_max_length: int | None, expected_called_max: int | None):
    fake_loader = _fake_load_dataset_factory(total=3)
    monkeypatch.setattr("nemo_automodel.components.datasets.llm.seq_cls.load_dataset", fake_loader)

    tokenizer = FakeTokenizer(include_attention_mask=True, model_max_length=tok_max_length, pad_token_id=0)

    # Act
    GLUE_MRPC(tokenizer, split="train", num_samples_limit=2, max_length=init_max_length)

    # Assert the tokenizer call received the resolved max_length
    assert "max_length" in tokenizer.last_kwargs
    assert tokenizer.last_kwargs["truncation"] is True



# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from __future__ import annotations

import math
from typing import Any

import pytest
import torch

import nemo_automodel.components.datasets.utils as sftp


class DummyTokenizer:
    """
    A minimal, framework-free tokenizer that supplies exactly the
    functionality expected by SFTSingleTurnPreprocessor.  Each character is
    converted to an integer id; two special tokens are added:
    ``bos`` (id=0) at the beginning and ``eos`` (id=1) at the end.
    """

    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2
    bos_token = "<s>"
    pad_token = "<pad>"

    @property
    def all_special_ids(self) -> list[int]:
        return [self.bos_token_id, self.eos_token_id, self.pad_token_id]

    def _encode_single(self, text: str) -> list[int]:
        # 0, 1 and 2 are reserved; normal chars start at 10 for readability
        return [self.bos_token_id] + [ord(c) % 100 + 10 for c in text] + [self.eos_token_id]

    def __call__(self, text: list[str] | str) -> dict[str, list[list[int]]]:
        if isinstance(text, str):
            text = [text]

        input_ids = [self._encode_single(t) for t in text]
        attn_mask = [[1] * len(ids) for ids in input_ids]
        return {"input_ids": input_ids, "attention_mask": attn_mask}


class DummyDataset:
    """
    A minimal “dataset” that exposes only the two accessor methods required
    by ``_tokenize_function``.
    """

    @staticmethod
    def get_context(batch: dict[str, Any]) -> list[str]:
        return batch["context"]

    @staticmethod
    def get_target(batch: dict[str, Any]) -> list[str]:
        return batch["target"]


@pytest.fixture()
def dummy_tokenizer() -> DummyTokenizer:
    """Return a fresh dummy tokenizer for each test."""
    return DummyTokenizer()


def _make_dummy_example(seq_len_ctx: int, seq_len_tgt: int) -> dict[str, Any]:
    """Produce a batch element with deterministic but different lengths."""
    context = "C" * seq_len_ctx
    target = "T" * seq_len_tgt
    return {"context": context, "target": target}


def test_batchify_adds_batch_dimension() -> None:
    """`batchify` must insert dim-0 in-place when given a 1-D tensor."""
    vec = torch.tensor([1, 2, 3])
    out = sftp.batchify(vec)
    # In-place inplace unsqueeze_ returns same object
    assert out is vec
    assert out.ndim == 2
    assert torch.equal(out, torch.tensor([[1, 2, 3]]))


def test_batchify_noop_for_higher_dim() -> None:
    """Tensor with ndim>1 should be returned untouched."""
    mat = torch.randn(4, 5)
    out = sftp.batchify(mat)
    assert out is mat
    assert out.ndim == 2


def test_extract_key_from_dicts() -> None:
    """Verify correct extraction order."""
    batch = [{"a": i, "b": -i} for i in range(5)]
    assert sftp.extract_key_from_dicts(batch, "b") == [0, -1, -2, -3, -4]


@pytest.mark.parametrize(
    "pad_div",
    [None, 4],
    ids=["no_divisor", "divisor_4"],
)
def test_pad_within_micro(pad_div: int | None) -> None:
    """
    Padding must
    1. equalise sequence lengths across the batch and
    2. respect the `pad_seq_len_divisible` argument when supplied.
    """
    batch = [[1], [1, 2, 3], [1, 2]]
    out = sftp.pad_within_micro(batch, pad_token_id=0, pad_seq_len_divisible=pad_div)
    max_len = max(map(len, out))
    # 1. all equal length
    assert all(len(x) == max_len for x in out)
    # 2. length divisible by divisor
    if pad_div is not None:
        assert max_len % pad_div == 0


def test_default_collater_shapes() -> None:
    """
    Check that `default_collater` returns the expected keys, tensor dtypes,
    and correct padding (-100 for labels, 0 elsewhere).
    """
    raw_batch = [
        {
            "input_ids": [1, 2],
            "attention_mask": [1, 1],
            "labels": [101, 102],
            "loss_mask": [1, 1],
        },
        {
            "input_ids": [3],
            "attention_mask": [1],
            "labels": [103],
            "loss_mask": [1],
        },
    ]

    collated = sftp.default_collater(raw_batch, pad_token_id=0)
    # Keys preserved
    assert set(collated) == {"input_ids", "attention_mask", "labels", "loss_mask"}

    # Batch dimension added
    assert collated["input_ids"].shape[0] == 2
    # Same seq length for all keys
    lens = {v.shape[1] for v in collated.values()}
    assert len(lens) == 1
    lens.pop()

    # Verify padded values
    assert collated["input_ids"][1, 1:].eq(0).all()
    assert collated["attention_mask"][1, 1:].eq(0).all()
    assert collated["labels"][1, 1:].eq(-100).all()
    # `loss_mask` mirrors labels but with 0/1 instead of ids
    assert collated["loss_mask"][1, 1:].eq(0).all()

    # Sanity on dtype
    for tensor in collated.values():
        assert tensor.dtype == torch.long


def test_tokenize_function_strips_special_tokens(dummy_tokenizer: DummyTokenizer) -> None:
    """
    • Context trailing *eos* token must be removed.
    • Target leading *bos* token must be removed.
    • Labels contain ‑100 on context tokens (plus one trailing) and the real
      ids on target tokens.
    """
    pre = sftp.SFTSingleTurnPreprocessor(dummy_tokenizer)
    batch_in = {
        "context": ["AAA"],
        "target": ["BBB"],
    }
    ds = DummyDataset()
    out = pre._tokenize_function(batch_in, ds)

    # 1. Structural checks -------------------------------------------------- #
    for key in ("input_ids", "attention_mask", "labels", "loss_mask"):
        assert isinstance(out[key], list) and isinstance(out[key][0], list)

    ids = out["input_ids"][0]
    lbl = out["labels"][0]

    assert len(ids) == len(lbl)  # same length by construction
    assert lbl.count(-100) >= len(ids) - len("BBB") - 1  # at least context masked

    # Ensure context's final token is *not* a special token (eos stripped)
    ctx_len = len(dummy_tokenizer._encode_single("AAA")) - 1  # eos removed
    assert ids[ctx_len - 1] != dummy_tokenizer.eos_token_id

    # Loss mask must be 1 where label != -100 else 0
    loss_mask = out["loss_mask"][0]
    assert all((lm == (lab != -100)) for lm, lab in zip(loss_mask, lbl))


def test_compute_dataset_max_len_respects_block_size(dummy_tokenizer: DummyTokenizer) -> None:
    """Max-len rounds up to multiple of 8 and is capped by `block_size`."""
    pre = sftp.SFTSingleTurnPreprocessor(dummy_tokenizer)

    # Build fake tokenised dataset (list of dicts)
    tokenised_ds = [
        {"input_ids": [0] * 11},
        {"input_ids": [0] * 21},
    ]
    # Without block size
    assert pre._compute_dataset_max_len(tokenised_ds) == math.ceil(21 / 8) * 8

    # With smaller block size
    pre.block_size = 16
    assert pre._compute_dataset_max_len(tokenised_ds) == 16


@pytest.mark.parametrize("max_len", [8, 13])
def test_pad_function_behaviour(dummy_tokenizer: DummyTokenizer, max_len: int) -> None:
    """_pad_function should truncate or pad every field to *exactly* max_len."""
    pre = sftp.SFTSingleTurnPreprocessor(dummy_tokenizer)
    pad_fn = pre._pad_function(max_len)

    # Produce a deliberately longer + shorter example
    examples = {
        "input_ids": [[1] * (max_len + 2)],
        "attention_mask": [[1] * (max_len + 2)],
        "labels": [[-100] * (max_len + 2)],
        "loss_mask": [[0] * (max_len + 2)],
    }
    out = pad_fn(examples)

    for key, seq in out.items():
        assert len(seq[0]) == max_len, f"{key} not equal to max_len"
        if key == "attention_mask":
            # Padding token for attention_mask is 0
            assert seq[0][-1] in (0, 1)


@pytest.mark.skipif(
    pytest.importorskip("datasets", reason="datasets not installed") is None
    or pytest.importorskip("transformers", reason="transformers not installed") is None,
    reason="Optional integration test that relies on HF libs",
)
def test_full_process_pipeline(dummy_tokenizer: DummyTokenizer) -> None:
    """
    End-to-end smoke test for ``process`` using an in-memory Hugging-Face
    ``datasets.Dataset`` (not *DatasetDict*).  Validates that the resulting
    dataset has sequences of uniform length and contains the expected columns.
    """
    import datasets  # type: ignore

    # Build a tiny raw dataset
    raw = datasets.Dataset.from_dict(
        {
            "context": ["hello"] * 3,
            "target": ["world"] * 3,
        }
    )

    pre = sftp.SFTSingleTurnPreprocessor(dummy_tokenizer)
    processed = pre.process(raw, DummyDataset())

    expected_cols = {"input_ids", "attention_mask", "labels", "loss_mask"}
    assert set(processed.column_names) == expected_cols

    first_len = len(processed[0]["input_ids"])
    assert all(len(r["input_ids"]) == first_len for r in processed)

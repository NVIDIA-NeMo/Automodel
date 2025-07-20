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
import pytest
from datasets import Dataset

import nemo_automodel.components.datasets.llm.mock as mock


# ---------- make_vocab --------------------------------------------------------
def test_make_vocab_basic():
    vocab = mock.make_vocab(vocab_size=10)
    # size and required specials
    assert len(vocab) == 10
    assert vocab["<pad>"] == 0
    assert vocab["<eos>"] == 1
    # every key is unique and values are contiguous ints
    assert sorted(vocab.values()) == list(range(10))


# ---------- gen_sentence_ids ---------------------------------------------------
def test_gen_sentence_ids_returns_ints_and_ends_with_eos():
    vocab = mock.make_vocab(20)
    sent = mock.gen_sentence_ids(vocab, mean_len=5.0, std_len=1.0, max_len=8)

    # last token must be <eos>
    assert sent[-1] == vocab["<eos>"]

    # sentence length within the requested range (1-max_len + eos)
    assert 2 <= len(sent) <= 9  # at least one word + eos

    # All tokens must exist in the vocab values
    assert all(tok in vocab.values() for tok in sent)


# ---------- build_unpacked_dataset --------------------------------------------
@pytest.fixture(scope="module")
def toy_ds() -> Dataset:
    return mock.build_unpacked_dataset(
        num_sentences=4,
        mean_len=6.0,
        std_len=1.5,
        vocab_size=32,
        max_sentence_len=10,
        seed=42,
    )


def test_dataset_length(toy_ds):
    assert len(toy_ds) == 4


def test_example_fields_exist_and_lengths_match(toy_ds):
    ex = toy_ds[0]
    expected_fields = {"input_ids", "attention_mask", "labels", "position_ids"}
    assert expected_fields.issubset(ex.keys())

    L = len(ex["input_ids"])
    # all sequence fields must be same length
    assert len(ex["attention_mask"]) == L
    assert len(ex["labels"]) == L
    assert len(ex["position_ids"]) == L


def test_attention_mask_is_all_ones(toy_ds):
    for ex in toy_ds:
        assert set(ex["attention_mask"]) == {1}


def test_labels_equal_inputs(toy_ds):
    for ex in toy_ds:
        assert ex["labels"] == ex["input_ids"]


def test_position_ids_reset_after_eos(toy_ds):
    eos = mock.make_vocab(32)["<eos>"]
    for ex in toy_ds:
        running_pos = 0
        for tok, pos in zip(ex["input_ids"], ex["position_ids"]):
            assert pos == running_pos
            running_pos = 0 if tok == eos else running_pos + 1


def test_dataset_determinism_with_seed():
    ds1 = mock.build_unpacked_dataset(seed=123)
    ds2 = mock.build_unpacked_dataset(seed=123)
    # Convert to list of dicts to avoid arrow table pointer equality pitfalls
    assert ds1.to_list() == ds2.to_list()

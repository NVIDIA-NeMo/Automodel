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

import nemo_automodel.components.datasets.llm.squad as mqd

make_squad_dataset = mqd.make_squad_dataset


class DummyTokenizer:
    """
    A *very* small tokenizer good enough for unit-testing the logic of
    `make_squad_dataset`.

    • Each whitespace-separated token becomes one integer id (lookup table).
    • Provides eos/bos ids.
    • Optionally provides a `chat_template` and a very small
      `apply_chat_template` implementation to trigger the code-path that
      computes `response_start`.
    """

    def __init__(self, with_chat_template=False, start_of_turn="▸"):
        self._vocab = {"<eos>": 0, "<bos>": 1, start_of_turn: 2}
        self.eos_token_id = 0
        self.bos_token_id = 1
        if with_chat_template:
            # merely a flag that tells make_squad_dataset to use the chat path
            self.chat_template = True
            self._start_tok = start_of_turn
            self.start_of_turn = start_of_turn

    def __call__(self, text, add_special_tokens=True):
        ids = [self._tok_to_id(t) for t in text.strip().split()]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return {"input_ids": ids}

    def _tok_to_id(self, tok):
        idx = self._vocab.get(tok)
        if idx is None:
            idx = len(self._vocab)
            self._vocab[tok] = idx
        return idx

    # Mini implementation of apply_chat_template. The contract is:
    #  - Accept list[dict{role, content}]
    #  - Prepend start_of_turn token before each role
    #  - Append eos at very end
    #  - Return list[int] token ids
    def apply_chat_template(self, messages, **kwargs):
        ids = []
        for msg in messages:
            ids.append(self._tok_to_id(self._start_tok))
            ids.extend(self(msg["content"], add_special_tokens=False)["input_ids"])
        ids.append(self.eos_token_id)
        return ids


@pytest.fixture(scope="function")
def tiny_hf_dataset():
    """
    Return an in-memory datasets.Dataset with exactly two rows, mimicking the
    SQuAD schema the function expects.

    We rely on automatic feature inference, which correctly handles the nested
    “answers” field without any manual Feature specification.
    """
    data = {
        "id": ["0", "1"],
        "title": ["t0", "t1"],
        "context": ["Earth is round.", "Sky is blue."],
        "question": ["What shape is Earth?", "What color is the sky?"],
        "answers": [
            {"text": ["round"], "answer_start": [9]},
            {"text": ["blue"], "answer_start": [7]},
        ],
    }
    return Dataset.from_dict(data)


@pytest.fixture(autouse=True)
def patch_load_dataset(monkeypatch, tiny_hf_dataset):
    """
    Monkey-patch datasets.load_dataset so no network call happens and emulate
    slice syntax like "train[:1]".
    """

    def _fake_load_dataset(name, split=None, **kw):
        if isinstance(split, str) and "[" in split:
            # e.g. "train[:3]"  → keep upper bound 3
            upper = int(split.split("[")[1].split(":")[1].rstrip("]"))
            return tiny_hf_dataset.select(range(upper))
        return tiny_hf_dataset

    monkeypatch.setattr(mqd, "load_dataset", _fake_load_dataset)
    yield


def test_plain_tokenizer_basic():
    """
    The “no chat template” branch should:
      • concatenate context+question+space with answer
      • drop EOS from context, BOS from answer
      • produce loss_mask = [0]*len(context_ids) + [1]*len(answer_ids)
    """
    tok = DummyTokenizer()
    ds = make_squad_dataset(tok, split="train", seq_length=None)
    # The dataset should have 2 examples (mocked dataset length)
    assert len(ds) == 2
    sample = ds[0]
    # keys present?
    assert set(sample) == {"input_ids", "labels", "loss_mask"}
    # loss_mask correct length
    assert len(sample["input_ids"]) == len(sample["loss_mask"]) == len(sample["labels"])
    # Verify at least one 1 exists in loss_mask (answer tokens)
    assert 1 in sample["loss_mask"]
    # Context tokens (loss_mask==0) must precede answer tokens (loss_mask==1)
    first_one = sample["loss_mask"].index(1)
    assert all(v == 0 for v in sample["loss_mask"][:first_one])
    assert all(v == 1 for v in sample["loss_mask"][first_one:])


def test_sequence_padding():
    """
    When `seq_length` is supplied, every field must be padded to that exact
    length; `loss_mask` should be padded with zeros; `input_ids` & `labels`
    with eos.
    """
    tok = DummyTokenizer()
    pad_len = 32
    ds = make_squad_dataset(tok, seq_length=pad_len)
    for row in ds:
        for key, val in row.items():
            assert len(val) == pad_len
        # last id in labels must equal eos
        assert row["labels"][-1] == tok.eos_token_id
        # loss mask padding must be zeros
        assert row["loss_mask"][-1] == 0


def test_limit_dataset_samples(monkeypatch):
    """
    `limit_dataset_samples` should translate into slice-syntax and therefore
    load only the requested number of rows.
    """
    tok = DummyTokenizer()

    ds = make_squad_dataset(tok, limit_dataset_samples=1)
    assert len(ds) == 1


def test_chat_template_path():
    """
    With `chat_template`, the code path that uses
    `formatting_prompts_func_with_chat_template` must be executed.

    We also test that:
      • `start_of_turn_token` is respected when computing `loss_mask`
      • everything after the second start-of-turn token gets loss_mask==1
    """
    start_token = "▸"
    tok = DummyTokenizer(with_chat_template=True, start_of_turn=start_token)

    ds = make_squad_dataset(
        tok,
        start_of_turn_token=start_token,
        seq_length=None,  # no padding
    )
    row = ds[0]
    sot_id = tok(start_token, add_special_tokens=False)["input_ids"][0]

    # The index of the *second* SOT token +1 is response_start
    idx_first = row["input_ids"].index(sot_id)
    idx_second = row["input_ids"].index(sot_id, idx_first + 1)
    response_start = idx_second + 1

    # Everything before response_start must have loss_mask==0; after ==>1
    assert all(v == 0 for v in row["loss_mask"][:response_start])
    assert all(v == 1 for v in row["loss_mask"][response_start:])


def test_fp8_flag_is_noop():
    """
    The `fp8` flag exists for future use. Setting it should not alter
    functional behaviour nor raise.
    """
    tok = DummyTokenizer()
    ds = make_squad_dataset(tok, fp8=True)
    # still returns a dataset
    assert isinstance(ds, Dataset)
    assert len(ds) == 2

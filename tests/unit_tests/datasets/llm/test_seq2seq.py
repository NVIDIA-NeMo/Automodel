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

import pytest
import torch
from datasets import Dataset

import nemo_automodel.components.datasets.llm.seq2seq as s2s
from nemo_automodel.components.datasets.lazy_mapped_dataset import LazyMappedDataset
from nemo_automodel.components.datasets.utils import default_collater
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

make_seq2seq_dataset = s2s.make_seq2seq_dataset


class DummyTokenizer:
    """Whitespace tokenizer good enough to exercise the seq2seq data path.

    Mirrors the parts of the HF tokenizer interface that ``make_seq2seq_dataset``
    uses: callable with either ``text`` or ``text_target``, returns
    ``input_ids`` + ``attention_mask``, appends eos, and exposes pad/eos ids.
    Defaults match T5 (pad=0, eos=1, no bos).
    """

    def __init__(self):
        self._vocab = {"<pad>": 0, "</s>": 1}
        self.pad_token_id = 0
        self.pad_token = "<pad>"
        self.eos_token_id = 1
        self.eos_token = "</s>"
        self.bos_token_id = None
        self.chat_template = None

    def _tok_to_id(self, tok):
        idx = self._vocab.get(tok)
        if idx is None:
            idx = len(self._vocab)
            self._vocab[tok] = idx
        return idx

    def __call__(self, text=None, text_target=None, add_special_tokens=True, max_length=None, truncation=False, **kw):
        s = text if text is not None else text_target
        ids = [self._tok_to_id(t) for t in s.strip().split()]
        if add_special_tokens:
            ids.append(self.eos_token_id)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


@pytest.fixture(scope="function")
def tiny_hf_dataset():
    """Two-row in-memory dataset with the SQuAD schema."""
    data = {
        "id": ["0", "1"],
        "title": ["t0", "t1"],
        "context": ["Earth is round", "Sky is very blue today"],
        "question": ["shape of Earth", "color of sky"],
        "answers": [
            {"text": ["round"], "answer_start": [9]},
            {"text": ["blue today"], "answer_start": [7]},
        ],
    }
    return Dataset.from_dict(data)


@pytest.fixture(autouse=True)
def patch_load_dataset(monkeypatch, tiny_hf_dataset):
    """Avoid any network call; honor ``train[:n]`` slice syntax."""

    def _fake_load_dataset(name, split=None, **kw):
        if isinstance(split, str) and "[" in split:
            upper = int(split.split("[")[1].split(":")[1].rstrip("]"))
            return tiny_hf_dataset.select(range(upper))
        return tiny_hf_dataset

    monkeypatch.setattr(s2s, "load_dataset", _fake_load_dataset)
    yield


def _seq2seq_sample(input_ids, labels, pad_id=0, decoder_start=0):
    """Build a per-sample dict in the shape make_seq2seq_dataset produces."""
    return {
        "input_ids": list(input_ids),
        "attention_mask": [1] * len(input_ids),
        "labels": list(labels),
        "decoder_input_ids": s2s._shift_right(labels, decoder_start),
        "___PAD_TOKEN_IDS___": {
            "input_ids": pad_id,
            "attention_mask": 0,
            "labels": -100,
            "decoder_input_ids": pad_id,
        },
    }


def test_shift_right():
    # position i of the result is the input that predicts token_ids[i]
    assert s2s._shift_right([7, 8, 9], 0) == [0, 7, 8]
    assert s2s._shift_right([5], 0) == [0]
    assert s2s._shift_right([], 0) == [0]


def test_make_dataset_keys_and_no_shift():
    tok = DummyTokenizer()
    ds = make_seq2seq_dataset(tok, split="train", seq_length=None)
    assert isinstance(ds, LazyMappedDataset)
    assert len(ds) == 2

    sample = ds[0]
    assert set(sample.keys()) == {
        "input_ids",
        "attention_mask",
        "labels",
        "decoder_input_ids",
        "___PAD_TOKEN_IDS___",
    }

    labels = sample["labels"]
    dec_in = sample["decoder_input_ids"]
    # labels are NOT shifted: same length, ending in eos.
    assert len(dec_in) == len(labels)
    assert labels[-1] == tok.eos_token_id
    # decoder_input_ids is the right-shift of labels with decoder_start (== pad == 0).
    assert dec_in[0] == 0
    assert dec_in[1:] == labels[:-1]
    # encoder side is independent of the decoder side.
    assert len(sample["input_ids"]) == len(sample["attention_mask"])
    assert all(m == 1 for m in sample["attention_mask"])
    # no -100 leaks into the input id streams at the sample level.
    assert -100 not in sample["input_ids"]
    assert -100 not in dec_in


def test_pad_token_ids_metadata():
    tok = DummyTokenizer()
    sample = make_seq2seq_dataset(tok)[0]
    pads = sample["___PAD_TOKEN_IDS___"]
    assert pads["labels"] == -100
    assert pads["input_ids"] == tok.pad_token_id
    assert pads["decoder_input_ids"] == tok.pad_token_id
    assert pads["attention_mask"] == 0


def test_decoder_start_token_id_override():
    tok = DummyTokenizer()
    ds = make_seq2seq_dataset(tok, decoder_start_token_id=7)
    dec_in = ds[0]["decoder_input_ids"]
    assert dec_in[0] == 7
    assert dec_in[1:] == ds[0]["labels"][:-1]


def test_limit_dataset_samples():
    tok = DummyTokenizer()
    ds = make_seq2seq_dataset(tok, limit_dataset_samples=1)
    assert len(ds) == 1


def test_default_collater_pads_seq2seq():
    # different encoder and decoder lengths across the two samples.
    s1 = _seq2seq_sample(input_ids=[5, 6, 7, 1], labels=[8, 9, 1])
    s2 = _seq2seq_sample(input_ids=[5, 1], labels=[8, 1])
    batch = default_collater([s1, s2])

    assert set(batch.keys()) >= {"input_ids", "attention_mask", "labels", "decoder_input_ids", "padding_mask"}
    # each field is padded to its own max length (encoder=4, decoder=3).
    assert batch["input_ids"].shape == (2, 4)
    assert batch["attention_mask"].shape == (2, 4)
    assert batch["labels"].shape == (2, 3)
    assert batch["decoder_input_ids"].shape == (2, 3)

    # labels pad with -100; id fields pad with pad id 0; attention with 0.
    assert batch["labels"][1].tolist() == [8, 1, -100]
    assert batch["decoder_input_ids"][1].tolist() == [0, 8, 0]
    assert batch["input_ids"][1].tolist() == [5, 1, 0, 0]
    assert batch["attention_mask"][1].tolist() == [1, 1, 0, 0]


def test_tiny_t5_forward_loss():
    """End-to-end alignment: a tiny T5 trains on unshifted labels via MaskedCrossEntropy."""
    transformers = pytest.importorskip("transformers")
    T5Config = transformers.T5Config
    T5ForConditionalGeneration = transformers.T5ForConditionalGeneration

    torch.manual_seed(0)
    config = T5Config(
        vocab_size=32,
        d_model=16,
        d_kv=8,
        d_ff=32,
        num_layers=2,
        num_heads=2,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    model = T5ForConditionalGeneration(config)
    model.train()

    s1 = _seq2seq_sample(input_ids=[5, 6, 7, 1], labels=[8, 9, 1])
    s2 = _seq2seq_sample(input_ids=[5, 4, 1], labels=[8, 1])
    batch = default_collater([s1, s2])

    labels = batch.pop("labels")
    batch.pop("padding_mask", None)  # T5 ignores it; drop so we pass a clean kwarg set
    out = model(**batch)

    # logits align position-for-position with the unshifted labels.
    assert out.logits.shape[:2] == labels.shape
    assert out.logits.shape[-1] == config.vocab_size

    loss_fn = MaskedCrossEntropy()
    num_label_tokens = int((labels != -100).sum())
    loss = loss_fn(out.logits, labels, num_label_tokens=num_label_tokens)
    assert torch.isfinite(loss)
    assert loss.item() > 0

    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0

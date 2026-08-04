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
import math
from typing import Any, Dict, List

import torch

import nemo_automodel.components.datasets.llm.retrieval_collator as rc


class FakeTokenizer:
    """Minimal tokenizer that records the texts it receives for assertion."""

    def __init__(self):
        self.last_texts = None

    def __call__(
        self,
        texts: List[str],
        max_length: int,
        padding: Any,
        truncation: bool,
    ) -> Dict[str, List[List[int]]]:
        self.last_texts = texts
        input_ids = []
        attention_masks = []
        for t in texts:
            tokens = t.split()
            if truncation:
                tokens = tokens[:max_length]
            ids = list(range(len(tokens)))
            mask = [1] * len(ids)
            input_ids.append(ids)
            attention_masks.append(mask)
        return {"input_ids": input_ids, "attention_mask": attention_masks}

    def pad(
        self,
        features: List[Dict[str, List[int]]],
        padding: Any,
        pad_to_multiple_of: int,
        return_tensors: str,
    ) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features) if features else 0
        if pad_to_multiple_of and max_len % pad_to_multiple_of != 0:
            max_len = int(math.ceil(max_len / pad_to_multiple_of) * pad_to_multiple_of)
        input_ids = []
        attention_masks = []
        for f in features:
            ids = list(f["input_ids"])
            mask = list(f["attention_mask"])
            pad_len = max_len - len(ids)
            ids = ids + [0] * pad_len
            mask = mask + [0] * pad_len
            input_ids.append(ids)
            attention_masks.append(mask)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        }


def _make_cross_encoder_batch(num_examples: int = 2, num_labels: int = None) -> List[Dict[str, Any]]:
    batch = []
    for i in range(num_examples):
        item = {"question": f"what is item {i}", "doc_text": f"document about item {i}"}
        if num_labels is not None:
            item["num_labels"] = num_labels
        batch.append(item)
    return batch


def test_cross_encoder_default_prompt_template():
    """Verify the default template produces the original hardcoded format."""
    tok = FakeTokenizer()
    collator = rc.CrossEncoderCollator(rerank_max_length=512, tokenizer=tok, padding=True)
    batch = _make_cross_encoder_batch(num_examples=1)
    collator(batch)

    expected = "question:what is item 0 \n \n passage:document about item 0"
    assert tok.last_texts == [expected]


def test_cross_encoder_custom_prompt_template():
    """Verify a custom template is used for formatting."""
    tok = FakeTokenizer()
    collator = rc.CrossEncoderCollator(
        rerank_max_length=512,
        prompt_template="Q: {query} P: {passage}",
        tokenizer=tok,
        padding=True,
    )
    batch = _make_cross_encoder_batch(num_examples=2)
    collator(batch)

    assert tok.last_texts == [
        "Q: what is item 0 P: document about item 0",
        "Q: what is item 1 P: document about item 1",
    ]


def test_cross_encoder_collator_output_keys_and_shapes():
    tok = FakeTokenizer()
    collator = rc.CrossEncoderCollator(rerank_max_length=512, tokenizer=tok, padding=True)
    batch = _make_cross_encoder_batch(num_examples=3)
    output = collator(batch)

    assert "input_ids" in output
    assert "attention_mask" in output
    assert output["input_ids"].shape[0] == 3
    assert output["attention_mask"].shape == output["input_ids"].shape
    assert output["input_ids"].dtype == torch.long
    assert "labels" not in output


def test_cross_encoder_collator_labels_from_num_labels():
    tok = FakeTokenizer()
    collator = rc.CrossEncoderCollator(rerank_max_length=512, tokenizer=tok, padding=True)
    batch = _make_cross_encoder_batch(num_examples=2, num_labels=4)
    output = collator(batch)

    assert "labels" in output
    assert output["labels"].shape == (4,)
    assert output["labels"].dtype == torch.long
    assert torch.all(output["labels"] == 0)


def test_cross_encoder_collator_pad_to_multiple_of():
    tok = FakeTokenizer()
    collator = rc.CrossEncoderCollator(rerank_max_length=512, tokenizer=tok, padding=True, pad_to_multiple_of=8)
    batch = [
        {"question": "short", "doc_text": "short"},
        {"question": "this is a much longer query text about many things", "doc_text": "document"},
    ]
    output = collator(batch)

    assert output["input_ids"].shape[1] % 8 == 0

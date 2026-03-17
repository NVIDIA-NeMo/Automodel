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

import pytest
import torch

import nemo_automodel.components.datasets.llm.retrieval_collator as rc


class FakeTokenizer:
    def __call__(
        self,
        texts: List[str],
        max_length: int,
        padding: Any,
        truncation: bool,
        return_token_type_ids: bool,
    ) -> Dict[str, List[List[int]]]:
        # Simple whitespace tokenizer: ids are range(len(tokens))
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
        # Determine max length and round to multiple if requested
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


def test_unpack_doc_values():
    features = [
        {"input_ids": [[1, 2], [3]], "attention_mask": [[1, 1], [1]]},
    ]
    out = rc._unpack_doc_values(features)
    assert out == [{"input_ids": [1, 2], "attention_mask": [1, 1]}, {"input_ids": [3], "attention_mask": [1]}]


def test_merge_and_convert_helpers():
    collator = rc.RetrievalBiencoderCollator(FakeTokenizer())
    query_batch = {"input_ids": [[10], [20]], "attention_mask": [[1], [1]]}  # batch_size = 2
    # 2 examples * train_n_passages(=2) = 4 document rows
    doc_batch = {"input_ids": [[100], [101], [110], [111]], "attention_mask": [[1], [1], [1], [1]]}
    merged = collator._merge_batch_dict(query_batch, doc_batch, train_n_passages=2)
    # Ensure query keys are prefixed and doc keys reshaped to [batch, passages, seq]
    assert "q_input_ids" in merged and "d_input_ids" in merged
    assert merged["d_input_ids"] == [[[100], [101]], [[110], [111]]]
    # Convert dict-of-lists to list-of-dicts
    lst = collator._convert_dict_to_list({"a": [1, 2], "b": [3, 4]})
    assert lst == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]


def _make_batch(num_examples: int = 2, docs_per_example: int = 3) -> List[Dict[str, Any]]:
    batch = []
    for i in range(num_examples):
        question = f"what is item {i}"
        docs = [f"doc {i}-{j}" for j in range(docs_per_example)]
        batch.append({"question": question, "doc_text": docs, "doc_image": [""] * docs_per_example})
    return batch


def test_collator_end_to_end_no_prefix():
    tok = FakeTokenizer()
    collator = rc.RetrievalBiencoderCollator(tokenizer=tok, q_max_len=16, p_max_len=16, padding=True)
    batch = _make_batch(num_examples=2, docs_per_example=3)
    out = collator(batch)
    # Expected keys
    for k in ["q_input_ids", "q_attention_mask", "d_input_ids", "d_attention_mask", "labels"]:
        assert k in out
    # Shapes: queries [B, Lq], docs [B * P, Ld], labels [B]
    assert out["q_input_ids"].shape[0] == 2
    assert out["d_input_ids"].shape[0] == 2 * 3
    assert out["labels"].dtype == torch.long and out["labels"].shape[0] == 2 and torch.all(out["labels"] == 0)
    # Ensure attention masks align with input_ids shapes
    assert out["q_input_ids"].shape == out["q_attention_mask"].shape
    assert out["d_input_ids"].shape == out["d_attention_mask"].shape


def test_collator_with_prefix_and_pad_multiple():
    tok = FakeTokenizer()
    collator = rc.RetrievalBiencoderCollator(
        tokenizer=tok, q_max_len=32, p_max_len=32, query_prefix="Q:", passage_prefix="D:", padding=True, pad_to_multiple_of=4
    )
    # Make varying lengths so padding is exercised and rounded to multiple-of 4
    batch = [
        {"question": "short", "doc_text": ["tiny", "a bit longer"], "doc_image": ["", ""]},
        {"question": "this is a somewhat longer question", "doc_text": ["short doc", "this is a longish doc text"], "doc_image": ["", ""]},
    ]
    out = collator(batch)
    # Verify padding rounded to multiple of 4
    assert out["q_input_ids"].shape[1] % 4 == 0
    assert out["d_input_ids"].shape[1] % 4 == 0
    # Still produces expected label size
    assert out["labels"].shape[0] == 2


def test_collator_with_dataset_instruction():
    """Test that use_dataset_instruction prepends query/passage instructions from dataset metadata."""
    tok = FakeTokenizer()
    collator = rc.RetrievalBiencoderCollator(
        tokenizer=tok,
        q_max_len=64,
        p_max_len=64,
        padding=True,
        use_dataset_instruction=True,
    )

    # Simulate the instruction from merlin_metadata.json
    query_instruction = "Instruct: Given a question, retrieve Wikipedia passages that answer the question\nQuery:"
    passage_instruction = ""

    # Create a batch with query_instruction and passage_instruction fields
    # as they would come from the dataset transform function
    batch = [
        {
            "question": "What is the capital of France?",
            "doc_text": ["Paris is the capital", "Lyon is a city"],
            "doc_image": ["", ""],
            "query_instruction": query_instruction,
            "passage_instruction": passage_instruction,
        },
        {
            "question": "Who invented the telephone?",
            "doc_text": ["Alexander Graham Bell invented", "The telephone was invented"],
            "doc_image": ["", ""],
            "query_instruction": query_instruction,
            "passage_instruction": passage_instruction,
        },
    ]

    out = collator(batch)

    # Verify output has expected keys
    for k in ["q_input_ids", "q_attention_mask", "d_input_ids", "d_attention_mask", "labels"]:
        assert k in out

    # Verify batch size
    assert out["q_input_ids"].shape[0] == 2
    assert out["d_input_ids"].shape[0] == 2 * 2  # 2 examples * 2 docs each

    # The key test: verify that the tokenizer received queries with instructions prepended
    # Since FakeTokenizer splits on whitespace, we can check the number of tokens
    # The instruction has many tokens, so the query with instruction should have more tokens
    # than just the question alone

    # Create a second batch without instructions to compare
    collator_no_instruction = rc.RetrievalBiencoderCollator(
        tokenizer=tok,
        q_max_len=64,
        p_max_len=64,
        padding=True,
        use_dataset_instruction=False,
    )

    batch_no_instruction = [
        {
            "question": "What is the capital of France?",
            "doc_text": ["Paris is the capital", "Lyon is a city"],
            "doc_image": ["", ""],
            "query_instruction": "",
            "passage_instruction": "",
        },
        {
            "question": "Who invented the telephone?",
            "doc_text": ["Alexander Graham Bell invented", "The telephone was invented"],
            "doc_image": ["", ""],
            "query_instruction": "",
            "passage_instruction": "",
        },
    ]

    out_no_instruction = collator_no_instruction(batch_no_instruction)

    # With instructions, queries should have more tokens (instruction adds many words)
    # Count non-zero tokens in first query
    with_instruction_tokens = (out["q_attention_mask"][0] == 1).sum().item()
    without_instruction_tokens = (out_no_instruction["q_attention_mask"][0] == 1).sum().item()

    # The instruction adds 12 words, so we should see significantly more tokens
    assert with_instruction_tokens > without_instruction_tokens
    assert with_instruction_tokens - without_instruction_tokens > 10  # At least 10 more tokens from instruction

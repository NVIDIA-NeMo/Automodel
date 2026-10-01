# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Real dataset batching and persistent-worker context dropout, without downloads."""

import json
from dataclasses import asdict

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast

from nemo_automodel.components.datasets.llm.retrieval_dataset_inline import ContextAwareRetrievalDatasetConfig
from nemo_automodel.components.models.qwen3_reranker.collator import Qwen3ContextAwareRerankerCollator


@pytest.fixture
def tokenizer():
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "TRACE": 2, "GLOBAL": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")


def test_real_jsonl_preserves_missing_context_through_batched_loading(tmp_path, tokenizer):
    rows = [
        {"query": "q0", "pos_doc": ["positive 0"], "neg_doc": ["negative 0"], "trace": "TRACE", "origin": "GLOBAL"},
        {"query": "q1", "pos_doc": ["positive 1"], "neg_doc": ["negative 1"]},
    ]
    path = tmp_path / "inline.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = ContextAwareRetrievalDatasetConfig(
        data_dir_list=str(path), n_passages=2, reasoning_column="trace", global_query_column="origin"
    )
    original_config = asdict(config)
    dataset = config.build()
    features = dataset.__getitems__([0, 1])
    assert [(row["question"], row["reasoning"], row["global_query"]) for row in features] == [
        ("q0", "TRACE", "GLOBAL"),
        ("q0", "TRACE", "GLOBAL"),
        ("q1", None, None),
        ("q1", None, None),
    ]
    assert [row["doc_text"] for row in features] == ["positive 0", "negative 0", "positive 1", "negative 1"]
    collator = Qwen3ContextAwareRerankerCollator(
        rerank_max_length=256, tokenizer=tokenizer, reasoning_drop_prob=0.0, global_query_drop_prob=0.0
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collator)))
    assert batch["input_ids"].shape[0] == 4
    assert torch.equal(batch["labels"], torch.zeros(2, dtype=torch.long))
    for token in ("TRACE", "GLOBAL"):
        present = (batch["input_ids"] == tokenizer.convert_tokens_to_ids(token)).any(dim=-1)
        assert present.tolist() == [True, True, False, False]
    assert asdict(config) == original_config


@pytest.mark.runtime_budget(
    20,
    reason="A real spawned DataLoader worker imports PyTorch and Transformers before exercising shared epoch state.",
)
def test_persistent_worker_observes_epoch_changes_in_actual_prompts(tokenizer):
    features = [
        {"question": f"query {i}", "doc_text": "document", "reasoning": "TRACE", "global_query": "GLOBAL"}
        for i in range(64)
    ]
    collator = Qwen3ContextAwareRerankerCollator(rerank_max_length=256, tokenizer=tokenizer)
    loader = DataLoader(
        features,
        batch_size=64,
        collate_fn=collator,
        num_workers=1,
        persistent_workers=True,
        multiprocessing_context="spawn",
        timeout=30,
    )
    try:
        (epoch0,) = list(loader)
        collator.set_epoch(1)
        (epoch1,) = list(loader)
        expected = collator(features)
        torch.testing.assert_close(epoch1["input_ids"], expected["input_ids"], rtol=0, atol=0)
        torch.testing.assert_close(epoch1["attention_mask"], expected["attention_mask"], rtol=0, atol=0)
        trace_id = tokenizer.convert_tokens_to_ids("TRACE")
        kept0 = (epoch0["input_ids"] == trace_id).any(dim=-1)
        kept1 = (epoch1["input_ids"] == trace_id).any(dim=-1)
        assert not torch.equal(kept0, kept1)
    finally:
        # DataLoader owns the persistent worker and shuts it down on destruction.
        del loader

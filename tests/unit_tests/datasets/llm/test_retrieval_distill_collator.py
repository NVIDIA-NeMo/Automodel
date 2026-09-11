# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from nemo_automodel.components.datasets.llm.retrieval_distill_collator import (
    BiEncoderDistillCollator,
    _content_hash,
)


class _TinyTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]

    def __call__(
        self,
        texts,
        max_length=None,
        padding=None,
        truncation=False,
        return_token_type_ids=True,
    ):
        assert return_token_type_ids is False
        if isinstance(texts, str):
            texts = [texts]
        input_ids = []
        attention_mask = []
        for text in texts:
            ids = [i + 1 for i, _ in enumerate(text.split())]
            if truncation and max_length is not None:
                ids = ids[:max_length]
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def pad(self, features, padding=True, pad_to_multiple_of=None, return_tensors=None):
        max_len = max((len(f["input_ids"]) for f in features), default=0)
        if pad_to_multiple_of and max_len % pad_to_multiple_of:
            max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        input_ids = []
        attention_mask = []
        for feature in features:
            ids = list(feature["input_ids"])
            mask = list(feature["attention_mask"])
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _TokenizerWithoutTokenTypeIds(_TinyTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __call__(self, texts, max_length=None, padding=None, truncation=False):
        return super().__call__(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_token_type_ids=False,
        )


@pytest.mark.parametrize("tokenizer_file", ["mistral_instruct_tokenizer_240323.model.v3", "tekken_240911.json"])
def test_distill_collator_with_mistral_common(tokenizer_file):
    # Mistral's backend is optional; the tokenizers are bundled with mistral-common.
    mistral_common = pytest.importorskip("mistral_common")
    from transformers import MistralCommonBackend

    tokenizer = MistralCommonBackend(Path(mistral_common.__file__).parent / "data" / tokenizer_file)
    tokenizer.pad_token = tokenizer.eos_token
    collator = BiEncoderDistillCollator(tokenizer=tokenizer, q_max_len=32, p_max_len=32, pad_to_multiple_of=4)
    batch = [
        {"question": "what is alpha", "doc_text": ["alpha positive", "alpha hard one", "alpha hard two"]},
        {"question": "what is beta", "doc_text": ["beta positive", "beta hard one"]},
    ]

    out = collator(batch)

    assert out["n_mask"].tolist() == [[1, 1], [1, 0]]
    for i, example in enumerate(batch):
        for prefix, text in [("q", example["question"]), ("d", example["doc_text"][0])]:
            expected = tokenizer(text, max_length=32, truncation=True)["input_ids"]
            actual = out[f"{prefix}_input_ids"][i][out[f"{prefix}_attention_mask"][i].bool()]
            assert actual.tolist() == expected
        for j, text in enumerate(example["doc_text"][1:]):
            expected = tokenizer(text, max_length=32, truncation=True)["input_ids"]
            actual = out["n_input_ids"][i, j][out["n_attention_mask"][i, j].bool()]
            assert actual.tolist() == expected


def _write_teacher_cache(cache_dir: Path, queries: list[str], docs: list[str], dim: int = 3):
    cache_dir.mkdir(parents=True, exist_ok=True)

    query_vectors = np.arange(len(queries) * dim, dtype=np.float16).reshape(len(queries), dim)
    doc_vectors = (100 + np.arange(len(docs) * dim, dtype=np.float16)).reshape(len(docs), dim)

    np.save(cache_dir / "queries.npy", query_vectors)
    np.save(cache_dir / "docs.npy", doc_vectors)
    (cache_dir / "index.json").write_text(
        json.dumps(
            {
                "dim": dim,
                "dtype": "float16",
                "num_queries": len(queries),
                "num_docs": len(docs),
                "query_hashes": {_content_hash(text): i for i, text in enumerate(queries)},
                "doc_hashes": {_content_hash(text): i for i, text in enumerate(docs)},
            }
        ),
        encoding="utf-8",
    )
    (cache_dir / "manifest.json").write_text(json.dumps({"teacher_path": "toy-teacher"}), encoding="utf-8")
    return query_vectors, doc_vectors


@pytest.mark.parametrize("tokenizer_cls", [_TinyTokenizer, _TokenizerWithoutTokenTypeIds])
def test_distill_collator_adds_cached_teacher_embeddings_with_raw_text_lookup(tmp_path: Path, tokenizer_cls):
    queries = ["what is alpha", "what is beta"]
    docs = ["alpha positive", "alpha hard one", "alpha hard two", "beta positive", "beta hard one"]
    query_vectors, doc_vectors = _write_teacher_cache(tmp_path / "cache", queries, docs)

    collator = BiEncoderDistillCollator(
        tokenizer=tokenizer_cls(),
        q_max_len=16,
        p_max_len=16,
        query_prefix="query:",
        passage_prefix="passage:",
        pad_to_multiple_of=4,
        teacher_embeddings_cache=str(tmp_path / "cache"),
    )

    batch = [
        {"question": queries[0], "doc_text": [docs[0], docs[1], docs[2]]},
        {"question": queries[1], "doc_text": [docs[3], docs[4]]},
    ]
    out = collator(batch)

    assert out["q_input_ids"].shape[0] == 2
    assert out["d_input_ids"].shape[0] == 2
    assert out["n_mask"].tolist() == [[1, 1], [1, 0]]

    torch.testing.assert_close(out["t_q_pool"], torch.tensor(query_vectors, dtype=torch.float32))
    torch.testing.assert_close(out["t_d_pool"], torch.tensor([doc_vectors[0], doc_vectors[3]], dtype=torch.float32))
    torch.testing.assert_close(out["t_n_pool"][0, 0], torch.tensor(doc_vectors[1], dtype=torch.float32))
    torch.testing.assert_close(out["t_n_pool"][0, 1], torch.tensor(doc_vectors[2], dtype=torch.float32))
    torch.testing.assert_close(out["t_n_pool"][1, 0], torch.tensor(doc_vectors[4], dtype=torch.float32))
    torch.testing.assert_close(out["t_n_pool"][1, 1], torch.zeros(3))

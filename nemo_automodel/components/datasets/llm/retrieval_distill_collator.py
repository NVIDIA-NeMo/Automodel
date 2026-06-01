# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


def _content_hash(text: str) -> str:
    """SHA-256 digest compatible with ministral3_distillation teacher caches."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CachedTeacherEmbeddings:
    """Lazy memory-mapped teacher embedding cache.

    This reads the cache format produced by
    ``ministral3_distillation/scripts/precompute_teacher_embeddings.py``:
    ``queries.npy``, ``docs.npy``, ``index.json``, and ``manifest.json``.
    Lookups are keyed by raw query/document text content hashes.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        cache_dir = Path(cache_dir).resolve()
        if not cache_dir.is_dir():
            raise FileNotFoundError(f"Teacher embedding cache directory not found: {cache_dir}")

        index_path = cache_dir / "index.json"
        manifest_path = cache_dir / "manifest.json"
        queries_path = cache_dir / "queries.npy"
        docs_path = cache_dir / "docs.npy"
        for required in (index_path, manifest_path, queries_path, docs_path):
            if not required.is_file():
                raise FileNotFoundError(f"Missing teacher cache file: {required}")

        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        with manifest_path.open("r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.cache_dir = cache_dir
        self.dim = int(index["dim"])
        self.dtype = np.dtype(index.get("dtype", "float16"))
        self.query_hashes: dict[str, int] = index["query_hashes"]
        self.doc_hashes: dict[str, int] = index["doc_hashes"]
        self.num_queries = int(index["num_queries"])
        self.num_docs = int(index["num_docs"])

        self._queries = np.load(queries_path, mmap_mode="r")
        self._docs = np.load(docs_path, mmap_mode="r")
        if self._queries.shape != (self.num_queries, self.dim):
            raise RuntimeError(
                f"queries.npy shape {self._queries.shape} does not match "
                f"index ({self.num_queries}, {self.dim})"
            )
        if self._docs.shape != (self.num_docs, self.dim):
            raise RuntimeError(
                f"docs.npy shape {self._docs.shape} does not match "
                f"index ({self.num_docs}, {self.dim})"
            )

    def lookup_queries(self, texts: list[str]) -> np.ndarray:
        rows = np.empty(len(texts), dtype=np.int64)
        for i, text in enumerate(texts):
            h = _content_hash(text)
            try:
                rows[i] = self.query_hashes[h]
            except KeyError as e:
                raise KeyError(
                    f"Query text not found in teacher cache {self.cache_dir} "
                    f"(hash={h[:12]}). Text: {text[:80]!r}..."
                ) from e
        return self._queries[rows]

    def lookup_docs(self, texts: list[str]) -> np.ndarray:
        rows = np.empty(len(texts), dtype=np.int64)
        for i, text in enumerate(texts):
            h = _content_hash(text)
            try:
                rows[i] = self.doc_hashes[h]
            except KeyError as e:
                raise KeyError(
                    f"Document text not found in teacher cache {self.cache_dir} "
                    f"(hash={h[:12]}). Text: {text[:80]!r}..."
                ) from e
        return self._docs[rows]

    def lookup_negatives(self, neg_lists: list[list[str]], k: int) -> np.ndarray:
        out = np.zeros((len(neg_lists), k, self.dim), dtype=self.dtype)
        for i, negs in enumerate(neg_lists):
            for j, text in enumerate(negs[:k]):
                if not text:
                    continue
                h = _content_hash(text)
                try:
                    out[i, j] = self._docs[self.doc_hashes[h]]
                except KeyError as e:
                    raise KeyError(
                        f"Hard-negative text not found in teacher cache {self.cache_dir} "
                        f"(row={i}, negative={j}, hash={h[:12]}). Text: {text[:80]!r}..."
                    ) from e
        return out


class BiEncoderDistillCollator:
    """Collator for bi-encoder distillation with explicit positive/negative splits."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        q_max_len: int = 512,
        p_max_len: int = 512,
        query_prefix: str = "",
        passage_prefix: str = "",
        padding: PaddingStrategy | str | bool = True,
        pad_to_multiple_of: int | None = None,
        use_dataset_instruction: bool = False,
        teacher_embeddings_cache: str | None = None,
        teacher_cache_lookup_with_prefixes: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.q_max_len = q_max_len
        self.p_max_len = p_max_len
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.use_dataset_instruction = use_dataset_instruction
        self.teacher_embeddings_cache = teacher_embeddings_cache
        self.teacher_cache_lookup_with_prefixes = teacher_cache_lookup_with_prefixes
        self._teacher_cache: CachedTeacherEmbeddings | None = None

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # Dataloader workers should open their own read-only memmaps.
        state["_teacher_cache"] = None
        return state

    @property
    def teacher_cache(self) -> CachedTeacherEmbeddings | None:
        if not self.teacher_embeddings_cache:
            return None
        if self._teacher_cache is None:
            self._teacher_cache = CachedTeacherEmbeddings(self.teacher_embeddings_cache)
        return self._teacher_cache

    def _apply_query_prefix(self, text: str, instruction: str | None = None) -> str:
        if self.use_dataset_instruction and instruction:
            return f"{instruction} {text}"
        if self.query_prefix:
            return f"{self.query_prefix} {text}"
        return text

    def _apply_passage_prefix(self, text: str, instruction: str | None = None) -> str:
        if self.use_dataset_instruction and instruction:
            return f"{instruction} {text}"
        if self.passage_prefix:
            return f"{self.passage_prefix} {text}"
        return text

    def _add_cached_teacher_embeddings(
        self,
        out: dict[str, torch.Tensor],
        *,
        raw_queries: list[str],
        raw_pos_docs: list[str],
        raw_neg_lists: list[list[str]],
        query_texts: list[str],
        pos_docs: list[str],
        neg_lists: list[list[str]],
        max_negs: int,
    ) -> None:
        cache = self.teacher_cache
        if cache is None:
            return

        if self.teacher_cache_lookup_with_prefixes:
            cache_queries = query_texts
            cache_pos_docs = pos_docs
            cache_neg_lists = neg_lists
        else:
            # The standalone precompute cache hashes raw dataset text; teacher
            # prompt/prefix handling is baked into the cached embedding itself.
            cache_queries = raw_queries
            cache_pos_docs = raw_pos_docs
            cache_neg_lists = raw_neg_lists

        out["t_q_pool"] = torch.from_numpy(np.asarray(cache.lookup_queries(cache_queries), dtype=np.float32))
        out["t_d_pool"] = torch.from_numpy(np.asarray(cache.lookup_docs(cache_pos_docs), dtype=np.float32))
        out["t_n_pool"] = torch.from_numpy(
            np.asarray(cache.lookup_negatives(cache_neg_lists, max_negs), dtype=np.float32)
        )

    def __call__(self, batch: List[dict[str, Any]]) -> dict[str, torch.Tensor]:
        queries = [x["question"] for x in batch]
        doc_lists = [x["doc_text"] for x in batch]

        if any(len(docs) < 1 for docs in doc_lists):
            raise ValueError("BiEncoderDistillCollator expects at least one document per query")

        if self.use_dataset_instruction:
            query_inst = [x.get("query_instruction", "") for x in batch]
            passage_inst = [x.get("passage_instruction", "") for x in batch]
        else:
            query_inst = [""] * len(batch)
            passage_inst = [""] * len(batch)

        query_texts = [self._apply_query_prefix(q, qi) for q, qi in zip(queries, query_inst)]
        pos_docs = [self._apply_passage_prefix(docs[0], pi) for docs, pi in zip(doc_lists, passage_inst)]
        raw_pos_docs = [docs[0] for docs in doc_lists]

        neg_lists_raw = [docs[1:] for docs in doc_lists]
        neg_lists = [
            [self._apply_passage_prefix(n, pi) for n in negs] for negs, pi in zip(neg_lists_raw, passage_inst)
        ]

        q_enc = self.tokenizer(
            query_texts,
            max_length=self.q_max_len,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=True,
            return_token_type_ids=False,
        )
        d_enc = self.tokenizer(
            pos_docs,
            max_length=self.p_max_len,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=True,
            return_token_type_ids=False,
        )

        q_features = [{k: q_enc[k][i] for k in q_enc.keys()} for i in range(len(query_texts))]
        d_features = [{k: d_enc[k][i] for k in d_enc.keys()} for i in range(len(pos_docs))]

        q_batch = self.tokenizer.pad(
            q_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        d_batch = self.tokenizer.pad(
            d_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        out: dict[str, torch.Tensor] = {
            "q_input_ids": q_batch["input_ids"],
            "q_attention_mask": q_batch["attention_mask"],
            "d_input_ids": d_batch["input_ids"],
            "d_attention_mask": d_batch["attention_mask"],
            "labels": torch.zeros(len(batch), dtype=torch.long),
        }

        max_negs = max((len(n) for n in neg_lists), default=0)
        if max_negs == 0:
            out["n_input_ids"] = torch.zeros((len(batch), 0, 0), dtype=torch.long)
            out["n_attention_mask"] = torch.zeros((len(batch), 0, 0), dtype=torch.long)
            out["n_mask"] = torch.zeros((len(batch), 0), dtype=torch.long)
            self._add_cached_teacher_embeddings(
                out,
                raw_queries=queries,
                raw_pos_docs=raw_pos_docs,
                raw_neg_lists=neg_lists_raw,
                query_texts=query_texts,
                pos_docs=pos_docs,
                neg_lists=neg_lists,
                max_negs=0,
            )
            return out

        neg_mask = torch.zeros((len(batch), max_negs), dtype=torch.long)
        flat_negs: list[str] = []
        for i, negs in enumerate(neg_lists):
            take = min(len(negs), max_negs)
            if take > 0:
                neg_mask[i, :take] = 1
                flat_negs.extend(negs[:take])
            if take < max_negs:
                flat_negs.extend([""] * (max_negs - take))

        if int(neg_mask.sum().item()) == 0:
            out["n_input_ids"] = torch.zeros((len(batch), max_negs, 0), dtype=torch.long)
            out["n_attention_mask"] = torch.zeros((len(batch), max_negs, 0), dtype=torch.long)
            out["n_mask"] = neg_mask
            self._add_cached_teacher_embeddings(
                out,
                raw_queries=queries,
                raw_pos_docs=raw_pos_docs,
                raw_neg_lists=neg_lists_raw,
                query_texts=query_texts,
                pos_docs=pos_docs,
                neg_lists=neg_lists,
                max_negs=max_negs,
            )
            return out

        n_enc = self.tokenizer(
            flat_negs,
            max_length=self.p_max_len,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=True,
            return_token_type_ids=False,
        )
        n_features = [{k: n_enc[k][i] for k in n_enc.keys()} for i in range(len(flat_negs))]
        n_batch = self.tokenizer.pad(
            n_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        seq_len = n_batch["input_ids"].shape[-1]
        out["n_input_ids"] = n_batch["input_ids"].view(len(batch), max_negs, seq_len)
        out["n_attention_mask"] = n_batch["attention_mask"].view(len(batch), max_negs, seq_len)
        out["n_mask"] = neg_mask
        self._add_cached_teacher_embeddings(
            out,
            raw_queries=queries,
            raw_pos_docs=raw_pos_docs,
            raw_neg_lists=neg_lists_raw,
            query_texts=query_texts,
            pos_docs=pos_docs,
            neg_lists=neg_lists,
            max_negs=max_negs,
        )
        return out

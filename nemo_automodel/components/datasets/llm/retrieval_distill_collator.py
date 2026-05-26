# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

from typing import Any, List

import torch
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy


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
    ) -> None:
        self.tokenizer = tokenizer
        self.q_max_len = q_max_len
        self.p_max_len = p_max_len
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.use_dataset_instruction = use_dataset_instruction

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
        return out

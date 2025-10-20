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

from __future__ import annotations

from typing import Optional

from datasets import load_dataset


class _BaseHFSeqClsDataset:
    """
    Base wrapper for HF sequence classification datasets.

    Tokenization and padding are performed via the tokenizer passed into the constructor
    using padding="max_length" with max_length if provided on the tokenizer.
    """

    def __init__(
        self,
        path_or_dataset: str,
        tokenizer,
        *,
        split: str = "train",
        text_field: str = "text",
        label_field: str = "label",
        num_samples_limit: Optional[int] = None,
        trust_remote_code: bool = True,
        max_length: Optional[int] = None,
    ) -> None:
        if isinstance(num_samples_limit, int):
            split = f"{split}[:{num_samples_limit}]"
        self.raw = load_dataset(path_or_dataset, split=split, trust_remote_code=trust_remote_code)

        # Resolve max_length
        if max_length is None:
            max_length = getattr(tokenizer, "model_max_length", None)
            # Avoid super large default
            if isinstance(max_length, int) and max_length > 8192:
                max_length = 1024

        def _tokenize(batch):
            tk_kwargs = dict(padding="max_length", truncation=True)
            if isinstance(max_length, int):
                tk_kwargs["max_length"] = max_length
            out = tokenizer(batch[text_field], **tk_kwargs)
            out["labels"] = batch[label_field]
            return out

        remove_cols = [c for c in self.raw.column_names if c not in (text_field, label_field)]
        tokenized = self.raw.map(_tokenize, batched=True, remove_columns=remove_cols)

        self.dataset = tokenized

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # HF returns lists; map to simple python types to be turned into tensors by collater
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item.get("attention_mask", [1] * len(item["input_ids"])),
            "labels": int(item["labels"]),
        }


class YelpReviewFull(_BaseHFSeqClsDataset):
    """Yelp Review Full dataset for sequence classification (5 classes)."""

    def __init__(
        self,
        tokenizer,
        *,
        split: str = "train",
        num_samples_limit: Optional[int] = None,
        trust_remote_code: bool = True,
        max_length: Optional[int] = 256,
    ) -> None:
        super().__init__(
            "yelp_review_full",
            tokenizer,
            split=split,
            text_field="text",
            label_field="label",
            num_samples_limit=num_samples_limit,
            trust_remote_code=trust_remote_code,
            max_length=max_length,
        )


class IMDB(_BaseHFSeqClsDataset):
    """IMDB binary sentiment dataset for sequence classification (2 classes)."""

    def __init__(
        self,
        tokenizer,
        *,
        split: str = "train",
        num_samples_limit: Optional[int] = None,
        trust_remote_code: bool = True,
        max_length: Optional[int] = 256,
    ) -> None:
        super().__init__(
            "imdb",
            tokenizer,
            split=split,
            text_field="text",
            label_field="label",
            num_samples_limit=num_samples_limit,
            trust_remote_code=trust_remote_code,
            max_length=max_length,
        )



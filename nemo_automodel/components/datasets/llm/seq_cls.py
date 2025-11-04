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


class GLUE_MRPC:
    """GLUE MRPC dataset (sentence pair classification).

    Produces tokenized inputs with both sentence1 and sentence2 using the provided tokenizer.
    """

    def __init__(
        self,
        tokenizer,
        *,
        split: str = "train",
        num_samples_limit: Optional[int] = None,
        trust_remote_code: bool = True,
        max_length: Optional[int] = 256,
    ) -> None:
        if isinstance(num_samples_limit, int):
            split = f"{split}[:{num_samples_limit}]"
        raw = load_dataset("glue", "mrpc", split=split, trust_remote_code=trust_remote_code)

        # Resolve max_length
        if max_length is None:
            max_length = getattr(tokenizer, "model_max_length", None)
            if isinstance(max_length, int) and max_length > 8192:
                max_length = 1024

        def _tokenize(batch):
            tk_kwargs = {
                "truncation": True,
                "max_length": None if max_length is None else max_length,
            }
            out = tokenizer(batch["sentence1"], batch["sentence2"], **tk_kwargs)
            out["labels"] = [[x] for x in batch["label"]]
            return out

        remove_cols = [c for c in raw.column_names if c not in ("sentence1", "sentence2", "label")]
        self.dataset = raw.map(_tokenize, batched=True, remove_columns=remove_cols)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item.get("attention_mask", [1] * len(item["input_ids"])),
            "labels": item["labels"],
            "___PAD_TOKEN_IDS___": {
                "input_ids": self.tokenizer.pad_token_id,
                "labels": -100,
                "attention_mask": 0,
            },
        }

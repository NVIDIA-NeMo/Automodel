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

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from torch.utils.data import Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass
class MockSequenceClassificationDatasetConfig:
    """Construction-time configuration for :class:`MockSequenceClassificationDataset`."""

    accepts_tokenizer: ClassVar[bool] = True

    num_samples: int = 64
    """Number of synthetic samples to generate."""
    num_labels: int = 2
    """Number of classification labels."""
    vocab_size: int = 256
    """Vocabulary size for the random token ids."""
    max_seq_len: int = 32
    """Maximum sequence length (each sample length is sampled in ``[4, max_seq_len]``)."""
    seed: int = 0
    """Seed for the random generator."""

    def build(self, *, tokenizer: "PreTrainedTokenizerBase | None" = None) -> "MockSequenceClassificationDataset":
        """Build a :class:`MockSequenceClassificationDataset` from this :class:`MockSequenceClassificationDatasetConfig`."""
        return MockSequenceClassificationDataset(
            num_samples=self.num_samples,
            num_labels=self.num_labels,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            seed=self.seed,
            tokenizer=tokenizer,
        )


class MockSequenceClassificationDataset(Dataset):
    """Mock dataset for sequence classification functional tests.

    Generates random token sequences with binary labels.
    Does not require a tokenizer or network access.
    """

    def __init__(
        self,
        *,
        num_samples: int = 64,
        num_labels: int = 2,
        vocab_size: int = 256,
        max_seq_len: int = 32,
        seed: int = 0,
        tokenizer=None,
    ):
        random.seed(seed)
        self.samples = []
        for _ in range(num_samples):
            seq_len = random.randint(4, max_seq_len)
            input_ids = [random.randint(2, vocab_size - 1) for _ in range(seq_len)]
            self.samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * seq_len,
                    "labels": [random.randint(0, num_labels - 1)],
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "input_ids": item["input_ids"],
            "attention_mask": item["attention_mask"],
            "labels": item["labels"],
            "___PAD_TOKEN_IDS___": {
                "input_ids": 0,
                "labels": -100,
                "attention_mask": 0,
            },
        }

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

import random

from torch.utils.data import Dataset


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

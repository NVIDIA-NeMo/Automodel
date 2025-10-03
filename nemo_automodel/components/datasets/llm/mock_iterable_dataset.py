# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import torch
from torch.utils.data import IterableDataset


class MockIterableDataset(IterableDataset):
    """Mock dataset that generates synthetic data for benchmarking.

    This dataset generates random tokens similar to the benchmarking script,
    creating input_ids, labels, and position_ids for each sample.
    """

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 1000000, batch_size: int = 1):
        """Initialize the mock dataset.

        Args:
            vocab_size: Size of the vocabulary for generating random tokens
            seq_len: Sequence length for each sample
            num_samples: Total number of samples to generate (default: 1M for infinite-like dataset)
            batch_size: Batch size to yield (default: 1 for unbatched samples)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.batch_size = batch_size

    def __iter__(self):
        """Generate synthetic batches."""
        for _ in range(self.num_samples):
            # Generate random tokens for the batch
            tokens = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))

            # Create labels by shifting tokens and padding last position with -100
            labels = torch.cat([tokens[:, 1:], torch.full((self.batch_size, 1), -100, dtype=tokens.dtype)], dim=1)

            # Create position ids
            position_ids = torch.arange(self.seq_len).unsqueeze(0).expand(self.batch_size, -1)

            yield {
                "input_ids": tokens,
                "labels": labels,
                "position_ids": position_ids,
            }

    def __len__(self):
        """Return the number of samples."""
        return self.num_samples

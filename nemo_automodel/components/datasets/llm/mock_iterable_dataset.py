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

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from copy import copy
from dataclasses import dataclass

import torch
from torch.utils.data import IterableDataset


@dataclass
class MockIterableDatasetConfig:
    """Construction-time configuration for :class:`MockIterableDataset`."""

    vocab_size: int = 1024
    """Size of the vocabulary for generating random tokens."""
    seq_len: int = 1024
    """Sequence length for each sample."""
    num_samples: int = 1000000
    """Total number of samples to generate (1M for an infinite-like dataset)."""
    batch_size: int = 1
    """Batch size to yield (1 for unbatched samples)."""
    seed: int = 0
    """Base seed for deterministic token generation."""

    def build(self) -> "MockIterableDataset":
        """Build a :class:`MockIterableDataset` from this :class:`MockIterableDatasetConfig`."""
        return MockIterableDataset(
            vocab_size=self.vocab_size,
            seq_len=self.seq_len,
            num_samples=self.num_samples,
            batch_size=self.batch_size,
            seed=self.seed,
        )


class MockIterableDataset(IterableDataset):
    """Mock dataset that generates synthetic data for benchmarking.

    This dataset generates random tokens similar to the benchmarking script,
    creating input_ids, labels, and position_ids for each sample.
    """

    def __init__(
        self,
        vocab_size: int = 1024,
        seq_len: int = 1024,
        num_samples: int = 1000000,
        batch_size: int = 1,
        seed: int = 0,
    ) -> None:
        """Initialize the mock dataset.

        Args:
            vocab_size: Size of the vocabulary for generating random tokens.
            seq_len: Sequence length for each sample.
            num_samples: Total number of batches to generate.
            batch_size: Number of sequences in each generated batch.
            seed: Base seed for deterministic token generation.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.seed = seed
        self._shard_index = 0

    def shard(self, num_shards: int, index: int) -> MockIterableDataset:
        """Return a deterministic data-parallel shard without mutating this dataset.

        Args:
            num_shards: Total number of data-parallel shards.
            index: Zero-based index of the requested shard.

        Returns:
            Dataset whose random stream is unique to ``index``.

        Raises:
            ValueError: If ``num_shards`` is not positive or ``index`` is out of range.
        """
        if not isinstance(num_shards, int) or isinstance(num_shards, bool) or num_shards <= 0:
            raise ValueError(f"num_shards must be a positive integer, got {num_shards!r}")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < num_shards:
            raise ValueError(f"index must be an integer in [0, {num_shards}), got {index!r}")

        sharded = copy(self)
        sharded._shard_index = index
        return sharded

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | str]]:
        """Generate deterministic synthetic batches.

        Yields:
            Mapping with ``input_ids``, ``labels``, and ``position_ids`` tensors of shape
            [batch, sequence] on CPU. Labels are shifted input IDs with the last token set to -100.
        """
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self._shard_index)
        for _ in range(self.num_samples):
            # Generate random tokens for the batch
            tokens = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), generator=generator)

            # Create labels by shifting tokens and padding last position with -100
            labels = torch.cat([tokens[:, 1:], torch.full((self.batch_size, 1), -100, dtype=tokens.dtype)], dim=1)

            # Create position ids
            position_ids = torch.arange(self.seq_len).unsqueeze(0).expand(self.batch_size, -1)

            fingerprint = hashlib.sha256()
            fingerprint.update(tokens.numpy().tobytes())
            fingerprint.update(labels.numpy().tobytes())

            yield {
                "input_ids": tokens,
                "labels": labels,
                "position_ids": position_ids,
                "mock_data_fingerprint": fingerprint.hexdigest(),
            }

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.num_samples

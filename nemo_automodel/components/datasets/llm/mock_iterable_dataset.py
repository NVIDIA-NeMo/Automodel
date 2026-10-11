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

from collections.abc import Iterator
from copy import copy
from dataclasses import dataclass

import torch
from torch.utils.data import IterableDataset, get_worker_info


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
    exclude_token_ids: tuple[int, ...] | list[int] | None = None
    """Token ids never emitted in ``input_ids`` (e.g. the model's ``pad_token_id``)."""

    seed: int | None = None
    """Optional deterministic sample seed, independent of model/process RNG and PP/CP rank."""

    def build(self) -> "MockIterableDataset":
        """Build a :class:`MockIterableDataset` from this :class:`MockIterableDatasetConfig`."""
        return MockIterableDataset(
            vocab_size=self.vocab_size,
            seq_len=self.seq_len,
            num_samples=self.num_samples,
            batch_size=self.batch_size,
            exclude_token_ids=self.exclude_token_ids,
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
        exclude_token_ids: tuple[int, ...] | list[int] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the mock dataset.

        Args:
            vocab_size: Size of the vocabulary for generating random tokens (default: 1024)
            seq_len: Sequence length for each sample (default: 1024)
            num_samples: Total number of samples to generate (default: 1M for infinite-like dataset)
            batch_size: Batch size to yield (default: 1 for unbatched samples)
            exclude_token_ids: Token ids that must never appear in ``input_ids``. Use it for the
                model's ``pad_token_id``: models built with ``nn.Embedding(padding_idx=...)`` keep
                that embedding row at zero, so a randomly placed padding token is a zero vector at
                arbitrary positions (including sequence starts) — a shape real data never has.
                Sampling stays uniform over the remaining ids.
            seed: Optional seed in [0, 2**64). When set, each global sample has its own
                deterministic CPU RNG stream and ``num_samples`` counts global samples
                before DP/worker sharding. Model-parallel peers receive the same stream.
                None preserves the historical process-global RNG and unsharded stream.
        """
        super().__init__()
        if seed is not None and not 0 <= seed < 2**64:
            raise ValueError("seed must be in [0, 2**64) or None")
        self.seed = seed
        self._num_shards = 1
        self._shard_index = 0
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.exclude_token_ids = tuple(sorted(set(int(t) for t in exclude_token_ids))) if exclude_token_ids else ()
        for token_id in self.exclude_token_ids:
            if not 0 <= token_id < vocab_size:
                raise ValueError(f"exclude_token_ids entry {token_id} is outside [0, {vocab_size})")
        if len(self.exclude_token_ids) >= vocab_size:
            raise ValueError("exclude_token_ids removes every id in the vocabulary")
        self._allowed_ids: torch.Tensor | None = None
        if self.exclude_token_ids:
            keep = torch.ones(vocab_size, dtype=torch.bool)
            keep[list(self.exclude_token_ids)] = False
            self._allowed_ids = torch.nonzero(keep).flatten()

    def shard(self, num_shards: int, index: int) -> "MockIterableDataset":
        """Return a DP shard of the deterministic sample stream.

        Args:
            num_shards: Number of data-parallel shards, strictly positive.
            index: Data-parallel shard index in [0, num_shards). PP/CP peers
                must pass the same index. Further sharding composes with this shard.

        Returns:
            An independent dataset view. With seed=None the legacy stream is
            unchanged, since it has no rank-independent global sample identity.
        """
        if num_shards < 1 or not 0 <= index < num_shards:
            raise ValueError("shard requires num_shards > 0 and 0 <= index < num_shards")
        dataset = copy(self)
        if self.seed is not None:
            dataset._shard_index += dataset._num_shards * index
            dataset._num_shards *= num_shards
        return dataset

    def _sample_tokens(self, generator: torch.Generator | None = None) -> torch.Tensor:
        """Uniform random ``[batch_size, seq_len]`` ids over the vocabulary minus ``exclude_token_ids``."""
        shape = (self.batch_size, self.seq_len)
        if self._allowed_ids is None:
            return torch.randint(0, self.vocab_size, shape, generator=generator)
        return self._allowed_ids[torch.randint(0, self._allowed_ids.numel(), shape, generator=generator)]

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Generate this shard independently of other model-parallel processes.

        Yields:
            Mapping with CPU int64 input_ids, labels, and position_ids, each
            [batch_size, seq_len]. Labels shift input IDs left and end with -100.
        """
        start, stride = self._shard_index, self._num_shards
        generator = None
        if self.seed is not None:
            worker = get_worker_info()
            if worker is not None:
                start += worker.id * stride
                stride *= worker.num_workers
            generator = torch.Generator(device="cpu")
        for sample_index in range(start, self.num_samples, stride):
            if generator is not None:
                generator.manual_seed((self.seed + sample_index) % 2**64)
            tokens = self._sample_tokens(generator)

            # Create labels by shifting tokens and padding last position with -100
            labels = torch.cat([tokens[:, 1:], torch.full((self.batch_size, 1), -100, dtype=tokens.dtype)], dim=1)

            # Create position ids
            position_ids = torch.arange(self.seq_len).unsqueeze(0).expand(self.batch_size, -1)

            yield {
                "input_ids": tokens,
                "labels": labels,
                "position_ids": position_ids,
            }

    def __len__(self) -> int:
        """Return the number of samples in this DP shard, summed over its workers."""
        return len(range(self._shard_index, self.num_samples, self._num_shards))

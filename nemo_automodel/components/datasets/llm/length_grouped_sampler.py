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

"""Length-grouped sampler for LLM training.

Groups samples by token count so that batches contain similar-length
sequences, minimizing padding waste.  Adapted from the VLM
``LengthGroupedSampler`` but simplified for text-only datasets.

Usage::

    sampler = LengthGroupedSampler(
        dataset=ds,
        batch_size=4,
        seed=42,
        num_replicas=world_size,
        rank=rank,
    )
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=4)
"""

from __future__ import annotations

import logging
import time
from typing import Iterator

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


class LengthGroupedSampler(Sampler[int]):
    """Sampler that groups samples by sequence length for balanced batches.

    Sorts samples by length, chunks into groups of ``batch_size``, then
    shuffles at the chunk level each epoch.  This preserves intra-batch
    length similarity (less padding) while adding per-epoch randomness.

    For distributed training, each rank gets an interleaved shard of the
    sorted indices.  All ranks use the same ``seed + epoch`` so chunk *K*
    on every rank corresponds to similar-length samples, keeping
    cross-rank padding minimal.

    Args:
        dataset: The dataset to sample from.  Samples must have an
            ``input_ids`` key (list or tensor) whose length is used
            for sorting.
        batch_size: Local batch size per rank.
        seed: Base random seed (must be the same on all ranks).
        num_replicas: Number of distributed ranks (default: world size).
        rank: This rank's index (default: current rank).
        drop_last: Drop the tail indices that don't fill a full batch
            across all ranks.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        seed: int = 42,
        num_replicas: int | None = None,
        rank: int | None = None,
        drop_last: bool = True,
    ) -> None:
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0

        self.dataset = dataset
        self.batch_size = max(1, batch_size)
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.epoch = 0

        # Compute lengths
        self.lengths = self._compute_lengths(dataset)

        # Sort by length (descending) and shard across ranks
        sorted_all = sorted(range(len(dataset)), key=lambda i: self.lengths[i], reverse=True)

        # Interleaved sharding: rank 0 gets [0, N, 2N, ...], rank 1 gets [1, N+1, 2N+1, ...]
        # This ensures each rank gets a balanced mix from the sorted order.
        self.sorted_indices = sorted_all[self.rank :: self.num_replicas]

        # Drop tail so all ranks have the same length
        if self.drop_last:
            total = len(self.sorted_indices)
            usable = (total // self.batch_size) * self.batch_size
            if usable < total:
                self.sorted_indices = self.sorted_indices[:usable]

        # Cross-rank alignment
        if dist.is_initialized() and self.num_replicas > 1:
            count = torch.tensor(len(self.sorted_indices), dtype=torch.long).cuda()
            dist.all_reduce(count, op=dist.ReduceOp.MIN)
            min_count = count.item()
            if min_count < len(self.sorted_indices):
                logger.info(
                    "LengthGroupedSampler: truncating from %d to %d samples to align across ranks.",
                    len(self.sorted_indices),
                    min_count,
                )
                self.sorted_indices = self.sorted_indices[:min_count]

        logger.info(
            "LengthGroupedSampler: rank=%d, %d samples, batch_size=%d, %d chunks",
            self.rank,
            len(self.sorted_indices),
            self.batch_size,
            len(self.sorted_indices) // max(self.batch_size, 1),
        )

    @staticmethod
    def _compute_lengths(dataset: Dataset) -> list[int]:
        """Compute token lengths for all samples."""
        # Fast path: access underlying list directly if available
        raw = dataset
        while hasattr(raw, "dataset"):
            raw = raw.dataset
        if not isinstance(raw, list):
            raw = None

        n = len(dataset)
        logger.info("Computing token lengths for %d samples...", n)
        t0 = time.monotonic()
        lengths = [0] * n

        for i in range(n):
            sample = raw[i] if raw is not None else dataset[i]
            ids = sample.get("input_ids")
            if ids is not None:
                lengths[i] = len(ids) if isinstance(ids, list) else ids.numel()
            if (i + 1) % 100_000 == 0 or i == n - 1:
                elapsed = time.monotonic() - t0
                logger.info(
                    "  %d/%d samples (%.1fs, %.0f samples/s)",
                    i + 1,
                    n,
                    elapsed,
                    (i + 1) / max(elapsed, 1e-6),
                )

        logger.info("Length computation done in %.1fs", time.monotonic() - t0)
        return lengths

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic per-epoch shuffling."""
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        # Chunk sorted indices into groups of batch_size, then shuffle chunks
        bs = self.batch_size
        chunks = [self.sorted_indices[i : i + bs] for i in range(0, len(self.sorted_indices), bs)]
        chunk_perm = torch.randperm(len(chunks), generator=g)
        indices = []
        for ci in chunk_perm:
            indices.extend(chunks[ci])

        return iter(indices)

    def __len__(self) -> int:
        return len(self.sorted_indices)

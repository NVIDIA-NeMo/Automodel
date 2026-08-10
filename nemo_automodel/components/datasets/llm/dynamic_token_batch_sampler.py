# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Token-budget batch sampler: variable sample count, near-constant tokens per batch.

A fixed ``batch_size`` sizes every micro-batch for the longest sequence it might
contain, so a batch of short samples wastes both padding and memory headroom. This
sampler instead fills each batch up to a *padded* token budget, so short samples
batch together in large groups and long samples in small ones.

The gradient is unaffected: the training recipe normalizes by the globally
all-reduced label-token count, not by the batch count, so a variable number of
samples per optimizer step leaves the effective loss scale unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from nemo_automodel.components.datasets.llm.sample_lengths import compute_sample_lengths

logger = logging.getLogger(__name__)


def plan_token_budget_batches(
    order: Sequence[int],
    lengths: Sequence[int],
    *,
    max_tokens_per_batch: int,
    max_batch_size: int | None = None,
) -> list[list[int]]:
    """Greedily group ``order`` into batches that fit a padded-token budget.

    The cost model is the *padded* one the collater actually materializes,
    ``len(batch) * max(lengths in batch)``, not the sum of the raw lengths: that is
    what sizes the activation memory and what a fixed ``batch_size`` overpays for.

    A sample longer than the budget on its own becomes a single-sample batch rather
    than being dropped, so no data disappears silently.

    Args:
        order: Dataset indices, in the order they should be considered.
        lengths: Per-dataset-index token counts (indexed by dataset index, not by
            position within ``order``).
        max_tokens_per_batch: Padded-token ceiling for one batch.
        max_batch_size: Optional cap on the sample count of a single batch.

    Returns:
        A list of batches, each a list of dataset indices.
    """
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0

    for index in order:
        length = lengths[index]
        candidate_max = max(current_max, length)
        exceeds_tokens = (len(current) + 1) * candidate_max > max_tokens_per_batch
        exceeds_count = max_batch_size is not None and len(current) + 1 > max_batch_size
        if current and (exceeds_tokens or exceeds_count):
            batches.append(current)
            current = [index]
            current_max = length
        else:
            current.append(index)
            current_max = candidate_max

    if current:
        batches.append(current)
    return batches


class DynamicTokenBatchSampler(Sampler[list[int]]):
    """Batch sampler that fills each batch to a padded-token budget.

    Every rank plans the *same* global batch list from the same seed, lengths, and
    algorithm, then takes a strided slice of it. Two properties follow, and both are
    load-bearing:

    - Each rank ends up with exactly ``len(global_batches) // world_size`` batches, so
      the data-parallel ranks never disagree on the step count. A rank running out of
      batches early would desync the next collective and hang the job, which is the
      usual way a dynamic batch size breaks distributed training.
    - No collective is needed to reach that agreement, so the sampler stays usable
      before the process group exists (and in single-process tests).

    Indices are shuffled per epoch, then sorted by length inside a window of
    ``sort_window`` samples before batching. The window keeps each batch
    length-homogeneous (which is what makes the token budget efficient) while the
    shuffle keeps epoch-to-epoch order random. Consecutive batches hold similar-length
    samples, so the strided rank assignment also balances work across ranks.

    Args:
        dataset: Map-style dataset whose samples carry ``input_ids``.
        max_tokens_per_batch: Padded-token ceiling for one batch.
        max_batch_size: Optional cap on the sample count of a single batch.
        sort_window: Number of shuffled samples sorted together before batching.
            ``1`` leaves the shuffled order untouched.
        seed: Base seed; must match across ranks.
        num_replicas: Data-parallel world size.
        rank: This rank's data-parallel index.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        max_tokens_per_batch: int,
        max_batch_size: int | None = None,
        sort_window: int = 2048,
        seed: int = 42,
        num_replicas: int = 1,
        rank: int = 0,
    ) -> None:
        if max_tokens_per_batch <= 0:
            raise ValueError(f"max_tokens_per_batch must be positive, got {max_tokens_per_batch}")
        if max_batch_size is not None and max_batch_size <= 0:
            raise ValueError(f"max_batch_size must be positive when set, got {max_batch_size}")
        if sort_window < 1:
            raise ValueError(f"sort_window must be >= 1, got {sort_window}")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be positive, got {num_replicas}")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"rank {rank} is out of range for num_replicas={num_replicas}")

        self.max_tokens_per_batch = int(max_tokens_per_batch)
        self.max_batch_size = max_batch_size
        self.sort_window = int(sort_window)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0
        self.batches_yielded = 0
        self._next_batches_yielded: int | None = None

        self.lengths = compute_sample_lengths(dataset)
        oversized = sum(1 for length in self.lengths if length > self.max_tokens_per_batch)
        if oversized:
            logger.warning(
                "DynamicTokenBatchSampler: %d/%d samples exceed max_tokens_per_batch=%d and will each "
                "form a single-sample batch that overruns the budget. Raise max_tokens_per_batch or "
                "truncate the dataset if that batch does not fit.",
                oversized,
                len(self.lengths),
                self.max_tokens_per_batch,
            )

        self._planned_epoch: int | None = None
        self._local_batches: list[list[int]] = []

    def _plan(self, epoch: int) -> None:
        """Compute this rank's batch list for ``epoch`` (identical inputs on every rank)."""
        if self._planned_epoch == epoch:
            return

        generator = torch.Generator()
        generator.manual_seed(self.seed + epoch)
        order = torch.randperm(len(self.lengths), generator=generator).tolist()

        if self.sort_window > 1:
            length_of = self.lengths.__getitem__
            ordered: list[int] = []
            for start in range(0, len(order), self.sort_window):
                window = order[start : start + self.sort_window]
                window.sort(key=length_of, reverse=True)
                ordered.extend(window)
            order = ordered

        global_batches = plan_token_budget_batches(
            order,
            self.lengths,
            max_tokens_per_batch=self.max_tokens_per_batch,
            max_batch_size=self.max_batch_size,
        )
        # Truncate to a whole number of rounds so every rank yields the same count.
        per_rank = len(global_batches) // self.num_replicas
        self._local_batches = global_batches[self.rank : per_rank * self.num_replicas : self.num_replicas]
        self._planned_epoch = epoch

        if per_rank == 0 and global_batches:
            logger.warning(
                "DynamicTokenBatchSampler: %d batches over %d ranks leaves every rank empty; "
                "lower max_tokens_per_batch or use more data.",
                len(global_batches),
                self.num_replicas,
            )
        logger.info(
            "DynamicTokenBatchSampler: epoch=%d rank=%d %d batches (%d global, %d dropped to align ranks)",
            epoch,
            self.rank,
            len(self._local_batches),
            len(global_batches),
            len(global_batches) - per_rank * self.num_replicas,
        )

    def set_epoch(self, epoch: int) -> None:
        """Re-plan the batches for ``epoch``; all ranks must call this with the same value."""
        self.epoch = int(epoch)
        self._plan(self.epoch)

    def state_dict(self) -> dict[str, Any]:
        """Return the resume state (epoch plus batches already consumed this epoch)."""
        return {"epoch": self.epoch, "batches_yielded": self.batches_yielded}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore ``state_dict``; the next iteration resumes mid-epoch."""
        self.epoch = int(state_dict["epoch"])
        self._plan(self.epoch)
        self._next_batches_yielded = int(state_dict["batches_yielded"])

    def __iter__(self) -> Iterator[list[int]]:
        """Yield this rank's batches, resuming from a loaded state when present."""
        self._plan(self.epoch)
        start = 0
        if self._next_batches_yielded is not None:
            start = self._next_batches_yielded
            self._next_batches_yielded = None
        self.batches_yielded = start
        for batch in self._local_batches[start:]:
            self.batches_yielded += 1
            yield list(batch)

    def __len__(self) -> int:
        """Number of batches this rank yields in the current epoch."""
        self._plan(self.epoch)
        return len(self._local_batches)


@dataclass
class DynamicTokenBatchSamplerConfig:
    """Declarative config for :class:`DynamicTokenBatchSampler`.

    Satisfies the ``BatchSamplerConfig`` protocol consumed by ``DataloaderConfig``.
    """

    max_tokens_per_batch: int
    max_batch_size: int | None = None
    sort_window: int = 2048
    seed: int = 42

    def build(self, *, dataset: Dataset, rank: int, world_size: int) -> DynamicTokenBatchSampler:
        """Build the sampler for one data-parallel rank.

        Args:
            dataset: The materialized dataset, read for per-sample token counts.
            rank: Rank within the data-parallel group.
            world_size: Size of the data-parallel group.

        Returns:
            A sampler yielding one local batch of dataset indices at a time.
        """
        return DynamicTokenBatchSampler(
            dataset,
            max_tokens_per_batch=self.max_tokens_per_batch,
            max_batch_size=self.max_batch_size,
            sort_window=self.sort_window,
            seed=self.seed,
            num_replicas=world_size,
            rank=rank,
        )


__all__ = [
    "DynamicTokenBatchSampler",
    "DynamicTokenBatchSamplerConfig",
    "plan_token_budget_batches",
]

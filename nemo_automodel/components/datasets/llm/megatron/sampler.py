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

import abc
import logging
from itertools import chain
from typing import Literal, Optional

import torch


class BaseMegatronSampler:
    """Base class for Megatron batch samplers.

    Provides common validation and shared behavior for Megatron samplers.
    Implementations must yield lists of dataset indices that correspond to
    one micro-batch for a single data-parallel rank.

    Args:
        total_samples: Total available samples in the dataset.
        micro_batch_size: Number of samples per micro-batch on each data-parallel
            rank.
        data_parallel_rank: Rank id in the data-parallel group that this sampler
            will serve.
        data_parallel_size: World size of the data-parallel group.
        drop_last: If True, drop incomplete batches. If False, implementations
            may yield a final partial micro-batch (subject to their constraints).
        global_batch_size: Effective global batch size across all data-parallel
            ranks; when provided, length is computed in global-batch units and
            converted to micro-batches.
        pad_samples_to_global_batch_size: If True and supported by the sampler,
            the last incomplete global batch will be padded to `global_batch_size`
            when `drop_last` is False.
    """

    def __init__(
        self,
        total_samples: int,
        micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        drop_last: bool = True,
        global_batch_size: Optional[int] = None,
        pad_samples_to_global_batch_size: Optional[bool] = False,
    ) -> None:
        # Sanity checks.
        if total_samples <= 0:
            raise RuntimeError(f"no sample to consume: {total_samples}")
        if micro_batch_size <= 0:
            raise RuntimeError(f"micro_batch_size size must be greater than 0, but {micro_batch_size}")
        if data_parallel_size <= 0:
            raise RuntimeError(f"data parallel size must be greater than 0, but {data_parallel_size}")
        if data_parallel_rank >= data_parallel_size:
            raise RuntimeError(
                f"data_parallel_rank should be smaller than data size, but {data_parallel_rank} >= {data_parallel_size}"
            )
        if global_batch_size is not None:
            if global_batch_size % (micro_batch_size * data_parallel_size) != 0:
                raise RuntimeError(
                    f"`global_batch_size` ({global_batch_size}) is not divisible by "
                    f"`micro_batch_size ({micro_batch_size}) x data_parallel_size "
                    f"({data_parallel_size})`"
                )
        if pad_samples_to_global_batch_size and global_batch_size is None:
            raise RuntimeError(
                "`pad_samples_to_global_batch_size` can be `True` only when "
                "`global_batch_size` is set to an integer value"
            )

        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last
        self.global_batch_size = global_batch_size
        self.pad_samples_to_global_batch_size = pad_samples_to_global_batch_size

        logging.info(f"Instantiating MegatronPretrainingSampler with total_samples: {total_samples}")

    def __len__(self):
        """Return the number of micro-batches this sampler will yield.

        If `global_batch_size` is provided, the length is computed in terms of
        global batches and converted to micro-batches to align with training
        loops that iterate by micro-batch.
        """
        if self.global_batch_size is not None:
            if self.drop_last:
                num_global_batches = self.total_samples // self.global_batch_size
            else:
                num_global_batches = (self.total_samples + self.global_batch_size - 1) // self.global_batch_size
            # return len of dataloader in terms of micro batches to avoid discrepancy between len of dataloader and
            # num of batches fetched (as training step fetches in terms of micro batches)
            return num_global_batches * (self.global_batch_size // self.micro_batch_times_data_parallel_size)
        else:
            return (self.total_samples - 1) // self.micro_batch_times_data_parallel_size + 1

    @abc.abstractmethod
    def __iter__(self): ...


class MegatronPretrainingSampler(BaseMegatronSampler):
    """Deterministic sequential sampler with per-rank slicing.

    Iterates deterministically over sample indices, splits each global batch
    across data-parallel ranks, and yields per-rank micro-batches. When
    `drop_last` is False and `pad_samples_to_global_batch_size` is True, the
    final global batch is padded to a full size so that all ranks emit complete
    micro-batches.

    Raises:
        RuntimeError: If there are no samples left to consume.
    """

    def __init__(
        self,
        total_samples: int,
        micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        drop_last: bool = True,
        global_batch_size: Optional[int] = None,
        pad_samples_to_global_batch_size: Optional[bool] = False,
    ):
        super().__init__(
            total_samples=total_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            drop_last=drop_last,
            global_batch_size=global_batch_size,
            pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
        )

    def get_start_end_idx(self):
        """Return slice boundaries for this rank within a global batch.

        Returns:
            Tuple of `(start_idx, end_idx)` used to extract this rank's
            micro-batch from a concatenated global batch buffer.
        """
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        """Yield lists of indices forming per-rank micro-batches.

        Iterates up to `total_samples`. Optionally pads
        the last global batch when `drop_last` is False and
        `pad_samples_to_global_batch_size` is True.
        """
        batch = []
        # Last batch will be dropped if drop_last is not set False
        indices = range(0, self.total_samples)
        if (not self.drop_last) and self.pad_samples_to_global_batch_size:
            pad_samples_num = -len(indices) % self.global_batch_size
            pad_indices = range(-1, -pad_samples_num - 1, -1)
            indices = chain(indices, pad_indices)

        for idx in indices:
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx, end_idx = self.get_start_end_idx()
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            assert not self.pad_samples_to_global_batch_size, (
                "with pad_samples_to_global_batch_size all batches should be complete"
            )
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class MegatronPretrainingRandomSampler(BaseMegatronSampler):
    """Randomized sampler with per-epoch shuffling and per-rank slicing.

    Uses a deterministic seed schedule `seed + epoch` to randomize indices
    within each data-parallel shard (bucket). Notably, this sampler:

    - Does not support padding the last global batch.
    - Requires `drop_last=True` when the product `micro_batch_size *
      data_parallel_size > 1`.
    """

    def __init__(
        self,
        total_samples: int,
        micro_batch_size: int,
        data_parallel_rank: int,
        data_parallel_size: int,
        drop_last: bool = True,
        global_batch_size: Optional[int] = None,
        pad_samples_to_global_batch_size: Optional[bool] = False,
        seed: int = 0,
    ) -> None:
        super().__init__(
            total_samples=total_samples,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=data_parallel_rank,
            data_parallel_size=data_parallel_size,
            drop_last=drop_last,
            global_batch_size=global_batch_size,
            pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
        )
        assert not pad_samples_to_global_batch_size, (
            "`MegatronPretrainingRandomSampler` does not support sample padding"
        )
        if (not drop_last) and self.micro_batch_times_data_parallel_size > 1:
            raise RuntimeError(
                "`MegatronPretrainingRandomSampler` does not support drop_last=False when micro_batch_size * data_parallel_size > 1. \
                  please reduce your MBS and data parallelism to 1 if you want to use drop_last=False, or switch to drop_last=True to avoid this error"
            )
        self.last_batch_size = self.total_samples % self.micro_batch_times_data_parallel_size
        self.seed = seed
        self.consumed_samples = 0

    def __len__(self):
        """Return the number of micro-batches that will be produced.

        Accounts for `drop_last` by excluding a trailing incomplete global batch.
        When `global_batch_size` is provided, converts global batches to
        micro-batches.
        """
        active_total_samples = self.total_samples - (self.last_batch_size if self.drop_last else 0)
        num_available_samples = active_total_samples - self.consumed_samples % active_total_samples
        if self.global_batch_size is not None:
            if self.drop_last:
                num_global_batches = num_available_samples // self.global_batch_size
            else:
                num_global_batches = (num_available_samples + self.global_batch_size - 1) // self.global_batch_size
            # return len of dataloader in terms of micro batches to avoid discrepancy between len of dataloader and
            # num of batches fetched (as training step fetches in terms of micro batches)
            return num_global_batches * (self.global_batch_size // self.micro_batch_times_data_parallel_size)
        else:
            if self.drop_last:
                return num_available_samples // self.micro_batch_times_data_parallel_size
            else:
                return (num_available_samples - 1) // self.micro_batch_times_data_parallel_size

    def __iter__(self):
        """Yield randomized micro-batches for this rank.

        Each epoch shuffles indices within the per-rank bucket using
        `torch.randperm` seeded by `seed + epoch`. The sampler then emits
        contiguous micro-batches of size `micro_batch_size` for this rank.
        """
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        # data sharding and random sampling
        data_parallel_size = self.micro_batch_times_data_parallel_size // self.micro_batch_size
        bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) * self.micro_batch_size
        bucket_offset = current_epoch_samples // data_parallel_size
        start_idx = self.data_parallel_rank * bucket_size

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        random_idx = torch.randperm(bucket_size, generator=g).tolist()
        idx_range = [start_idx + x for x in random_idx[bucket_offset:]]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            yield batch


def create_megatron_sampler(
    dataset_len: int,
    micro_batch_size: int,
    global_batch_size: int,
    dataloader_type: Literal["single", "cyclic"] = "single",
    drop_last: bool = True,
    pad_samples_to_global_batch_size: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> BaseMegatronSampler:
    """Factory for Megatron samplers.

    Constructs and returns a Megatron-compatible sampler for a dataset of a
    given length and batch configuration. The returned sampler yields lists of
    indices per micro-batch for a single data-parallel rank.

    Args:
        dataset_len: Number of samples in the underlying dataset.
        micro_batch_size: Number of samples per micro-batch on each
            data-parallel rank.
        global_batch_size: Effective global batch size across all
            data-parallel ranks (`micro_batch_size * world_size * grad_accum`).
        dataloader_type: Sampler type to construct. Supported values:
            - "single": Deterministic sequential sampling
              (`MegatronPretrainingSampler`).
            - "cyclic": Randomized per-epoch sampling
              (`MegatronPretrainingRandomSampler`).
            The value "batch" is not supported in this implementation.
        drop_last: When True, drop a trailing incomplete batch.
        pad_samples_to_global_batch_size: When True and supported by the sampler,
            pad the final global batch to `global_batch_size` if `drop_last` is
            False.
        rank: Data-parallel rank id for this process.
        world_size: Number of data-parallel ranks.

    Returns:
        BaseMegatronSampler: Configured sampler instance for the requested type.

    Raises:
        Exception: If an unsupported `dataloader_type` is provided.
    """
    if dataloader_type == "single":
        batch_sampler = MegatronPretrainingSampler(
            total_samples=dataset_len,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            data_parallel_rank=rank,
            data_parallel_size=world_size,
            drop_last=drop_last,
            pad_samples_to_global_batch_size=pad_samples_to_global_batch_size,
        )
    elif dataloader_type == "cyclic":
        batch_sampler = MegatronPretrainingRandomSampler(
            total_samples=dataset_len,
            micro_batch_size=micro_batch_size,
            data_parallel_rank=rank,
            data_parallel_size=world_size,
            drop_last=drop_last,
        )
    else:
        raise Exception(f"{dataloader_type} dataloader type is not supported.")
    return batch_sampler

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

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
import torch.distributed as dist

BatchT = TypeVar("BatchT")


@dataclass(frozen=True)
class ValidationBatch(Generic[BatchT]):
    """A forward input and whether its loss belongs to this rank's validation shard.

    Tensor fields in ``batch`` retain their loader-defined shapes and dtypes.
    Placeholder batches participate in model collectives but never contribute metrics.
    """

    batch: BatchT
    contributes: bool


def synchronized_validation_batches(loader: Iterable[BatchT]) -> Iterator[ValidationBatch[BatchT]]:
    """Keep forward collectives aligned when validation shards have unequal lengths.

    All ranks in the default process group must call this iterator. Exhausted ranks
    forward a live rank's CPU batch, without counting it in validation metrics.
    No batch is transferred on steps where every rank has data. The loader's batch
    layout is preserved; callers must not mutate the yielded input in place.
    Loader exceptions are propagated collectively before any forward starts.
    """
    distributed = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if distributed else 1
    rank = dist.get_rank() if distributed else 0
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if distributed and dist.get_backend() == "nccl"
        else torch.device("cpu")
    )
    iterator = _validation_source(loader)
    while True:
        batch, error = _read_validation_batch(iterator)
        # MIN obtains an error flag, the first live source rank, and whether every rank has data.
        status = torch.tensor(
            [int(error is None), rank if batch is not None else world_size, int(batch is not None)], device=device
        )
        if distributed:
            dist.all_reduce(status, op=dist.ReduceOp.MIN)
        ok, source, all_present = status.tolist()
        if not ok:
            raise RuntimeError("Reading the validation loader failed on at least one rank") from error
        if source == world_size:
            return
        contributes = batch is not None
        if not all_present:
            batch = _complete_validation_batch(batch, source=source)
        if batch is None:
            raise RuntimeError("The live validation rank did not provide a batch")
        yield ValidationBatch(batch, contributes)


def _complete_validation_batch(batch: BatchT | None, *, source: int) -> BatchT | None:
    """Broadcast one live input, retaining local inputs on ranks that still have data."""
    payload = [batch if dist.get_rank() == source else None]
    dist.broadcast_object_list(payload, src=source)
    return payload[0] if batch is None else batch


def _validation_source(loader: Iterable[BatchT]) -> Iterator[BatchT]:
    """Defer loader construction so iterator failures reach the collective error check."""
    yield from loader


def _read_validation_batch(iterator: Iterator[BatchT]) -> tuple[BatchT | None, Exception | None]:
    """Read locally without allowing an exception to bypass the next collective."""
    try:
        return next(iterator, None), None
    except Exception as error:
        return None, error

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

"""Two-rank CPU regression tests for the EAGLE-3 TTT shift under context parallelism.

The TTT recurrence left-shifts the supervision by one position per step. Sharded
across cp that shift crosses rank boundaries, and a wrong boundary token is nearly
invisible in the loss (it perturbs a handful of positions out of ``S``, and the loss
is a mean over all of them) -- only the gradients move. These tests therefore compare
the sharded shift against the full-sequence shift *elementwise*, which is what
actually pins the boundary behaviour.

The shift is pure tensor ops plus a P2P exchange, so gloo on CPU exercises the real
cross-rank logic without a GPU.
"""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_automodel.components.speculative.eagle.core import (
    _cp_shift_left,
    _cp_shift_left_zigzag,
    _shift_left_with_zero,
)

# Run only on the GPU job. Each test mp.spawns gloo worker processes that re-import
# the full package; context parallelism is a multi-GPU feature, so skip on CPU.
pytestmark = pytest.mark.run_only_on("GPU")

B, S, STEPS = 2, 8, 3


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _init_gloo(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _zigzag_idx(rank: int, world_size: int, seq: int) -> list[int]:
    """Global indices rank ``r`` owns under zig-zag: chunk ``r`` plus chunk ``2W-1-r``."""
    c = seq // (2 * world_size)
    return list(range(rank * c, (rank + 1) * c)) + list(
        range((2 * world_size - 1 - rank) * c, (2 * world_size - rank) * c)
    )


def _full_sequence() -> torch.Tensor:
    # Nonzero values so the zero-fill at the global tail is observable.
    return torch.arange(1, B * S + 1, dtype=torch.float32).view(B, S)


def _contiguous_worker(rank: int, world_size: int, port: int) -> None:
    try:
        _init_gloo(rank, world_size, port)
        full = _full_sequence()
        lo, hi = rank * S // world_size, (rank + 1) * S // world_size
        ref, local = full, full[:, lo:hi].contiguous()
        for _ in range(STEPS):
            ref = _shift_left_with_zero(ref)
            local = _cp_shift_left(local, dist.group.WORLD)
            torch.testing.assert_close(local, ref[:, lo:hi])
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _zigzag_worker(rank: int, world_size: int, port: int) -> None:
    try:
        _init_gloo(rank, world_size, port)
        full = _full_sequence()
        idx = _zigzag_idx(rank, world_size, S)
        local = full[:, idx].contiguous()

        # Teeth: the contiguous shift pulls the wrong boundary tokens under a zig-zag
        # shard, so it must disagree -- otherwise these tests could not catch a
        # regression that swapped one shift for the other.
        wrong = _cp_shift_left(local.clone(), dist.group.WORLD)
        right = _cp_shift_left_zigzag(local.clone(), dist.group.WORLD)
        assert not torch.equal(wrong, right)

        ref = full
        for _ in range(STEPS):
            ref = _shift_left_with_zero(ref)
            local = _cp_shift_left_zigzag(local, dist.group.WORLD)
            torch.testing.assert_close(local, ref[:, idx])
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _zigzag_extra_dim_worker(rank: int, world_size: int, port: int) -> None:
    """``position_mask`` is ``[B, S, 1]``; the shift must handle a trailing dim."""
    try:
        _init_gloo(rank, world_size, port)
        full = _full_sequence().unsqueeze(-1)
        idx = _zigzag_idx(rank, world_size, S)
        ref, local = full, full[:, idx].contiguous()
        for _ in range(STEPS):
            ref = _shift_left_with_zero(ref)
            local = _cp_shift_left_zigzag(local, dist.group.WORLD)
            torch.testing.assert_close(local, ref[:, idx])
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _zigzag_single_rank_worker(rank: int, world_size: int, port: int) -> None:
    """At cp_size=1 rank 0 owns both chunks in order, so the shift is the local one."""
    try:
        _init_gloo(rank, world_size, port)
        full = _full_sequence()
        shifted = _cp_shift_left_zigzag(full.clone(), dist.group.WORLD)
        torch.testing.assert_close(shifted, _shift_left_with_zero(full))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_cp_shift_left_matches_full_shift_on_contiguous_shards():
    mp.spawn(_contiguous_worker, args=(2, _free_port()), nprocs=2, join=True)


def test_cp_shift_left_zigzag_matches_full_shift_on_zigzag_shards():
    mp.spawn(_zigzag_worker, args=(2, _free_port()), nprocs=2, join=True)


def test_cp_shift_left_zigzag_handles_trailing_dim():
    mp.spawn(_zigzag_extra_dim_worker, args=(2, _free_port()), nprocs=2, join=True)


def test_cp_shift_left_zigzag_single_rank_matches_local_shift():
    mp.spawn(_zigzag_single_rank_worker, args=(1, _free_port()), nprocs=1, join=True)

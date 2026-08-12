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

"""Two-rank CPU regression tests for Kimi Linear context parallelism.

The rest of ``test_cp.py`` drives a fake mesh with no process group, so the one
collective Kimi Linear owns end to end -- the ``_AllGatherSequence`` autograd
Function that MLA uses to gather the compressed KV latent -- is never actually
executed. These tests run it on two gloo ranks, where a wrong backward (missing
all-reduce, or narrowing to the wrong shard) shows up immediately.
"""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_automodel._transformers.models.kimi_linear.cp import all_gather_sequence, shard_batch_for_kimi_cp

# Run only on the GPU job. Each test mp.spawns two gloo worker processes that
# re-import the full package, and these cover a multi-GPU feature, so they are
# skipped on the CPU unit-test job.
pytestmark = pytest.mark.run_only_on("GPU")

WORLD_SIZE = 2
LOCAL_LEN = 3
HIDDEN = 4


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


def _global_sequence() -> torch.Tensor:
    """A deterministic [1, world * local, hidden] tensor every rank agrees on."""
    return torch.arange(WORLD_SIZE * LOCAL_LEN * HIDDEN, dtype=torch.float32).reshape(1, -1, HIDDEN)


def _upstream_for(rank: int) -> torch.Tensor:
    """Per-rank upstream gradient, different on each rank so the all-reduce matters."""
    return (rank + 1) * torch.arange(1, WORLD_SIZE * LOCAL_LEN * HIDDEN + 1, dtype=torch.float32).reshape(1, -1, HIDDEN)


def _all_gather_worker(rank: int, world_size: int, port: int) -> None:
    try:
        _init_gloo(rank, world_size, port)

        full = _global_sequence()
        start = rank * LOCAL_LEN
        local = full[:, start : start + LOCAL_LEN].clone().requires_grad_(True)

        gathered = all_gather_sequence(local, dist.group.WORLD, dim=1)

        # Forward: shards land in rank order, rebuilding the global sequence.
        torch.testing.assert_close(gathered, full)

        (gathered * _upstream_for(rank)).sum().backward()

        # Backward: every rank's contribution is summed, then narrowed back to the
        # shard this rank owns.
        expected = sum(_upstream_for(other) for other in range(world_size))[:, start : start + LOCAL_LEN]
        torch.testing.assert_close(local.grad, expected)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _sharder_worker(rank: int, world_size: int, port: int) -> None:
    try:
        _init_gloo(rank, world_size, port)
        from torch.distributed.device_mesh import init_device_mesh

        cp_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("cp",))["cp"]
        seq_len = WORLD_SIZE * LOCAL_LEN
        input_ids = torch.arange(seq_len, dtype=torch.long).reshape(1, seq_len)
        attention_mask = torch.ones(1, seq_len, dtype=torch.int32)
        attention_mask[:, -1] = 0

        _, sharded, layout = shard_batch_for_kimi_cp(
            cp_mesh,
            None,
            {"input_ids": input_ids.clone(), "labels": input_ids.clone(), "attention_mask": attention_mask.clone()},
        )

        # The real mesh reports the same contiguous slice the fake mesh does, and
        # gathering the shards back through the CP collective restores the batch.
        start = rank * LOCAL_LEN
        torch.testing.assert_close(sharded["input_ids"], input_ids[:, start : start + LOCAL_LEN])
        assert sharded["kimi_packed_context"].seq_start == start
        assert layout.original_seq_len == seq_len

        gathered = all_gather_sequence(sharded["input_ids"].float(), dist.group.WORLD, dim=1)
        torch.testing.assert_close(gathered, input_ids.float())
        # Every rank keeps the whole document map so MLA can mask the gathered keys.
        torch.testing.assert_close(
            sharded["kimi_packed_context"].doc_ids,
            attention_mask,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_all_gather_sequence_round_trips_and_reduces_gradients():
    mp.spawn(_all_gather_worker, args=(WORLD_SIZE, _free_port()), nprocs=WORLD_SIZE, join=True)


def test_shard_batch_matches_the_collective_layout_on_a_real_mesh():
    mp.spawn(_sharder_worker, args=(WORLD_SIZE, _free_port()), nprocs=WORLD_SIZE, join=True)

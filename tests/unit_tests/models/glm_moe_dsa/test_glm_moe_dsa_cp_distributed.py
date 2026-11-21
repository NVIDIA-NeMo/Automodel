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

"""Two-rank NCCL regressions for GLM DSA's differentiable CP gather (rank order and head/tail global order)."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_automodel.components.models.glm_moe_dsa.cp import gather_order, glm_dsa_cp_all_gather, layout_indices_for_rank
from nemo_automodel.shared.recompute_replay import RecomputeReplay, RecomputeReplayRecorder

# Over the default 5s budget on purpose: this module spawns worker processes; every child re-imports torch from scratch.
# Shrink the work or the process count before raising this further.
pytestmark = [
    pytest.mark.timeout(70),
    pytest.mark.run_only_on("GPU"),
]

_WORLD_SIZE = 2


def _free_port() -> int:
    """Return an unused localhost port for the process-group rendezvous."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _init(rank: int, port: int) -> torch.device:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(_WORLD_SIZE)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=_WORLD_SIZE)
    return torch.device("cuda", rank)


def _gather_worker(rank: int, port: int) -> None:
    try:
        device = _init(rank, port)
        local_k = (torch.arange(4, device=device, dtype=torch.float32).reshape(2, 2) + 10 * rank).requires_grad_()
        local_v = (torch.arange(2, device=device, dtype=torch.float32).reshape(2, 1) + 100 * rank).requires_grad_()

        gathered_k = glm_dsa_cp_all_gather(local_k, dim=0, cp_group=dist.group.WORLD)
        gathered_v = glm_dsa_cp_all_gather(local_v, dim=0, cp_group=dist.group.WORLD)

        expected_k = torch.cat(
            [torch.arange(4, device=device, dtype=torch.float32).reshape(2, 2) + 10 * source for source in range(2)]
        )
        expected_v = torch.cat(
            [torch.arange(2, device=device, dtype=torch.float32).reshape(2, 1) + 100 * source for source in range(2)]
        )
        torch.testing.assert_close(gathered_k, expected_k)
        torch.testing.assert_close(gathered_v, expected_v)

        k_weight = rank + 1
        v_weight = 2 * rank + 1
        (k_weight * gathered_k.sum() + v_weight * gathered_v.sum()).backward()

        # Differentiable all-gather sums every consumer's contribution before
        # returning the gradient shard owned by this rank.
        torch.testing.assert_close(local_k.grad, torch.full_like(local_k, 3.0))
        torch.testing.assert_close(local_v.grad, torch.full_like(local_v, 4.0))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _head_tail_worker(rank: int, port: int) -> None:
    try:
        device = _init(rank, port)
        total, width = 16, 3
        # Row i of the global tensor identifies global token i in every column.
        global_rows = torch.arange(total, device=device, dtype=torch.float32).view(total, 1).repeat(1, width)
        global_rows = global_rows + torch.arange(width, device=device, dtype=torch.float32).view(1, width) / 10
        mine = layout_indices_for_rank("head_tail", rank, _WORLD_SIZE, total, device)
        order = gather_order("head_tail", _WORLD_SIZE, total, device)
        assert order is not None
        from_global, _ = order
        # Every consumer weights global token i by (i + 1) on rank 0 and by 2 (i + 1) on rank 1, so the
        # gradient a token's owner receives is 3 (i + 1): the sum of both ranks' consumption.
        weights = (rank + 1) * torch.arange(1.0, total + 1, device=device).view(total, 1).expand(total, width)
        expected_grad = 3.0 * torch.arange(1.0, total + 1, device=device).view(total, 1).expand(total, width)

        local = global_rows.index_select(0, mine).clone().requires_grad_()
        gathered = glm_dsa_cp_all_gather(local, dim=0, cp_group=dist.group.WORLD, order=order)
        torch.testing.assert_close(gathered, global_rows)
        (gathered * weights).sum().backward()
        torch.testing.assert_close(local.grad, expected_grad.index_select(0, mine))

        # The plain gather returns rank order: rank 0's head and tail, then rank 1's chunks.
        plain = glm_dsa_cp_all_gather(local.detach(), dim=0, cp_group=dist.group.WORLD)
        torch.testing.assert_close(plain, global_rows.index_select(0, from_global))
        assert not torch.equal(plain, global_rows)

        # Record in the checkpoint forward, replay in the recompute: same tensor, same gradient, no collective.
        replay = RecomputeReplay("test head/tail gather")
        recorder = RecomputeReplayRecorder()
        recorded_local = global_rows.index_select(0, mine).clone().requires_grad_()
        with replay.scope(recorder, "record"):
            recorded = glm_dsa_cp_all_gather(
                recorded_local, dim=0, cp_group=dist.group.WORLD, replay=replay, order=order
            )
        torch.testing.assert_close(recorded, global_rows)
        recorder.rewind()
        replayed_local = global_rows.index_select(0, mine).clone().requires_grad_()
        with replay.scope(recorder, "replay"):
            replayed = glm_dsa_cp_all_gather(
                replayed_local, dim=0, cp_group=dist.group.WORLD, replay=replay, order=order
            )
        torch.testing.assert_close(replayed, global_rows)
        (replayed * weights).sum().backward()
        torch.testing.assert_close(replayed_local.grad, expected_grad.index_select(0, mine))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_nccl_available() or torch.cuda.device_count() < _WORLD_SIZE,
    reason="requires NCCL and at least two CUDA devices",
)
def test_glm_dsa_cp_all_gather_preserves_rank_order_and_sums_kv_gradients() -> None:
    mp.spawn(_gather_worker, args=(_free_port(),), nprocs=_WORLD_SIZE, join=True)


@pytest.mark.skipif(
    not dist.is_nccl_available() or torch.cuda.device_count() < _WORLD_SIZE,
    reason="requires NCCL and at least two CUDA devices",
)
def test_glm_dsa_cp_all_gather_reorders_head_tail_shards_into_global_order_with_owner_gradients() -> None:
    mp.spawn(_head_tail_worker, args=(_free_port(),), nprocs=_WORLD_SIZE, join=True)

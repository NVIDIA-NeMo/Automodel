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

"""Scheduled CPU coverage for separate student and teacher KD meshes."""

from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_automodel.components.loss.kd_loss import KDLoss
from nemo_automodel.recipes import kd_utils


def _teacher_logits(input_ids: torch.Tensor) -> torch.Tensor:
    """Return deterministic teacher logits.

    Args:
        input_ids: Tensor of shape [batch, sequence] containing token IDs.

    Returns:
        Tensor of shape [batch, sequence, vocab], where vocab is 3.
    """
    values = input_ids.float()
    return torch.stack((values * 0.5, values * -0.25 + 1.0, values * 0.125 - 0.5), dim=-1)


def _run_kd_bridge_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        cfg = {
            "separate_meshes": True,
            "distributed": {"strategy": "fsdp2", "dp_size": 2},
            "teacher_distributed": {"strategy": "fsdp2", "dp_size": 1, "tp_size": 2},
        }
        setups = kd_utils.create_kd_distributed_setups(cfg, world_size=world_size)
        bridge = kd_utils.KDMeshBridge(setups, device=torch.device("cpu"))
        batch = None
        if bridge.is_student:
            input_ids = torch.tensor([[rank + 1, rank + 2]], dtype=torch.long)
            batch = {"input_ids": input_ids, "labels": input_ids.clone()}

        bridge.broadcast_command(kd_utils.RUN_TEACHER if bridge.is_student else None)
        received_teacher_logits = None
        for wave in range(bridge.num_waves):
            teacher_batch = bridge.send_batch(wave, batch)
            logits = _teacher_logits(teacher_batch["input_ids"]) if bridge.is_teacher else None
            received = bridge.send_logits(wave, logits)
            if received is not None:
                received_teacher_logits = received

        if bridge.is_student:
            assert received_teacher_logits is not None
            expected = _teacher_logits(batch["input_ids"])
            torch.testing.assert_close(received_teacher_logits, expected)
            scale = torch.tensor(0.75, requires_grad=True)
            loss = KDLoss()(expected.detach() * scale, received_teacher_logits, batch["labels"])
            loss.backward()
            assert torch.isfinite(loss)
            assert scale.grad is not None and torch.isfinite(scale.grad)

        bridge.synchronize()
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_kd_mesh_bridge_routes_two_student_replicas_through_teacher_tp2(tmp_path: Path) -> None:
    """Exercise explicit subset meshes and bridge collectives on four CPU ranks."""
    mp.spawn(_run_kd_bridge_worker, args=(4, str(tmp_path / "process_group")), nprocs=4, join=True)


def _run_subset_ep_mesh_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        cfg = {
            "separate_meshes": True,
            "distributed": {"strategy": "fsdp2", "dp_size": 2},
            "teacher_distributed": {
                "strategy": "fsdp2",
                "dp_size": 4,
                "pp_size": 2,
                "ep_size": 2,
                "pipeline": {},
            },
        }
        setups = kd_utils.create_kd_distributed_setups(cfg, world_size=world_size)
        if rank in setups.teacher_ranks:
            device_mesh = setups.teacher.mesh_context.device_mesh
            moe_mesh = setups.teacher.mesh_context.moe_mesh
            assert device_mesh["pp"].size() == 2
            assert moe_mesh is not None
            assert moe_mesh["ep"].size() == 2
            assert set(dist.get_process_group_ranks(moe_mesh["ep"].get_group())).issubset(setups.teacher_ranks)
    finally:
        dist.destroy_process_group()


def test_separate_kd_setup_builds_teacher_ep_mesh_on_rank_subset(tmp_path: Path) -> None:
    """Teacher PP and EP groups may occupy ranks disjoint from the student mesh."""
    mp.spawn(_run_subset_ep_mesh_worker, args=(10, str(tmp_path / "ep_process_group")), nprocs=10, join=True)

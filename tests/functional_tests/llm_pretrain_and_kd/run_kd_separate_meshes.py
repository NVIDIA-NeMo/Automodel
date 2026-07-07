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

"""Four-rank correctness smoke for KD meshes with different parallelism."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from nemo_automodel.components.loss.kd_loss import KDLoss
from nemo_automodel.recipes.kd_utils import RUN_TEACHER, KDMeshBridge, create_kd_distributed_setups


def _teacher_logits(input_ids: torch.Tensor) -> torch.Tensor:
    values = input_ids.float()
    return torch.stack((values * 0.5, values * -0.25 + 1.0, values * 0.125 - 0.5), dim=-1)


def main() -> None:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise ValueError(f"This smoke test requires 4 ranks, got {world_size}")

    if backend == "nccl":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    cfg = {
        "separate_meshes": True,
        "distributed": {
            "strategy": "fsdp2",
            "dp_size": 2,
            "tp_size": 1,
            "cp_size": 1,
            "pp_size": 1,
        },
        "teacher_distributed": {
            "strategy": "fsdp2",
            "dp_size": 1,
            "tp_size": 2,
            "cp_size": 1,
            "pp_size": 1,
        },
    }
    setups = create_kd_distributed_setups(cfg, world_size=world_size)
    bridge = KDMeshBridge(setups, device=device)

    batch = None
    if bridge.is_student:
        input_ids = torch.tensor([[rank + 1, rank + 2]], dtype=torch.long, device=device)
        batch = {"input_ids": input_ids, "labels": input_ids.clone()}

    bridge.broadcast_command(RUN_TEACHER if bridge.is_student else None)
    received_teacher_logits = None
    for wave in range(bridge.num_waves):
        teacher_batch = bridge.send_batch(wave, batch)
        logits = _teacher_logits(teacher_batch["input_ids"]) if bridge.is_teacher else None
        received = bridge.send_logits(wave, logits)
        if received is not None:
            received_teacher_logits = received

    local_result = None
    if bridge.is_student:
        assert received_teacher_logits is not None
        expected = _teacher_logits(batch["input_ids"])
        torch.testing.assert_close(received_teacher_logits, expected)
        scale = torch.tensor(0.75, device=device, requires_grad=True)
        student_logits = expected.detach() * scale
        loss = KDLoss()(student_logits, received_teacher_logits, batch["labels"])
        loss.backward()
        if not torch.isfinite(loss) or scale.grad is None or not torch.isfinite(scale.grad):
            raise AssertionError(f"Non-finite KD result on rank {rank}: loss={loss}, grad={scale.grad}")
        local_result = {"rank": rank, "loss": loss.item(), "grad": scale.grad.item()}

    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_result)
    if rank == 0:
        student_results = [item for item in gathered if item is not None]
        assert len(student_results) == 2
        print(
            "KD_SEPARATE_MESH_RESULT "
            + json.dumps(
                {
                    "pass": True,
                    "student_ranks": list(setups.student_ranks),
                    "teacher_ranks": list(setups.teacher_ranks),
                    "student_dp": 2,
                    "teacher_tp": 2,
                    "results": student_results,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

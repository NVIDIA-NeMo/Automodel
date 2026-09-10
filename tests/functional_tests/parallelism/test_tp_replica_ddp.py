# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native DDP initialization/updates over combined DP/TP replica groups (L2)."""

import copy
import sys
from contextlib import nullcontext
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.parallel import DistributedDataParallel

from nemo_automodel.components.distributed.ddp import fp32_allreduce_hook
from nemo_automodel.components.distributed.mesh_utils import get_dp_tp_group


def _replica_group_worker(rank: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=4, timeout=timedelta(seconds=60)
    )
    try:
        # DP x TP, then CP and PP exclusion, then replicated-DP composition.
        for shape in ((1, 1, 2, 1, 2), (1, 1, 1, 2, 2), (2, 1, 1, 1, 2), (1, 2, 1, 1, 2)):
            mesh = init_device_mesh("cpu", shape, mesh_dim_names=("pp", "dp_replicate", "dp_shard", "cp", "tp"))
            group = get_dp_tp_group(mesh)
            members = dist.get_process_group_ranks(group or dist.group.WORLD)
            coordinate = mesh.get_coordinate()
            expected_members = [
                int(mesh.mesh[coordinate[0], replicate, shard, coordinate[3], tp])
                for replicate in range(shape[1])
                for shard in range(shape[2])
                for tp in range(shape[4])
            ]
            assert members == expected_members
            model = nn.Linear(4, 2)
            with torch.no_grad():
                model.weight.fill_(rank + 0.5)
                model.bias.fill_(0.25)
            model.register_buffer("scale", torch.tensor(float(rank + 1)))
            model = DistributedDataParallel(model, process_group=group, broadcast_buffers=True)
            model.broadcast_buffers = False
            model.register_comm_hook(model.process_group, fp32_allreduce_hook)
            # No separate initialization broadcast: DDP canonicalizes the weights.
            torch.testing.assert_close(
                model.module.weight, torch.full_like(model.module.weight, min(members) + 0.5), rtol=0, atol=0
            )
            torch.testing.assert_close(model.module.scale, torch.tensor(float(min(members) + 1)), rtol=0, atol=0)
            reference = copy.deepcopy(model.module)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.01)
            for window in (3, 1):
                for microbatch in range(window):
                    sync_context = model.no_sync() if microbatch < window - 1 else nullcontext()
                    inputs = torch.arange(4).float().reshape(1, 4) + microbatch + rank // 2
                    with sync_context:
                        # Synthetic TP-varying upstream gradient; DDP must average it.
                        model(inputs).square().sum().mul((0.75 + 0.5 * (rank % 2)) / window).backward()
                    for member in members:
                        peer_inputs = torch.arange(4).float().reshape(1, 4) + microbatch + member // 2
                        reference(peer_inputs).square().sum().mul(
                            (0.75 + 0.5 * (member % 2)) / (window * len(members))
                        ).backward()
                for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-6)
                actual_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.5)
                torch.testing.assert_close(actual_norm, expected_norm, rtol=1e-6, atol=1e-6)
                optimizer.step()
                reference_optimizer.step()
                for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
                    for key in ("exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(
                            optimizer.state[actual][key], reference_optimizer.state[expected][key], rtol=1e-6, atol=1e-6
                        )
                optimizer.zero_grad(set_to_none=True)
                reference_optimizer.zero_grad(set_to_none=True)
            dist.barrier()
        # BF16 storage must not turn the new cross-TP reduction into a BF16 sum.
        # Cancellation makes low-precision intermediate sums observably different.
        all_inputs = torch.tensor([[1024.0, 1.0], [-1024.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        for max_norm in (float("inf"), 0.5):
            model = nn.Linear(2, 1, bias=False, dtype=torch.bfloat16)
            with torch.no_grad():
                model.weight.fill_(1)
            model = DistributedDataParallel(model)
            model.register_comm_hook(model.process_group, fp32_allreduce_hook)
            reference = copy.deepcopy(model.module)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            reference_optimizer = torch.optim.Adam(reference.parameters(), lr=0.01)
            for window in (3, 1):
                for microbatch in range(window):
                    sync_context = model.no_sync() if microbatch < window - 1 else nullcontext()
                    with sync_context:
                        model(all_inputs[rank : rank + 1].bfloat16()).sum().backward()
                expected_grad = (all_inputs.mean(0) * window).reshape(1, 2).bfloat16()
                torch.testing.assert_close(model.module.weight.grad, expected_grad, rtol=0, atol=0)
                reference.weight.grad = expected_grad.clone()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                ref_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), max_norm)
                torch.testing.assert_close(norm, ref_norm, rtol=0, atol=0)
                optimizer.step()
                reference_optimizer.step()
                torch.testing.assert_close(model.module.weight, reference.weight, rtol=0, atol=0)
                for key in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(
                        optimizer.state[model.module.weight][key],
                        reference_optimizer.state[reference.weight][key],
                        rtol=0,
                        atol=0,
                    )
                optimizer.zero_grad(set_to_none=True)
                reference_optimizer.zero_grad(set_to_none=True)
    finally:
        dist.destroy_process_group()


def test_ddp_owns_all_tp_replicas_without_recipe_sync(tmp_path, monkeypatch):
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    torch.multiprocessing.spawn(_replica_group_worker, args=(str(tmp_path / "replica-rdzv"),), nprocs=4, join=True)

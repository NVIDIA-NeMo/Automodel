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

"""Preserve CPU collectives when CUDA meshes override process-group timeouts."""

import datetime
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard

from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh


def _check_mesh_backends(rank: int, store_path: str, bind_device: bool, layout: str) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    default_timeout = datetime.timedelta(minutes=7)
    dist.init_process_group(
        backend="cuda:nccl,cpu:gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=2,
        timeout=default_timeout,
        device_id=device if bind_device else None,
    )
    try:
        context = MeshContext.build(
            FSDP2Config(),
            ParallelismSizes(cp_size=2 if layout == "cp" else 1, ep_size=2 if layout == "ep" else 1),
            world_size=2,
            timeout_minutes=45,
        )
        meshes = [context.device_mesh]
        if context.moe_mesh is not None:
            meshes.append(context.moe_mesh)
        for mesh in meshes:
            for axis in mesh.mesh_dim_names:
                group = mesh[axis].get_group()
                # World-sized axes reuse the default group, which already owns its
                # timeout, instead of allocating another communicator. All newly
                # created groups use 45 min.
                assert (group is dist.group.WORLD) == (mesh[axis].size() == dist.get_world_size())
                expected_timeout = default_timeout if group is dist.group.WORLD else datetime.timedelta(minutes=45)
                for device_type in ("cpu", "cuda"):
                    tensor = torch.ones(1, device=device_type)
                    dist.all_reduce(tensor, group=group)
                    torch.testing.assert_close(tensor, torch.full_like(tensor, dist.get_world_size(group)))
                    assert group._get_backend(torch.device(device_type)).options._timeout == expected_timeout

        # Exercise the actual failure from #4601: CPU-offloaded FSDP state-dict
        # reconstruction uses a CPU all-gather through the selected DP mesh.
        torch.manual_seed(17)
        model = torch.nn.Linear(4, 4, bias=False, device=device)
        expected = model.weight.detach().cpu().clone()
        fully_shard(
            model, mesh=get_fsdp_dp_mesh(context.device_mesh), offload_policy=CPUOffloadPolicy(pin_memory=False)
        )
        model.to("cpu")
        torch.testing.assert_close(model.state_dict()["weight"].full_tensor(), expected)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
@pytest.mark.timeout(120)
@pytest.mark.parametrize("bind_device", [False, True], ids=["lazy", "bound"])
@pytest.mark.parametrize("layout", ["dp", "cp", "ep"])
def test_mesh_timeout_preserves_cpu_collectives(tmp_path: Path, bind_device: bool, layout: str) -> None:
    mp.spawn(_check_mesh_backends, args=(str(tmp_path / "store"), bind_device, layout), nprocs=2, join=True)

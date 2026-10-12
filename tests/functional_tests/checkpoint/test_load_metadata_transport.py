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


"""Exercise checkpoint load transport with CUDA DTensors on at most two GPUs."""

import socket
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.checkpoint.config import CheckpointingConfig


def _worker(rank: int, world_size: int, port: int, root: str, use_gloo: bool) -> None:
    """Load known parameter shards twice and verify the caller's group survives close."""
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    control = dist.new_group(backend="gloo")
    checkpointer = None
    try:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))
        path = Path(root) / "model"
        path.mkdir(exist_ok=True)
        weight = torch.arange(32 * 16, dtype=torch.float32).reshape(32, 16) / 100
        bias = torch.arange(32, dtype=torch.float32) / 100
        if rank == 0:
            save_file({"weight": weight, "bias": bias}, str(path / "model.safetensors"))
        dist.barrier(group=control)
        model = torch.nn.Linear(16, 32, device="meta")
        for name, full in (("weight", weight), ("bias", bias)):
            local = torch.zeros_like(full.chunk(world_size)[rank], device="cuda")
            setattr(model, name, torch.nn.Parameter(DTensor.from_local(local, mesh, [Shard(0)])))
        for _ in range(2):
            checkpointer = CheckpointingConfig(
                enabled=False,
                save_consolidated=False,
                load_metadata_with_gloo=use_gloo,
                checkpoint_dir=str(Path(root) / "unused"),
                model_repo_id=str(path),
            ).build(rank, 0, 0, process_group=dist.group.WORLD)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.to_local().zero_()
            checkpointer.load_model(model, str(path), is_init_step=True)
            for name, expected in (("weight", weight), ("bias", bias)):
                torch.testing.assert_close(
                    getattr(model, name).to_local().cpu(), expected.chunk(world_size)[rank], rtol=0, atol=0
                )
            metadata_group = checkpointer._load_process_group
            if use_gloo and world_size > 1:
                assert metadata_group is not None and dist.get_backend(metadata_group) == "gloo"
                assert dist.get_process_group_ranks(metadata_group) == list(range(world_size))
            else:
                assert metadata_group is None
            checkpointer.close()
            checkpointer.close()
        check = torch.ones(1, device="cuda")
        dist.all_reduce(check)
        assert check.item() == world_size
    finally:
        if checkpointer is not None:
            checkpointer.close()
        dist.destroy_process_group(control)
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("use_gloo", [False, True])
def test_load_metadata_transport(tmp_path: Path, world_size: int, use_gloo: bool) -> None:
    """Changing the planning backend must preserve every loaded weight and group ownership."""
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} CUDA GPUs")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(world_size, port, str(tmp_path), use_gloo), nprocs=world_size, join=True)

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Scheduled DCP round-trip coverage for rank-asymmetric optimizer parameters."""

import os
import socket
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState
from nemo_automodel.components.optim.optimizer import FusedAdamConfig, _drop_empty_local_shards
from tests.unit_tests.optim.test_fused_adam_empty_local_shards import (
    _RecordingFusedAdam,
    _te_stub_modules,
)


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


class _ShardedTwoParamModel(nn.Module):
    """Two dim-0 ``Shard(0)`` DTensor parameters over a two-rank mesh.

    ``weight`` has global shape [4, 3] and a non-empty local shard on every
    rank. ``tiny`` has global shape [1, 3], so rank 1 owns an empty local shard.
    """

    def __init__(self, mesh: DeviceMesh, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.weight = nn.Parameter(distribute_tensor(torch.randn(4, 3), mesh, [Shard(0)]))
        self.tiny = nn.Parameter(distribute_tensor(torch.randn(1, 3), mesh, [Shard(0)]))


def _asymmetric_shard_roundtrip_worker(rank: int, world_size: int, port: int, checkpoint_dir: str) -> None:
    try:
        _init_gloo(rank, world_size, port)
        mesh = init_device_mesh("cpu", (world_size,))
        num_local = 2 if rank == 0 else 1

        sys.modules.update(_te_stub_modules())
        model = _ShardedTwoParamModel(mesh, seed=17)
        FusedAdamConfig()._build_optimizer(list(model.parameters()))
        assert len(_RecordingFusedAdam.last_params) == num_local

        params = _drop_empty_local_shards(list(model.parameters()))
        assert len(params) == num_local
        opt = torch.optim.Adam(params, lr=1e-2, foreach=False)
        for index, parameter in enumerate(params):
            parameter.grad = torch.full_like(parameter, float(index + 1))
        opt.step()
        saved_state = [
            (
                opt.state[parameter]["exp_avg"].to_local().clone(),
                opt.state[parameter]["exp_avg_sq"].to_local().clone(),
            )
            for parameter in params
        ]
        dcp.save({"optim": OptimizerState(model, opt)}, checkpoint_id=checkpoint_dir)

        model2 = _ShardedTwoParamModel(mesh, seed=23)
        params2 = _drop_empty_local_shards(list(model2.parameters()))
        assert len(params2) == num_local
        opt2 = torch.optim.Adam(params2, lr=1e-2, foreach=False)
        dcp.load({"optim": OptimizerState(model2, opt2)}, checkpoint_id=checkpoint_dir)

        for parameter, (exp_avg, exp_avg_sq) in zip(params2, saved_state):
            torch.testing.assert_close(opt2.state[parameter]["exp_avg"].to_local(), exp_avg)
            torch.testing.assert_close(opt2.state[parameter]["exp_avg_sq"].to_local(), exp_avg_sq)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_two_rank_asymmetric_shards_optimizer_state_roundtrip(tmp_path: Path) -> None:
    """Rank-asymmetric optimizers survive an OptimizerState DCP round trip."""
    mp.spawn(
        _asymmetric_shard_roundtrip_worker,
        args=(2, _free_port(), str(tmp_path)),
        nprocs=2,
        join=True,
    )

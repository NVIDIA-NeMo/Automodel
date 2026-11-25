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

"""DCP round-trip coverage for optimizer parameters with an empty local shard."""

import os
import socket
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState
from nemo_automodel.components.optim.optimizer import FusedAdamConfig
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
    With ``tiny_first`` the optimizer sees ``tiny`` before ``weight``, as it
    sees a LoRA ``lora_A`` weight before its ``lora_B``.
    """

    def __init__(self, mesh: DeviceMesh, seed: int, tiny_first: bool) -> None:
        super().__init__()
        torch.manual_seed(seed)
        weight = nn.Parameter(distribute_tensor(torch.randn(4, 3), mesh, [Shard(0)]))
        tiny = nn.Parameter(distribute_tensor(torch.randn(1, 3), mesh, [Shard(0)]))
        if tiny_first:
            self.tiny = tiny
            self.weight = weight
        else:
            self.weight = weight
            self.tiny = tiny


def _te_fused_adam_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return the parameters that ``FusedAdamConfig`` hands to TE FusedAdam on this rank."""
    FusedAdamConfig()._build_optimizer(list(model.parameters()))
    return _RecordingFusedAdam.last_params


def _empty_shard_roundtrip_worker(
    rank: int, world_size: int, port: int, checkpoint_dir: str, tiny_first: bool, native: bool
) -> None:
    try:
        _init_gloo(rank, world_size, port)
        mesh = init_device_mesh("cpu", (world_size,))

        sys.modules.update(_te_stub_modules())
        model = _ShardedTwoParamModel(mesh, seed=17, tiny_first=tiny_first)
        expected_names = ["tiny", "weight"] if tiny_first else ["weight", "tiny"]
        assert [name for name, _ in model.named_parameters()] == expected_names
        # Every rank holds every parameter at the same position, the empty shard of rank 1 included.
        params = _te_fused_adam_parameters(model)
        assert len(params) == 2
        assert all(held is owned for held, owned in zip(params, model.parameters()))
        assert model.tiny.to_local().numel() == (3 if rank == 0 else 0)

        # torch.optim.Adam stands in for TE FusedAdam, which needs CUDA. Both key
        # state_dict() by parameter position.
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
        # PEFT with expert parallelism saves the optimizer's own state_dict().
        wrapper = OptimizerState(model, opt, is_peft=native, has_expert_parallelism=native)
        assert wrapper._use_native_optimizer_state is native
        dcp.save({"optim": wrapper}, checkpoint_id=checkpoint_dir)

        model2 = _ShardedTwoParamModel(mesh, seed=23, tiny_first=tiny_first)
        params2 = _te_fused_adam_parameters(model2)
        assert len(params2) == 2
        opt2 = torch.optim.Adam(params2, lr=1e-2, foreach=False)
        dcp.load(
            {"optim": OptimizerState(model2, opt2, is_peft=native, has_expert_parallelism=native)},
            checkpoint_id=checkpoint_dir,
        )

        for parameter, (exp_avg, exp_avg_sq) in zip(params2, saved_state):
            torch.testing.assert_close(opt2.state[parameter]["exp_avg"].to_local(), exp_avg)
            torch.testing.assert_close(opt2.state[parameter]["exp_avg_sq"].to_local(), exp_avg_sq)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize("native", [False, True], ids=["by-name", "by-position"])
@pytest.mark.parametrize("tiny_first", [False, True], ids=["tiny-last", "tiny-first"])
def test_two_rank_asymmetric_shards_optimizer_state_roundtrip(tmp_path: Path, tiny_first: bool, native: bool) -> None:
    """Optimizer state survives an OptimizerState DCP round trip when rank 1 owns an empty shard.

    ``by-position`` is the branch PEFT with expert parallelism takes: the saved
    keys carry the parameter's position in the optimizer, so with ``tiny-first``
    a rank that left ``tiny`` out would save ``weight`` under the key of ``tiny``.
    """
    mp.spawn(
        _empty_shard_roundtrip_worker,
        args=(2, _free_port(), str(tmp_path), tiny_first, native),
        nprocs=2,
        join=True,
    )


def _name_keyed_checkpoint_without_empty_shard_worker(
    rank: int, world_size: int, port: int, checkpoint_dir: str
) -> None:
    try:
        _init_gloo(rank, world_size, port)
        mesh = init_device_mesh("cpu", (world_size,))

        sys.modules.update(_te_stub_modules())
        gradients = {"tiny": 1.0, "weight": 2.0}
        model = _ShardedTwoParamModel(mesh, seed=17, tiny_first=True)
        # A checkpoint written before empty shards were kept: the saving optimizer of
        # rank 1 does not hold ``tiny``, whose local shard is empty there.
        saved_names = [name for name, parameter in model.named_parameters() if parameter.to_local().numel() > 0]
        assert saved_names == (["tiny", "weight"] if rank == 0 else ["weight"])
        opt = torch.optim.Adam([getattr(model, name) for name in saved_names], lr=1e-2, foreach=False)
        for name in saved_names:
            getattr(model, name).grad = torch.full_like(getattr(model, name), gradients[name])
        opt.step()
        saved_state = {
            name: (
                opt.state[getattr(model, name)]["exp_avg"].to_local().clone(),
                opt.state[getattr(model, name)]["exp_avg_sq"].to_local().clone(),
            )
            for name in saved_names
        }
        dcp.save({"optim": OptimizerState(model, opt)}, checkpoint_id=checkpoint_dir)

        # The loading optimizer holds every parameter, as FusedAdamConfig hands them to TE.
        model2 = _ShardedTwoParamModel(mesh, seed=23, tiny_first=True)
        params2 = _te_fused_adam_parameters(model2)
        assert len(params2) == 2
        opt2 = torch.optim.Adam(params2, lr=1e-2, foreach=False)
        dcp.load({"optim": OptimizerState(model2, opt2)}, checkpoint_id=checkpoint_dir)

        for name, parameter in model2.named_parameters():
            exp_avg = opt2.state[parameter]["exp_avg"].to_local()
            exp_avg_sq = opt2.state[parameter]["exp_avg_sq"].to_local()
            if name in saved_state:
                torch.testing.assert_close(exp_avg, saved_state[name][0])
                torch.testing.assert_close(exp_avg_sq, saved_state[name][1])
            else:
                # ``tiny`` on rank 1 was not in the saving optimizer; its state is empty.
                assert exp_avg.numel() == 0
                assert exp_avg_sq.numel() == 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_two_rank_name_keyed_checkpoint_without_empty_shard_loads_into_full_lists(tmp_path: Path) -> None:
    """A name-keyed checkpoint saved without rank 1's empty shard loads into optimizers that hold it."""
    mp.spawn(
        _name_keyed_checkpoint_without_empty_shard_worker,
        args=(2, _free_port(), str(tmp_path)),
        nprocs=2,
        join=True,
    )

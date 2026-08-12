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

"""Scheduled numerical regression test for MoE HSDP replica gradient synchronization.

Four Gloo ranks form a logical ``dp_replicate=2 x dp_shard=2`` mesh while
``ep_size=2`` keeps the MoE parallelization path active. Rank-specific inputs
produce different local gradients; corresponding parameter shards must remain
identical after the optimizer step only when replica synchronization is present.
"""

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.components.moe.parallelizer import parallelize_model

_WORLD_SIZE = 4
_HIDDEN_SIZE = 8


class _TinyExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(4, _HIDDEN_SIZE, _HIDDEN_SIZE))


class _TinyMoE(MoE):
    """Minimal MoE recognized by the production parallelizer."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.experts = _TinyExperts()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _TinyMoE()
        self.proj = nn.Linear(_HIDDEN_SIZE, _HIDDEN_SIZE, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(hidden_states))


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_TinyBlock()])
        self.moe_config = type("MoeConfig", (), {"n_routed_experts": 4})()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyBackbone()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model(hidden_states)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _init_process_group(rank: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=_WORLD_SIZE)


def _replica_sync_worker(rank: int, port: int) -> None:
    try:
        _init_process_group(rank, port)
        torch.manual_seed(1234)
        model = _TinyModel()
        mesh_context = MeshContext.build(
            FSDP2Config(),
            ParallelismSizes(
                dp_size=_WORLD_SIZE,
                dp_replicate_size=2,
                ep_size=2,
            ),
            world_size=_WORLD_SIZE,
        )

        parallelize_model(
            model,
            mesh_context.device_mesh,
            mesh_context.moe_mesh,
            **mesh_context.parallelize_axis_kwargs(),
        )

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        # Distinct inputs make the test sensitive to a missing replica all-reduce.
        generator = torch.Generator().manual_seed(9000 + rank)
        hidden_states = torch.randn(4, _HIDDEN_SIZE, generator=generator)
        loss = model(hidden_states).square().mean()
        loss.backward()
        optimizer.step()

        local_weight = model.model.layers[0].proj.weight.to_local().detach()
        gathered_weights = [torch.empty_like(local_weight) for _ in range(_WORLD_SIZE)]
        dist.all_gather(gathered_weights, local_weight)

        if rank == 0:
            # Ranks 0/2 and 1/3 own corresponding shards in the two replica groups.
            torch.testing.assert_close(gathered_weights[0], gathered_weights[2], rtol=0, atol=0)
            torch.testing.assert_close(gathered_weights[1], gathered_weights[3], rtol=0, atol=0)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_moe_hsdp_optimizer_step_synchronizes_replicas():
    """Different replica inputs must still produce identical replicated shards."""
    mp.spawn(
        _replica_sync_worker,
        args=(_free_port(),),
        nprocs=_WORLD_SIZE,
        join=True,
    )

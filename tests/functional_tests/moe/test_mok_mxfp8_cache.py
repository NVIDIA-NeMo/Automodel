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

"""Regression coverage for MXFP8 cache invalidation with EP DTensor weights."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe import mok_experts
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.mok_experts import GroupedExpertsMoK


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _config() -> MoEConfig:
    return MoEConfig(
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=256,
        inter_dim=512,
        moe_inter_dim=256,
        norm_topk_prob=True,
        dtype=torch.bfloat16,
    )


def _cache_worker(rank: int, world_size: int, port: int) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("ep",))
        experts = GroupedExpertsMoK(
            _config(),
            BackendConfig(dispatcher="mok", mok={"precision": "mxfp8"}),
        ).cuda()
        with torch.no_grad():
            for parameter in experts.parameters():
                parameter.normal_(mean=0.0, std=0.1)
        for name in ("routed_gate_weights", "routed_up_weights", "routed_down_weights"):
            parameter = getattr(experts, name)
            setattr(experts, name, nn.Parameter(distribute_tensor(parameter.detach(), mesh, [Shard(0)])))

        quantization_calls = 0

        def fake_quantize(
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Return cloned normal/transposed layouts for a local expert weight.

            Args:
                weight: Rank-local BF16 tensor of shape
                    [local_experts, output_features, input_features].

            Returns:
                Cloned normal and transposed weights plus placeholder scale tensors.
            """
            nonlocal quantization_calls
            quantization_calls += 1
            scale = torch.zeros(1, device=weight.device)
            return weight.clone(), scale, weight.transpose(-1, -2).contiguous(), scale.clone()

        mok_experts._mxfp8_weight_both = fake_quantize
        optimizer = torch.optim.AdamW(experts.parameters(), lr=1.0, foreach=False)
        mok_experts.enable_mok_mxfp8_optimizer_step_cache([experts], [optimizer])
        local_weights = tuple(mok_experts._local_tensor(parameter) for parameter in experts.parameters())
        first = experts.runtime._get_mxfp8_weights(*local_weights)
        snapshot = local_weights[0].clone()

        sum(parameter.float().square().sum() for parameter in experts.parameters()).backward()
        optimizer.step()

        updated_local_weights = tuple(mok_experts._local_tensor(parameter) for parameter in experts.parameters())
        assert not torch.equal(updated_local_weights[0], snapshot)
        assert experts.runtime._mxfp8_weights is None
        refreshed = experts.runtime._get_mxfp8_weights(*updated_local_weights)
        assert quantization_calls == 6
        assert not torch.equal(first[0][0], refreshed[0][0])
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_mok_mxfp8_cache_refreshes_after_dtensor_optimizer_step() -> None:
    mp.spawn(_cache_worker, args=(1, _free_port()), nprocs=1, join=True)

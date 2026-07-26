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

"""Scheduled two-rank CPU regression for router gradients through the EP all-gather.

``GroupedExperts.forward`` all-gathers the per-token routing probabilities across
the expert-parallel group before dispatching tokens to local experts. Routing
probabilities participate in the main-loss gradient, so the gather must be
autograd-safe: a plain ``dist.all_gather`` detaches every gathered tensor and
silently leaves the router trainable only through auxiliary losses.
"""

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

from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts

_N_EXPERTS = 4
_TOP_K = 2
_DIM = 16
_MOE_INTER_DIM = 32
# Uneven per-rank token counts exercise the variable-length gather path.
_TOKENS_PER_RANK = (3, 2)


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _tiny_moe_config() -> MoEConfig:
    return MoEConfig(
        n_routed_experts=_N_EXPERTS,
        n_shared_experts=0,
        n_activated_experts=_TOP_K,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=_DIM,
        inter_dim=_MOE_INTER_DIM,
        moe_inter_dim=_MOE_INTER_DIM,
        norm_topk_prob=False,
        expert_bias=False,
        expert_activation="swiglu",
        dtype=torch.float32,
    )


def _global_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic global batch shared by the reference and EP runs."""
    generator = torch.Generator().manual_seed(1234)
    num_tokens = sum(_TOKENS_PER_RANK)
    x = torch.randn(num_tokens, _DIM, generator=generator)
    router_logits = torch.randn(num_tokens, _N_EXPERTS, generator=generator)
    weights, indices = router_logits.softmax(dim=-1).topk(_TOP_K, dim=-1)
    token_mask = torch.ones(num_tokens, dtype=torch.bool)
    return x, weights, indices, token_mask


def _build_experts(config: MoEConfig) -> GroupedExperts:
    generator = torch.Generator().manual_seed(4321)
    experts = GroupedExperts(config)
    with torch.no_grad():
        experts.gate_and_up_projs.copy_(torch.randn(experts.gate_and_up_projs.shape, generator=generator) * 0.05)
        experts.down_projs.copy_(torch.randn(experts.down_projs.shape, generator=generator) * 0.05)
    return experts


def _reference_forward_backward() -> tuple[torch.Tensor, torch.Tensor]:
    """Single-process (ep_size=1) forward/backward as ground truth."""
    experts = _build_experts(_tiny_moe_config())
    x, weights, indices, token_mask = _global_inputs()
    weights = weights.clone().requires_grad_(True)
    y = experts(x, token_mask, weights, indices)
    y.sum().backward()
    assert weights.grad is not None
    return y.detach(), weights.grad.detach()


def _ep_router_grad_worker(rank: int, world_size: int, port: int) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

        y_ref, weights_grad_ref = _reference_forward_backward()

        ep_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("ep",))
        experts = _build_experts(_tiny_moe_config())
        experts.gate_and_up_projs = nn.Parameter(
            distribute_tensor(experts.gate_and_up_projs.detach(), ep_mesh, [Shard(0)])
        )
        experts.down_projs = nn.Parameter(distribute_tensor(experts.down_projs.detach(), ep_mesh, [Shard(0)]))

        x, weights, indices, token_mask = _global_inputs()
        start = sum(_TOKENS_PER_RANK[:rank])
        end = start + _TOKENS_PER_RANK[rank]
        local_weights = weights[start:end].clone().requires_grad_(True)

        y_local = experts(x[start:end], token_mask[start:end], local_weights, indices[start:end])
        torch.testing.assert_close(y_local, y_ref[start:end], rtol=1e-4, atol=1e-5)

        y_local.sum().backward()

        # Pre-fix, the routing weights were gathered with a non-differentiable
        # ``dist.all_gather`` and the local router leaf received no gradient.
        assert local_weights.grad is not None, "router weights received no gradient through the EP all-gather"
        torch.testing.assert_close(local_weights.grad, weights_grad_ref[start:end], rtol=1e-4, atol=1e-5)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
def test_ep_all_gather_propagates_router_weight_gradients():
    mp.spawn(
        _ep_router_grad_worker, args=(len(_TOKENS_PER_RANK), _free_port()), nprocs=len(_TOKENS_PER_RANK), join=True
    )

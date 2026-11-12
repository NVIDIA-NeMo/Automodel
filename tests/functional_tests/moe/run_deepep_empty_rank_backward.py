#!/usr/bin/env python
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

"""Two-rank DeepEP regression for an EP rank that receives no expert tokens.

All tokens are routed to expert 0, so rank 1 takes the zero-token expert path.
Backward must still traverse DeepEP's reverse-dispatch collective on both ranks
and materialize explicit zero gradients for rank 1's local expert parameters.
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor

from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExpertsDeepEP
from nemo_automodel.components.moe.megatron.fused_a2a import free_buffer


def _config() -> MoEConfig:
    return MoEConfig(
        n_routed_experts=2,
        n_shared_experts=0,
        n_activated_experts=1,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=16,
        inter_dim=32,
        moe_inter_dim=32,
        norm_topk_prob=False,
        expert_bias=True,
        expert_activation="swiglu",
        dtype=torch.bfloat16,
    )


def main() -> int:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))

    try:
        if dist.get_world_size() != 2:
            if dist.get_rank() == 0:
                print("ERROR: this regression requires exactly two ranks", file=sys.stderr)
            return 1

        ep_mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("ep",))
        experts = GroupedExpertsDeepEP(_config()).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            experts.init_weights(device)
        for name, parameter in list(experts.named_parameters(recurse=False)):
            sharded = nn.Parameter(distribute_tensor(parameter.detach(), ep_mesh, [Shard(0)]))
            sharded.requires_grad = parameter.requires_grad
            experts.register_parameter(name, sharded)
        experts.init_token_dispatcher(ep_mesh)

        generator = torch.Generator(device=device).manual_seed(20260825 + dist.get_rank())
        hidden_states = torch.randn(4, 16, generator=generator, device=device, dtype=torch.bfloat16)
        hidden_states.requires_grad_(True)
        routing_probs = torch.ones(4, 1, device=device, dtype=torch.float32, requires_grad=True)
        expert_indices = torch.zeros(4, 1, device=device, dtype=torch.long)
        token_mask = torch.ones(4, device=device, dtype=torch.bool)

        output = experts(hidden_states, token_mask, routing_probs, expert_indices)
        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()
        output.float().square().sum().backward()

        assert hidden_states.grad is not None and torch.isfinite(hidden_states.grad).all()
        assert routing_probs.grad is not None and torch.isfinite(routing_probs.grad).all()
        has_nonzero_expert_grad = False
        for parameter in experts.parameters():
            assert parameter.grad is not None
            local_grad = parameter.grad.to_local()
            assert torch.isfinite(local_grad).all()
            if dist.get_rank() == 1:
                assert torch.count_nonzero(local_grad) == 0
            else:
                has_nonzero_expert_grad |= bool(torch.count_nonzero(local_grad))
        if dist.get_rank() == 0:
            assert has_nonzero_expert_grad

        dist.barrier()
        if dist.get_rank() == 0:
            print("PASS: DeepEP backward completed with an empty expert rank")
        return 0
    finally:
        free_buffer()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())

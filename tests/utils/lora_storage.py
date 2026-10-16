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

"""Shared setup for LoRA storage tests on CPU and CUDA."""

import copy

import torch
from torch import nn

from nemo_automodel.components.moe.config import MoEConfig


def tiny_moe_config(*, dtype: torch.dtype = torch.bfloat16, bias: bool = False) -> MoEConfig:
    """Return the four-expert configuration used by storage tests."""
    return MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=1,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=False,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=16,
        inter_dim=32,
        moe_inter_dim=32,
        norm_topk_prob=False,
        dtype=dtype,
        expert_bias=bias,
    )


def randomize_and_copy(module: nn.Module) -> nn.Module:
    """Initialize every parameter in place, then copy the module as a reference."""
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.05)
    return copy.deepcopy(module)


def expert_routing_inputs(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Route each token to one of four experts in round-robin order.

    Args:
        x: Activations of shape [tokens, hidden]; supplies the device and dtype.

    Returns:
        Boolean mask [tokens], unit weights [tokens, 1] in x's dtype, and
        int64 expert indices [tokens, 1], all on x's device.
    """
    tokens = x.shape[0]
    return (
        torch.ones(tokens, device=x.device, dtype=torch.bool),
        torch.ones(tokens, 1, device=x.device, dtype=x.dtype),
        torch.arange(tokens, device=x.device).remainder(4).view(-1, 1),
    )

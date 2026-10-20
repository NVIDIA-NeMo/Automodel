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

"""A rank whose local experts received no tokens runs the regular expert path on a phantom token."""

import pytest
import torch

from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import (
    GroupedExperts,
    _phantom_dispatch,
    _phantom_local_routing,
)


def _config(**overrides) -> MoEConfig:
    settings = dict(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
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
        norm_topk_prob=True,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        activation_alpha=1.702,
        activation_limit=7.0,
        dtype=torch.float32,
    )
    settings.update(overrides)
    return MoEConfig(**settings)


def _unrouted_batch(config: MoEConfig, device: torch.device):
    """Eight valid tokens whose routes all point at experts this rank does not own."""
    torch.manual_seed(3)
    x = torch.randn(8, config.dim, device=device, requires_grad=True)
    token_mask = torch.ones(8, dtype=torch.bool, device=device)
    weights = torch.rand(8, config.n_activated_experts, device=device, requires_grad=True)
    indices = torch.full((8, config.n_activated_experts), config.n_routed_experts + 5, dtype=torch.long, device=device)
    return x, token_mask, weights, indices


@pytest.mark.parametrize("lora", [False, True], ids=["base", "lora"])
@pytest.mark.parametrize("use_torch_mm", [False, True], ids=["loop", "grouped_mm"])
@pytest.mark.parametrize("after_down", [False, True], ids=["weight_in_activation", "weight_after_down"])
def test_unrouted_rank_outputs_zero_and_keeps_every_parameter_and_routing_weight_in_the_graph(
    lora: bool, use_torch_mm: bool, after_down: bool
) -> None:
    if use_torch_mm and not torch.cuda.is_available():
        pytest.skip("torch._grouped_mm needs CUDA")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = _config(apply_router_weight_after_down=after_down)
    experts = GroupedExperts(config).to(device)
    with torch.no_grad():
        experts.init_weights(buffer_device=device)
    experts.use_torch_mm = use_torch_mm
    if lora:
        experts = GroupedExpertsLoRA(experts, lora_dim=4, alpha=8).to(device)
        with torch.no_grad():
            experts.lora_gate_and_up_B.normal_()
            experts.lora_down_B.normal_()
    x, token_mask, weights, indices = _unrouted_batch(config, device)

    out = experts(x, token_mask, weights, indices)

    assert out.shape == x.shape
    torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)
    out.sum().backward()
    assert weights.grad is not None and torch.count_nonzero(weights.grad) == 0
    for name, parameter in experts.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all() and torch.count_nonzero(parameter.grad) == 0, name


def test_phantom_local_routing_describes_one_token_for_local_expert_zero() -> None:
    weights = torch.rand(3, 2, requires_grad=True)
    tokens_per_expert = torch.zeros(4, dtype=torch.long)
    token_ids, slot_ids, probs, counts, offs = _phantom_local_routing(weights, tokens_per_expert)
    assert token_ids.tolist() == [0] and slot_ids.tolist() == [0]
    assert counts.tolist() == [1, 0, 0, 0] and offs.tolist() == [1, 1, 1, 1] and offs.dtype == torch.int32
    assert probs.shape == (1,) and probs.item() == 0 and probs.requires_grad
    probs.sum().backward()
    assert weights.grad is not None and torch.count_nonzero(weights.grad) == 0


def test_phantom_dispatch_appends_one_zero_row_and_keeps_dispatched_tensors() -> None:
    weights = torch.rand(5, 2, requires_grad=True)
    hidden = torch.empty(0, 16, dtype=torch.bfloat16)
    probs = torch.empty(0, 1, dtype=torch.bfloat16)
    counts = torch.zeros(4, dtype=torch.long)
    hidden, counts, probs = _phantom_dispatch(hidden, counts, probs, weights)
    assert hidden.shape == (1, 16) and hidden.dtype == torch.bfloat16 and torch.count_nonzero(hidden) == 0
    assert counts.tolist() == [1, 0, 0, 0]
    assert probs.shape == (1, 1) and probs.dtype == torch.bfloat16 and probs.item() == 0 and probs.requires_grad
    _, counts_none, _ = _phantom_dispatch(torch.empty(0, 16), None, torch.empty(0, 1), weights)
    assert counts_none is None


def test_phantom_weight_without_routing_weights_is_a_plain_zero() -> None:
    _, _, probs, _, _ = _phantom_local_routing(torch.empty(0, 2), torch.zeros(2, dtype=torch.long))
    assert probs.shape == (1,) and probs.item() == 0 and not probs.requires_grad

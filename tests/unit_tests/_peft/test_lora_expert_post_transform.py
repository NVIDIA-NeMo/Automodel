# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Numerical contract for LoRA expert transforms and routing order."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components._peft.lora_experts import GroupedExpertsDeepEPLoRA, GroupedExpertsLoRA
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP


def _cpu_grouped_mm(x: torch.Tensor, weight: torch.Tensor, *, offs: torch.Tensor) -> torch.Tensor:
    """Provide CPU GEMMs for the production LoRA routing math.

    Args:
        x: Expert-sorted activations [routed_tokens, input].
        weight: Expert projections [experts, input, output].
        offs: Cumulative token counts [experts].

    Returns:
        Expert-sorted results [routed_tokens, output].
    """
    bounds = [0, *offs.tolist()]
    return torch.cat([x[bounds[i] : bounds[i + 1]] @ weight[i] for i in range(weight.shape[0])])


@pytest.mark.parametrize("after_down", [False, True])
@pytest.mark.parametrize("with_transform", [False, True])
@pytest.mark.parametrize("zero_adapter", [False, True])
def test_lora_expert_transform_forward_and_gradients(
    monkeypatch: pytest.MonkeyPatch, after_down: bool, with_transform: bool, zero_adapter: bool
) -> None:
    """Compare grouped execution with explicit per-expert effective weights."""
    torch.manual_seed(19)
    config = MoEConfig(
        dim=8,
        inter_dim=8,
        moe_inter_dim=8,
        n_routed_experts=2,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=False,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        norm_topk_prob=False,
        expert_activation="swiglu",
        apply_router_weight_after_down=after_down,
        dtype=torch.float32,
    )
    base = GroupedExpertsDeepEP(config)
    with torch.no_grad():
        for parameter in base.parameters():
            parameter.normal_(std=0.2)
    module = GroupedExpertsDeepEPLoRA(base, lora_dim=4, alpha=8, lora_dtype=torch.float32)
    with torch.no_grad():
        if not zero_adapter:
            module.lora_gate_and_up_B.normal_(std=0.1)
            module.lora_down_B.normal_(std=0.1)
    module.ep_size = 1
    tokens, dim = 3, 8

    def dispatch(
        x: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Arrange [tokens, hidden] and [tokens, topk] inputs by expert.

        Args:
            x: Local activations [tokens, hidden].
            mask: Valid-token flags [tokens].
            weights: Probabilities [tokens, topk].
            indices: Expert IDs [tokens, topk].

        Returns:
            Activations [experts * tokens, hidden], counts [experts], and
            probabilities [experts * tokens, 1].
        """
        return x.repeat(2, 1), torch.tensor([tokens, tokens]), weights.T.reshape(-1, 1)

    def combine(x: torch.Tensor) -> torch.Tensor:
        """Sum expert-sorted [experts * tokens, hidden] into [tokens, hidden]."""
        return x.reshape(2, tokens, dim).sum(0)

    monkeypatch.setattr(module, "_dispatch_tokens", dispatch)
    module.token_dispatcher = SimpleNamespace(token_unpermutation=combine)
    monkeypatch.setattr(torch, "_grouped_mm", _cpu_grouped_mm)
    x = torch.randn(tokens, dim, requires_grad=True)
    probabilities = torch.rand(tokens, 2, requires_grad=True)
    transform = torch.tanh if with_transform else None
    output = module(
        x,
        torch.ones(tokens, dtype=torch.bool),
        probabilities,
        torch.tensor([[0, 1]] * tokens),
        post_expert_output=transform,
    )
    x_ref = x.detach().clone().requires_grad_()
    p_ref = probabilities.detach().clone().requires_grad_()
    names = ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B")
    reference_adapters = [getattr(module, name).detach().clone().requires_grad_() for name in names]
    up_a, up_b, down_a, down_b = reference_adapters
    expert_outputs = []
    for expert in range(2):
        up = module.gate_and_up_projs[expert].detach() + module.scale * (up_a[expert] @ up_b[expert])
        down = module.down_projs[expert].detach() + module.scale * (down_a[expert] @ down_b[expert])
        gate, value = (x_ref @ up).chunk(2, -1)
        activation = F.silu(gate) * value
        if not after_down:
            activation = activation * p_ref[:, expert : expert + 1]
        result = activation @ down
        if transform is not None:
            result = transform(result)
        if after_down:
            result = result * p_ref[:, expert : expert + 1]
        expert_outputs.append(result)
    reference = sum(expert_outputs)
    torch.testing.assert_close(output, reference, rtol=2e-5, atol=2e-6)
    cotangent = torch.randn_like(output)
    actual_gradients = torch.autograd.grad(
        (output * cotangent).sum(), [x, probabilities, *[getattr(module, name) for name in names]]
    )
    reference_gradients = torch.autograd.grad((reference * cotangent).sum(), [x_ref, p_ref, *reference_adapters])
    for actual, expected in zip(actual_gradients, reference_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    assert module.gate_and_up_projs.grad is None
    assert module.down_projs.grad is None


@pytest.mark.parametrize("use_torch_mm", [False, True], ids=["loop", "grouped-mm"])
def test_plain_lora_post_transform_precedes_after_down_router_weight(
    monkeypatch: pytest.MonkeyPatch, use_torch_mm: bool
) -> None:
    """Plain LoRA expert paths preserve the post-transform routing order and gradients."""
    torch.manual_seed(29)
    config = MoEConfig(
        dim=8,
        inter_dim=8,
        moe_inter_dim=8,
        n_routed_experts=2,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=False,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        norm_topk_prob=False,
        expert_activation="swiglu",
        apply_router_weight_after_down=True,
        dtype=torch.float32,
    )
    base = GroupedExperts(config)
    base.use_torch_mm = use_torch_mm
    with torch.no_grad():
        for parameter in base.parameters():
            parameter.normal_(std=0.2)
    module = GroupedExpertsLoRA(base, lora_dim=4, alpha=8, lora_dtype=torch.float32)
    with torch.no_grad():
        module.lora_gate_and_up_B.normal_(std=0.1)
        module.lora_down_B.normal_(std=0.1)

    if use_torch_mm:
        monkeypatch.setattr(torch, "_grouped_mm", _cpu_grouped_mm)

    tokens, dim = 3, config.dim
    x = torch.randn(tokens, dim, requires_grad=True)
    probabilities = torch.rand(tokens, 2, requires_grad=True)
    indices = torch.tensor([[0, 1]] * tokens)
    output = module(
        x,
        torch.ones(tokens, dtype=torch.bool),
        probabilities,
        indices,
        post_expert_output=torch.tanh,
    )

    x_ref = x.detach().clone().requires_grad_()
    p_ref = probabilities.detach().clone().requires_grad_()
    names = ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B")
    reference_adapters = [getattr(module, name).detach().clone().requires_grad_() for name in names]
    up_a, up_b, down_a, down_b = reference_adapters
    expert_outputs = []
    for expert in range(2):
        up = module.gate_and_up_projs[expert].detach() + module.scale * (up_a[expert] @ up_b[expert])
        down = module.down_projs[expert].detach() + module.scale * (down_a[expert] @ down_b[expert])
        gate, value = (x_ref @ up).chunk(2, -1)
        expert_output = torch.tanh((F.silu(gate) * value) @ down)
        expert_outputs.append(expert_output * p_ref[:, expert : expert + 1])
    reference = sum(expert_outputs)

    torch.testing.assert_close(output, reference, rtol=2e-5, atol=2e-6)
    cotangent = torch.randn_like(output)
    actual_gradients = torch.autograd.grad(
        (output * cotangent).sum(), [x, probabilities, *[getattr(module, name) for name in names]]
    )
    reference_gradients = torch.autograd.grad((reference * cotangent).sum(), [x_ref, p_ref, *reference_adapters])
    for actual, expected in zip(actual_gradients, reference_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)

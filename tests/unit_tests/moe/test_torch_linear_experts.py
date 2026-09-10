# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the separate eager expert backend against independent linear weights."""

from dataclasses import replace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP, _torch_linear_experts_fwd


def _config(dim=8, inter_dim=12, **kwargs):
    return MoEConfig(
        n_routed_experts=3,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.5,
        dim=dim,
        inter_dim=inter_dim,
        moe_inter_dim=inter_dim,
        norm_topk_prob=True,
        swiglu_limit=1.5,
        combine_in_fp32=True,
        **kwargs,
    )


class _SeparateExpert(nn.Module):
    """Literal released Expert arithmetic with independent contiguous weights."""

    def __init__(self, config, device, dtype):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.moe_inter_dim, bias=False, device=device, dtype=dtype)
        self.w3 = nn.Linear(config.dim, config.moe_inter_dim, bias=False, device=device, dtype=dtype)
        self.w2 = nn.Linear(config.moe_inter_dim, config.dim, bias=False, device=device, dtype=dtype)
        self.limit = config.swiglu_limit

    def forward(self, hidden, weights):
        gate = self.w1(hidden).float()
        up = self.w3(hidden).float()
        if self.limit > 0:
            up = torch.clamp(up, min=-self.limit, max=self.limit)
            gate = torch.clamp(gate, max=self.limit)
        activated = F.silu(gate) * up
        activated = weights * activated
        return self.w2(activated.to(hidden.dtype))


def _compare_oracle(device, dtype, counts, dim=8, inter_dim=12, limit=1.5, padded_counts=None):
    torch.manual_seed(419)
    config = replace(_config(dim, inter_dim), swiglu_limit=limit)
    experts = nn.ModuleList(_SeparateExpert(config, device, dtype) for _ in counts)
    storage_counts = padded_counts or counts
    hidden = (torch.randn(sum(storage_counts), dim, device=device, dtype=dtype) * 4).requires_grad_()
    probs = torch.rand(sum(storage_counts), 1, device=device, dtype=torch.float32, requires_grad=True)
    actual_hidden = hidden.detach().clone().requires_grad_()
    actual_probs = probs.detach().clone().requires_grad_()
    packed = torch.stack([torch.cat((expert.w1.weight.T, expert.w3.weight.T), -1) for expert in experts])
    packed = packed.detach().requires_grad_()
    down = torch.stack([expert.w2.weight.T for expert in experts]).detach().requires_grad_()
    expected_parts = []
    for expert, x, weights, count in zip(experts, hidden.split(storage_counts), probs.split(storage_counts), counts):
        expected_parts.extend([expert(x[:count], weights[:count]), torch.zeros_like(x[count:])])
    expected = torch.cat(expected_parts)
    actual = _torch_linear_experts_fwd(
        actual_hidden,
        packed,
        down,
        storage_counts,
        actual_probs,
        config,
        unpadded_tokens_per_expert=counts if padded_counts else None,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(expected)
    expected.backward(upstream)
    actual.backward(upstream)
    torch.testing.assert_close(actual_hidden.grad, hidden.grad, rtol=0, atol=0)
    torch.testing.assert_close(actual_probs.grad, probs.grad, rtol=0, atol=0)
    for index, expert in enumerate(experts):
        torch.testing.assert_close(packed.grad[index, :, :inter_dim].T, expert.w1.weight.grad, rtol=0, atol=0)
        torch.testing.assert_close(packed.grad[index, :, inter_dim:].T, expert.w3.weight.grad, rtol=0, atol=0)
        torch.testing.assert_close(down.grad[index].T, expert.w2.weight.grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("counts", [(3, 0, 5), (0, 0, 0)])
@pytest.mark.parametrize("limit", [0.0, 1.5])
@pytest.mark.parametrize("padded_counts", [None, (8, 8, 8)])
def test_eager_expert_values_and_every_gradient_match_separate_weights(dtype, counts, limit, padded_counts):
    _compare_oracle("cpu", dtype, counts, limit=limit, padded_counts=padded_counts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for original BF16 linear kernels")
@pytest.mark.parametrize("dim,inter_dim", [(64, 128), (5120, 2304)])
@pytest.mark.parametrize("padded_counts", [None, (16, 16, 16)])
def test_cuda_eager_expert_values_and_gradients_match_original_layout(dim, inter_dim, padded_counts):
    _compare_oracle(
        "cuda", torch.bfloat16, (7, 0, 13), dim=dim, inter_dim=inter_dim, limit=10.0, padded_counts=padded_counts
    )


@pytest.mark.parametrize("module", [GroupedExperts, GroupedExpertsDeepEP])
@pytest.mark.parametrize(
    "change", [{"expert_activation": "relu2"}, {"expert_bias": True}, {"apply_router_weight_after_down": True}]
)
def test_unsupported_variants_fail_before_dispatcher_construction(module, change):
    with pytest.raises(ValueError, match="torch_linear.*requires SwiGLU"):
        module(replace(_config(), **change), BackendConfig(experts="torch_linear", dispatcher="torch"))


@pytest.mark.parametrize("all_masked", [False, True])
def test_local_grouped_experts_dispatch_and_padding_preserve_gradients(all_masked):
    torch.manual_seed(723)
    config = _config(dtype=torch.float32)
    model = GroupedExperts(config, BackendConfig(experts="torch_linear", dispatcher="torch"))
    oracle = nn.ModuleList(_SeparateExpert(config, "cpu", torch.float32) for _ in range(3))
    with torch.no_grad():
        model.gate_and_up_projs.copy_(torch.stack([torch.cat((e.w1.weight.T, e.w3.weight.T), -1) for e in oracle]))
        model.down_projs.copy_(torch.stack([e.w2.weight.T for e in oracle]))
    hidden = torch.randn(4, config.dim, requires_grad=True)
    weights = torch.rand(4, 2, requires_grad=True)
    indices = torch.tensor([[2, 0], [1, 2], [0, 1], [2, 0]])
    mask = torch.tensor([not all_masked, False, not all_masked, not all_masked])
    expected = torch.zeros_like(hidden, dtype=torch.float32)
    for expert_id, expert in enumerate(oracle):
        row, slot = torch.where((indices == expert_id) & mask[:, None])
        expected[row] += expert(hidden[row], weights[row, slot, None])
    actual = model(hidden, mask, weights, indices)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    for value in (hidden, weights, model.gate_and_up_projs, model.down_projs):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        if all_masked:
            assert torch.count_nonzero(value.grad) == 0
    assert torch.count_nonzero(weights.grad[~mask]) == 0

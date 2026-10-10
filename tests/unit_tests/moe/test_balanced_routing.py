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

"""CPU coverage for balanced dispatch that retains learned router work."""

import copy

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import FakeBalancedGate, Gate, MoE


@pytest.fixture(autouse=True)
def _eager_cpu_kernels():
    # Exercise real expert dispatch/autograd without compiling unrelated activation kernels.
    with torch.compiler.set_stance("force_eager"):
        yield


def _build_moe(*, score_func="softmax", dtype=torch.float32, force_balance=True, topk=2, noise=0.0, static=False):
    config = MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=topk,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func=score_func,
        route_scale=1.7,
        dim=8,
        inter_dim=16,
        moe_inter_dim=16,
        norm_topk_prob=True,
        router_bias=True,
        force_e_score_correction_bias=True,
        dtype=dtype,
    )
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=force_balance,
        fake_gate_noise=noise,
        benchmark_static_routing=static,
    )
    moe = MoE(config, backend)
    with torch.no_grad():
        for parameter in moe.parameters():
            parameter.normal_(std=0.1)
    return moe


def _assert_balanced(indices: torch.Tensor, n_experts: int):
    """Check assignment counts and distinct expert selections.

    Args:
        indices: Expert IDs of shape [tokens, activated_experts].
        n_experts: Number of routed experts.
    """
    counts = torch.bincount(indices.flatten(), minlength=n_experts)
    assert counts.max() - counts.min() <= 1
    if indices.numel() % n_experts == 0:
        assert (counts == indices.numel() // n_experts).all()
    sorted_indices = indices.sort(dim=-1).values
    assert (sorted_indices[:, 1:] != sorted_indices[:, :-1]).all()
    assert ((indices >= 0) & (indices < n_experts)).all()


@pytest.mark.parametrize("n_tokens,topk", [(0, 2), (1, 1), (3, 2), (8, 2), (5, 4)])
def test_assignments_are_balanced_and_distinct(n_tokens, topk):
    moe = _build_moe(topk=topk)
    indices = torch.zeros(n_tokens, topk, dtype=torch.int64)
    actual = moe._maybe_balance_routing(indices, torch.randn(n_tokens, moe.dim))
    assert actual.shape == indices.shape
    assert actual.dtype == indices.dtype
    _assert_balanced(actual, moe.n_routed_experts)


def test_default_routing_is_unchanged():
    moe = _build_moe(force_balance=False)
    indices = torch.tensor([[3, 1], [2, 0]])
    assert moe._maybe_balance_routing(indices, torch.randn(2, moe.dim)) is indices
    assert isinstance(moe.gate, Gate)


@pytest.mark.parametrize("score_func", ["softmax", "sigmoid", "sigmoid_with_bias", "softmax_with_bias", "sqrtsoftplus"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("noise", [0.0, 0.3])
def test_real_expert_dispatch_retains_learned_weights_and_gate_gradients(score_func, dtype, noise):
    torch.manual_seed(123)
    moe = _build_moe(score_func=score_func, dtype=dtype, noise=noise)
    reference_gate = copy.deepcopy(moe.gate)
    hidden_states = torch.randn(1, 7, moe.dim, dtype=dtype, requires_grad=True)
    reference_weights, _, _ = reference_gate(
        hidden_states.detach().view(-1, moe.dim),
        torch.ones(7, dtype=torch.bool),
        None,
    )
    dispatch = {}

    def capture_dispatch(module, args):
        weights, indices = args[2:4]
        weights.retain_grad()
        dispatch["weights"] = weights
        dispatch["indices"] = indices.detach().clone()

    handle = moe.experts.register_forward_pre_hook(capture_dispatch)
    try:
        output = moe(hidden_states)
    finally:
        handle.remove()
    assert torch.isfinite(output).all()
    assert torch.equal(dispatch["weights"], reference_weights)
    if noise == 0.0:
        _assert_balanced(dispatch["indices"], moe.n_routed_experts)
    else:
        assert (dispatch["indices"].sort(dim=-1).values.diff(dim=-1) > 0).all()
    output.backward(torch.randn_like(output))
    reference_weights.backward(dispatch["weights"].grad)
    for name, parameter in moe.gate.named_parameters():
        expected = dict(reference_gate.named_parameters())[name].grad
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
        # Identical learned gate and upstream weight gradient: exact equality is expected.
        torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
    assert torch.isfinite(hidden_states.grad).all()


@pytest.mark.parametrize("noise", [0.0, 0.3])
def test_activation_checkpointing_preserves_outputs_and_gradients(noise):
    torch.manual_seed(42)
    eager = _build_moe(noise=noise)
    recomputed = copy.deepcopy(eager)
    hidden_states = torch.randn(1, 6, eager.dim)
    x_eager = hidden_states.clone().requires_grad_()
    x_recomputed = hidden_states.clone().requires_grad_()
    actual = checkpoint(recomputed, x_recomputed, use_reentrant=False)
    expected = eager(x_eager)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(x_recomputed.grad, x_eager.grad, rtol=0, atol=0)
    for actual_parameter, expected_parameter in zip(recomputed.parameters(), eager.parameters()):
        torch.testing.assert_close(actual_parameter.grad, expected_parameter.grad, rtol=0, atol=0)


def test_balanced_gate_preserves_learned_router_state_dict():
    balanced = _build_moe()
    learned = _build_moe(force_balance=False)
    assert isinstance(balanced.gate, Gate)
    assert isinstance(balanced.balanced_gate, FakeBalancedGate)
    assert not list(balanced.balanced_gate.parameters())
    assert balanced.state_dict().keys() == learned.state_dict().keys()
    balanced.load_state_dict(learned.state_dict(), strict=True)


def test_static_balanced_dispatch_retains_gate_gradients():
    moe = _build_moe(static=True)
    for _ in range(2):
        output = moe(torch.randn(1, 8, moe.dim))
        output.backward(torch.randn_like(output))
        assert torch.isfinite(moe.gate.weight.grad).all()
        assert moe.gate.weight.grad.abs().sum() > 0
        moe.zero_grad(set_to_none=True)


@pytest.mark.parametrize("overrides", [{"fake_balanced_gate": False}, {"fake_gate_noise": 0.1}])
def test_incompatible_static_routing_settings_are_rejected(overrides):
    settings = {"fake_balanced_gate": True, "benchmark_static_routing": True, **overrides}
    with pytest.raises(ValueError, match="requires fake_balanced_gate=True"):
        BackendConfig(**settings)

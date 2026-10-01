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


def _build_moe(*, score_func="softmax", dtype=torch.float32, force_balance=True, topk=2):
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
        force_balanced_routing=force_balance,
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
    actual = moe._maybe_balance_routing(indices)
    assert actual.shape == indices.shape
    assert actual.dtype == indices.dtype
    _assert_balanced(actual, moe.n_routed_experts)


def test_default_routing_is_unchanged():
    moe = _build_moe(force_balance=False)
    indices = torch.tensor([[3, 1], [2, 0]])
    assert moe._maybe_balance_routing(indices) is indices
    assert isinstance(moe.gate, Gate)


@pytest.mark.parametrize("score_func", ["softmax", "sigmoid", "sigmoid_with_bias", "softmax_with_bias", "sqrtsoftplus"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_real_expert_dispatch_retains_learned_weights_and_gate_gradients(score_func, dtype):
    torch.manual_seed(123)
    moe = _build_moe(score_func=score_func, dtype=dtype)
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
    _assert_balanced(dispatch["indices"], moe.n_routed_experts)
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


def test_activation_checkpointing_preserves_outputs_and_gradients():
    torch.manual_seed(42)
    eager = _build_moe()
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


def test_fake_gate_still_omits_learned_router():
    moe = _build_moe()
    fake = MoE(moe.experts.config, BackendConfig(fake_balanced_gate=True, experts="torch", dispatcher="torch"))
    assert isinstance(fake.gate, FakeBalancedGate)
    assert not list(fake.gate.parameters())


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"fake_balanced_gate": True}, "mutually exclusive"),
        ({"fake_gate_noise": 0.1}, "fake_gate_noise=0.0"),
        ({"benchmark_static_routing": True}, "requires fake_balanced_gate=True"),
    ],
)
def test_incompatible_benchmark_settings_are_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        BackendConfig(force_balanced_routing=True, **overrides)

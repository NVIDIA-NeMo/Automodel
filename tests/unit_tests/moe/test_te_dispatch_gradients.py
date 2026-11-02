# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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


"""CPU coverage for preserving routing gradients at the TE dispatcher boundary."""

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig


@pytest.fixture
def moe_config():
    return MoEConfig(
        n_routed_experts=8,
        n_shared_experts=2,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.1,
        aux_loss_coeff=0.01,
        score_func="softmax",
        route_scale=1.0,
        dim=128,
        inter_dim=256,
        moe_inter_dim=256,
        norm_topk_prob=False,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        activation_alpha=1.702,
        activation_limit=7.0,
        dtype=torch.bfloat16,
    )


@pytest.fixture
def fake_legacy_te(monkeypatch):
    """Exercise Automodel's TE integration on CPU, including coupled autograd outputs."""

    class GroupedLinear(torch.nn.Module):
        def __init__(self, *, num_gemms, in_features, out_features, bias, params_dtype, device):
            super().__init__()
            self.num_gemms = num_gemms
            self.weight = torch.nn.Parameter(torch.randn(out_features, in_features, dtype=params_dtype) * 0.02)

        def forward(self, values, splits):
            assert isinstance(splits, list)
            return torch.nn.functional.linear(values, self.weight)

    class Padding(torch.nn.Module):
        def __init__(self, num_gemms, align_size=None):
            super().__init__()
            self.num_gemms = num_gemms
            self.align_size = align_size
            self.calls = 0

        def forward(self, values, splits):
            self.calls += 1
            return values, splits

    class Unpadding(Padding):
        def forward(self, values, splits):
            self.calls += 1
            return values

    class CoupledDispatch(torch.autograd.Function):
        @staticmethod
        def forward(ctx, hidden, probabilities):
            ctx.set_materialize_grads(False)
            return hidden.clone(), probabilities[:, 0].clone()

        @staticmethod
        def backward(ctx, hidden_grad, probability_grad):
            return hidden_grad, None if probability_grad is None else probability_grad.unsqueeze(-1)

    class Dispatcher:
        def __init__(self, num_local_experts, **kwargs):
            self.num_local_experts = num_local_experts

        def token_permutation2(self, hidden_states, token_probs, **kwargs):
            hidden, probs = CoupledDispatch.apply(hidden_states, token_probs)
            assert probs.requires_grad  # Even when the input probabilities are frozen.
            return hidden, [hidden.shape[0]] + [0] * (self.num_local_experts - 1), probs

        def token_unpermutation(self, values):
            return values

    state = Mock()
    state.is_fp8_enabled.return_value = False
    te = ModuleType("transformer_engine.pytorch")
    te.GroupedLinear, te.Fp8Padding, te.Fp8Unpadding = GroupedLinear, Padding, Unpadding
    quantization = ModuleType("transformer_engine.pytorch.quantization")
    quantization.FP8GlobalStateManager = state
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", te)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.quantization", quantization)
    monkeypatch.setattr("nemo_automodel.components.models.common.utils._patch_te_modules", lambda: None)
    monkeypatch.setattr("nemo_automodel.components.moe.experts.MoEFlexTokenDispatcher", Dispatcher)
    return state


@pytest.mark.parametrize("train_router", [False, True])
def test_te_dispatch_preserves_routing_gradient_requirement(moe_config, fake_legacy_te, train_router):
    from nemo_automodel.components.moe.experts import GroupedExpertsTE

    backend = BackendConfig(experts="te", dispatcher="deepep")
    experts = GroupedExpertsTE(moe_config, backend=backend)
    for ep_size in (1, 2):
        if ep_size == 2:
            mesh = Mock()
            mesh.size.return_value = ep_size
            mesh.get_local_rank.return_value = 0
            mesh.mesh_dim_names = ("ep",)
            experts.init_token_dispatcher(mesh)
        for layer in (experts.gate_up_linear, experts.down_linear):
            assert layer.num_gemms == moe_config.n_routed_experts // ep_size
        assert experts.fp8_padding.num_gemms == moe_config.n_routed_experts // ep_size

    observed = []

    def activation(values, probabilities):
        observed.append(probabilities.requires_grad)
        return values.chunk(2, -1)[0] * probabilities

    experts.expert_activation = activation
    x = torch.randn(3, moe_config.dim, dtype=moe_config.dtype, requires_grad=True)
    weights = torch.rand(3, 1, dtype=moe_config.dtype, requires_grad=train_router)
    experts(x, torch.ones(3, dtype=torch.bool), weights, torch.zeros(3, 1, dtype=torch.long)).sum().backward()
    assert observed == [train_router]
    assert x.grad is not None
    assert (weights.grad is not None) == train_router

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

"""Transformer Engine experts under NVFP4: at most 64 experts per rank."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from nemo_automodel.components.models.common.utils import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExpertsTE


class _FakeGroupedLinear(torch.nn.Module):
    """CPU stand-in for TE ``GroupedLinear``: one ``weight{i}`` per expert."""

    def __init__(self, *, num_gemms, in_features, out_features, bias, params_dtype, device, use_grouped_tensor=False):
        super().__init__()
        self.num_gemms = num_gemms
        self.use_bias = bias
        for index in range(num_gemms):
            weight = torch.nn.Parameter(torch.empty(out_features, in_features, dtype=params_dtype))
            self.register_parameter(f"weight{index}", weight)


class _FakePadding(torch.nn.Module):
    def __init__(self, num_gemms, align_size=None):
        super().__init__()


class _FakeMesh:
    """Expert-parallel mesh of ``size`` ranks seen from rank 0."""

    def __init__(self, size: int) -> None:
        self._size = size

    def get_local_rank(self) -> int:
        return 0

    def size(self) -> int:
        return self._size

    def get_group(self) -> None:
        return None


@pytest.fixture
def fake_te(monkeypatch):
    """Stand in for Transformer Engine and the token dispatcher so the experts are built on CPU."""
    te = ModuleType("transformer_engine.pytorch")
    te.GroupedLinear, te.Fp8Padding, te.Fp8Unpadding = _FakeGroupedLinear, _FakePadding, _FakePadding
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", te)
    monkeypatch.setattr("nemo_automodel.components.moe.experts.is_te_min_version", lambda version: True)
    monkeypatch.setattr("nemo_automodel.components.models.common.utils._patch_te_modules", lambda: None)
    monkeypatch.setattr(GroupedExpertsTE, "_normalize_moe_mesh", lambda self, mesh: mesh)
    monkeypatch.setattr(
        "nemo_automodel.components.moe.experts.MoEFlexTokenDispatcher", lambda **kwargs: SimpleNamespace(**kwargs)
    )


def _moe_config(n_routed_experts: int) -> MoEConfig:
    return MoEConfig(
        n_routed_experts=n_routed_experts,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=8,
        inter_dim=4,
        moe_inter_dim=4,
        norm_topk_prob=False,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        dtype=torch.float32,
    )


@pytest.mark.parametrize(
    ("te_block", "n_experts", "ep_size", "allowed"),
    [
        ({"te_nvfp4": {}}, 96, 1, False),
        ({"te_nvfp4": {}}, 130, 2, False),
        ({"te_nvfp4": {}}, 128, 2, True),
        ({"te_fp8": {"recipe": "mxfp8"}}, 96, 1, True),
        ({}, 96, 1, True),
    ],
)
def test_only_nvfp4_limits_the_experts_per_rank(fake_te, te_block, n_experts, ep_size, allowed):
    backend = BackendConfig(experts="te", dispatcher="deepep", **te_block)
    experts = GroupedExpertsTE(_moe_config(n_experts), backend=backend)

    if not allowed:
        with pytest.raises(ValueError, match="te_nvfp4 supports at most 64 experts per rank"):
            experts.init_token_dispatcher(_FakeMesh(ep_size))
        return

    experts.init_token_dispatcher(_FakeMesh(ep_size))
    assert experts.gate_up_linear.num_gemms == n_experts // ep_size
    assert experts.token_dispatcher.num_local_experts == n_experts // ep_size

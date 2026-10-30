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

"""Configuration guards for the fused expert backend."""

import dataclasses
from unittest.mock import Mock

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.dist_moe_experts import GroupedExpertsDistMoE
from nemo_automodel.shared.dist_moe import DistMoEConfig


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_invalid_token_capacity(capacity: bool | int | float) -> None:
    """Reject invalid static extents before any distributed allocation."""
    with pytest.raises(ValueError, match="positive integer"):
        DistMoEConfig(max_tokens_per_rank=capacity)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_invalid_scratch_capacity(value: float) -> None:
    """Reject invalid scratch planning bounds."""
    with pytest.raises(ValueError, match="finite"):
        DistMoEConfig(device_scratch_capacity_factor=value)


def test_backend_requires_token_capacity() -> None:
    """Selecting the fused backend requires an explicit physical extent."""
    with pytest.raises(ValueError, match="max_tokens_per_rank"):
        BackendConfig(dispatcher="dist_moe", experts="torch_mm")


def test_backend_resolves_typed_configuration() -> None:
    """The recipe boundary resolves the optional configuration into its type."""
    selected = BackendConfig(dispatcher="dist_moe", experts="torch_mm", dist_moe={"max_tokens_per_rank": 32})
    assert selected.dist_moe.max_tokens_per_rank == 32


@pytest.mark.parametrize("limit", [0.0, 0.0625, 10.0])
def test_accepts_pre_activation_clamp(limit: float) -> None:
    """Construct clamped experts without changing canonical parameter ownership."""
    config = MoEConfig(
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
        dim=256,
        inter_dim=256,
        moe_inter_dim=256,
        norm_topk_prob=False,
        swiglu_limit=limit,
    )
    backend = BackendConfig(dispatcher="dist_moe", experts="torch_mm", dist_moe={"max_tokens_per_rank": 32})
    module = GroupedExpertsDistMoE(config, backend)
    assert module.gate_and_up_projs.shape == (4, 256, 512)
    assert module.down_projs.shape == (4, 256, 256)
    with pytest.raises(ValueError, match="after the activation"):
        GroupedExpertsDistMoE(dataclasses.replace(config, swiglu_clamp_after_activation=True), backend)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        GroupedExpertsDistMoE(dataclasses.replace(config, swiglu_limit=-1.0), backend)
    with pytest.raises(ValueError, match="biased experts"):
        GroupedExpertsDistMoE(dataclasses.replace(config, swiglu_limit=0.0, expert_bias=True), backend)


@pytest.mark.parametrize("budget", [-1, True, 1.5])
def test_invalid_activation_byte_budget(budget: bool | int | float) -> None:
    """Byte budgets reject invalid values before device initialization."""
    with pytest.raises(ValueError, match="activation_slot_bytes"):
        DistMoEConfig(activation_slot_bytes=budget)


def test_conflicting_activation_policies() -> None:
    """An explicit byte budget cannot silently override a capacity factor."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        DistMoEConfig(activation_slot_bytes=1024, activation_slot_capacity_factor=1.0)


@pytest.mark.parametrize("sms", [0, -1, True, 2.5])
def test_invalid_sm_budget(sms: bool | int | float) -> None:
    """SM budgets must be positive counts."""
    with pytest.raises(ValueError, match="num_sms"):
        DistMoEConfig(num_sms=sms)


def test_fast_math_requires_low_precision() -> None:
    """Reject a precision flag that cannot affect BF16 computation."""
    with pytest.raises(ValueError, match="requires mxfp8"):
        DistMoEConfig(fast_math=True)


@pytest.mark.parametrize("capacity", [1.0, float("nan"), float("inf")])
def test_invalid_host_capacity(capacity: float) -> None:
    """Host plus device capacity must cover the device allocation."""
    with pytest.raises(ValueError, match="host_scratch_capacity_factor"):
        DistMoEConfig(device_scratch_capacity_factor=2.0, host_scratch_capacity_factor=capacity)


@pytest.mark.parametrize("limit, activation", [(0.0, "swiglu"), (0.0625, "swiglu_deepseek"), (10.0, "swiglu_deepseek")])
def test_context_receives_model_activation(monkeypatch: pytest.MonkeyPatch, limit: float, activation: str) -> None:
    """The optional backend receives the model's clamp with ordinary SiLU slope."""
    package = Mock()
    monkeypatch.setattr("nemo_automodel.shared.dist_moe.safe_import", lambda *args, **kwargs: (True, package))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 0))
    settings = DistMoEConfig(max_tokens_per_rank=32)
    settings.build(
        group=Mock(spec=torch.distributed.ProcessGroup),
        device=torch.device("cuda", 0),
        hidden_dim=256,
        intermediate_dim=256,
        num_experts=4,
        top_k=2,
        num_layers=4,
        swiglu_limit=limit,
    )
    kwargs = package.Config.call_args.kwargs
    assert kwargs["activation"] == activation
    assert kwargs["swiglu_alpha"] == 1.0
    assert kwargs["swiglu_limit"] == limit
    assert kwargs["max_moe_layers_per_activation_slot"] == 4

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

import pytest

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


def test_rejects_different_clamped_swiglu_semantics() -> None:
    """The upstream plus-one clamped activation cannot implement this clamp."""
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
        swiglu_limit=7.0,
    )
    backend = BackendConfig(dispatcher="dist_moe", experts="torch_mm", dist_moe={"max_tokens_per_rank": 32})
    with pytest.raises(ValueError, match="clamped formula"):
        GroupedExpertsDistMoE(config, backend)
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

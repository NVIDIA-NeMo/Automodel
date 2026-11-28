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

"""Unit tests for per-layer MoE shapes on ``MoEConfig``."""

from dataclasses import replace

import pytest
import torch

from nemo_automodel.components.moe.config import MoEConfig


def _config(**overrides) -> MoEConfig:
    fields = dict(
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="sigmoid",
        route_scale=1.0,
        dim=16,
        inter_dim=32,
        moe_inter_dim=8,
        norm_topk_prob=False,
        dtype=torch.float32,
    )
    fields.update(overrides)
    return MoEConfig(**fields)


def test_uniform_config_is_its_own_layer_config():
    config = _config()

    assert config.layer_configs is None
    assert config.for_layer(0) is config
    assert config.for_layer("7") is config
    assert config.routed_expert_counts == [4]


def test_per_layer_configs_resolve_by_layer_index():
    config = _config()
    config.layer_configs = {
        1: replace(config, n_routed_experts=8, moe_inter_dim=12),
        3: replace(config, n_routed_experts=6, n_activated_experts=3),
        5: replace(config, n_routed_experts=8),
    }

    assert config.for_layer(1).n_routed_experts == 8
    assert config.for_layer(1).moe_inter_dim == 12
    assert config.for_layer("3").n_activated_experts == 3
    assert config.routed_expert_counts == [6, 8]
    with pytest.raises(KeyError):
        config.for_layer(0)

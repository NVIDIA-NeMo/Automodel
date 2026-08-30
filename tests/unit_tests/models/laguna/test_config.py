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

import pytest

from nemo_automodel.components.models.laguna.config import LagunaConfig


def test_laguna_config_clips_full_checkpoint_layer_lists_for_proxy_models():
    cfg = LagunaConfig(
        num_hidden_layers=2,
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        num_attention_heads_per_layer=[2, 4, 2],
        gating_types=["per_head", "per_head", "per_head"],
        mlp_layer_types=["dense", "sparse", "sparse"],
    )

    assert cfg.layer_types == ["full_attention", "sliding_attention"]
    assert cfg.num_attention_heads_per_layer == [2, 4]
    assert cfg.gating_types == ["per_head", "per_head"]
    assert cfg.mlp_only_layers == [0]


def test_laguna_config_rejects_short_layer_lists():
    with pytest.raises(ValueError, match="layer_types"):
        LagunaConfig(num_hidden_layers=3, layer_types=["full_attention"])


def test_laguna_config_derives_swa_rope_from_nested_rope_parameters():
    cfg = LagunaConfig(
        rope_parameters={
            "full_attention": {"rope_type": "yarn", "rope_theta": 500000.0, "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
        },
    )

    assert cfg.swa_rope_parameters == {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 1.0,
    }

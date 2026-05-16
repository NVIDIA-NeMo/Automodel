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

from nemo_automodel.components.models.mimo_v2_flash.config import MiMoV2FlashConfig


class TestMiMoV2FlashConfig:
    def test_model_type(self):
        cfg = MiMoV2FlashConfig()
        assert cfg.model_type == "mimo_v2_flash"

    def test_num_local_experts_alias(self):
        """attribute_map should expose num_local_experts as n_routed_experts."""
        cfg = MiMoV2FlashConfig(n_routed_experts=16, num_hidden_layers=4)
        assert cfg.num_local_experts == 16

    def test_default_kv_heads_falls_back_to_attention_heads(self):
        cfg = MiMoV2FlashConfig(num_attention_heads=8, num_key_value_heads=None, num_hidden_layers=2)
        assert cfg.num_key_value_heads == 8

    def test_rms_norm_eps_defaults_to_layernorm_epsilon(self):
        cfg = MiMoV2FlashConfig(layernorm_epsilon=1.5e-5, rms_norm_eps=None, num_hidden_layers=2)
        assert cfg.rms_norm_eps == 1.5e-5

    def test_rms_norm_eps_explicit_override(self):
        cfg = MiMoV2FlashConfig(layernorm_epsilon=1e-5, rms_norm_eps=2e-6, num_hidden_layers=2)
        assert cfg.rms_norm_eps == 2e-6

    def test_routed_scaling_factor_none_defaults_to_one(self):
        cfg = MiMoV2FlashConfig(routed_scaling_factor=None, num_hidden_layers=2)
        assert cfg.routed_scaling_factor == 1.0

    def test_routed_scaling_factor_explicit(self):
        cfg = MiMoV2FlashConfig(routed_scaling_factor=2.5, num_hidden_layers=2)
        assert cfg.routed_scaling_factor == 2.5

    def test_hybrid_layer_pattern_from_block_size(self):
        """When only hybrid_block_size is given, every block_size'th layer becomes full-attention."""
        cfg = MiMoV2FlashConfig(num_hidden_layers=6, hybrid_block_size=3)
        # (i+1) % 3 == 0 -> 0 (full-attention). So layers [2, 5] are full.
        assert cfg.hybrid_layer_pattern == [1, 1, 0, 1, 1, 0]
        assert cfg.layer_types == [
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]

    def test_hybrid_layer_pattern_default(self):
        """Without block_size or pattern, layer 0, every 6th layer, and the last are full-attention."""
        cfg = MiMoV2FlashConfig(num_hidden_layers=8)
        # i==0 or i%6==0 or i==7 → 0 (full)
        assert cfg.hybrid_layer_pattern == [0, 1, 1, 1, 1, 1, 0, 0]

    def test_explicit_hybrid_layer_pattern_passthrough(self):
        pattern = [0, 1, 0, 1]
        cfg = MiMoV2FlashConfig(num_hidden_layers=4, hybrid_layer_pattern=pattern)
        assert cfg.hybrid_layer_pattern == pattern
        assert cfg.layer_types == [
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ]

    def test_moe_layer_freq_default_makes_layer_0_dense(self):
        cfg = MiMoV2FlashConfig(num_hidden_layers=4)
        # default: [0] + [1]*(N-1)
        assert cfg.moe_layer_freq == [0, 1, 1, 1]

    def test_moe_layer_freq_explicit_passthrough(self):
        cfg = MiMoV2FlashConfig(num_hidden_layers=3, moe_layer_freq=[1, 0, 1])
        assert cfg.moe_layer_freq == [1, 0, 1]

    def test_rope_parameters_populated(self):
        cfg = MiMoV2FlashConfig(rope_theta=4242.0, partial_rotary_factor=0.5, num_hidden_layers=2)
        assert cfg.rope_parameters == {
            "rope_theta": 4242.0,
            "partial_rotary_factor": 0.5,
            "rope_type": "default",
        }

    def test_keys_to_ignore_at_inference(self):
        assert MiMoV2FlashConfig.keys_to_ignore_at_inference == ["past_key_values"]

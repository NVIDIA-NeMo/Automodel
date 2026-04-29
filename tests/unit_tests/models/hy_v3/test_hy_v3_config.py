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

"""Unit tests for ``HYV3Config``."""

import pytest
from transformers import PretrainedConfig

from nemo_automodel.components.models.hy_v3.config import HYV3Config


class TestDefaults:
    def test_model_type(self):
        assert HYV3Config.model_type == "hy_v3"

    def test_inherits_pretrained_config(self):
        cfg = HYV3Config()
        assert isinstance(cfg, PretrainedConfig)

    def test_default_attributes_match_295b(self):
        cfg = HYV3Config()
        # Architecture defaults from the published Hy3-preview spec.
        assert cfg.vocab_size == 129280
        assert cfg.hidden_size == 4096
        assert cfg.intermediate_size == 1536
        assert cfg.moe_intermediate_size == 1536
        assert cfg.num_hidden_layers == 80
        assert cfg.num_attention_heads == 64
        assert cfg.num_key_value_heads == 8
        assert cfg.head_dim == 128
        assert cfg.num_experts == 192
        assert cfg.num_shared_experts == 1
        assert cfg.num_experts_per_tok == 8
        assert cfg.first_k_dense_replace == 1
        assert cfg.max_position_embeddings == 262144
        assert cfg.rope_theta == 11158840.0
        assert cfg.rms_norm_eps == 1e-6
        assert cfg.attention_bias is False
        assert cfg.hidden_act == "silu"
        # torch_dtype is auto-coerced by PretrainedConfig (deprecated -> dtype);
        # accept either the string we set or whatever the base class normalizes to.
        assert cfg.torch_dtype in ("bfloat16", None) or str(cfg.torch_dtype).endswith("bfloat16")
        assert cfg.tie_word_embeddings is False
        assert cfg.moe_router_enable_expert_bias is True

    def test_keys_to_ignore_at_inference(self):
        assert HYV3Config.keys_to_ignore_at_inference == ["past_key_values"]


class TestOverrides:
    def test_override_attention_dims(self):
        cfg = HYV3Config(
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
            hidden_size=512,
        )
        assert cfg.num_attention_heads == 8
        assert cfg.num_key_value_heads == 2
        assert cfg.head_dim == 64
        assert cfg.hidden_size == 512

    def test_override_moe_routing(self):
        cfg = HYV3Config(num_experts=64, num_experts_per_tok=4, num_shared_experts=2, router_scaling_factor=1.5)
        assert cfg.num_experts == 64
        assert cfg.num_experts_per_tok == 4
        assert cfg.num_shared_experts == 2
        assert cfg.router_scaling_factor == 1.5

    def test_truncated_layer_count(self):
        cfg = HYV3Config(num_hidden_layers=4)
        assert cfg.num_hidden_layers == 4

    def test_first_k_dense_replace(self):
        cfg = HYV3Config(first_k_dense_replace=3)
        assert cfg.first_k_dense_replace == 3

    def test_router_flags(self):
        cfg = HYV3Config(route_norm=True, moe_router_enable_expert_bias=False)
        assert cfg.route_norm is True
        assert cfg.moe_router_enable_expert_bias is False

    def test_rope_overrides(self):
        cfg = HYV3Config(rope_theta=500000.0, max_position_embeddings=4096)
        assert cfg.rope_theta == 500000.0
        assert cfg.max_position_embeddings == 4096

    def test_rope_scaling_dict(self):
        scaling = {"factor": 8.0, "rope_type": "yarn"}
        cfg = HYV3Config(rope_scaling=scaling)
        assert cfg.rope_scaling == scaling

    def test_token_ids(self):
        cfg = HYV3Config(pad_token_id=0, bos_token_id=10, eos_token_id=11)
        assert cfg.pad_token_id == 0
        assert cfg.bos_token_id == 10
        assert cfg.eos_token_id == 11

    def test_super_init_kwargs_accepted(self):
        # Verify that PretrainedConfig-recognized kwargs (here: use_cache,
        # tie_word_embeddings) flow through __init__ without raising.
        HYV3Config(use_cache=False, tie_word_embeddings=True)

    def test_extra_kwargs_pass_through_super_init(self):
        # PretrainedConfig **kwargs in newer transformers no longer attaches
        # arbitrary fields to the instance, but the call should still succeed.
        cfg = HYV3Config(custom_field="abc")
        assert isinstance(cfg, HYV3Config)


class TestSerialization:
    def test_to_dict_round_trip(self):
        cfg = HYV3Config(num_hidden_layers=4, num_experts=8, hidden_size=256)
        d = cfg.to_dict()
        assert d["model_type"] == "hy_v3"
        assert d["num_hidden_layers"] == 4
        assert d["num_experts"] == 8

        rebuilt = HYV3Config(**{k: v for k, v in d.items() if k != "model_type"})
        assert rebuilt.num_hidden_layers == 4
        assert rebuilt.num_experts == 8
        assert rebuilt.hidden_size == 256

    def test_model_type_class_attribute_not_overridden_by_instance(self):
        cfg = HYV3Config()
        # model_type is a class-level attribute that AutoConfig dispatches on.
        assert cfg.model_type == "hy_v3"
        assert HYV3Config.model_type == "hy_v3"

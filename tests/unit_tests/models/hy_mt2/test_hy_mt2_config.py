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

"""Unit tests for ``HyMT2Config``."""

from transformers import PretrainedConfig

from nemo_automodel.components.models.hy_mt2.config import HyMT2Config


class TestDefaults:
    def test_model_type(self):
        assert HyMT2Config.model_type == "hy_mt2"

    def test_inherits_pretrained_config(self):
        cfg = HyMT2Config()
        assert isinstance(cfg, PretrainedConfig)

    def test_default_attributes_match_30b_a3b(self):
        cfg = HyMT2Config()
        # Architecture defaults from the published Hy-MT2-30B-A3B config.json.
        assert cfg.vocab_size == 120832
        assert cfg.hidden_size == 2048
        assert cfg.intermediate_size == 6912
        assert cfg.moe_intermediate_size == 768
        assert cfg.expert_hidden_dim == 768
        assert cfg.num_hidden_layers == 48
        assert cfg.num_attention_heads == 32
        assert cfg.num_key_value_heads == 4
        assert cfg.head_dim == 128
        assert cfg.num_experts == 128
        assert cfg.num_shared_experts == 1
        assert cfg.num_experts_per_tok == 8
        assert cfg.first_k_dense_replace == 1
        assert cfg.max_position_embeddings == 262144
        assert cfg.rope_theta == 11158840.0
        assert cfg.rms_norm_eps == 1e-5
        assert cfg.attention_bias is False
        assert cfg.hidden_act == "silu"
        assert cfg.qk_norm is True
        assert cfg.route_norm is True
        assert cfg.router_scaling_factor == 2.826
        assert cfg.moe_router_use_sigmoid is True
        assert cfg.moe_router_enable_expert_bias is True
        assert cfg.enable_lm_head_fp32 is True
        assert cfg.enable_attention_fp32_softmax is False
        assert cfg.enable_moe_fp32_combine is False
        assert cfg.tie_word_embeddings is False
        # torch_dtype is auto-coerced by PretrainedConfig in newer transformers
        # (deprecated -> dtype); accept either the string we set or whatever
        # the base class normalizes to.
        assert cfg.torch_dtype in ("bfloat16", None) or str(cfg.torch_dtype).endswith("bfloat16")

    def test_default_token_ids(self):
        cfg = HyMT2Config()
        assert cfg.pad_token_id == 120002
        assert cfg.bos_token_id == 120000
        assert cfg.eos_token_id == 120025

    def test_keys_to_ignore_at_inference(self):
        assert HyMT2Config.keys_to_ignore_at_inference == ["past_key_values"]


class TestOverrides:
    def test_override_attention_dims(self):
        cfg = HyMT2Config(num_attention_heads=8, num_key_value_heads=2, head_dim=64, hidden_size=512)
        assert cfg.num_attention_heads == 8
        assert cfg.num_key_value_heads == 2
        assert cfg.head_dim == 64
        assert cfg.hidden_size == 512

    def test_override_moe_routing(self):
        cfg = HyMT2Config(
            num_experts=64,
            num_experts_per_tok=4,
            num_shared_experts=2,
            router_scaling_factor=1.5,
            route_norm=False,
        )
        assert cfg.num_experts == 64
        assert cfg.num_experts_per_tok == 4
        assert cfg.num_shared_experts == 2
        assert cfg.router_scaling_factor == 1.5
        assert cfg.route_norm is False

    def test_override_router_flavor(self):
        cfg = HyMT2Config(moe_router_use_sigmoid=False, moe_router_enable_expert_bias=False)
        assert cfg.moe_router_use_sigmoid is False
        assert cfg.moe_router_enable_expert_bias is False

    def test_truncated_layer_count(self):
        cfg = HyMT2Config(num_hidden_layers=4)
        assert cfg.num_hidden_layers == 4

    def test_first_k_dense_replace(self):
        cfg = HyMT2Config(first_k_dense_replace=3)
        assert cfg.first_k_dense_replace == 3

    def test_rope_overrides(self):
        cfg = HyMT2Config(rope_theta=500000.0, max_position_embeddings=4096)
        assert cfg.rope_theta == 500000.0
        assert cfg.max_position_embeddings == 4096

    def test_rope_scaling_dict(self):
        scaling = {"factor": 8.0, "rope_type": "yarn"}
        cfg = HyMT2Config(rope_scaling=scaling)
        assert cfg.rope_scaling == scaling

    def test_qk_norm_override(self):
        cfg = HyMT2Config(qk_norm=False)
        assert cfg.qk_norm is False

    def test_lm_head_fp32_override(self):
        cfg = HyMT2Config(enable_lm_head_fp32=False)
        assert cfg.enable_lm_head_fp32 is False

    def test_expert_hidden_dim_override(self):
        cfg = HyMT2Config(expert_hidden_dim=1024)
        assert cfg.expert_hidden_dim == 1024

    def test_token_ids(self):
        cfg = HyMT2Config(pad_token_id=0, bos_token_id=10, eos_token_id=11)
        assert cfg.pad_token_id == 0
        assert cfg.bos_token_id == 10
        assert cfg.eos_token_id == 11

    def test_super_init_kwargs_accepted(self):
        # Verify PretrainedConfig kwargs flow through without raising.
        HyMT2Config(use_cache=False, tie_word_embeddings=True)

    def test_extra_kwargs_pass_through_super_init(self):
        # PretrainedConfig **kwargs in newer transformers no longer attaches
        # arbitrary fields; the call should still succeed.
        cfg = HyMT2Config(custom_field="abc")
        assert isinstance(cfg, HyMT2Config)


class TestSerialization:
    def test_to_dict_round_trip(self):
        cfg = HyMT2Config(num_hidden_layers=4, num_experts=8, hidden_size=256)
        d = cfg.to_dict()
        assert d["model_type"] == "hy_mt2"
        assert d["num_hidden_layers"] == 4
        assert d["num_experts"] == 8

        rebuilt = HyMT2Config(**{k: v for k, v in d.items() if k != "model_type"})
        assert rebuilt.num_hidden_layers == 4
        assert rebuilt.num_experts == 8
        assert rebuilt.hidden_size == 256

    def test_model_type_class_attribute_not_overridden_by_instance(self):
        cfg = HyMT2Config()
        assert cfg.model_type == "hy_mt2"
        assert HyMT2Config.model_type == "hy_mt2"

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

from nemo_automodel.components.models.ling_v2.config import BailingMoeV2Config


def test_default_matches_ling_mini_2_0():
    """The dataclass defaults must reproduce the inclusionAI/Ling-mini-2.0 config.json."""
    cfg = BailingMoeV2Config()
    assert cfg.model_type == "bailing_moe"
    assert cfg.vocab_size == 157184
    assert cfg.hidden_size == 2048
    assert cfg.intermediate_size == 5120
    assert cfg.moe_intermediate_size == 512
    assert cfg.num_hidden_layers == 20
    assert cfg.num_attention_heads == 16
    assert cfg.num_key_value_heads == 4
    assert cfg.head_dim == 128
    assert cfg.num_experts == 256
    assert cfg.num_experts_per_tok == 8
    assert cfg.num_shared_experts == 1
    assert cfg.first_k_dense_replace == 1
    assert cfg.n_group == 8
    assert cfg.topk_group == 4
    assert cfg.score_function == "sigmoid"
    assert cfg.use_qk_norm is True
    assert cfg.moe_router_enable_expert_bias is True
    assert cfg.tie_word_embeddings is False


def test_auto_config_registration():
    """Importing nemo_automodel must register 'bailing_moe' with transformers' AutoConfig."""
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    # Touch the registry to trigger registration.
    import nemo_automodel._transformers.registry  # noqa: F401

    assert "bailing_moe" in CONFIG_MAPPING
    assert CONFIG_MAPPING["bailing_moe"] is BailingMoeV2Config


def test_head_dim_falls_back_to_hidden_over_heads():
    cfg = BailingMoeV2Config(hidden_size=512, num_attention_heads=8, head_dim=0)
    assert cfg.head_dim == 64


def test_overriding_first_k_dense_replace():
    cfg = BailingMoeV2Config(first_k_dense_replace=4, num_hidden_layers=80)
    assert cfg.first_k_dense_replace == 4
    assert cfg.num_hidden_layers == 80


def test_rotary_dim_derives_partial_rotary_factor():
    """Ling-1T's config.json uses ``rotary_dim`` (e.g. 64) instead of
    ``partial_rotary_factor`` — we must derive the latter or layers will
    silently apply full RoPE."""
    cfg = BailingMoeV2Config(head_dim=128, rotary_dim=64)
    assert cfg.partial_rotary_factor == 0.5
    assert cfg.rotary_dim == 64


def test_partial_rotary_factor_preserved_when_rotary_dim_absent():
    cfg = BailingMoeV2Config(head_dim=128, partial_rotary_factor=0.5)
    assert cfg.partial_rotary_factor == 0.5
    assert cfg.rotary_dim is None


def test_rotary_dim_wins_over_partial_rotary_factor():
    """If both fields are present, the explicit dim is authoritative — matches
    HF reference modeling which reads rotary_dim directly when set."""
    cfg = BailingMoeV2Config(head_dim=128, partial_rotary_factor=1.0, rotary_dim=64)
    assert cfg.partial_rotary_factor == 0.5

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

"""KimiK3TextConfig defaults must describe the released checkpoint, so from_config == from_pretrained shape."""

from nemo_automodel.components.models.kimi_k3.config import KimiK3Config, KimiK3TextConfig

# moonshotai/Kimi-K3 config.json (text_config) at snapshot 9f62e4e9fffbd0a83ddd60e1c209d828994b3569,
# cross-checked against the safetensors shapes (q_b_proj [18432, 1536] = 96 heads x 192,
# layers.0.mlp.gate_proj [33792, 7168]) and the tech report's Table 1.
RELEASED_TEXT_CONFIG = {
    "vocab_size": 163840,
    "hidden_size": 7168,
    "intermediate_size": 33792,
    "num_hidden_layers": 93,
    "num_attention_heads": 96,
    "num_key_value_heads": 96,
    "moe_intermediate_size": 3072,
    "num_experts": 896,
    "num_experts_per_token": 16,
    "num_shared_experts": 2,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "routed_expert_hidden_size": 3584,
    "latent_moe_use_norm": True,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "mla_use_nope": True,
    "mla_use_output_gate": True,
    "attn_res_block_size": 12,
    "activation_situ_beta": 4.0,
    "activation_situ_linear_beta": 25.0,
    "num_nextn_predict_layers": 0,
    "rms_norm_eps": 1e-5,
    "max_position_embeddings": 1048576,
    "tie_word_embeddings": False,
}
RELEASED_LINEAR_ATTN = {
    "head_dim": 128,
    "num_heads": 96,
    "short_conv_kernel_size": 4,
    "use_full_rank_gate": True,
    "gate_lower_bound": -5.0,
}


def test_text_config_defaults_match_the_released_checkpoint():
    cfg = KimiK3TextConfig()
    mismatches = {k: (getattr(cfg, k), v) for k, v in RELEASED_TEXT_CONFIG.items() if getattr(cfg, k) != v}
    assert not mismatches, f"defaults differ from moonshotai/Kimi-K3 config.json: {mismatches}"
    for k, v in RELEASED_LINEAR_ATTN.items():
        assert cfg.linear_attn_config[k] == v, k
    full = cfg.linear_attn_config["full_attn_layers"]
    assert full == list(range(4, 93, 4)) + [93]  # every 4th layer plus the final one: 24 gated-MLA layers
    assert len(cfg.linear_attn_config["kda_layers"]) == 69
    assert sum(cfg.is_kda_layer(i) for i in range(cfg.num_hidden_layers)) == 69


def test_mla_projection_shapes_implied_by_defaults():
    """The head count shows up in the MLA projection widths; pin them to the checkpoint tensors."""
    cfg = KimiK3TextConfig()
    q_head_dim = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    assert cfg.num_attention_heads * q_head_dim == 18432  # q_b_proj.weight: [18432, 1536]
    assert cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim) == 24576  # kv_b_proj.weight: [24576, 512]
    assert cfg.num_attention_heads * cfg.v_head_dim == 12288  # o_proj / g_proj: [7168, 12288] / [12288, 7168]
    assert cfg.intermediate_size == 33792  # layers.0.mlp.gate_proj.weight: [33792, 7168]


def test_multimodal_wrapper_defaults_follow_the_text_config():
    assert KimiK3Config().text_config.num_attention_heads == 96
    assert KimiK3Config().text_config.intermediate_size == 33792

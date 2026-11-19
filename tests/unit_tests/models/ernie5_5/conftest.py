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

"""Checkpoint-style configuration shared by the ERNIE 5.5 unit tests."""

from nemo_automodel.components.models.ernie5_5.config import Ernie5_5Config, HFConfigValue

_DECODER_LAYERS = 43
# Decoder layers with dynamic sparse attention (DSA); the others use heavily compressed attention (HCA).
_DSA_LAYERS = (7, 14, 21, 28, 35, 42)
_HCA_RATIO, _DSA_RATIO = 128, -2


def checkpoint_style_config() -> dict[str, object]:
    """Return a fresh ``config.json``-style mapping with the 43-layer ERNIE-5.5-Flash schedule.

    The values are those of the source checkpoint. The mapping is written here instead of copying the
    checkpoint's own file. It carries the fields that ``Ernie5_5Config`` declares and the checkpoint states, and
    three keys that name the class and the dtype. Keys of the checkpoint's file that the class does not declare
    are left out.
    """
    decoder_ratios = [_DSA_RATIO if index in _DSA_LAYERS else _HCA_RATIO for index in range(_DECODER_LAYERS)]
    return {
        "architectures": ["Ernie5_5_MoeForCausalLM"],
        "model_type": "ernie5_5_moe",
        "torch_dtype": "bfloat16",
        # Sizes of the decoder
        "vocab_size": 201216,
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_hidden_layers": _DECODER_LAYERS,
        "max_position_embeddings": 1048576,
        # Layer schedule: one entry per decoder layer, then one for the MTP layer, which is a DSA layer.
        "compress_ratios": [*decoder_ratios, _DSA_RATIO],
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "num_nextn_predict_layers": 1,
        "mtp_shared_last_layer": True,
        # HCA layers
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1024,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "sliding_window": 128,
        "compress_rope_theta": 160000.0,
        "hca_rope_type": "yarn",
        "use_qk_norm": True,
        # Attention variant, shared by both kinds of layer
        "qk_norm_type": "per_head",
        "use_gated_attn": True,
        "gated_attn_act": "sigmoid",
        "gated_attn_use_q_lora": True,
        "use_vha_attention": True,
        "add_full_attention_sink_bias": True,
        "attention_bias": False,
        # DSA layers
        "hybrid_mla_attention": "mqa_dsa",
        "mqa_split_kv_b_proj": True,
        "use_fast_hadamard": True,
        "hybrid_mla_num_attention_heads": 64,
        "hybrid_mla_q_lora_rank": 1024,
        "hybrid_mla_kv_lora_rank": 512,
        "hybrid_mla_qk_nope_head_dim": 192,
        "hybrid_mla_qk_rope_head_dim": 64,
        "hybrid_mla_v_head_dim": 256,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 2048,
        "rope_interleave": True,
        # Rotary position embedding: the main base is stated twice, the YaRN settings belong to the HCA layers.
        "rope_theta": 8000000,
        "rotary_base": 8000000,
        "rope_scaling": {
            "type": "yarn",
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        # Hyper-connections and normalization
        "enable_hyper_connections": True,
        "hc_mult": 4,
        "hc_sinkhorn_iters": 20,
        "hc_eps": 1e-06,
        "rms_norm_eps": 1e-06,
        # Feed-forward and mixture of experts
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "n_routed_experts": 512,
        "n_shared_experts": 1,
        "num_experts_per_tok": 10,
        "moe_intermediate_size": 2048,
        "moe_latent_size": 2048,
        "routed_scaling_factor": 3.0,
        "routed_scaling_factor_learnable": True,
        "scoring_func": "sigmoid",
        "norm_topk_prob": True,
        "n_group": 1,
        "topk_group": 1,
        "moe_split_feature_routing": True,
        "latent_moe_use_norm": True,
        # Output head, tokens, weight tying and the inference default
        "multimax": "lm_head",
        "pad_token_id": 0,
        "tie_word_embeddings": False,
        "use_cache": True,
    }


def checkpoint_config(**overrides: HFConfigValue) -> Ernie5_5Config:
    """Build the configuration of ``checkpoint_style_config``, with ``overrides`` replacing its entries."""
    return Ernie5_5Config(**{**checkpoint_style_config(), **overrides})

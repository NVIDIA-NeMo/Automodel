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

"""Mistral4 configuration for AutoConfig registration.

Needed so that ``AutoConfig.from_pretrained`` can resolve ``model_type: "mistral4"``
in the text_config of Mistral3-wrapped checkpoints, even when the HF transformers
version does not ship a native Mistral4Config (added only in the custom fork).
"""

from transformers import PretrainedConfig


class Mistral4Config(PretrainedConfig):
    model_type = "mistral4"

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=4096,
        intermediate_size=12288,
        moe_intermediate_size=2048,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=32,
        n_shared_experts=1,
        n_routed_experts=128,
        routed_scaling_factor=1.0,
        kv_lora_rank=256,
        q_lora_rank=1024,
        qk_rope_head_dim=64,
        v_head_dim=128,
        qk_nope_head_dim=64,
        n_group=8,
        topk_group=1,
        num_experts_per_tok=4,
        first_k_dense_replace=0,
        norm_topk_prob=True,
        hidden_act="silu",
        max_position_embeddings=1048576,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=11,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_parameters=None,
        rope_interleave=True,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.head_dim = v_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.rope_interleave = rope_interleave
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        if rope_parameters is None:
            rope_parameters = {
                "type": "yarn",
                "rope_theta": 10000.0,
                "factor": 128.0,
                "original_max_position_embeddings": 8192,
                "max_position_embeddings": max_position_embeddings,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale_all_dim": 1.0,
                "mscale": 1.0,
                "llama_4_scaling_beta": 0.1,
            }
        self.rope_parameters = rope_parameters
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

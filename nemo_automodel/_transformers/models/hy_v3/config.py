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

from __future__ import annotations

from transformers import PretrainedConfig


class HYV3Config(PretrainedConfig):
    """Configuration class for Tencent Hy3-preview (295B MoE).

    Architecture:
      - 80 transformer layers; layer 0 is dense, layers 1-79 are MoE
      - MoE: 192 routed experts + 1 shared expert, top-8 activated
      - Sigmoid routing with expert-bias correction (e_score_correction_bias)
      - GQA: 64 Q heads, 8 KV heads, head_dim=128
      - Per-head QK RMSNorm before RoPE
      - 256K context, rope_theta=11158840
    """

    model_type = "hy_v3"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 4096,
        intermediate_size: int = 1536,
        moe_intermediate_size: int = 1536,
        num_hidden_layers: int = 80,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        # MoE routing
        num_experts: int = 192,
        num_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        router_scaling_factor: float = 1.0,
        route_norm: bool = False,
        moe_router_enable_expert_bias: bool = True,
        # Dense layers
        first_k_dense_replace: int = 1,
        # Position encoding
        max_position_embeddings: int = 262144,
        rope_theta: float = 11158840.0,
        rope_scaling: dict | None = None,
        # Standard options
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        hidden_act: str = "silu",
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.router_scaling_factor = router_scaling_factor
        self.route_norm = route_norm
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.first_k_dense_replace = first_k_dense_replace
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.hidden_act = hidden_act
        self.torch_dtype = torch_dtype

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            **kwargs,
        )

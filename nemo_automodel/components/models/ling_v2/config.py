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

"""Configuration for BailingMoeV2 (Ling 2.0 family: Ling-mini, Ling-flash, Ling-1T).

Mirrors the ``BailingMoeV2Config`` shipped in the official HuggingFace checkpoints'
``configuration_bailing_moe_v2.py``.  Registered against ``AutoConfig`` so that
``AutoConfig.from_pretrained(...)`` resolves without ``trust_remote_code``.
"""

from transformers.configuration_utils import PretrainedConfig


class BailingMoeV2Config(PretrainedConfig):
    """Configuration class for the BailingMoeV2 model (Ling 2.0).

    The defaults reflect the ``Ling-mini-2.0`` (16B-A1.4B) variant.  Larger
    variants (``Ling-flash-2.0`` 100B-A6B and ``Ling-1T`` 1T-A50B) override
    sizing knobs but share the same architecture: GQA attention with per-head
    QK-RMSNorm, partial RoPE, sigmoid-routed grouped MoE with shared experts,
    and ``first_k_dense_replace`` dense MLP layers at the start.
    """

    model_type = "bailing_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 157184,
        hidden_size: int = 2048,
        intermediate_size: int = 5120,
        num_hidden_layers: int = 20,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        hidden_act: str = "silu",
        use_qkv_bias: bool = False,
        use_bias: bool = False,
        rms_norm_eps: float = 1e-06,
        tie_word_embeddings: bool = False,
        embedding_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        output_dropout: float = 0.0,
        initializer_range: float = 0.02,
        max_position_embeddings: int = 32768,
        rope_theta: float = 600000.0,
        use_cache: bool = True,
        max_window_layers: int = 20,
        rope_scaling: dict | None = None,
        pad_token_id: int = 156892,
        eos_token_id: int = 156892,
        num_experts: int = 256,
        num_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        n_group: int = 8,
        topk_group: int = 4,
        moe_intermediate_size: int = 512,
        first_k_dense_replace: int = 1,
        head_dim: int = 128,
        output_router_logits: bool = False,
        use_qk_norm: bool = True,
        partial_rotary_factor: float = 1.0,
        num_nextn_predict_layers: int = 0,
        mtp_loss_scaling_factor: float = 0,
        moe_router_enable_expert_bias: bool = True,
        routed_scaling_factor: float = 1.0,
        norm_topk_prob: bool = True,
        score_function: str = "sigmoid",
        rotary_dim: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.use_qkv_bias = use_qkv_bias
        self.use_bias = use_bias
        self.rms_norm_eps = rms_norm_eps
        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.use_cache = use_cache
        self.max_window_layers = max_window_layers
        self.rope_scaling = rope_scaling
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.use_qk_norm = use_qk_norm

        # The Ling 2.0 checkpoints disagree on how to express half-RoPE.  Mini
        # and Flash set ``partial_rotary_factor`` (e.g. 0.5); Ling-1T omits it
        # and uses an explicit ``rotary_dim`` field instead (e.g. 64 with
        # head_dim=128 is also a 0.5 factor).  Prefer the explicit ``rotary_dim``
        # when it is present so that the two layouts produce the same rope_dim.
        if rotary_dim is not None and rotary_dim > 0:
            partial_rotary_factor = float(rotary_dim) / float(self.head_dim)
        self.partial_rotary_factor = partial_rotary_factor
        self.rotary_dim = rotary_dim

        # MoE
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_group = n_group
        self.topk_group = topk_group
        self.moe_intermediate_size = moe_intermediate_size
        self.first_k_dense_replace = first_k_dense_replace
        self.output_router_logits = output_router_logits
        self.moe_router_enable_expert_bias = moe_router_enable_expert_bias
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.score_function = score_function

        # MTP (disabled in all published Ling 2.0 checkpoints; reserved)
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_loss_scaling_factor = mtp_loss_scaling_factor

        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

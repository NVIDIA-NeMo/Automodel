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

from __future__ import annotations

from transformers import PretrainedConfig


class HyMT2Config(PretrainedConfig):
    """Configuration class for Tencent Hy-MT2-30B-A3B (translation MoE).

    Architecture (from tencent/Hy-MT2-30B-A3B config.json):
      - 48 transformer layers; layer 0 is dense, layers 1-47 are MoE
      - MoE: 128 routed experts + 1 shared expert, top-8 activated
      - Sigmoid routing with expert-bias correction (e_score_correction_bias)
        and router_scaling_factor = 2.826
      - route_norm = True (normalize top-k routing weights)
      - GQA: 32 Q heads, 4 KV heads, head_dim=128, hidden_size=2048
      - Per-head Q/K RMSNorm before RoPE (qk_norm)
      - 256K context, rope_theta=11158840
      - vocab_size=120832, dense intermediate_size=6912, moe_intermediate_size=768
      - enable_lm_head_fp32 = True (HF reference upcasts lm_head to fp32)

    Note: the on-disk HF checkpoint declares ``model_type: "hy_v3"`` and
    ``architectures: ["HYV3ForCausalLM"]``. NeMo AutoModel's existing
    ``HYV3Config`` therefore wins ``AutoConfig.from_pretrained``. This class
    is provided for tests and for standalone instantiation; the model code in
    ``model.py`` is duck-typed against ``config.<field>`` and works with either
    config class.
    """

    model_type = "hy_mt2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 120832,
        hidden_size: int = 2048,
        intermediate_size: int = 6912,
        moe_intermediate_size: int = 768,
        expert_hidden_dim: int = 768,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        # MoE routing
        num_experts: int = 128,
        num_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        router_scaling_factor: float = 2.826,
        route_norm: bool = True,
        moe_router_enable_expert_bias: bool = True,
        moe_router_use_sigmoid: bool = True,
        # Dense layers
        first_k_dense_replace: int = 1,
        # Position encoding
        max_position_embeddings: int = 262144,
        rope_theta: float = 11158840.0,
        rope_scaling: dict | None = None,
        # Norm / attention
        rms_norm_eps: float = 1e-5,
        qk_norm: bool = True,
        attention_bias: bool = False,
        hidden_act: str = "silu",
        # FP32 upcast hints (mirroring HF config). NeMo AutoModel wires
        # ``enable_lm_head_fp32`` either via the YAML ``lm_head_precision: float32``
        # (preferred, handled by the MoE parallelizer) or via the in-model
        # cast in ``HyMT2ForCausalLM.forward`` when ``lm_head_precision`` is
        # unset.
        enable_lm_head_fp32: bool = True,
        enable_attention_fp32_softmax: bool = False,
        enable_moe_fp32_combine: bool = False,
        # Standard options
        use_cache: bool = True,
        pad_token_id: int | None = 120002,
        bos_token_id: int = 120000,
        eos_token_id: int = 120025,
        tie_word_embeddings: bool = False,
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.expert_hidden_dim = expert_hidden_dim
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
        self.moe_router_use_sigmoid = moe_router_use_sigmoid
        self.first_k_dense_replace = first_k_dense_replace
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rms_norm_eps = rms_norm_eps
        self.qk_norm = qk_norm
        self.attention_bias = attention_bias
        self.hidden_act = hidden_act
        self.enable_lm_head_fp32 = enable_lm_head_fp32
        self.enable_attention_fp32_softmax = enable_attention_fp32_softmax
        self.enable_moe_fp32_combine = enable_moe_fp32_combine
        self.torch_dtype = torch_dtype

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            **kwargs,
        )

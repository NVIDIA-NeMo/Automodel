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

"""AutoModel-owned configuration for Tencent HY V4.

The class intentionally does not inherit from Transformers' HY V4 config. It
keeps checkpoint parsing stable while the model implementation remains wholly
owned by AutoModel. Architecture and logits parity are pinned to vLLM commit
``b2f685834a6456197e7033966fdef52a23f1abcd``; no Transformers HY V4 model
implementation is imported at runtime.
"""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig

__all__ = ["HyV4Config"]


class HyV4Config(PretrainedConfig):
    """Configuration for HY V4's gated DSA, iHC, MoE, and MTP stack."""

    model_type = "hy_v4"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts"}

    def __init__(
        self,
        vocab_size: int = 120832,
        hidden_size: int = 6144,
        intermediate_size: int = 18432,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 78,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        head_dim: int = 64,
        hidden_act: str = "silu",
        max_position_embeddings: int = 1048576,
        initializer_range: float = 0.006,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rope_parameters: dict[str, float | str] | None = None,
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 8,
        routed_scaling_factor: float = 2.827,
        norm_topk_prob: bool = True,
        n_group: int = 1,
        topk_group: int = 1,
        mlp_layer_types: list[str] | None = None,
        q_lora_rank: int = 2048,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 192,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 256,
        layer_types: list[str] | None = None,
        index_topk: int = 2048,
        index_head_dim: int = 128,
        index_n_heads: int = 32,
        indexer_types: list[str] | None = None,
        use_dsa: bool = True,
        use_mla: bool = True,
        gated_mla: bool = True,
        gating_type: str = "elementwise",
        learnable_sink: bool = True,
        learnable_sink_init: float = 0.0,
        enable_ihc: bool = True,
        hc_mult: int = 4,
        hc_magnitude: float = 2.0,
        hc_eps: float = 1e-6,
        swiglu_limit: float = 10.0,
        enable_lm_head_fp32: bool = True,
        bitwise_backward_align: bool = False,
        num_nextn_predict_layers: int = 1,
        mtp_loss_factor: float = 0.1,
        pad_token_id: int | None = 120002,
        bos_token_id: int | None = 120000,
        eos_token_id: int | list[int] | None = 120025,
        tie_word_embeddings: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_parameters = dict(rope_parameters or {"rope_type": "default", "rope_theta": 10_000_000.0})

        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.n_group = n_group
        self.topk_group = topk_group

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.use_dsa = use_dsa
        self.use_mla = use_mla

        if mlp_layer_types is None:
            mlp_layer_types = ["dense"] * min(num_hidden_layers, 1) + ["sparse"] * max(num_hidden_layers - 1, 0)
        if layer_types is None:
            layer_types = ["deepseek_sparse_attention"] * num_hidden_layers
        if indexer_types is None:
            indexer_types = [
                "full" if layer_idx == 0 or (layer_idx - 1) % 4 == 0 else "shared"
                for layer_idx in range(num_hidden_layers)
            ]
        if len(mlp_layer_types) != num_hidden_layers:
            raise ValueError("mlp_layer_types must contain one entry per hidden layer")
        if len(layer_types) != num_hidden_layers:
            raise ValueError("layer_types must contain one entry per hidden layer")
        if len(indexer_types) != num_hidden_layers:
            raise ValueError("indexer_types must contain one entry per hidden layer")
        if set(mlp_layer_types) - {"dense", "sparse"}:
            raise ValueError("mlp_layer_types entries must be 'dense' or 'sparse'")
        expected_mlp_types = ["dense"] + ["sparse"] * max(num_hidden_layers - 1, 0)
        if mlp_layer_types != expected_mlp_types:
            raise ValueError("HY4-preview requires layer 0 to be dense and every later MLP layer to be sparse")
        if set(layer_types) != {"deepseek_sparse_attention"}:
            raise ValueError("HY4-preview supports only 'deepseek_sparse_attention' layer_types")
        if set(indexer_types) - {"full", "shared"}:
            raise ValueError("indexer_types entries must be 'full' or 'shared'")
        expected_indexer_types = [
            "full" if layer_idx == 0 or (layer_idx - 1) % 4 == 0 else "shared" for layer_idx in range(num_hidden_layers)
        ]
        if indexer_types != expected_indexer_types:
            raise ValueError("HY4-preview indexer_types must follow the checkpoint's IndexShare schedule")
        self.mlp_layer_types = list(mlp_layer_types)
        self.layer_types = list(layer_types)
        self.indexer_types = list(indexer_types)

        if gating_type != "elementwise":
            raise ValueError("HY4-preview requires elementwise MLA gating")
        self.gated_mla = gated_mla
        self.gating_type = gating_type
        self.learnable_sink = learnable_sink
        self.learnable_sink_init = learnable_sink_init
        self.enable_ihc = enable_ihc
        self.hc_mult = hc_mult
        self.hc_magnitude = hc_magnitude
        self.hc_eps = hc_eps
        self.swiglu_limit = swiglu_limit
        self.enable_lm_head_fp32 = enable_lm_head_fp32
        self.bitwise_backward_align = bitwise_backward_align
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.mtp_loss_factor = mtp_loss_factor

        if not use_dsa or not use_mla:
            raise ValueError("HY4-preview requires both use_dsa=true and use_mla=true")
        if attention_bias or attention_dropout != 0.0:
            raise ValueError("HY4-preview requires bias-free attention with attention_dropout=0")
        if hidden_act != "silu":
            raise ValueError("HY4-preview requires hidden_act='silu'")
        if not gated_mla or not learnable_sink or not enable_ihc or not enable_lm_head_fp32:
            raise ValueError("HY4-preview requires gated_mla, learnable_sink, enable_ihc, and enable_lm_head_fp32")
        if not norm_topk_prob or n_group != 1 or topk_group != 1:
            raise ValueError("HY4-preview requires normalized top-k routing with n_group=topk_group=1")
        if num_nextn_predict_layers not in (0, 1):
            raise ValueError("HY4-preview supports zero or one checkpoint-native MTP layer")
        if set(self.rope_parameters) != {"rope_type", "rope_theta"}:
            raise ValueError("HY4-preview supports only rope_parameters with rope_type and rope_theta")
        if self.rope_parameters["rope_type"] != "default":
            raise ValueError("HY4-preview supports only default interleaved RoPE")

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

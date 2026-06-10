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

from transformers.configuration_utils import PretrainedConfig

_MIMOV2_ATTENTION_PROJECTION_LAYOUTS = {"split", "fused_qkv"}


class MiMoV2Config(PretrainedConfig):
    """Configuration for XiaomiMiMo/MiMo-V2.5-Pro."""

    model_type = "mimo_v2"
    keys_to_ignore_at_inference = ["past_key_values"]

    attribute_map = {
        "num_local_experts": "n_routed_experts",
    }

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        intermediate_size: int = 22016,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 32,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        layernorm_epsilon: float = 1e-6,
        rms_norm_eps: float | None = None,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        attention_value_scale: float | None = None,
        head_dim: int | None = None,
        v_head_dim: int | None = None,
        swa_num_attention_heads: int | None = None,
        swa_num_key_value_heads: int | None = None,
        swa_head_dim: int | None = None,
        swa_v_head_dim: int | None = None,
        swa_rope_theta: float | None = None,
        sliding_window: int | None = None,
        sliding_window_size: int | None = None,
        attention_chunk_size: int | None = None,
        add_full_attention_sink_bias: bool = False,
        add_swa_attention_sink_bias: bool = False,
        hybrid_block_size: int | None = None,
        hybrid_layer_pattern: list[int] | None = None,
        partial_rotary_factor: float = 1.0,
        n_routed_experts: int | None = None,
        n_shared_experts: int | None = None,
        moe_intermediate_size: int | None = None,
        num_experts_per_tok: int | None = None,
        routed_scaling_factor: float | None = None,
        scoring_func: str = "sigmoid",
        topk_method: str = "noaux_tc",
        n_group: int | None = None,
        topk_group: int | None = None,
        norm_topk_prob: bool = True,
        moe_layer_freq: list[int] | None = None,
        attention_projection_layout: str = "split",
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        rope_parameters = kwargs.pop("rope_parameters", None)
        if rope_scaling is None and rope_parameters is not None:
            rope_scaling = rope_parameters

        if attention_projection_layout is None:
            attention_projection_layout = "split"
        if attention_projection_layout not in _MIMOV2_ATTENTION_PROJECTION_LAYOUTS:
            raise ValueError(f"Unsupported MiMoV2 attention projection layout: {attention_projection_layout}")

        self.attention_projection_layout = attention_projection_layout

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layernorm_epsilon = layernorm_epsilon
        self.rms_norm_eps = layernorm_epsilon if rms_norm_eps is None else rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.attention_value_scale = attention_value_scale

        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        self.v_head_dim = v_head_dim if v_head_dim is not None else self.head_dim
        self.swa_num_attention_heads = (
            swa_num_attention_heads if swa_num_attention_heads is not None else num_attention_heads
        )
        self.swa_num_key_value_heads = (
            swa_num_key_value_heads if swa_num_key_value_heads is not None else num_key_value_heads
        )
        if self.swa_num_attention_heads % self.swa_num_key_value_heads != 0:
            raise ValueError("swa_num_attention_heads must be divisible by swa_num_key_value_heads")
        self.swa_head_dim = swa_head_dim if swa_head_dim is not None else self.head_dim
        self.swa_v_head_dim = swa_v_head_dim if swa_v_head_dim is not None else self.swa_head_dim
        self.swa_rope_theta = swa_rope_theta if swa_rope_theta is not None else rope_theta

        if sliding_window is None:
            sliding_window = sliding_window_size
        self.sliding_window = sliding_window
        self.sliding_window_size = sliding_window_size if sliding_window_size is not None else sliding_window
        self.attention_chunk_size = attention_chunk_size
        self.add_full_attention_sink_bias = add_full_attention_sink_bias
        self.add_swa_attention_sink_bias = add_swa_attention_sink_bias

        if hybrid_block_size is not None and hybrid_layer_pattern is None:
            hybrid_layer_pattern = [0 if ((i + 1) % hybrid_block_size == 0) else 1 for i in range(num_hidden_layers)]
        elif hybrid_layer_pattern is None:
            hybrid_layer_pattern = [0] * num_hidden_layers
        if len(hybrid_layer_pattern) != num_hidden_layers:
            raise ValueError("hybrid_layer_pattern length must match num_hidden_layers")
        self.hybrid_block_size = hybrid_block_size
        self.hybrid_layer_pattern = hybrid_layer_pattern

        self.partial_rotary_factor = partial_rotary_factor

        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.moe_intermediate_size = moe_intermediate_size if moe_intermediate_size is not None else intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob
        if isinstance(moe_layer_freq, int):
            moe_layer_freq = [moe_layer_freq > 0 and i % moe_layer_freq == 0 for i in range(num_hidden_layers)]
        elif moe_layer_freq is None:
            moe_layer_freq = [False] * num_hidden_layers
        if len(moe_layer_freq) != num_hidden_layers:
            raise ValueError("moe_layer_freq length must match num_hidden_layers")
        self.moe_layer_freq = moe_layer_freq

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        # Assign after super().__init__() so our string value wins over any
        # dtype conversion done by PretrainedConfig.
        self.torch_dtype = torch_dtype

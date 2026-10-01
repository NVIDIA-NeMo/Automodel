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

"""Configuration for tencent/HunyuanImage-3.0 (``model_type: hunyuan_image_3_moe``)."""

from typing import Any

from transformers import PretrainedConfig


def per_layer(value: Any, layer_idx: int) -> Any:
    """HunyuanImage-3.0 stores some MoE fields either as a scalar or as a per-layer list."""
    return value[layer_idx] if isinstance(value, (list, tuple)) else value


class HunyuanImage3Config(PretrainedConfig):
    """Backbone fields used by the native implementation.

    Everything else in the released ``config.json`` (``vae``, ``vit``, ``vit_aligner``, tokenizer and image-size
    fields) is kept verbatim as attributes and forwarded to the image-side modules.
    """

    model_type = "hunyuan_image_3_moe"

    def __init__(
        self,
        vocab_size: int = 133120,
        hidden_size: int = 4096,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        attention_head_dim: int = 128,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        hidden_act: str = "silu",
        intermediate_size: int = 3072,
        moe_intermediate_size: int | list[int] = 3072,
        num_experts: int = 64,
        moe_topk: int | list[int] = 8,
        num_shared_expert: int | list[int] = 1,
        use_mixed_mlp_moe: bool = True,
        moe_layer_num_skipped: int = 0,
        norm_topk_prob: bool = True,
        use_qk_norm: bool = True,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 22800,
        patch_size: int = 1,
        patch_embed_hidden_dim: int = 1024,
        img_proj_type: str = "unet",
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_head_dim = attention_head_dim
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts = num_experts
        self.moe_topk = moe_topk
        self.num_shared_expert = num_shared_expert
        self.use_mixed_mlp_moe = use_mixed_mlp_moe
        self.moe_layer_num_skipped = moe_layer_num_skipped
        self.norm_topk_prob = norm_topk_prob
        self.use_qk_norm = use_qk_norm
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.patch_size = patch_size
        self.patch_embed_hidden_dim = patch_embed_hidden_dim
        self.img_proj_type = img_proj_type
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

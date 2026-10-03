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

"""Configuration for tencent/HunyuanImage-3.0.

The checkpoint's ``config.json`` carries ``auto_map`` entries that point at remote code. Registering this class
for ``model_type="hunyuan_image_3_moe"`` lets ``AutoConfig`` read the checkpoint without ``trust_remote_code``.
Fields not listed here (VAE / ViT settings, special token ids, ...) are kept as plain attributes by
``PretrainedConfig``.
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


def per_layer(value: int | list[int], layer_idx: int) -> int:
    """Resolve a config field that is either a scalar or a per-layer list."""
    return value[layer_idx] if isinstance(value, (list, tuple)) else value


class HunyuanImage3Config(PretrainedConfig):
    """Configuration of the HunyuanImage-3.0 multimodal MoE transformer.

    Architecture (tencent/HunyuanImage-3.0):
      - 32 decoder layers, hidden size 4096, GQA with 32 query / 8 KV heads, head dim 128
      - Every layer is MoE: 64 routed experts (top-8, softmax, renormalized) plus one shared expert
      - Per-head QK RMSNorm applied after RoPE; 2D RoPE over the joint text / image sequence
      - Image tokens come from VAE latents through a UNet patch embedding, and the diffusion velocity is read out
        through a UNet final layer, both conditioned on the flow-matching timestep
    """

    model_type = "hunyuan_image_3_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 133120,
        hidden_size: int = 4096,
        intermediate_size: int = 3072,
        moe_intermediate_size: int | list[int] = 3072,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        attention_head_dim: int = 128,
        num_experts: int | list[int] = 64,
        moe_topk: int | list[int] = 8,
        num_shared_expert: int | list[int] = 1,
        use_mixed_mlp_moe: bool = True,
        norm_topk_prob: bool = True,
        router_aux_loss_coef: float = 0.0,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        use_qk_norm: bool = True,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        max_position_embeddings: int = 22800,
        img_proj_type: str = "unet",
        patch_size: int = 1,
        patch_embed_hidden_dim: int = 1024,
        image_base_size: int = 1024,
        image_token_id: int = 128006,
        vae: dict[str, Any] | None = None,
        pad_token_id: int | None = 128009,
        bos_token_id: int | None = 127958,
        eos_token_id: int | list[int] | None = 127957,
        tie_word_embeddings: bool = False,
        torch_dtype: str = "bfloat16",
        **kwargs: Any,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_head_dim = attention_head_dim
        self.num_experts = num_experts
        self.moe_topk = moe_topk
        self.num_shared_expert = num_shared_expert
        self.use_mixed_mlp_moe = use_mixed_mlp_moe
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.use_qk_norm = use_qk_norm
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias
        self.max_position_embeddings = max_position_embeddings
        self.img_proj_type = img_proj_type
        self.patch_size = patch_size
        self.patch_embed_hidden_dim = patch_embed_hidden_dim
        self.image_base_size = image_base_size
        self.image_token_id = image_token_id
        self.vae = vae if vae is not None else {"latent_channels": 32}
        kwargs.pop("head_dim", None)
        # Let PretrainedConfig own the dtype (transformers 5 stores it as ``dtype``, 4.x as ``torch_dtype``).
        kwargs["dtype"] = kwargs.get("dtype") or torch_dtype
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def head_dim(self) -> int:
        return self.attention_head_dim

    @property
    def latent_channels(self) -> int:
        return int(self.vae["latent_channels"])

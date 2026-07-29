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

"""Checkpoint-compatible configuration classes for Moonshot Kimi K3."""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig


class KimiK3TextConfig(PretrainedConfig):
    """Configuration for the Kimi K3 hybrid KDA/MLA text backbone."""

    model_type = "kimi_linear"
    architectures = ["KimiK3ForCausalLM"]
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 163840,
        hidden_size: int = 7168,
        head_dim: int | None = None,
        intermediate_size: int = 18432,
        num_hidden_layers: int = 93,
        num_attention_heads: int = 56,
        num_key_value_heads: int | None = None,
        hidden_act: str = "situ",
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        architectures: list[str] | None = None,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, Any] | None = None,
        tie_word_embeddings: bool = False,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 1048576,
        # MoE.
        moe_intermediate_size: int | None = 3072,
        moe_renormalize: bool = True,
        moe_router_activation_func: str = "sigmoid",
        num_experts: int | None = 896,
        num_experts_per_token: int | None = 16,
        num_shared_experts: int = 2,
        routed_scaling_factor: float = 1.0,
        first_k_dense_replace: int = 1,
        moe_layer_freq: int = 1,
        use_grouped_topk: bool = True,
        num_expert_group: int = 1,
        topk_group: int = 1,
        topk_method: str = "noaux_tc",
        routed_expert_hidden_size: int | None = 3584,
        latent_moe_use_norm: bool = True,
        # MLA.
        q_lora_rank: int | None = 1536,
        kv_lora_rank: int | None = 512,
        qk_nope_head_dim: int | None = 128,
        qk_rope_head_dim: int | None = 64,
        v_head_dim: int | None = 128,
        mla_use_nope: bool = True,
        mla_use_output_gate: bool = True,
        # KDA and residual mixing.
        linear_attn_config: dict[str, Any] | None = None,
        kda_mode: str = "chunk",
        kda_unpad_inputs: bool = True,
        kda_use_fused_gate: bool = True,
        kda_use_qk_l2norm_in_kernel: bool = True,
        attn_res_block_size: int | None = 12,
        activation_situ_beta: float | None = 4.0,
        activation_situ_linear_beta: float | None = 25.0,
        num_nextn_predict_layers: int = 0,
        **kwargs: Any,
    ) -> None:
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if head_dim is None:
            head_dim = hidden_size // num_attention_heads
        if linear_attn_config is None:
            full_attn_layers = list(range(4, num_hidden_layers + 1, 4))
            if num_hidden_layers not in full_attn_layers:
                full_attn_layers.append(num_hidden_layers)
            linear_attn_config = {
                "head_dim": 128,
                "num_heads": 96,
                "short_conv_kernel_size": 4,
                "kda_layers": [
                    layer for layer in range(1, num_hidden_layers + 1) if layer not in set(full_attn_layers)
                ],
                "full_attn_layers": full_attn_layers,
                "use_full_rank_gate": True,
                "gate_lower_bound": -5.0,
            }

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings

        self.moe_intermediate_size = moe_intermediate_size
        self.moe_renormalize = moe_renormalize
        self.moe_router_activation_func = moe_router_activation_func
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.num_shared_experts = num_shared_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.first_k_dense_replace = first_k_dense_replace
        self.moe_layer_freq = moe_layer_freq
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.topk_method = topk_method
        self.routed_expert_hidden_size = routed_expert_hidden_size
        self.latent_moe_use_norm = latent_moe_use_norm

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.mla_use_nope = mla_use_nope
        self.mla_use_output_gate = mla_use_output_gate

        self.linear_attn_config = linear_attn_config
        self.kda_mode = kda_mode
        self.kda_unpad_inputs = kda_unpad_inputs
        self.kda_use_fused_gate = kda_use_fused_gate
        self.kda_use_qk_l2norm_in_kernel = kda_use_qk_l2norm_in_kernel
        self.attn_res_block_size = attn_res_block_size
        self.activation_situ_beta = activation_situ_beta
        self.activation_situ_linear_beta = activation_situ_linear_beta
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self._validate()

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            architectures=architectures or ["KimiK3ForCausalLM"],
            **kwargs,
        )

    def _validate(self) -> None:
        if self.moe_router_activation_func not in {"sigmoid", "softmax"}:
            raise ValueError("moe_router_activation_func must be 'sigmoid' or 'softmax'.")
        if self.kda_mode not in {"chunk", "fused_recurrent"}:
            raise ValueError("kda_mode must be 'chunk' or 'fused_recurrent'.")
        if self.linear_attn_config is None:
            return
        if "kda_layers" not in self.linear_attn_config or "full_attn_layers" not in self.linear_attn_config:
            raise ValueError("linear_attn_config must define kda_layers and full_attn_layers.")
        configured = set(self.linear_attn_config["kda_layers"]) | set(self.linear_attn_config["full_attn_layers"])
        expected = set(range(1, self.num_hidden_layers + 1))
        if configured != expected:
            raise ValueError("KDA and full-attention layer lists must cover every decoder layer exactly.")
        overlap = set(self.linear_attn_config["kda_layers"]) & set(self.linear_attn_config["full_attn_layers"])
        if overlap:
            raise ValueError(f"KDA and full-attention layer lists overlap: {sorted(overlap)}")

    @property
    def is_mla(self) -> bool:
        """Whether full-attention layers use Kimi multi-latent attention."""
        return (
            self.q_lora_rank is not None
            or self.kv_lora_rank is not None
            or self.qk_nope_head_dim is not None
            or self.qk_rope_head_dim is not None
            or self.v_head_dim is not None
            or self.mla_use_nope
        )

    @property
    def is_moe(self) -> bool:
        """Whether the text checkpoint has routed experts."""
        return self.num_experts is not None

    @property
    def is_linear_attn(self) -> bool:
        """Whether any decoder layer uses Kimi Delta Attention."""
        return bool(self.linear_attn_config and self.linear_attn_config["kda_layers"])

    def is_kda_layer(self, layer_idx: int) -> bool:
        """Return whether zero-based ``layer_idx`` is a KDA layer."""
        return bool(self.linear_attn_config and layer_idx + 1 in self.linear_attn_config["kda_layers"])


class KimiK3VisionConfig(PretrainedConfig):
    """Configuration for the Kimi K3 MoonViT3d vision tower and projector."""

    model_type = "kimi_k3_vision"

    def __init__(
        self,
        patch_size: int = 14,
        init_pos_emb_height: int = 64,
        init_pos_emb_width: int = 64,
        init_pos_emb_time: int = 4,
        pos_emb_type: str = "divided_fixed",
        vt_num_attention_heads: int = 12,
        vt_num_hidden_layers: int = 27,
        vt_hidden_size: int = 1024,
        vt_intermediate_size: int = 4096,
        merge_kernel_size: tuple[int, int] | list[int] = (2, 2),
        merge_type: str = "sd2_tpool",
        attn_implementation: str = "flash_attention_2",
        mm_projector_type: str = "patchmergerv2",
        mm_hidden_size: int | None = None,
        projector_hidden_act: str = "gelu",
        projector_ln_eps: float = 1e-5,
        qkv_hidden_size: int = 1536,
        norm_type: str = "rmsnorm",
        attn_bias: bool = False,
        patch_embed_proj_bias: bool = False,
        mlp_type: str = "mlp2",
        linear_bias: bool = False,
        activation_func: str = "gelu_pytorch_tanh",
        pos_emb_interpolation_mode: str = "bilinear",
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        text_hidden_size: int = 7168,
        **kwargs: Any,
    ) -> None:
        checkpoint_attn = kwargs.pop("_attn_implementation", None)
        self.patch_size = patch_size
        self.init_pos_emb_height = init_pos_emb_height
        self.init_pos_emb_width = init_pos_emb_width
        self.init_pos_emb_time = init_pos_emb_time
        self.pos_emb_type = pos_emb_type
        self.vt_num_attention_heads = vt_num_attention_heads
        self.vt_num_hidden_layers = vt_num_hidden_layers
        self.vt_hidden_size = vt_hidden_size
        self.vt_intermediate_size = vt_intermediate_size
        self.merge_kernel_size = list(merge_kernel_size)
        self.merge_type = merge_type
        requested_attn_implementation = checkpoint_attn or attn_implementation
        self.mm_projector_type = mm_projector_type
        self.mm_hidden_size = mm_hidden_size if mm_hidden_size is not None else vt_hidden_size
        self.projector_hidden_act = projector_hidden_act
        self.projector_ln_eps = projector_ln_eps
        self.qkv_hidden_size = qkv_hidden_size
        self.norm_type = norm_type
        self.attn_bias = attn_bias
        self.patch_embed_proj_bias = patch_embed_proj_bias
        self.mlp_type = mlp_type
        self.linear_bias = linear_bias
        self.activation_func = activation_func
        self.pos_emb_interpolation_mode = pos_emb_interpolation_mode
        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        self.text_hidden_size = text_hidden_size
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self._attn_implementation = requested_attn_implementation


class KimiK3Config(PretrainedConfig):
    """Top-level Kimi K3 vision-language checkpoint configuration."""

    model_type = "kimi_k3"
    architectures = ["KimiK3ForConditionalGeneration"]
    sub_configs = {"text_config": KimiK3TextConfig, "vision_config": KimiK3VisionConfig}

    def __init__(
        self,
        text_config: dict[str, Any] | KimiK3TextConfig | None = None,
        vision_config: dict[str, Any] | KimiK3VisionConfig | None = None,
        ignore_index: int = -100,
        media_placeholder_token_id: int = 163605,
        pad_token_id: int = 0,
        architectures: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if text_config is None:
            text_config = KimiK3TextConfig()
        elif isinstance(text_config, dict):
            text_config = KimiK3TextConfig(**text_config)
        if vision_config is None:
            vision_config = KimiK3VisionConfig(text_hidden_size=text_config.hidden_size)
        elif isinstance(vision_config, dict):
            vision_config = KimiK3VisionConfig(**vision_config)

        self.text_config = text_config
        self.vision_config = vision_config
        self.ignore_index = ignore_index
        self.media_placeholder_token_id = media_placeholder_token_id
        self.hidden_size = text_config.hidden_size
        self.vocab_size = text_config.vocab_size
        self.max_position_embeddings = text_config.max_position_embeddings
        if getattr(text_config, "quantization_config", None) is not None:
            self.quantization_config = text_config.quantization_config
        kwargs.pop("tie_word_embeddings", None)
        vision_attn_implementation = vision_config._attn_implementation

        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=text_config.tie_word_embeddings,
            architectures=architectures or ["KimiK3ForConditionalGeneration"],
            **kwargs,
        )
        self.vision_config._attn_implementation = vision_attn_implementation

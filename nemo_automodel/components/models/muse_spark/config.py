# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Muse Spark Open Omni checkpoints."""

from __future__ import annotations

import math
from typing import Any

from transformers import PretrainedConfig


def expand_global_attention_schedule(
    schedule: list[int] | tuple[int, ...] | None,
    num_hidden_layers: int,
    every_n_layers_nope: int | None,
) -> list[int] | None:
    """Expand a periodic attention-window schedule to one entry per layer.

    Args:
        schedule: Periodic window sizes. Zero denotes global attention.
        num_hidden_layers: Number of decoder layers.
        every_n_layers_nope: Period used to align no-RoPE layers from the end.

    Returns:
        A list of ``num_hidden_layers`` window sizes, or ``None`` when no
        valid periodic schedule was supplied.
    """
    if not schedule or not all(isinstance(window, int) for window in schedule):
        return None
    offset = 0 if every_n_layers_nope is None else every_n_layers_nope - num_hidden_layers % every_n_layers_nope
    return [schedule[(layer_idx + offset) % len(schedule)] for layer_idx in range(num_hidden_layers)]


class MuseSpark1xOmniConfig(PretrainedConfig):
    """Configuration matching the released Muse Spark Open Omni dump."""

    model_type = "musespark1x_omni"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 202048,
        hidden_size: int = 8192,
        num_hidden_layers: int = 62,
        schedule_num_hidden_layers: int | None = None,
        num_attention_heads: int = 128,
        num_key_value_heads: int = 16,
        head_dim: int = 64,
        max_position_embeddings: int = 1048576,
        rms_norm_eps: float = 1e-5,
        rms_post_norm_eps: float = 1e-8,
        rope_theta: float = 500000.0,
        use_qk_norm: bool = True,
        use_rope: bool = True,
        use_sqrt_attn_scaling: bool = False,
        every_n_layers_nope: int | None = 4,
        global_attn_cfg: list[int] | None = None,
        use_attn_o_gate: bool = True,
        use_attn_o_norm: bool = True,
        ffn_dim_multiplier: float = 2.0,
        non_linearity: str = "swiglu",
        use_ffn_learnable_scales: bool = True,
        norm_affine: bool = True,
        post_norm_affine: bool = False,
        norm_zero_centered_gamma: bool = True,
        post_norm_gain_center_type: str = "one",
        output_norm_gain_center_type: str = "zero",
        normalize_tok_embeddings: bool = True,
        use_residual_gate: bool = True,
        residual_gate_temperature: float = 0.3,
        residual_dtype: str = "fp32",
        output_soft_cap_temp: float | None = 20.0,
        use_fp32_for_output: bool = False,
        moe: dict[str, Any] | None = None,
        metap: dict[str, Any] | None = None,
        numerics_args: dict[str, Any] | None = None,
        vision_config: dict[str, Any] | None = None,
        audio_perception_modality_args: dict[str, Any] | None = None,
        use_audio_perception: bool = True,
        image_token_id: int = 200090,
        patch_token_id: int = 200092,
        video_token_id: int = 200091,
        audio_token_id: int = 200038,
        initializer_range: float = 0.02,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        bos_token_id: int = 200000,
        eos_token_id: int = 200001,
        **kwargs: Any,
    ) -> None:
        if global_attn_cfg is None:
            global_attn_cfg = [2048, 2048, 2048, 0]
        if moe is None:
            moe = {
                "is_enabled": True,
                "topk": 8,
                "num_experts": 256,
                "use_shared_expert": False,
                "router_in_fp32": True,
                "route_scale": 1.0,
                "override_routed_hidden_dim": 4096,
                "interleave_moe_layer_step": 1,
                "n_dense_layers_before_moe": 0,
                "router_score_normalization": "sigmoid",
                "router_score_division_eps": 1e-15,
                "compression_factor": 2,
                "use_pre_expert_norm": True,
                "use_post_expert_norm": True,
            }
        if metap is None:
            metap = {
                "use_metap": True,
                "metap_mode": "sp",
                "base_width": 256.0,
                "router_has_one_inf_dim": True,
                "m_emb": 1.0,
            }
        if vision_config is None:
            vision_config = {
                "downsample_factor": 2,
                "patch_size": 14,
                "patch_temporal": 2,
                "projector_input_dim": 4096,
                "navit_config": {
                    "downsample_factor": 2,
                    "downsample_strategy": "ps",
                    "grid_sample_mode": "bilinear",
                    "hidden_act": "gelu",
                    "hidden_size": 1536,
                    "image_mean": [0.5, 0.5, 0.5],
                    "image_size": 448,
                    "image_std": [0.5, 0.5, 0.5],
                    "intermediate_size": 8960,
                    "max_num_image_tokens": 4096,
                    "max_num_video_tokens": 4096,
                    "min_num_image_tokens": 1,
                    "num_attention_heads": 16,
                    "num_channels": 3,
                    "num_hidden_layers": 50,
                    "patch_size": 14,
                    "patch_temporal": 2,
                    "rope_theta": 10000.0,
                    "sparse_attention_factor": 4,
                    "use_cls_token": False,
                    "use_rope": True,
                    "use_rowwise_vision_projection": True,
                    "vision_output_dim": 6144,
                },
            }
        if audio_perception_modality_args is None:
            audio_perception_modality_args = {
                "encoder_family": "wav_ae",
                "input_mode": "wav",
                "use_patch_encoder": False,
            }

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        # Diagnostic few-layer runs can instantiate only a checkpoint prefix
        # while retaining the released 62-layer RoPE/window phase.
        self.schedule_num_hidden_layers = schedule_num_hidden_layers or num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rms_post_norm_eps = rms_post_norm_eps
        self.rope_theta = rope_theta
        self.use_qk_norm = use_qk_norm
        self.use_rope = use_rope
        self.use_sqrt_attn_scaling = use_sqrt_attn_scaling
        self.every_n_layers_nope = every_n_layers_nope
        self.global_attn_cfg = list(global_attn_cfg)
        self.sliding_window = max(self.global_attn_cfg, default=0)
        self.use_attn_o_gate = use_attn_o_gate
        self.use_attn_o_norm = use_attn_o_norm
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.intermediate_size = int(hidden_size * ffn_dim_multiplier)
        self.non_linearity = non_linearity
        self.use_ffn_learnable_scales = use_ffn_learnable_scales
        self.norm_affine = norm_affine
        self.post_norm_affine = post_norm_affine
        self.norm_zero_centered_gamma = norm_zero_centered_gamma
        self.post_norm_gain_center_type = post_norm_gain_center_type
        self.output_norm_gain_center_type = output_norm_gain_center_type
        self.normalize_tok_embeddings = normalize_tok_embeddings
        self.use_residual_gate = use_residual_gate
        self.residual_gate_temperature = residual_gate_temperature
        self.residual_dtype = residual_dtype
        self.output_soft_cap_temp = output_soft_cap_temp
        self.final_logit_softcapping = output_soft_cap_temp
        self.use_fp32_for_output = use_fp32_for_output
        self.moe = dict(moe)
        self.metap = dict(metap)
        self.numerics_args = dict(numerics_args or {})
        self.vision_config = dict(vision_config)
        self.audio_perception_modality_args = dict(audio_perception_modality_args)
        self.use_audio_perception = use_audio_perception
        self.image_token_id = image_token_id
        # The released config serializes this field as null even though its
        # tokenizer assigns <|patch|> to 200092.  Resolve the immutable
        # tokenizer id here so model-side media splicing never compares token
        # ids against None.
        self.patch_token_id = 200092 if patch_token_id is None else patch_token_id
        self.video_token_id = video_token_id
        self.audio_token_id = audio_token_id
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias

        expanded = expand_global_attention_schedule(
            self.global_attn_cfg,
            self.schedule_num_hidden_layers,
            every_n_layers_nope,
        )
        if expanded is not None:
            expanded = expanded[:num_hidden_layers]
        if expanded is None:
            self.layer_types = ["full_attention"] * num_hidden_layers
            self.layer_sliding_windows = [0] * num_hidden_layers
        else:
            self.layer_types = ["sliding_attention" if window > 0 else "full_attention" for window in expanded]
            self.layer_sliding_windows = expanded
        self.no_rope_layers = [not self.layer_uses_rope(layer_idx) for layer_idx in range(num_hidden_layers)]

        self._validate()
        kwargs.setdefault("architectures", ["MuseSpark1xOmniForCausalLM"])
        kwargs.setdefault("dtype", "bfloat16")
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    def _validate(self) -> None:
        """Reject checkpoint settings unsupported by the native implementation."""
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("Muse Spark Open hidden_size must equal num_attention_heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Muse Spark Open query heads must be divisible by key/value heads")
        if self.residual_dtype not in {"fp32", "bf16", "fp16"}:
            raise ValueError(f"Unsupported residual_dtype {self.residual_dtype!r}")
        if self.non_linearity != "swiglu":
            raise ValueError("Muse Spark Open AutoModel currently requires SwiGLU experts")
        if not self.use_qk_norm or not self.use_attn_o_gate or not self.use_attn_o_norm:
            raise ValueError("Muse Spark Open requires QK norm, attention output norm, and the output gate")
        if self.residual_gate_temperature <= 0:
            raise ValueError("Muse Spark Open residual_gate_temperature must be positive")
        if not self.use_residual_gate:
            raise ValueError("Muse Spark Open requires its learned norm-preserving residual gates")
        if not self.moe.get("is_enabled", False):
            raise ValueError("The released Muse Spark Open checkpoint requires MoE")
        if self.moe.get("use_shared_expert", False):
            raise ValueError("Muse Spark Open uses routed experts without a shared expert")
        if self.moe.get("router_score_normalization", "sigmoid") != "sigmoid":
            raise ValueError("Muse Spark Open requires sigmoid router scores")
        compression = int(self.moe.get("compression_factor", 1))
        if compression < 1 or self.hidden_size % compression:
            raise ValueError("Muse Spark Open MoE compression_factor must divide hidden_size")
        if self.schedule_num_hidden_layers < self.num_hidden_layers:
            raise ValueError("schedule_num_hidden_layers cannot be smaller than num_hidden_layers")

    def layer_uses_rope(self, layer_idx: int) -> bool:
        """Return whether decoder layer ``layer_idx`` applies text RoPE."""
        if not self.use_rope:
            return False
        if self.every_n_layers_nope is None:
            return True
        return (self.schedule_num_hidden_layers - layer_idx - 1) % self.every_n_layers_nope != 0

    @property
    def num_local_experts(self) -> int:
        """Return the total routed-expert count before EP sharding."""
        return int(self.moe["num_experts"])

    @property
    def num_experts_per_tok(self) -> int:
        """Return the number of selected experts per token."""
        return int(self.moe["topk"])

    @property
    def moe_intermediate_size(self) -> int:
        """Return each routed expert's SwiGLU intermediate width."""
        return int(self.moe.get("override_routed_hidden_dim", self.intermediate_size))

    @property
    def expert_hidden_size(self) -> int:
        """Return the pooler-compressed expert input/output width."""
        return self.hidden_size // int(self.moe.get("compression_factor", 1))

    @property
    def router_output_multiplier(self) -> float:
        """Return the MetaP multiplier applied to logits before sigmoid routing."""
        if not self.metap.get("use_metap", False) or not self.metap.get("router_has_one_inf_dim", False):
            return 1.0
        width_mult = self.hidden_size / float(self.metap.get("base_width", 256.0))
        mode = self.metap.get("metap_mode", "sp")
        if mode == "sp":
            return width_mult**-0.5
        if mode == "mup":
            return width_mult**-1.0
        return 1.0

    @property
    def output_multiplier(self) -> float:
        """Return the checkpoint's language-model output multiplier."""
        return self.router_output_multiplier

    @property
    def scale_query_by(self) -> float:
        """Return the post-QK-normalization query scale."""
        return 3.87 if self.use_sqrt_attn_scaling else 43.7840518911 / math.sqrt(self.head_dim)

    @property
    def vision_navit_config(self) -> dict[str, Any]:
        """Return the flattened NaViT configuration mapping."""
        raw = self.vision_config.get("navit_config", self.vision_config)
        navit = dict(raw)
        if "projector_input_dim" in self.vision_config:
            navit["projector_input_dim"] = self.vision_config["projector_input_dim"]
        return navit

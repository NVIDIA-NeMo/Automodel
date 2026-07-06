# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0.

from typing import Callable

import torch
import torch.nn as nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import eager_attention_forward
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module, initialize_rms_norm_module
from nemo_automodel.components.models.llama.rope_utils import (
    apply_rotary_pos_emb as _apply_rotary_pos_emb,
)
from nemo_automodel.components.models.llama.rope_utils import (
    apply_rotary_pos_emb_fused as _apply_rotary_pos_emb_fused,
)
from nemo_automodel.shared.import_utils import safe_import_te
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

logger = logging.get_logger(__name__)


def _delete_module_attr(module: nn.Module, name: str) -> None:
    if hasattr(module, name):
        delattr(module, name)


def _require_transformer_engine(feature_name: str) -> None:
    te_available, _ = safe_import_te()
    if not te_available:
        raise RuntimeError(
            f"{feature_name} requires Transformer Engine, but transformer_engine could not be imported. "
            "Disable the corresponding use_te_fused_* option or run in an environment with a working "
            "Transformer Engine installation."
        )


def _get_siglip_vision_transformer(model: nn.Module) -> nn.Module | None:
    """Return the SigLIP transformer for nested and flattened HF layouts."""
    vision_model = getattr(model, "vision_model", None)
    nested = getattr(vision_model, "vision_model", None)
    if getattr(nested, "encoder", None) is not None or getattr(nested, "head", None) is not None:
        return nested
    if getattr(vision_model, "encoder", None) is not None or getattr(vision_model, "head", None) is not None:
        return vision_model
    return None


class OptimizedLlamaAttention(nn.Module):
    """Local LLaMA attention used only by the optimized Nemotron VL retriever path."""

    def __init__(self, config: LlamaConfig, layer_idx: int, backend: BackendConfig):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.rope_fusion = getattr(backend, "rope_fusion", False)
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        self.q_proj = initialize_linear_module(
            backend.linear,
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.k_proj = initialize_linear_module(
            backend.linear,
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.v_proj = initialize_linear_module(
            backend.linear,
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.o_proj = initialize_linear_module(
            backend.linear,
            self.num_key_value_groups * config.num_key_value_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            dtype=dtype,
        )
        self.pre_attention_qkv: nn.Module | None = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if self.pre_attention_qkv is not None:
            q, k, v = self.pre_attention_qkv(hidden_states)
        else:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

        query_states = q.view(hidden_shape).transpose(1, 2)
        key_states = k.view(hidden_shape).transpose(1, 2)
        value_states = v.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings[:2]
        if self.rope_fusion and len(position_embeddings) == 3:
            query_states, key_states = _apply_rotary_pos_emb_fused(query_states, key_states, position_embeddings[2])
        else:
            query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class OptimizedLlamaMLP(nn.Module):
    """Local SwiGLU MLP for the optimized VL-only LLaMA backend."""

    def __init__(self, config: LlamaConfig, backend: BackendConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.gate_proj = initialize_linear_module(
            backend.linear, self.hidden_size, self.intermediate_size, bias=config.mlp_bias, dtype=dtype
        )
        self.up_proj = initialize_linear_module(
            backend.linear, self.hidden_size, self.intermediate_size, bias=config.mlp_bias, dtype=dtype
        )
        self.down_proj = initialize_linear_module(
            backend.linear, self.intermediate_size, self.hidden_size, bias=config.mlp_bias, dtype=dtype
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class OptimizedFusedTERMSNormMLP(nn.Module):
    """Transformer Engine RMSNorm + SwiGLU replacement for the local optimized LLaMA MLP."""

    def __init__(self, source_norm: nn.Module, source_mlp: OptimizedLlamaMLP, eps: float):
        super().__init__()
        _require_transformer_engine(type(self).__name__)
        from transformer_engine.pytorch import LayerNormMLP

        norm_weight = source_norm.weight
        gate_proj = source_mlp.gate_proj
        up_proj = source_mlp.up_proj
        down_proj = source_mlp.down_proj
        gate_bias = getattr(gate_proj, "bias", None)
        use_bias = gate_bias is not None and gate_bias.numel() > 0
        self.fused = LayerNormMLP(
            hidden_size=source_mlp.hidden_size,
            ffn_hidden_size=source_mlp.intermediate_size,
            eps=eps,
            normalization="RMSNorm",
            activation="swiglu",
            bias=use_bias,
            params_dtype=norm_weight.dtype,
            device=norm_weight.device,
        )
        with torch.no_grad():
            self.fused.layer_norm_weight.copy_(norm_weight)
            self.fused.fc1_weight.copy_(torch.cat([gate_proj.weight, up_proj.weight], dim=0))
            self.fused.fc2_weight.copy_(down_proj.weight)
            if use_bias:
                self.fused.fc1_bias.copy_(torch.cat([gate_proj.bias, up_proj.bias], dim=0))
                self.fused.fc2_bias.copy_(down_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fused(x)


class OptimizedFusedTERMSNormQKV(nn.Module):
    """Transformer Engine RMSNorm + concatenated QKV projection for the local optimized LLaMA attention."""

    def __init__(self, source_norm: nn.Module, source_attn: OptimizedLlamaAttention, eps: float):
        super().__init__()
        _require_transformer_engine(type(self).__name__)
        from transformer_engine.pytorch import LayerNormLinear

        norm_weight = source_norm.weight
        q_proj = source_attn.q_proj
        k_proj = source_attn.k_proj
        v_proj = source_attn.v_proj
        q_size = q_proj.out_features
        k_size = k_proj.out_features
        v_size = v_proj.out_features
        q_bias = getattr(q_proj, "bias", None)
        use_bias = q_bias is not None and q_bias.numel() > 0
        self.sizes = (q_size, k_size, v_size)
        self.fused = LayerNormLinear(
            in_features=source_attn.config.hidden_size,
            out_features=q_size + k_size + v_size,
            eps=eps,
            normalization="RMSNorm",
            bias=use_bias,
            params_dtype=norm_weight.dtype,
            device=norm_weight.device,
        )
        with torch.no_grad():
            self.fused.layer_norm_weight.copy_(norm_weight)
            self.fused.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))
            if use_bias:
                self.fused.bias.copy_(torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.fused(x)
        return qkv.split(self.sizes, dim=-1)


class OptimizedLlamaDecoderLayer(nn.Module):
    """Local LLaMA decoder layer used only by the optimized Nemotron VL retriever path."""

    def __init__(self, config: LlamaConfig, layer_idx: int, backend: BackendConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = OptimizedLlamaAttention(config=config, layer_idx=layer_idx, backend=backend)
        self.mlp = OptimizedLlamaMLP(config=config, backend=backend)
        self.input_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, device=None
        )
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, device=None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        if getattr(self, "input_layernorm", None) is not None:
            hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        if getattr(self, "post_attention_layernorm", None) is not None:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


def replace_llama_mlp_with_te_fused(model: nn.Module) -> int:
    """Fuse post-attention RMSNorm and SwiGLU in the custom LLaMA stack."""
    _require_transformer_engine("use_te_fused_mlp")
    language_model = getattr(model, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        return 0

    fused_count = 0
    for layer in layers:
        norm = getattr(layer, "post_attention_layernorm", None)
        mlp = getattr(layer, "mlp", None)
        if norm is None or mlp is None or isinstance(mlp, OptimizedFusedTERMSNormMLP):
            continue
        if not hasattr(mlp, "gate_proj") or not hasattr(mlp, "up_proj") or not hasattr(mlp, "down_proj"):
            continue

        fused = OptimizedFusedTERMSNormMLP(norm, mlp, eps=language_model.config.rms_norm_eps)
        _delete_module_attr(layer, "post_attention_layernorm")
        layer.mlp = fused
        fused_count += 1

    logger.info("Fused custom LLaMA post-attention RMSNorm + SwiGLU layers: %d", fused_count)
    if fused_count:
        model._nemo_use_te_fused_llama_mlp = True
    return fused_count


def replace_llama_qkv_with_te_fused(model: nn.Module) -> int:
    """Fuse custom LLaMA input RMSNorm and Q/K/V projections with TE."""
    _require_transformer_engine("use_te_fused_qkv")
    language_model = getattr(model, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        return 0

    fused_count = 0
    for layer in layers:
        norm = getattr(layer, "input_layernorm", None)
        attention = getattr(layer, "self_attn", None)
        if norm is None or attention is None or getattr(attention, "pre_attention_qkv", None) is not None:
            continue
        if not all(hasattr(attention, name) for name in ("q_proj", "k_proj", "v_proj")):
            continue

        fused = OptimizedFusedTERMSNormQKV(norm, attention, eps=language_model.config.rms_norm_eps)
        _delete_module_attr(layer, "input_layernorm")
        attention.pre_attention_qkv = fused
        attention.q_proj = None
        attention.k_proj = None
        attention.v_proj = None
        fused_count += 1

    logger.info("Fused custom LLaMA input RMSNorm + QKV layers: %d", fused_count)
    if fused_count:
        model._nemo_use_te_fused_llama_qkv = True
    return fused_count


class FusedTESiglipTransformerLayer(nn.Module):
    """Transformer Engine implementation of one complete SigLIP encoder layer."""

    def __init__(self, source_layer: nn.Module):
        super().__init__()
        _require_transformer_engine(type(self).__name__)
        from transformer_engine.pytorch import TransformerLayer

        source_attn = source_layer.self_attn
        source_norm = source_layer.layer_norm1
        source_mlp = source_layer.mlp
        norm_weight = source_norm.weight
        self.fused = TransformerLayer(
            hidden_size=source_layer.embed_dim,
            ffn_hidden_size=source_mlp.fc1.out_features,
            num_attention_heads=source_attn.num_heads,
            kv_channels=source_attn.head_dim,
            layernorm_epsilon=source_norm.eps,
            hidden_dropout=0.0,
            attention_dropout=source_attn.dropout,
            self_attn_mask_type="no_mask",
            normalization="LayerNorm",
            activation="gelu",
            bias=True,
            fuse_qkv_params=True,
            qkv_weight_interleaved=False,
            attn_input_format="bshd",
            params_dtype=norm_weight.dtype,
            device=norm_weight.device,
        )

        with torch.no_grad():
            fused_qkv = self.fused.self_attention.layernorm_qkv
            fused_mlp = self.fused.layernorm_mlp
            fused_qkv.layer_norm_weight.copy_(source_norm.weight)
            fused_qkv.layer_norm_bias.copy_(source_norm.bias)
            fused_qkv.weight.copy_(
                torch.cat(
                    [source_attn.q_proj.weight, source_attn.k_proj.weight, source_attn.v_proj.weight],
                    dim=0,
                )
            )
            fused_qkv.bias.copy_(
                torch.cat(
                    [source_attn.q_proj.bias, source_attn.k_proj.bias, source_attn.v_proj.bias],
                    dim=0,
                )
            )
            self.fused.self_attention.proj.weight.copy_(source_attn.out_proj.weight)
            self.fused.self_attention.proj.bias.copy_(source_attn.out_proj.bias)
            fused_mlp.layer_norm_weight.copy_(source_layer.layer_norm2.weight)
            fused_mlp.layer_norm_bias.copy_(source_layer.layer_norm2.bias)
            fused_mlp.fc1_weight.copy_(source_mlp.fc1.weight)
            fused_mlp.fc1_bias.copy_(source_mlp.fc1.bias)
            fused_mlp.fc2_weight.copy_(source_mlp.fc2.weight)
            fused_mlp.fc2_bias.copy_(source_mlp.fc2.bias)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fused(
            hidden_states,
            attention_mask=None,
            self_attn_mask_type="no_mask",
            window_size=(-1, -1),
        )


class FusedTESiglipEncoderLayer(nn.Module):
    """Stock-compatible wrapper for a full TE SigLIP encoder layer."""

    def __init__(self, source_layer: nn.Module):
        super().__init__()
        self.embed_dim = source_layer.embed_dim
        # Keep the active fused block under a standard child name so the
        # generic DDP/FSDP activation-checkpointing path can wrap it.
        self.mlp = FusedTESiglipTransformerLayer(source_layer)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.mlp(hidden_states, *args, **kwargs)


def replace_siglip_encoder_layers_with_te_fused(model: nn.Module) -> int:
    """Replace loaded HF SigLIP encoder layers with full TE TransformerLayer equivalents."""
    select_layer = getattr(model, "select_layer", getattr(getattr(model, "config", None), "select_layer", -1))
    if select_layer != -1:
        raise ValueError(
            "use_te_fused_siglip_layer currently requires LlamaNemotronVL select_layer=-1. "
            "The TE SigLIP wrapper does not preserve intermediate hidden-state recording."
        )
    _require_transformer_engine("use_te_fused_siglip_layer")
    vision_transformer = _get_siglip_vision_transformer(model)
    encoder = getattr(vision_transformer, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if layers is None:
        return 0

    replaced = 0
    for idx, layer in enumerate(layers):
        if not all(hasattr(layer.self_attn, name) for name in ("q_proj", "k_proj", "v_proj", "out_proj")):
            continue
        layers[idx] = FusedTESiglipEncoderLayer(layer)
        replaced += 1

    logger.info("Replaced SigLIP encoder layers with TE TransformerLayer: layers=%d", replaced)
    if replaced:
        model._nemo_use_te_fused_siglip_layer = True
    return replaced


def disable_unused_siglip_pooling_head_grad(model: nn.Module) -> int:
    """Disable the unused SigLIP pooling head for retrieval feature extraction."""
    vision_transformer = _get_siglip_vision_transformer(model)
    pooling_head = getattr(vision_transformer, "head", None)
    if pooling_head is None:
        return 0

    # Retrieval consumes only ``last_hidden_state``. Avoid launching the
    # pooling head while retaining its parameters for checkpoint compatibility.
    if hasattr(vision_transformer, "use_head"):
        vision_transformer.use_head = False

    disabled = 0
    for param in pooling_head.parameters():
        if param.requires_grad:
            param.requires_grad_(False)
            disabled += param.numel()

    logger.info("Disabled unused SigLIP pooling head gradients: parameters=%d", disabled)
    return disabled


__all__ = [
    "FusedTESiglipEncoderLayer",
    "FusedTESiglipTransformerLayer",
    "OptimizedFusedTERMSNormMLP",
    "OptimizedFusedTERMSNormQKV",
    "OptimizedLlamaAttention",
    "OptimizedLlamaDecoderLayer",
    "OptimizedLlamaMLP",
    "disable_unused_siglip_pooling_head_grad",
    "replace_llama_mlp_with_te_fused",
    "replace_llama_qkv_with_te_fused",
    "replace_siglip_encoder_layers_with_te_fused",
]

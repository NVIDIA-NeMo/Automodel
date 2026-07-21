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

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import (
    _has_dtensor_params,
    cast_model_to_dtype,
    compute_lm_head_logits,
)
from nemo_automodel.components.models.laguna.config import LagunaConfig
from nemo_automodel.components.models.laguna.state_dict_adapter import LagunaStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MLP, MoE
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (_rotate_half(k_rot) * sin)
    return torch.cat([q_embed, q_pass], dim=-1), torch.cat([k_embed, k_pass], dim=-1)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def _convert_bool_4d_mask_to_additive(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if attention_mask.ndim != 4 or attention_mask.dtype != torch.bool:
        return attention_mask
    additive = torch.zeros(attention_mask.shape, dtype=dtype, device=attention_mask.device)
    return additive.masked_fill(~attention_mask, torch.finfo(dtype).min)


def _derive_padding_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim == 2:
        return attention_mask == 0
    if attention_mask.ndim == 4:
        diagonal = torch.diagonal(attention_mask[:, 0], dim1=-2, dim2=-1)
        if attention_mask.dtype == torch.bool:
            return diagonal.logical_not()
        return diagonal != 0
    return attention_mask.bool().logical_not()


def _fallback_additive_mask(
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: torch.Tensor | None = None,
    sliding_window: int | None = None,
) -> torch.Tensor:
    min_value = torch.finfo(dtype).min
    idx = torch.arange(seq_len, device=device)
    masked = idx.unsqueeze(0) > idx.unsqueeze(1)
    if sliding_window is not None and sliding_window > 0:
        masked = masked | ((idx.unsqueeze(1) - idx.unsqueeze(0)) >= sliding_window)
    additive = torch.zeros((seq_len, seq_len), dtype=dtype, device=device).masked_fill(masked, min_value)
    additive = additive.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len).contiguous()
    if attention_mask is not None and attention_mask.ndim == 2:
        pad = attention_mask.to(dtype=dtype, device=device)
        additive = additive + (1.0 - pad).unsqueeze(1).unsqueeze(2) * min_value
    return additive


def _ensure_additive_mask(
    mask: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: torch.Tensor | None,
    sliding_window: int | None,
) -> torch.Tensor:
    if mask is None or not isinstance(mask, torch.Tensor):
        return _fallback_additive_mask(batch_size, seq_len, dtype, device, attention_mask, sliding_window)
    return _convert_bool_4d_mask_to_additive(mask, dtype)


def _eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    del kwargs
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


def _sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    del scaling, kwargs
    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)
    attn_output = F.scaled_dot_product_attention(
        query,
        key_states,
        value_states,
        attn_mask=attention_mask[:, :, :, : key_states.shape[-2]] if attention_mask is not None else None,
        dropout_p=dropout,
        is_causal=False,
    )
    return attn_output.transpose(1, 2).contiguous(), None


def _normalize_gating_mode(gating: bool | str) -> str | None:
    if isinstance(gating, str):
        gating = gating.replace("_", "-").lower()
        if gating in {"false", "none", "off"}:
            return None
        if gating in {"per-head", "head"}:
            return "per-head"
        if gating in {"true", "per-element", "element"}:
            return "per-element"
        raise ValueError(f"Unsupported Laguna attention gating mode: {gating}")
    return "per-element" if gating else None


class LagunaRMSNorm(nn.Module):
    """RMSNorm with fp32 variance, matching the Laguna reference implementation."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LagunaRotaryEmbedding(nn.Module):
    """Rotary embedding with Laguna's nested full-attention/SWA RoPE support."""

    inv_freq: torch.Tensor

    def __init__(self, config: LagunaConfig, device: torch.device | None = None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_type = config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self._compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(config, device)
        self.register_buffer("inv_freq", inv_freq.float(), persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.float().clone(), persistent=False)

    @staticmethod
    def _compute_default_rope_parameters(
        config: LagunaConfig,
        device: torch.device | None = None,
        seq_len: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        del seq_len
        base = config.rope_parameters["rope_theta"]
        partial = config.rope_parameters.get("partial_rotary_factor", 1.0)
        dim = int(config.head_dim * partial)
        dim = dim - (dim % 2)
        if dim <= 0:
            raise ValueError(f"Invalid rotary dimension {dim} for head_dim={config.head_dim}")
        arange = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float()
        inv_freq = 1.0 / (base ** (arange / dim))
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq @ position_ids).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LagunaAttention(nn.Module):
    """Laguna attention: explicit per-layer heads, QK RMSNorm, and output gating."""

    def __init__(self, config: LagunaConfig, backend: BackendConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.backend = backend
        self.layer_idx = layer_idx
        per_layer_heads = config.num_attention_heads_per_layer
        self.num_heads = per_layer_heads[layer_idx] if per_layer_heads is not None else config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = float(config.attention_dropout or 0.0)
        self.is_causal = True

        layer_types = getattr(config, "layer_types", None)
        self.attention_type = layer_types[layer_idx] if layer_types is not None else "full_attention"
        self.is_sliding = self.attention_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.q_proj = initialize_linear_module(
            backend.linear,
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.qkv_bias,
            dtype=dtype,
        )
        self.k_proj = initialize_linear_module(
            backend.linear,
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.qkv_bias,
            dtype=dtype,
        )
        self.v_proj = initialize_linear_module(
            backend.linear,
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.qkv_bias,
            dtype=dtype,
        )
        self.o_proj = initialize_linear_module(
            backend.linear,
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            dtype=dtype,
        )

        gating = config.gating_types[layer_idx] if config.gating_types is not None else config.gating
        self.gating_mode = _normalize_gating_mode(gating)
        if self.gating_mode is not None:
            g_out = self.num_heads if self.gating_mode == "per-head" else self.num_heads * self.head_dim
            self.g_proj = initialize_linear_module(
                backend.linear,
                config.hidden_size,
                g_out,
                bias=False,
                dtype=dtype,
            )
        else:
            self.g_proj = None

        if self.is_sliding and getattr(config, "swa_attention_sink_enabled", False):
            self.sink = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        else:
            self.sink = None

        self.q_norm = LagunaRMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        self.k_norm = LagunaRMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch, seq_len = hidden_states.shape[:2]
        query_states = self.q_proj(hidden_states).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(
            batch,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        key_states = key_states.transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(
            batch,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        )
        value_states = value_states.transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attention_interface: Callable = _eager_attention_forward
        attn_impl = getattr(self.config, "_attn_implementation", "eager")
        if attn_impl == "sdpa":
            attention_interface = _sdpa_attention_forward
        elif attn_impl != "eager":
            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

            attention_interface = ALL_ATTENTION_FUNCTIONS[attn_impl]

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
        attn_output = attn_output.reshape(batch, seq_len, -1).contiguous()

        if self.g_proj is not None:
            gate = F.softplus(self.g_proj(hidden_states).float()).to(attn_output.dtype)
            if self.gating_mode == "per-head":
                attn_shape = attn_output.shape
                attn_output = (
                    attn_output.view(*attn_shape[:-1], self.num_heads, self.head_dim) * gate.unsqueeze(-1)
                ).view(attn_shape)
            else:
                attn_output = attn_output * gate

        return self.o_proj(attn_output), attn_weights

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        del buffer_device
        for linear in (self.q_proj, self.k_proj, self.v_proj, self.o_proj, self.g_proj):
            if linear is None:
                continue
            nn.init.normal_(linear.weight, mean=0.0, std=init_std)
            if getattr(linear, "bias", None) is not None:
                nn.init.zeros_(linear.bias)
        self.q_norm.reset_parameters()
        self.k_norm.reset_parameters()
        if self.sink is not None:
            nn.init.zeros_(self.sink)


class LagunaBlock(nn.Module):
    """Decoder block that uses dense MLP for configured layers and MoE otherwise."""

    def __init__(self, layer_idx: int, config: LagunaConfig, moe_config: MoEConfig, backend: BackendConfig):
        super().__init__()
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]
        self.self_attn = LagunaAttention(config, backend, layer_idx=layer_idx)
        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        is_moe_layer = layer_idx not in config.mlp_only_layers and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        )
        if is_moe_layer:
            self.mlp = MoE(moe_config, backend)
        else:
            self.mlp = MLP(
                config.hidden_size,
                config.intermediate_size,
                backend.linear,
                dtype=dtype,
                activation="swiglu",
                bias=False,
            )
        self.input_layernorm = LagunaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.post_attention_layernorm = LagunaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        padding_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, MoE):
            hidden_states = self.mlp(hidden_states, padding_mask)
        else:
            hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def init_weights(self, buffer_device: torch.device) -> None:
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
        self.self_attn.init_weights(buffer_device, init_std=0.02)
        self.mlp.init_weights(buffer_device)


def _config_with_rope(config: LagunaConfig, rope_parameters: dict[str, Any]) -> LagunaConfig:
    rope_config = copy.deepcopy(config)
    rope_config.rope_parameters = dict(rope_parameters)
    rope_config.partial_rotary_factor = rope_config.rope_parameters.get("partial_rotary_factor")
    return rope_config


class LagunaModel(nn.Module):
    """Backbone model for Laguna SFT."""

    def __init__(
        self,
        config: LagunaConfig,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict | None = None,
    ):
        super().__init__()
        self.config = config
        self.backend = backend
        if moe_config is not None and moe_overrides is not None:
            raise ValueError("Cannot pass both moe_config and moe_overrides; use one or the other.")
        if config.moe_apply_router_weight_on_input:
            raise NotImplementedError("Laguna moe_apply_router_weight_on_input=True is not supported in Automodel.")
        if float(config.moe_router_logit_softcapping or 0.0) > 0.0:
            raise NotImplementedError("Laguna router logit softcapping is not supported in Automodel.")
        if self.backend.gate_precision is None:
            self.backend.gate_precision = torch.float32

        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        moe_defaults = dict(
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            n_routed_experts=config.num_experts,
            n_shared_experts=1 if config.shared_expert_intermediate_size else 0,
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            score_func="sigmoid",
            route_scale=config.moe_routed_scaling_factor,
            aux_loss_coeff=config.router_aux_loss_coef,
            norm_topk_prob=config.norm_topk_prob,
            router_bias=False,
            expert_bias=False,
            expert_activation="swiglu",
            shared_expert_inter_dim=config.shared_expert_intermediate_size,
            shared_expert_activation="swiglu",
            softmax_before_topk=False,
            force_e_score_correction_bias=True,
            dtype=dtype,
        )
        if moe_overrides:
            moe_defaults.update(moe_overrides)
        self.moe_config = moe_config or MoEConfig(**moe_defaults)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id, dtype=dtype)
        self.layers = nn.ModuleDict(
            {
                str(layer_id): LagunaBlock(layer_id, config, self.moe_config, backend)
                for layer_id in range(config.num_hidden_layers)
            }
        )
        self.norm = LagunaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

        rope_parameters = config.rope_parameters
        full_rope = rope_parameters.get("full_attention") if isinstance(rope_parameters, dict) else None
        if isinstance(full_rope, dict):
            self.rotary_emb = LagunaRotaryEmbedding(_config_with_rope(config, full_rope))
        else:
            self.rotary_emb = LagunaRotaryEmbedding(config)

        if getattr(config, "swa_rope_parameters", None) is not None:
            self.swa_rotary_emb = LagunaRotaryEmbedding(_config_with_rope(config, config.swa_rope_parameters))
        else:
            self.swa_rotary_emb = None
        self.has_sliding_layers = "sliding_attention" in config.layer_types

    def _build_causal_mask_mapping(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None,
        position_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len = inputs_embeds.shape[:2]
        if isinstance(attention_mask, dict):
            full = attention_mask.get("full_attention")
            sliding = attention_mask.get("sliding_attention")
            if sliding is None:
                sliding = attention_mask.get("sliding_window_attention")
            return {
                "full_attention": _ensure_additive_mask(
                    full,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device,
                    attention_mask=None,
                    sliding_window=None,
                ),
                "sliding_attention": _ensure_additive_mask(
                    sliding,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device,
                    attention_mask=None,
                    sliding_window=self.config.sliding_window,
                ),
            }

        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        full = create_causal_mask(**mask_kwargs)
        sliding = create_sliding_window_causal_mask(**mask_kwargs) if self.has_sliding_layers else None
        tensor_attention_mask = attention_mask if isinstance(attention_mask, torch.Tensor) else None
        return {
            "full_attention": _ensure_additive_mask(
                full,
                batch_size=batch_size,
                seq_len=seq_len,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
                attention_mask=tensor_attention_mask,
                sliding_window=None,
            ),
            "sliding_attention": _ensure_additive_mask(
                sliding,
                batch_size=batch_size,
                seq_len=seq_len,
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
                attention_mask=tensor_attention_mask,
                sliding_window=self.config.sliding_window,
            ),
        }

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        if padding_mask is None and isinstance(attention_mask, torch.Tensor):
            padding_mask = _derive_padding_mask(attention_mask)

        causal_mask_mapping = self._build_causal_mask_mapping(
            inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        full_position_embeddings = self.rotary_emb(hidden_states, position_ids)
        if self.has_sliding_layers:
            sliding_position_embeddings = (
                self.swa_rotary_emb(hidden_states, position_ids)
                if self.swa_rotary_emb is not None
                else full_position_embeddings
            )
            position_embeddings_mapping = {
                "full_attention": full_position_embeddings,
                "sliding_attention": sliding_position_embeddings,
            }
        else:
            position_embeddings_mapping = {"full_attention": full_position_embeddings}

        for decoder_layer in self.layers.values():
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_embeddings=position_embeddings_mapping[decoder_layer.attention_type],
                padding_mask=padding_mask,
                **kwargs,
            )

        return self.norm(hidden_states) if self.norm is not None else hidden_states

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=self.config.initializer_range)
            if self.norm is not None:
                self.norm.reset_parameters()
        for layer in self.layers.values():
            layer.init_weights(buffer_device)


class LagunaForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Causal LM wrapper for Laguna with Automodel checkpoint adapters."""

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    _keep_in_fp32_modules_strict = ["mlp.gate.e_score_correction_bias", "rotary_emb", "sink"]
    _skip_init_weights_on_load = True

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = True

    @classmethod
    def from_config(
        cls,
        config: LagunaConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        return cls(config, moe_config, backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        config_kwargs = kwargs.pop("config_kwargs", {})
        config = LagunaConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: LagunaConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        reject_unsupported_tie_word_embeddings(type(self), config)
        self.backend = backend or BackendConfig()
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model = LagunaModel(config, self.backend, moe_config=moe_config, moe_overrides=moe_overrides)
        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.lm_head = initialize_linear_module(
            self.backend.linear,
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=dtype,
        )
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = LagunaStateDictAdapter(self.config, self.model.moe_config, self.backend, dtype)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def update_moe_gate_bias(self) -> None:
        for layer in self.model.layers.values():
            if isinstance(layer.mlp, MoE) and layer.mlp.gate.bias_update_factor > 0:
                layer.mlp.gate.update_bias()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        padding_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        if past_key_values is not None or use_cache:
            raise NotImplementedError("LagunaForCausalLM currently supports training forwards without KV cache.")
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )
        hidden = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **kwargs,
        )
        return compute_lm_head_logits(
            self.lm_head,
            hidden,
            logits_to_keep,
            output_hidden_states=output_hidden_states,
        )

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            self.model.init_weights(buffer_device)
            final_out_std = self.config.hidden_size**-0.5
            cutoff_factor = 3
            if self.lm_head is not None:
                nn.init.trunc_normal_(
                    self.lm_head.weight,
                    mean=0.0,
                    std=final_out_std,
                    a=-cutoff_factor * final_out_std,
                    b=cutoff_factor * final_out_std,
                )
        if _has_dtensor_params(self):
            return
        cast_model_to_dtype(self, dtype)


ModelClass = LagunaForCausalLM

__all__ = ["LagunaForCausalLM", "LagunaModel", "ModelClass"]

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Portions copyright 2026 Xiaomi Corporation.
# Portions copyright 2026 The HuggingFace Inc. team.
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

from copy import copy
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.utils import (
    _has_dtensor_params,
    cast_model_to_dtype,
    compute_lm_head_logits,
)
from nemo_automodel.components.models.mimo_v25.config import MiMoV2Config
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MLP, MoE
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def _convert_bool_4d_mask_to_additive(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if mask.ndim != 4 or mask.dtype != torch.bool:
        return mask
    additive = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    return additive.masked_fill(~mask, torch.finfo(dtype).min)


def _fallback_additive_mask(
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: torch.Tensor | None = None,
    sliding_window: int | None = None,
) -> torch.Tensor:
    min_val = torch.finfo(dtype).min
    idx = torch.arange(seq_len, device=device)
    masked = idx.unsqueeze(0) > idx.unsqueeze(1)
    if sliding_window is not None and sliding_window > 0:
        masked = masked | ((idx.unsqueeze(1) - idx.unsqueeze(0)) >= sliding_window)
    additive = torch.zeros((seq_len, seq_len), dtype=dtype, device=device).masked_fill(masked, min_val)
    additive = additive.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len).contiguous()
    if attention_mask is not None and attention_mask.ndim == 2:
        pad = (1.0 - attention_mask.to(dtype=dtype, device=device)).unsqueeze(1).unsqueeze(2) * min_val
        additive = additive + pad
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


def _derive_padding_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim == 2:
        return attention_mask == 0
    if attention_mask.ndim == 4:
        diagonal = torch.diagonal(attention_mask[:, 0], dim1=-2, dim2=-1)
        return diagonal.logical_not() if attention_mask.dtype == torch.bool else diagonal != 0
    return attention_mask.bool().logical_not()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    sinks: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

    if sinks is not None:
        sink_bias = module.attention_sink_bias.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        attn_weights = torch.cat([attn_weights, sink_bias.to(attn_weights.dtype)], dim=-1)

    attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values
    probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    if sinks is not None:
        probs = probs[..., :-1]

    probs = F.dropout(probs, p=dropout, training=module.training)
    attn_output = torch.matmul(probs, value_states)
    return attn_output.transpose(1, 2).contiguous(), probs


class MiMoV2Attention(nn.Module):
    """MiMoV2 hybrid attention (full or sliding-window)."""

    def __init__(
        self,
        config: MiMoV2Config,
        is_swa: bool,
        layer_idx: int,
        projection_layout: str,
        backend: BackendConfig,
        dtype: torch.dtype,
    ):
        super().__init__()
        if projection_layout not in {"split", "fused_qkv"}:
            raise ValueError(f"Unsupported MiMoV2 attention projection layout: {projection_layout}")

        self.layer_idx = layer_idx
        self.is_swa = is_swa
        self.is_causal = True
        self.projection_layout = projection_layout

        default_head_dim = config.hidden_size // config.num_attention_heads
        default_v_head_dim = getattr(config, "v_head_dim", default_head_dim)

        if is_swa:
            self.head_dim = getattr(config, "swa_head_dim", getattr(config, "head_dim", default_head_dim))
            self.v_head_dim = getattr(config, "swa_v_head_dim", default_v_head_dim)
            self.num_attention_heads = getattr(config, "swa_num_attention_heads", config.num_attention_heads)
            self.num_key_value_heads = getattr(config, "swa_num_key_value_heads", config.num_key_value_heads)
        else:
            self.head_dim = getattr(config, "head_dim", default_head_dim)
            self.v_head_dim = getattr(config, "v_head_dim", self.head_dim)
            self.num_attention_heads = config.num_attention_heads
            self.num_key_value_heads = config.num_key_value_heads

        self.rope_dim = int(self.head_dim * getattr(config, "partial_rotary_factor", 1.0))
        if self.rope_dim % 2 != 0:
            raise ValueError(
                f"MiMoV2 rotary dimension must be even, got {self.rope_dim} from "
                f"head_dim={self.head_dim} and partial_rotary_factor={getattr(config, 'partial_rotary_factor', 1.0)}"
            )

        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.scaling = self.head_dim**-0.5
        self.sliding_window = getattr(config, "sliding_window", None) if is_swa else None

        self.q_size = self.num_attention_heads * self.head_dim
        self.k_size = self.num_key_value_heads * self.head_dim
        self.v_size = self.num_key_value_heads * self.v_head_dim
        self.o_hidden_size = self.num_attention_heads * self.v_head_dim
        self.v_scale = getattr(config, "attention_value_scale", None)

        self.attention_sink_bias = (
            nn.Parameter(torch.empty(self.num_attention_heads), requires_grad=False)
            if (
                (getattr(config, "add_full_attention_sink_bias", False) and not is_swa)
                or (getattr(config, "add_swa_attention_sink_bias", False) and is_swa)
            )
            else None
        )

        attention_bias = getattr(config, "attention_bias", False)
        if self.projection_layout == "fused_qkv":
            self.qkv_proj = initialize_linear_module(
                backend.linear,
                config.hidden_size,
                self.q_size + self.k_size + self.v_size,
                bias=attention_bias,
                dtype=dtype,
            )
        else:
            self.q_proj = initialize_linear_module(
                backend.linear, config.hidden_size, self.q_size, bias=attention_bias, dtype=dtype
            )
            self.k_proj = initialize_linear_module(
                backend.linear, config.hidden_size, self.k_size, bias=attention_bias, dtype=dtype
            )
            self.v_proj = initialize_linear_module(
                backend.linear, config.hidden_size, self.v_size, bias=attention_bias, dtype=dtype
            )
        self.o_proj = initialize_linear_module(
            backend.linear, self.o_hidden_size, config.hidden_size, bias=False, dtype=dtype
        )

    def _forward_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        input_shape: torch.Size,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.v_scale is not None:
            value_states = value_states * self.v_scale

        cos, sin = position_embeddings
        query_rope, query_nope = query_states.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        key_rope, key_nope = key_states.split([self.rope_dim, self.head_dim - self.rope_dim], dim=-1)
        query_rope, key_rope = apply_rotary_pos_emb(query_rope, key_rope, cos, sin)
        query_states = torch.cat([query_rope, query_nope], dim=-1)
        key_states = torch.cat([key_rope, key_nope], dim=-1)

        attn_output, _ = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            sinks=self.attention_sink_bias,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        input_shape = hidden_states.shape[:-1]

        if self.projection_layout == "fused_qkv":
            qkv = self.qkv_proj(hidden_states)
            query_states, key_states, value_states = qkv.split([self.q_size, self.k_size, self.v_size], dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(*input_shape, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(*input_shape, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(*input_shape, self.num_key_value_heads, self.v_head_dim).transpose(1, 2)
        return self._forward_attention(
            query_states, key_states, value_states, input_shape, position_embeddings, attention_mask
        )


class MiMoV2RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: MiMoV2Config, is_swa: bool, device: Optional[torch.device] = None):
        super().__init__()
        self.rope_type = (
            config.rope_scaling.get("rope_type", config.rope_scaling.get("type", "default"))
            if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict)
            else "default"
        )
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = copy(config)
        self.config.rope_parameters = copy(getattr(config, "rope_parameters", None) or {})
        if is_swa:
            self.config.rope_theta = getattr(config, "swa_rope_theta", config.rope_theta)
            self.config.head_dim = getattr(config, "swa_head_dim", getattr(config, "head_dim", None))
            if self.config.rope_parameters:
                self.config.rope_parameters["rope_theta"] = self.config.rope_theta

        self.rope_init_fn = (
            self.compute_default_rope_parameters if self.rope_type == "default" else ROPE_INIT_FUNCTIONS[self.rope_type]
        )
        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: MiMoV2Config,
        device: Optional[torch.device] = None,
        seq_len: Optional[int] = None,
        layer_type: Optional[str] = None,
    ) -> tuple[torch.Tensor, float]:
        config.standardize_rope_params()
        rope_parameters = config.rope_parameters[layer_type] if layer_type is not None else config.rope_parameters
        base = rope_parameters["rope_theta"]
        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)
        if dim % 2 != 0:
            raise ValueError(
                f"MiMoV2 rotary dimension must be even, got {dim} from "
                f"head_dim={head_dim} and partial_rotary_factor={partial_rotary_factor}"
            )
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MiMoV2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: MiMoV2Config,
        layer_idx: int,
        moe_config: MoEConfig,
        backend: BackendConfig,
    ):
        super().__init__()
        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        is_swa_layer = config.hybrid_layer_pattern[layer_idx] == 1
        self.attention_type = "sliding_attention" if is_swa_layer else "full_attention"
        self.self_attn = MiMoV2Attention(
            config,
            is_swa=is_swa_layer,
            layer_idx=layer_idx,
            projection_layout=config.attention_projection_layout,
            backend=backend,
            dtype=dtype,
        )
        is_moe_layer = getattr(config, "n_routed_experts", None) is not None and config.moe_layer_freq[layer_idx]
        self.mlp = (
            MoE(moe_config, backend)
            if is_moe_layer
            else MLP(config.hidden_size, config.intermediate_size, backend.linear, dtype=dtype)
        )
        self.input_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.layernorm_epsilon, dtype=dtype
        )
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.layernorm_epsilon, dtype=dtype
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MiMoV2Model(nn.Module):
    def __init__(self, config: MiMoV2Config, moe_config: MoEConfig, backend: BackendConfig):
        super().__init__()
        self.config = config
        self.backend = backend

        if backend.gate_precision is None:
            backend.gate_precision = torch.float32

        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.layers = nn.ModuleDict(
            {str(i): MiMoV2DecoderLayer(config, i, moe_config, backend) for i in range(config.num_hidden_layers)}
        )
        self.norm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.layernorm_epsilon, dtype=dtype
        )
        self.rotary_emb = MiMoV2RotaryEmbedding(config=config, is_swa=False)
        self.swa_rotary_emb = MiMoV2RotaryEmbedding(config=config, is_swa=True)
        self.has_sliding_layers = any(p == 1 for p in config.hybrid_layer_pattern)

    def _build_causal_mask_mapping(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None,
        position_ids: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, seq_len = inputs_embeds.shape[:2]
        dtype = inputs_embeds.dtype
        device = inputs_embeds.device

        if isinstance(attention_mask, dict):
            return {
                "full_attention": _ensure_additive_mask(
                    attention_mask.get("full_attention"),
                    batch_size=batch_size,
                    seq_len=seq_len,
                    dtype=dtype,
                    device=device,
                    attention_mask=None,
                    sliding_window=None,
                ),
                "sliding_attention": _ensure_additive_mask(
                    attention_mask.get("sliding_attention", attention_mask.get("sliding_window_attention")),
                    batch_size=batch_size,
                    seq_len=seq_len,
                    dtype=dtype,
                    device=device,
                    attention_mask=None,
                    sliding_window=self.config.sliding_window,
                ),
            }

        mask_kwargs = dict(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
        pad_mask = attention_mask if isinstance(attention_mask, torch.Tensor) else None
        return {
            "full_attention": _ensure_additive_mask(
                create_causal_mask(**mask_kwargs),
                batch_size=batch_size,
                seq_len=seq_len,
                dtype=dtype,
                device=device,
                attention_mask=pad_mask,
                sliding_window=None,
            ),
            "sliding_attention": _ensure_additive_mask(
                create_sliding_window_causal_mask(**mask_kwargs) if self.has_sliding_layers else None,
                batch_size=batch_size,
                seq_len=seq_len,
                dtype=dtype,
                device=device,
                attention_mask=pad_mask,
                sliding_window=self.config.sliding_window,
            ),
        }

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        *,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kwargs
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds must be provided")
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if padding_mask is None and isinstance(attention_mask, torch.Tensor):
            padding_mask = _derive_padding_mask(attention_mask)

        causal_mask_mapping = self._build_causal_mask_mapping(
            inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        swa_position_embeddings = self.swa_rotary_emb(hidden_states, position_ids)

        for layer in self.layers.values():
            layer_position_embeddings = (
                swa_position_embeddings if layer.attention_type == "sliding_attention" else position_embeddings
            )
            hidden_states = layer(
                hidden_states,
                attention_mask=causal_mask_mapping[layer.attention_type],
                position_embeddings=layer_position_embeddings,
                padding_mask=padding_mask,
            )

        return self.norm(hidden_states)

    @torch.no_grad()
    def init_weights(self, buffer_device: Optional[torch.device] = None) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            nn.init.normal_(self.embed_tokens.weight)
            self.norm.reset_parameters()
        for layer in self.layers.values():
            layer.init_weights(buffer_device)


class MiMoV2ForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """NeMo AutoModel causal LM wrapper for MiMo-V2.5-Pro."""

    _keep_in_fp32_modules_strict = ["mlp.gate.e_score_correction_bias", "attention_sink_bias"]
    _pp_keep_self_forward = True
    _skip_init_weights_on_load = True

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = True
        supports_ep: bool = True

    @classmethod
    def from_config(
        cls,
        config: MiMoV2Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ) -> MiMoV2ForCausalLM:
        return cls(config, moe_config, backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ) -> MiMoV2ForCausalLM:
        config = MiMoV2Config.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: MiMoV2Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()
        moe_overrides = kwargs.pop("moe_overrides", None)

        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        moe_defaults = dict(
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            n_routed_experts=int(config.n_routed_experts or 0),
            n_shared_experts=int(config.n_shared_experts or 0),
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=config.n_group,
            n_limited_groups=config.topk_group,
            train_gate=True,
            gate_bias_update_factor=0.0,
            score_func="sigmoid_with_bias" if config.scoring_func == "sigmoid" else config.scoring_func,
            route_scale=config.routed_scaling_factor if config.routed_scaling_factor is not None else 1.0,
            aux_loss_coeff=0.0,
            norm_topk_prob=config.norm_topk_prob,
            router_bias=False,
            expert_bias=False,
            expert_activation="swiglu",
            softmax_before_topk=False,
            force_e_score_correction_bias=True,
            dtype=dtype,
        )
        if moe_overrides:
            moe_defaults.update(moe_overrides)
        resolved_moe_config = moe_config or MoEConfig(**moe_defaults)

        self.model = MiMoV2Model(config, resolved_moe_config, self.backend)
        self.lm_head = initialize_linear_module(
            self.backend.linear,
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=dtype,
        )

        if self.backend.enable_hf_state_dict_adapter:
            from nemo_automodel.components.models.mimo_v25.state_dict_adapter import MiMoV2StateDictAdapter

            self.state_dict_adapter = MiMoV2StateDictAdapter(
                self.config,
                self.model.moe_config if hasattr(self.model, "moe_config") else resolved_moe_config,
                self.backend,
                dtype=dtype,
            )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        *,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
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
        return compute_lm_head_logits(self.lm_head, hidden, logits_to_keep, output_hidden_states=output_hidden_states)

    def customize_pipeline_stage_modules(
        self,
        module_names_per_stage: list[list[str]],
        *,
        layers_prefix: str,
        text_model: Optional[nn.Module] = None,
    ) -> list[list[str]]:
        """Keep the SWA rotary embedding on every PP stage."""
        text_model = text_model or self.model
        stage_modules = [list(modules) for modules in module_names_per_stage]
        if getattr(text_model, "swa_rotary_emb", None) is not None:
            fqn = f"{layers_prefix}swa_rotary_emb"
            for modules in stage_modules:
                if fqn not in modules:
                    modules.append(fqn)
        return stage_modules

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            self.model.init_weights(buffer_device)
            final_out_std = self.config.hidden_size**-0.5
            cutoff_factor = 3
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


ModelClass = MiMoV2ForCausalLM

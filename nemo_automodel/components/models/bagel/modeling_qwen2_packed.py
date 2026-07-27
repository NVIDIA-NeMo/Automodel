# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
#
# Includes Apache-2.0 code adapted from ByteDance-Seed/Bagel and HuggingFace
# Transformers' Qwen2 implementation. Upstream copyright notices:
#   Copyright (c) 2024 The Qwen Team and The HuggingFace Inc. team.
#   Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
#
# This module implements a Qwen2-compatible decoder with packed-sequence
# attention, QK-norm, and MoT parameter siblings. Attention dispatch is routed
# through AM's ``FlexAttention`` wrapper for training and
# ``flash_attn_varlen_func`` for inference while preserving checkpoint key
# compatibility.

"""Qwen2 language backbone with packed-sequence attention and MoT shell.

Stage 1 uses ``PackedAttention`` + ``Qwen2DecoderLayer``. The
``PackedAttentionMoT`` / ``Qwen2MoTDecoderLayer`` shells are defined so that
the ``*_moe_gen`` parameter siblings exist in the module tree and survive
checkpoint round-tripping; they remain dormant in Stage 1 when
``packed_gen_token_indexes`` is empty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention
from transformers import Qwen2Config
from transformers.activations import ACT2FN
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from nemo_automodel.components.attention.flex_attention import FlexAttention
from nemo_automodel.components.models.bagel.configuration import BagelBackendConfig
from nemo_automodel.components.models.common import (
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.shared.import_utils import safe_import_from

__all__ = [
    "Qwen2RMSNorm",
    "Qwen2RotaryEmbedding",
    "Qwen2MLP",
    "PackedAttention",
    "PackedAttentionMoT",
    "Qwen2DecoderLayer",
    "Qwen2MoTDecoderLayer",
    "Qwen2Model",
    "Qwen2ForCausalLM",
    "NaiveCache",
    "BaseNavitOutputWithPast",
]


# ---------------------------------------------------------------------------
# Optional flash-attn import (inference path only).
# ---------------------------------------------------------------------------
_HAS_FLASH_ATTN, _flash_attn_varlen_func = safe_import_from(
    "flash_attn",
    "flash_attn_varlen_func",
    msg="flash_attn is required for BAGEL packed-sequence inference; install flash-attn>=2.0",
)

logger = logging.getLogger(__name__)
_WARNED_INDEX_PUT_DTYPE_CASTS: set[tuple[torch.dtype, torch.dtype]] = set()


def _flash_attn_varlen(*args, **kwargs):
    if not _HAS_FLASH_ATTN:
        raise ImportError(
            "PackedAttention.forward_inference requires flash_attn.flash_attn_varlen_func; "
            "install flash-attn>=2.0 or stay on forward_train."
        )
    return _flash_attn_varlen_func(*args, **kwargs)


def _initialize_linear(
    backend: BagelBackendConfig,
    in_features: int,
    out_features: int,
    *,
    bias: bool,
) -> nn.Module:
    """Construct a torch-compatible linear while preserving the configured parameter dtype."""
    if backend.linear == "torch":
        return nn.Linear(in_features, out_features, bias=bias)
    return initialize_linear_module(
        backend.linear,
        in_features,
        out_features,
        bias=bias,
        dtype=torch.get_default_dtype(),
    )


def _index_put_matching_dtype(destination: torch.Tensor, index: torch.Tensor, source: torch.Tensor) -> None:
    """Assign selected packed-token rows and surface backend dtype changes once.

    Args:
        destination: Packed tensor of shape ``[tokens, ...]``. This tensor is
            mutated in place.
        index: One-dimensional integer tensor of shape ``[selected_tokens]``
            selecting rows along the packed-token axis.
        source: Tensor of shape ``[selected_tokens, ...]`` whose trailing
            dimensions match ``destination``.
    """
    if destination.is_floating_point() and source.is_floating_point() and destination.dtype != source.dtype:
        dtype_pair = (source.dtype, destination.dtype)
        if dtype_pair not in _WARNED_INDEX_PUT_DTYPE_CASTS:
            is_rank_zero = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            if is_rank_zero:
                logger.warning(
                    "BAGEL packed scatter is casting backend output from %s to %s. "
                    "This may indicate an unintended backend precision change.",
                    source.dtype,
                    destination.dtype,
                )
            _WARNED_INDEX_PUT_DTYPE_CASTS.add(dtype_pair)
        source = source.to(dtype=destination.dtype)
    destination[index] = source


# Route flex_attention calls through AM's compiled wrapper. BAGEL upstream does
# ``flex_attention = torch.compile(flex_attention)`` at module import; AM's
# FlexAttention class does the equivalent once as a class attribute.
_flex_attention = FlexAttention.flex_attn


# ---------------------------------------------------------------------------
# Norms, RoPE, and MLP layers.
# ---------------------------------------------------------------------------


class Qwen2RMSNorm(nn.Module):
    """Qwen2 RMSNorm (equivalent to T5LayerNorm)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


def _initialize_rms_norm(backend: BagelBackendConfig, hidden_size: int, eps: float) -> nn.Module:
    """Construct the selected RMSNorm while preserving BAGEL's torch baseline."""
    if backend.rms_norm == "torch_fp32":
        return Qwen2RMSNorm(hidden_size, eps=eps)
    return initialize_rms_norm_module(
        backend.rms_norm,
        hidden_size,
        eps=eps,
        dtype=torch.get_default_dtype(),
    )


def _apply_qk_norm(
    norm: nn.Module,
    hidden_states: torch.Tensor,
    *,
    backend: BagelBackendConfig,
    eps: float,
) -> torch.Tensor:
    """Apply Q/K RMSNorm over each attention head's final dimension.

    Args:
        norm: RMSNorm module to apply.
        hidden_states: Tensor of shape ``[tokens, heads, head_dim]``. RMSNorm
            is computed over the final ``head_dim`` axis.
        backend: Resolved BAGEL backend configuration.
        eps: Epsilon used by the explicit FP32 Transformer Engine path.

    Returns:
        Tensor of shape ``[tokens, heads, head_dim]`` with the same layout as
        ``hidden_states``.
    """
    weight = getattr(norm, "weight", None)
    if backend.rms_norm == "te" and hidden_states.dtype == torch.float32 and weight is not None:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return weight.float() * hidden_states * torch.rsqrt(variance + eps)
    return norm(hidden_states)


def _extract_rope_config(config: Qwen2Config) -> dict:
    """Return a dict with ``rope_theta`` / scaling info, handling transformers 4.x and 5.x.

    DIVERGENCE: upstream BAGEL was written against transformers 4.4x where
    ``Qwen2Config`` exposes ``rope_theta`` and ``rope_scaling`` as top-level
    attributes. transformers 5.x moves these into a single ``rope_parameters``
    dict on Qwen2Config (Llama still keeps the old layout). AM's container runs
    transformers 5.x, so we normalize here instead of hard-coding one schema.
    """
    rope_params = getattr(config, "rope_parameters", None)
    if rope_params:
        return dict(rope_params)
    try:
        rope_theta = config.rope_theta
    except AttributeError:
        rope_theta = 10000.0
    try:
        rope_scaling = config.rope_scaling or {}
    except AttributeError:
        rope_scaling = {}
    return {"rope_theta": rope_theta, **rope_scaling}


def _compute_default_rope_parameters(
    config: Qwen2Config,
    device: Optional[torch.device] = None,
    seq_len: Optional[int] = None,
) -> Tuple[torch.Tensor, float]:
    """Local "default" RoPE init — transformers 5.x dropped it from ROPE_INIT_FUNCTIONS."""
    rope_params = _extract_rope_config(config)
    base = rope_params.get("rope_theta", 10000.0)
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32) / dim))
    return inv_freq, 1.0


class Qwen2RotaryEmbedding(nn.Module):
    """Qwen2 rotary embedding — delegates inv_freq init to HF ``ROPE_INIT_FUNCTIONS``.

    DIVERGENCE: transformers 5.x removed ``"default"`` from ``ROPE_INIT_FUNCTIONS``
    so we fall back to a local copy of the pre-5.x default implementation when
    the rope_type is unspecified.
    """

    def __init__(self, config: Qwen2Config, device: Optional[torch.device] = None) -> None:
        super().__init__()
        rope_params = _extract_rope_config(config)
        self.rope_type = rope_params.get("rope_type") or rope_params.get("type") or "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        # Ensure config.rope_theta is readable by rope_init_fn regardless of
        # transformers version. In 5.x, Qwen2Config stores this in rope_parameters
        # and accessing config.rope_theta raises AttributeError.
        try:
            _ = config.rope_theta
        except AttributeError:
            config.rope_theta = rope_params.get("rope_theta", 10000.0)

        if self.rope_type == "default" and self.rope_type not in ROPE_INIT_FUNCTIONS:
            self.rope_init_fn = _compute_default_rope_parameters
        else:
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids: torch.Tensor, device: torch.device) -> None:
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len
        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
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
    fused: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if fused:
        return _bagel_fused_rope(q, k, cos, sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Optional fused-kernel helpers. Pure functions; gated by BagelBackendConfig at
# the call sites (default OFF).
# ---------------------------------------------------------------------------


@torch.compile(dynamic=True)
def _bagel_fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused SwiGLU gate: ``silu(gate) * up`` in a single compiled pointwise kernel.

    Compiling ONLY this pointwise activation — over regular (non-DTensor) activation tensors,
    not the projections/attention that run on FSDP2 DTensor params — fuses the silu and the
    multiply into one kernel (fewer launches, no materialization of the silu output) without
    triggering the DTensor-spec recompile thrash that whole-layer ``torch.compile`` hits.
    """
    return torch.nn.functional.silu(gate) * up


@torch.compile(dynamic=True)
def _bagel_fused_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused RoPE apply: rotate ``q`` and ``k`` in a single compiled pointwise kernel.

    ``cos``/``sin`` arrive already unsqueezed to broadcast over the head dim. Compiling this
    pointwise region — like the SwiGLU gate, over regular activation tensors and no FSDP2
    DTensor parameters — fuses the ``rotate_half`` slice/negate/cat and the mul/add into far
    fewer kernels than eager ATen, with no recompile thrash.

    Args:
        q: Query states, shape ``[tokens, heads, head_dim]``.
        k: Key states, shape ``[tokens, kv_heads, head_dim]``.
        cos: Cosine table, broadcastable to ``q`` and ``k``.
        sin: Sine table, broadcastable to ``q`` and ``k``.

    Returns:
        Tuple of rotated ``(q_embed, k_embed)`` matching the inputs' shapes and dtypes.
    """
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen2MLP(nn.Module):
    """SwiGLU MLP with an independently configurable linear backend."""

    def __init__(self, config: Qwen2Config, backend: Optional[BagelBackendConfig] = None) -> None:
        super().__init__()
        self.backend = backend or BagelBackendConfig()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = _initialize_linear(self.backend, self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = _initialize_linear(self.backend, self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = _initialize_linear(self.backend, self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
        # Fuse silu(gate)*up only when the activation is silu (BAGEL/Qwen2 default).
        self._fuse_silu_mul = (
            getattr(self.backend, "fused_swiglu", False) and getattr(config, "hidden_act", "silu") == "silu"
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_state)
        up = self.up_proj(hidden_state)
        if self._fuse_silu_mul:
            return self.down_proj(_bagel_fused_silu_mul(gate, up))
        return self.down_proj(self.act_fn(gate) * up)


# ---------------------------------------------------------------------------
# Packed-sequence primitives.
# ---------------------------------------------------------------------------


class NaiveCache:
    """Dict-backed KV cache, one entry per layer (BAGEL inference helper)."""

    def __init__(self, num_layers: int) -> None:
        self.key_cache: dict[int, Optional[torch.Tensor]] = {k: None for k in range(num_layers)}
        self.value_cache: dict[int, Optional[torch.Tensor]] = {k: None for k in range(num_layers)}

    @property
    def num_layers(self) -> int:
        return len(self.key_cache)

    @property
    def seq_lens(self) -> int:
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        return 0


@dataclass
class BaseNavitOutputWithPast(ModelOutput):
    """BAGEL packed decoder output with optional past key-value cache."""

    packed_query_sequence: torch.FloatTensor = None
    past_key_values: Optional[NaiveCache] = None


def _pad_sequence(tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    H, L, D = tensor.shape
    pad_tensor = tensor.new_zeros((H, pad_size, D))
    return torch.cat([tensor, pad_tensor], dim=1)


class _PackedAttentionBase(nn.Module):
    """Common init for PackedAttention / PackedAttentionMoT (QKV shapes, RoPE, QK-norm)."""

    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: Optional[int] = None,
        backend: Optional[BagelBackendConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.backend = backend or BagelBackendConfig()

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = _extract_rope_config(config).get("rope_theta", 10000.0)
        # BAGEL's Qwen2Config extension fields — read with safe defaults so this
        # port accepts a vanilla transformers.Qwen2Config as well.
        self.is_causal = getattr(config, "is_causal", True)
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = _initialize_linear(self.backend, self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = _initialize_linear(
            self.backend, self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj = _initialize_linear(
            self.backend, self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj = _initialize_linear(self.backend, self.num_heads * self.head_dim, self.hidden_size, bias=False)


class PackedAttention(_PackedAttentionBase):
    """BAGEL's packed-sequence attention (UND path, no MoT)."""

    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: Optional[int] = None,
        backend: Optional[BagelBackendConfig] = None,
    ) -> None:
        super().__init__(config, layer_idx, backend=backend)
        if getattr(config, "qk_norm", False):
            self.q_norm = _initialize_rms_norm(self.backend, self.head_dim, config.rms_norm_eps)
            self.k_norm = _initialize_rms_norm(self.backend, self.head_dim, config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        packed_query_states = self.q_proj(packed_sequence).reshape(-1, self.num_heads, self.head_dim)
        packed_key_states = self.k_proj(packed_sequence).reshape(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = self.v_proj(packed_sequence).reshape(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states = _apply_qk_norm(
            self.q_norm, packed_query_states, backend=self.backend, eps=self.config.rms_norm_eps
        )
        packed_key_states = _apply_qk_norm(
            self.k_norm, packed_key_states, backend=self.backend, eps=self.config.rms_norm_eps
        )

        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states,
            packed_key_states,
            packed_cos,
            packed_sin,
            unsqueeze_dim=1,
            fused=self.backend.fused_rope,
        )

        if isinstance(attention_mask, list):
            packed_key_states = packed_key_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_key_states = packed_key_states.reshape(-1, self.num_heads, self.head_dim)
            packed_value_states = packed_value_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_value_states = packed_value_states.reshape(-1, self.num_heads, self.head_dim)

            unpacked_query_states = packed_query_states.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_key_states = packed_key_states.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_attn_output = []
            for query_states, key_states, value_states, attention_mask_per_sample in zip(
                unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0),
                        key_states.to(torch.bfloat16).unsqueeze(0),
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    )
                unpacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(unpacked_attn_output, dim=1)
        else:
            pad_size = sum(sample_lens) - packed_query_states.shape[0]
            packed_query_states = _pad_sequence(packed_query_states.permute(1, 0, 2), pad_size)
            packed_key_states = _pad_sequence(packed_key_states.permute(1, 0, 2), pad_size)
            packed_value_states = _pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)
            packed_attn_output = _flex_attention(
                packed_query_states.unsqueeze(0),
                packed_key_states.unsqueeze(0),
                packed_value_states.unsqueeze(0),
                enable_gqa=True,
                block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.hidden_size)
        packed_attn_output = self.o_proj(packed_attn_output)
        return packed_attn_output

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Optional[NaiveCache]]:
        packed_query_states = self.q_proj(packed_query_sequence).reshape(-1, self.num_heads, self.head_dim)
        packed_key_states = self.k_proj(packed_query_sequence).reshape(-1, self.num_key_value_heads, self.head_dim)
        packed_value_states = self.v_proj(packed_query_sequence).reshape(-1, self.num_key_value_heads, self.head_dim)

        packed_query_states = _apply_qk_norm(
            self.q_norm, packed_query_states, backend=self.backend, eps=self.config.rms_norm_eps
        )
        packed_key_states = _apply_qk_norm(
            self.k_norm, packed_key_states, backend=self.backend, eps=self.config.rms_norm_eps
        )

        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_value_states = past_key_states.new_zeros((seqlens, self.num_key_value_heads, self.head_dim))
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))

        packed_attn_output = _flash_attn_varlen(
            q=packed_query_states,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=max(query_lens).item(),
            max_seqlen_k=max(key_values_lens).item(),
            causal=is_causal,
        )
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        packed_attn_output = self.o_proj(packed_attn_output)

        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values


class PackedAttentionMoT(_PackedAttentionBase):
    """MoT variant: adds ``*_moe_gen`` siblings of every projection and QK-norm."""

    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: Optional[int] = None,
        backend: Optional[BagelBackendConfig] = None,
    ) -> None:
        super().__init__(config, layer_idx, backend=backend)
        if getattr(config, "qk_norm", False):
            self.q_norm = _initialize_rms_norm(self.backend, self.head_dim, config.rms_norm_eps)
            self.k_norm = _initialize_rms_norm(self.backend, self.head_dim, config.rms_norm_eps)
            self.q_norm_moe_gen = _initialize_rms_norm(self.backend, self.head_dim, config.rms_norm_eps)
            self.k_norm_moe_gen = _initialize_rms_norm(self.backend, self.head_dim, config.rms_norm_eps)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.q_norm_moe_gen = nn.Identity()
            self.k_norm_moe_gen = nn.Identity()

        self.q_proj_moe_gen = _initialize_linear(
            self.backend, self.hidden_size, self.num_heads * self.head_dim, bias=True
        )
        self.k_proj_moe_gen = _initialize_linear(
            self.backend, self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.v_proj_moe_gen = _initialize_linear(
            self.backend, self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True
        )
        self.o_proj_moe_gen = _initialize_linear(
            self.backend, self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
        mot_perm: Optional[torch.LongTensor] = None,
        mot_inv: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        freeze_und = getattr(self.config, "freeze_und", False)

        if mot_perm is not None:
            # MLM-style: und tokens are [:Lund], gen tokens [Lund:] — slice + concat, no gather/scatter.
            lund = packed_und_token_indexes.shape[0]
            seq_und = packed_sequence[:lund]
            seq_gen = packed_sequence[lund:]
            q = torch.cat([self.q_proj(seq_und), self.q_proj_moe_gen(seq_gen)], dim=0).view(
                -1, self.num_heads, self.head_dim
            )
            k = torch.cat([self.k_proj(seq_und), self.k_proj_moe_gen(seq_gen)], dim=0).view(
                -1, self.num_key_value_heads, self.head_dim
            )
            packed_value_states = torch.cat([self.v_proj(seq_und), self.v_proj_moe_gen(seq_gen)], dim=0).view(
                -1, self.num_key_value_heads, self.head_dim
            )
            packed_query_states_ = torch.cat(
                [
                    _apply_qk_norm(self.q_norm, q[:lund], backend=self.backend, eps=self.config.rms_norm_eps),
                    _apply_qk_norm(self.q_norm_moe_gen, q[lund:], backend=self.backend, eps=self.config.rms_norm_eps),
                ],
                dim=0,
            )
            packed_key_states_ = torch.cat(
                [
                    _apply_qk_norm(self.k_norm, k[:lund], backend=self.backend, eps=self.config.rms_norm_eps),
                    _apply_qk_norm(self.k_norm_moe_gen, k[lund:], backend=self.backend, eps=self.config.rms_norm_eps),
                ],
                dim=0,
            )
            if freeze_und:
                packed_value_states[:lund] = packed_value_states[:lund].detach()
                packed_query_states_[:lund] = packed_query_states_[:lund].detach()
                packed_key_states_[:lund] = packed_key_states_[:lund].detach()
        else:
            packed_sequence_und = packed_sequence[packed_und_token_indexes]
            packed_sequence_gen = packed_sequence[packed_gen_token_indexes]

            packed_query_states = packed_sequence.new_zeros((packed_sequence.shape[0], self.num_heads * self.head_dim))
            packed_key_states = packed_sequence.new_zeros(
                (packed_sequence.shape[0], self.num_key_value_heads * self.head_dim)
            )
            packed_value_states = packed_sequence.new_zeros(
                (packed_sequence.shape[0], self.num_key_value_heads * self.head_dim)
            )
            _index_put_matching_dtype(packed_query_states, packed_und_token_indexes, self.q_proj(packed_sequence_und))
            _index_put_matching_dtype(
                packed_query_states, packed_gen_token_indexes, self.q_proj_moe_gen(packed_sequence_gen)
            )
            _index_put_matching_dtype(packed_key_states, packed_und_token_indexes, self.k_proj(packed_sequence_und))
            _index_put_matching_dtype(
                packed_key_states, packed_gen_token_indexes, self.k_proj_moe_gen(packed_sequence_gen)
            )
            _index_put_matching_dtype(packed_value_states, packed_und_token_indexes, self.v_proj(packed_sequence_und))
            _index_put_matching_dtype(
                packed_value_states, packed_gen_token_indexes, self.v_proj_moe_gen(packed_sequence_gen)
            )

            packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
            packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)
            if freeze_und:
                packed_value_states[packed_und_token_indexes] = packed_value_states[packed_und_token_indexes].detach()

            packed_query_states_ = packed_query_states.new_zeros(packed_query_states.shape)
            packed_key_states_ = packed_key_states.new_zeros(packed_key_states.shape)

            _index_put_matching_dtype(
                packed_query_states_,
                packed_und_token_indexes,
                _apply_qk_norm(
                    self.q_norm,
                    packed_query_states[packed_und_token_indexes],
                    backend=self.backend,
                    eps=self.config.rms_norm_eps,
                ),
            )
            if freeze_und:
                packed_query_states_[packed_und_token_indexes] = packed_query_states_[packed_und_token_indexes].detach()
            _index_put_matching_dtype(
                packed_query_states_,
                packed_gen_token_indexes,
                _apply_qk_norm(
                    self.q_norm_moe_gen,
                    packed_query_states[packed_gen_token_indexes],
                    backend=self.backend,
                    eps=self.config.rms_norm_eps,
                ),
            )

            _index_put_matching_dtype(
                packed_key_states_,
                packed_und_token_indexes,
                _apply_qk_norm(
                    self.k_norm,
                    packed_key_states[packed_und_token_indexes],
                    backend=self.backend,
                    eps=self.config.rms_norm_eps,
                ),
            )
            if freeze_und:
                packed_key_states_[packed_und_token_indexes] = packed_key_states_[packed_und_token_indexes].detach()
            _index_put_matching_dtype(
                packed_key_states_,
                packed_gen_token_indexes,
                _apply_qk_norm(
                    self.k_norm_moe_gen,
                    packed_key_states[packed_gen_token_indexes],
                    backend=self.backend,
                    eps=self.config.rms_norm_eps,
                ),
            )

        packed_cos, packed_sin = packed_position_embeddings
        packed_query_states_, packed_key_states_ = apply_rotary_pos_emb(
            packed_query_states_,
            packed_key_states_,
            packed_cos,
            packed_sin,
            unsqueeze_dim=1,
            fused=self.backend.fused_rope,
        )

        if mot_inv is not None:
            # Restore ORIGINAL token order for the attention kernel so the block-diagonal mask
            # stays block-sparse (RoPE is already applied in grouped order with original position
            # values, so this only re-scatters rows). Re-grouped after attention for the O-proj.
            packed_query_states_ = packed_query_states_[mot_inv]
            packed_key_states_ = packed_key_states_[mot_inv]
            packed_value_states = packed_value_states[mot_inv]

        if isinstance(attention_mask, list):
            packed_key_states_ = packed_key_states_[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_key_states_ = packed_key_states_.reshape(-1, self.num_heads, self.head_dim)
            packed_value_states = packed_value_states[:, :, None, :].repeat(1, 1, self.num_key_value_groups, 1)
            packed_value_states = packed_value_states.reshape(-1, self.num_heads, self.head_dim)

            unpacked_query_states = packed_query_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_key_states = packed_key_states_.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_value_states = packed_value_states.transpose(0, 1).split(sample_lens, dim=1)
            unpacked_attn_output = []
            for query_states, key_states, value_states, attention_mask_per_sample in zip(
                unpacked_query_states, unpacked_key_states, unpacked_value_states, attention_mask
            ):
                with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                    attn_output = scaled_dot_product_attention(
                        query_states.to(torch.bfloat16).unsqueeze(0),
                        key_states.to(torch.bfloat16).unsqueeze(0),
                        value_states.to(torch.bfloat16).unsqueeze(0),
                        attention_mask_per_sample.to(torch.bfloat16).unsqueeze(0),
                    )
                unpacked_attn_output.append(attn_output.squeeze(0))
            packed_attn_output = torch.cat(unpacked_attn_output, dim=1)
        else:
            pad_size = sum(sample_lens) - packed_query_states_.shape[0]
            packed_query_states_ = _pad_sequence(packed_query_states_.permute(1, 0, 2), pad_size)
            packed_key_states_ = _pad_sequence(packed_key_states_.permute(1, 0, 2), pad_size)
            packed_value_states = _pad_sequence(packed_value_states.permute(1, 0, 2), pad_size)
            packed_attn_output = _flex_attention(
                packed_query_states_.unsqueeze(0),
                packed_key_states_.unsqueeze(0),
                packed_value_states.unsqueeze(0),
                enable_gqa=True,
                block_mask=attention_mask,
            )
            end_index = packed_attn_output.shape[2] - pad_size
            packed_attn_output = packed_attn_output[0, :, :end_index, :]

        packed_attn_output = packed_attn_output.transpose(0, 1).reshape(-1, self.num_heads * self.head_dim)
        if mot_perm is not None:
            # Re-group the attention output (original -> [und | gen]) for slice O-proj.
            packed_attn_output = packed_attn_output[mot_perm]
            lund = packed_und_token_indexes.shape[0]
            return torch.cat(
                [self.o_proj(packed_attn_output[:lund]), self.o_proj_moe_gen(packed_attn_output[lund:])],
                dim=0,
            )
        packed_attn_output_ = packed_attn_output.new_zeros(packed_attn_output.shape)
        _index_put_matching_dtype(
            packed_attn_output_,
            packed_und_token_indexes,
            self.o_proj(packed_attn_output[packed_und_token_indexes]),
        )
        _index_put_matching_dtype(
            packed_attn_output_,
            packed_gen_token_indexes,
            self.o_proj_moe_gen(packed_attn_output[packed_gen_token_indexes]),
        )
        return packed_attn_output_

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        packed_text_indexes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[NaiveCache]]:
        if mode == "und":
            packed_query_states = self.q_proj(packed_query_sequence).reshape(-1, self.num_heads, self.head_dim)
            packed_key_states = self.k_proj(packed_query_sequence).reshape(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = self.v_proj(packed_query_sequence).reshape(
                -1, self.num_key_value_heads, self.head_dim
            )
            packed_query_states = _apply_qk_norm(
                self.q_norm, packed_query_states, backend=self.backend, eps=self.config.rms_norm_eps
            )
            packed_key_states = _apply_qk_norm(
                self.k_norm, packed_key_states, backend=self.backend, eps=self.config.rms_norm_eps
            )
        elif mode == "gen":
            packed_query_sequence = packed_query_sequence.to(torch.bfloat16)
            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]

            packed_query_states = packed_query_sequence.new_zeros(
                (packed_query_sequence.shape[0], self.num_heads * self.head_dim)
            )
            packed_key_states = packed_query_sequence.new_zeros(
                (packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim)
            )
            packed_value_states = packed_query_sequence.new_zeros(
                (packed_query_sequence.shape[0], self.num_key_value_heads * self.head_dim)
            )
            _index_put_matching_dtype(packed_query_states, packed_text_indexes, self.q_proj(packed_text_query_sequence))
            _index_put_matching_dtype(
                packed_query_states,
                packed_vae_token_indexes,
                self.q_proj_moe_gen(packed_vae_query_sequence),
            )
            _index_put_matching_dtype(packed_key_states, packed_text_indexes, self.k_proj(packed_text_query_sequence))
            _index_put_matching_dtype(
                packed_key_states,
                packed_vae_token_indexes,
                self.k_proj_moe_gen(packed_vae_query_sequence),
            )
            _index_put_matching_dtype(packed_value_states, packed_text_indexes, self.v_proj(packed_text_query_sequence))
            _index_put_matching_dtype(
                packed_value_states,
                packed_vae_token_indexes,
                self.v_proj_moe_gen(packed_vae_query_sequence),
            )

            packed_query_states = packed_query_states.view(-1, self.num_heads, self.head_dim)
            packed_key_states = packed_key_states.view(-1, self.num_key_value_heads, self.head_dim)
            packed_value_states = packed_value_states.view(-1, self.num_key_value_heads, self.head_dim)

            packed_query_states = packed_query_states.to(torch.float32)
            _index_put_matching_dtype(
                packed_query_states,
                packed_text_indexes,
                _apply_qk_norm(
                    self.q_norm,
                    packed_query_states[packed_text_indexes],
                    backend=self.backend,
                    eps=self.config.rms_norm_eps,
                ),
            )
            _index_put_matching_dtype(
                packed_query_states,
                packed_vae_token_indexes,
                _apply_qk_norm(
                    self.q_norm_moe_gen,
                    packed_query_states[packed_vae_token_indexes],
                    backend=self.backend,
                    eps=self.config.rms_norm_eps,
                ),
            )

            packed_key_states = packed_key_states.to(torch.float32)
            _index_put_matching_dtype(
                packed_key_states,
                packed_text_indexes,
                _apply_qk_norm(
                    self.k_norm,
                    packed_key_states[packed_text_indexes],
                    backend=self.backend,
                    eps=self.config.rms_norm_eps,
                ),
            )
            _index_put_matching_dtype(
                packed_key_states,
                packed_vae_token_indexes,
                _apply_qk_norm(
                    self.k_norm_moe_gen,
                    packed_key_states[packed_vae_token_indexes],
                    backend=self.backend,
                    eps=self.config.rms_norm_eps,
                ),
            )
        else:
            raise ValueError(f"mode must be 'und' or 'gen', got {mode!r}")

        packed_cos, packed_sin = packed_query_position_embeddings
        packed_query_states, packed_key_states = apply_rotary_pos_emb(
            packed_query_states, packed_key_states, packed_cos, packed_sin, unsqueeze_dim=1
        )

        packed_query_states = packed_query_states.to(torch.bfloat16)
        packed_key_states = packed_key_states.to(torch.bfloat16)
        packed_value_states = packed_value_states.to(torch.bfloat16)

        if past_key_values is not None and past_key_values.key_cache[self.layer_idx] is not None:
            past_key_states = past_key_values.key_cache[self.layer_idx]
            past_value_states = past_key_values.value_cache[self.layer_idx]

            seqlens = sum(query_lens) + sum(key_values_lens)
            merged_key_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_value_states = past_key_states.new_zeros(size=[seqlens, self.num_key_value_heads, self.head_dim])
            merged_key_states[packed_query_indexes] = packed_key_states
            merged_key_states[packed_key_value_indexes] = past_key_states
            merged_value_states[packed_query_indexes] = packed_value_states
            merged_value_states[packed_key_value_indexes] = past_value_states
            key_values_lens = key_values_lens + query_lens
        else:
            merged_key_states = packed_key_states
            merged_value_states = packed_value_states
            key_values_lens = query_lens

        cu_seqlens_q = torch.nn.functional.pad(torch.cumsum(query_lens, dim=0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(torch.cumsum(key_values_lens, dim=0), (1, 0))

        packed_attn_output = _flash_attn_varlen(
            q=packed_query_states,
            k=merged_key_states,
            v=merged_value_states,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            max_seqlen_q=max(query_lens).item(),
            max_seqlen_k=max(key_values_lens).item(),
            causal=is_causal,
        )
        packed_attn_output = packed_attn_output.reshape(-1, self.hidden_size)
        if mode == "und":
            packed_attn_output = self.o_proj(packed_attn_output)
        elif mode == "gen":
            _index_put_matching_dtype(
                packed_attn_output,
                packed_text_indexes,
                self.o_proj(packed_attn_output[packed_text_indexes]),
            )
            _index_put_matching_dtype(
                packed_attn_output,
                packed_vae_token_indexes,
                self.o_proj_moe_gen(packed_attn_output[packed_vae_token_indexes]),
            )

        if update_past_key_values:
            past_key_values.key_cache[self.layer_idx] = merged_key_states
            past_key_values.value_cache[self.layer_idx] = merged_value_states

        return packed_attn_output, past_key_values


# ---------------------------------------------------------------------------
# Decoder layers.
# ---------------------------------------------------------------------------


class Qwen2DecoderLayer(nn.Module):
    """Standard (non-MoT) packed Qwen2 decoder block."""

    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: Optional[int] = None,
        backend: Optional[BagelBackendConfig] = None,
    ) -> None:
        super().__init__()
        self.backend = backend or BagelBackendConfig()
        self.hidden_size = config.hidden_size

        self.self_attn = PackedAttention(config, layer_idx, backend=self.backend)
        self.mlp = Qwen2MLP(config, backend=self.backend)
        self.input_layernorm = _initialize_rms_norm(self.backend, config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _initialize_rms_norm(self.backend, config.hidden_size, config.rms_norm_eps)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = packed_sequence
        packed_sequence = self.input_layernorm(packed_sequence)

        packed_sequence = self.self_attn(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
        )
        packed_sequence = residual + packed_sequence

        residual = packed_sequence
        packed_sequence = self.post_attention_layernorm(packed_sequence)
        packed_sequence = self.mlp(packed_sequence)
        packed_sequence = residual + packed_sequence
        return packed_sequence

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
    ) -> Tuple[torch.Tensor, Optional[NaiveCache]]:
        residual = packed_query_sequence
        packed_query_sequence = self.input_layernorm(packed_query_sequence)

        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
        )
        packed_query_sequence = residual + packed_query_sequence

        residual = packed_query_sequence
        packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
        packed_query_sequence = self.mlp(packed_query_sequence)
        packed_query_sequence = residual + packed_query_sequence
        return packed_query_sequence, past_key_values


class Qwen2MoTDecoderLayer(nn.Module):
    """MoT decoder: every norm/MLP is duplicated into ``*_moe_gen`` siblings."""

    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: Optional[int] = None,
        attn_module: type = PackedAttentionMoT,
        backend: Optional[BagelBackendConfig] = None,
    ) -> None:
        super().__init__()
        self.backend = backend or BagelBackendConfig()
        self.hidden_size = config.hidden_size
        self.freeze_und = getattr(config, "freeze_und", False)

        self.self_attn = attn_module(config, layer_idx, backend=self.backend)

        self.mlp = Qwen2MLP(config, backend=self.backend)
        self.mlp_moe_gen = Qwen2MLP(config, backend=self.backend)
        self.input_layernorm = _initialize_rms_norm(self.backend, config.hidden_size, config.rms_norm_eps)
        self.input_layernorm_moe_gen = _initialize_rms_norm(self.backend, config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = _initialize_rms_norm(self.backend, config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm_moe_gen = _initialize_rms_norm(
            self.backend, config.hidden_size, config.rms_norm_eps
        )

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_und_token_indexes: torch.LongTensor,
        packed_gen_token_indexes: torch.LongTensor,
        mot_perm: Optional[torch.LongTensor] = None,
        mot_inv: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if mot_perm is not None:
            # MLM-style slice+concat routing at the scalar und/gen boundary.
            lund = packed_und_token_indexes.shape[0]
            residual = packed_sequence
            packed_sequence_ = torch.cat(
                [
                    self.input_layernorm(packed_sequence[:lund]),
                    self.input_layernorm_moe_gen(packed_sequence[lund:]),
                ],
                dim=0,
            )
            packed_sequence_ = self.self_attn(
                packed_sequence=packed_sequence_,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_embeddings=packed_position_embeddings,
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
                mot_perm=mot_perm,
                mot_inv=mot_inv,
            )
            if self.freeze_und:
                packed_sequence_[:lund] = packed_sequence_[:lund].detach()
            packed_sequence = residual + packed_sequence_
            residual = packed_sequence
            und_out = self.mlp(self.post_attention_layernorm(packed_sequence[:lund]))
            gen_out = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(packed_sequence[lund:]))
            mlp_out = torch.cat([und_out, gen_out], dim=0)
            if self.freeze_und:
                mlp_out[:lund] = mlp_out[:lund].detach()
            packed_sequence = residual + mlp_out
            return packed_sequence

        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        _index_put_matching_dtype(
            packed_sequence_,
            packed_und_token_indexes,
            self.input_layernorm(packed_sequence[packed_und_token_indexes]),
        )
        _index_put_matching_dtype(
            packed_sequence_,
            packed_gen_token_indexes,
            self.input_layernorm_moe_gen(packed_sequence[packed_gen_token_indexes]),
        )

        packed_sequence_ = self.self_attn(
            packed_sequence=packed_sequence_,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_embeddings=packed_position_embeddings,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
        )
        if self.freeze_und:
            packed_sequence_[packed_und_token_indexes] = packed_sequence_[packed_und_token_indexes].detach()
        packed_sequence = residual + packed_sequence_

        residual = packed_sequence
        packed_sequence_ = packed_sequence.new_zeros(packed_sequence.shape)
        _index_put_matching_dtype(
            packed_sequence_,
            packed_und_token_indexes,
            self.mlp(self.post_attention_layernorm(packed_sequence[packed_und_token_indexes])),
        )
        if self.freeze_und:
            packed_sequence_[packed_und_token_indexes] = packed_sequence_[packed_und_token_indexes].detach()

        _index_put_matching_dtype(
            packed_sequence_,
            packed_gen_token_indexes,
            self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(packed_sequence[packed_gen_token_indexes])),
        )
        packed_sequence = residual + packed_sequence_
        return packed_sequence

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        packed_text_indexes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[NaiveCache]]:
        residual = packed_query_sequence
        if mode == "und":
            packed_query_sequence = self.input_layernorm(packed_query_sequence)
        elif mode == "gen":
            packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
            _index_put_matching_dtype(
                packed_query_sequence_,
                packed_text_indexes,
                self.input_layernorm(packed_query_sequence[packed_text_indexes]),
            )
            _index_put_matching_dtype(
                packed_query_sequence_,
                packed_vae_token_indexes,
                self.input_layernorm_moe_gen(packed_query_sequence[packed_vae_token_indexes]),
            )
            packed_query_sequence = packed_query_sequence_
        else:
            raise ValueError(f"mode must be 'und' or 'gen', got {mode!r}")

        packed_query_sequence, past_key_values = self.self_attn(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_embeddings=packed_query_position_embeddings,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )
        packed_query_sequence = residual + packed_query_sequence

        residual = packed_query_sequence
        if mode == "und":
            packed_query_sequence = self.post_attention_layernorm(packed_query_sequence)
            packed_query_sequence = self.mlp(packed_query_sequence)
        elif mode == "gen":
            packed_text_query_sequence = packed_query_sequence[packed_text_indexes]
            packed_vae_query_sequence = packed_query_sequence[packed_vae_token_indexes]
            packed_text_query_sequence = self.post_attention_layernorm(packed_text_query_sequence).to(torch.bfloat16)
            packed_vae_query_sequence = self.post_attention_layernorm_moe_gen(packed_vae_query_sequence).to(
                torch.bfloat16
            )

            packed_query_sequence_ = torch.zeros_like(packed_query_sequence).to(torch.bfloat16)
            _index_put_matching_dtype(packed_query_sequence_, packed_text_indexes, self.mlp(packed_text_query_sequence))
            _index_put_matching_dtype(
                packed_query_sequence_, packed_vae_token_indexes, self.mlp_moe_gen(packed_vae_query_sequence)
            )
            packed_query_sequence = packed_query_sequence_

        packed_query_sequence = residual + packed_query_sequence
        return packed_query_sequence, past_key_values


# ---------------------------------------------------------------------------
# Top-level model (packed navit variant).
# ---------------------------------------------------------------------------


_DECODER_LAYER_DICT = {
    "Qwen2DecoderLayer": Qwen2DecoderLayer,
    "Qwen2MoTDecoderLayer": partial(Qwen2MoTDecoderLayer, attn_module=PackedAttentionMoT),
}


class Qwen2PreTrainedModel(PreTrainedModel):
    """Abstract base class — mirrors HF Qwen2PreTrainedModel flags."""

    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer", "Qwen2MoTDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Qwen2Model(Qwen2PreTrainedModel):
    """Packed-sequence Qwen2 backbone.

    Selects ``Qwen2DecoderLayer`` or ``Qwen2MoTDecoderLayer`` per-layer based
    on ``config.layer_module`` (string -> class). When the MoT variant is
    active, ``self.use_moe == True`` and an extra ``norm_moe_gen`` sibling is
    created for the final RMSNorm.
    """

    def __init__(self, config: Qwen2Config, backend: Optional[BagelBackendConfig] = None) -> None:
        super().__init__(config)
        self.backend = backend or BagelBackendConfig()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        layer_module_name = getattr(config, "layer_module", "Qwen2DecoderLayer")
        self.use_moe = "Mo" in layer_module_name

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        layer_module = _DECODER_LAYER_DICT[layer_module_name]
        self.layers = nn.ModuleList(
            [layer_module(config, layer_idx, backend=self.backend) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = _initialize_rms_norm(self.backend, config.hidden_size, config.rms_norm_eps)
        if self.use_moe:
            self.norm_moe_gen = _initialize_rms_norm(self.backend, config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        self.post_init()

    def init_moe(self) -> None:
        """Copy UND weights into MoE-gen siblings (Stage 1 cold-start seeding)."""
        sd = self.state_dict()
        for name, param in self.named_parameters():
            if "moe_gen" in name:
                original_name = name.replace("_moe_gen", "")
                param.data.copy_(sd[original_name].data)

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
        mot_perm: Optional[torch.LongTensor] = None,
        mot_inv: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        freeze_und = getattr(self.config, "freeze_und", False)
        if freeze_und:
            if mot_perm is not None and packed_und_token_indexes is not None:
                lund = packed_und_token_indexes.shape[0]
                packed_sequence[:lund] = packed_sequence[:lund].detach()
            else:
                packed_sequence[packed_und_token_indexes] = packed_sequence[packed_und_token_indexes].detach()

        cos, sin = self.rotary_emb(packed_sequence, packed_position_ids.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_position_embeddings = (cos, sin)

        extra_inputs = {}
        if self.use_moe:
            if packed_und_token_indexes is None:
                raise ValueError("packed_und_token_indexes is required for MoT layers")
            if packed_gen_token_indexes is None:
                packed_gen_token_indexes = packed_und_token_indexes.new_ones(size=[0])
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_gen_token_indexes,
            )
            if mot_perm is not None:
                extra_inputs.update(mot_perm=mot_perm, mot_inv=mot_inv)

        for decoder_layer in self.layers:
            packed_sequence = decoder_layer(
                packed_sequence=packed_sequence,
                sample_lens=sample_lens,
                attention_mask=attention_mask,
                packed_position_embeddings=packed_position_embeddings,
                **extra_inputs,
            )

        if self.use_moe:
            if mot_perm is not None:
                lund = packed_und_token_indexes.shape[0]
                packed_sequence_ = torch.cat(
                    [self.norm(packed_sequence[:lund]), self.norm_moe_gen(packed_sequence[lund:])], dim=0
                )
                if freeze_und:
                    packed_sequence_[:lund] = packed_sequence_[:lund].detach()
                return packed_sequence_
            packed_sequence_ = torch.zeros_like(packed_sequence)
            _index_put_matching_dtype(
                packed_sequence_,
                packed_und_token_indexes,
                self.norm(packed_sequence[packed_und_token_indexes]),
            )
            if freeze_und:
                packed_sequence_[packed_und_token_indexes] = packed_sequence_[packed_und_token_indexes].detach()
            _index_put_matching_dtype(
                packed_sequence_,
                packed_gen_token_indexes,
                self.norm_moe_gen(packed_sequence[packed_gen_token_indexes]),
            )
            return packed_sequence_
        return self.norm(packed_sequence)

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        packed_text_indexes: Optional[torch.Tensor] = None,
    ) -> BaseNavitOutputWithPast:
        cos, sin = self.rotary_emb(packed_query_sequence, packed_query_position_ids.unsqueeze(0))
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
        packed_query_position_embeddings = (cos, sin)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs.update(mode=mode)
            if mode == "gen":
                if packed_vae_token_indexes is None or packed_text_indexes is None:
                    raise ValueError("gen-mode inference requires packed_vae_token_indexes and packed_text_indexes")
                extra_inputs.update(
                    packed_vae_token_indexes=packed_vae_token_indexes,
                    packed_text_indexes=packed_text_indexes,
                )

        for decoder_layer in self.layers:
            packed_query_sequence, past_key_values = decoder_layer(
                packed_query_sequence=packed_query_sequence,
                query_lens=query_lens,
                packed_query_position_embeddings=packed_query_position_embeddings,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=update_past_key_values,
                is_causal=is_causal,
                **extra_inputs,
            )

        if self.use_moe:
            if mode == "und":
                packed_query_sequence = self.norm(packed_query_sequence)
            elif mode == "gen":
                packed_query_sequence_ = torch.zeros_like(packed_query_sequence)
                _index_put_matching_dtype(
                    packed_query_sequence_,
                    packed_text_indexes,
                    self.norm(packed_query_sequence[packed_text_indexes]),
                )
                _index_put_matching_dtype(
                    packed_query_sequence_,
                    packed_vae_token_indexes,
                    self.norm_moe_gen(packed_query_sequence[packed_vae_token_indexes]),
                )
                packed_query_sequence = packed_query_sequence_
        else:
            packed_query_sequence = self.norm(packed_query_sequence)

        return BaseNavitOutputWithPast(
            packed_query_sequence=packed_query_sequence,
            past_key_values=past_key_values,
        )


class Qwen2ForCausalLM(Qwen2PreTrainedModel):
    """Packed-sequence Qwen2 LM head wrapper."""

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen2Config, backend: Optional[BagelBackendConfig] = None) -> None:
        super().__init__(config)
        self.backend = backend or BagelBackendConfig()
        self.model = Qwen2Model(config, backend=self.backend)
        self.vocab_size = config.vocab_size
        self.lm_head = _initialize_linear(
            self.backend,
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )
        self.post_init()

    def init_moe(self) -> None:
        """Seed ``*_moe_gen`` parameters from their UND siblings (Stage 1 cold-start)."""
        sd = self.state_dict()
        for name, param in self.named_parameters():
            if "moe_gen" in name:
                original_name = name.replace("_moe_gen", "")
                param.data.copy_(sd[original_name].data)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: nn.Module) -> None:
        self.model = decoder

    def get_decoder(self) -> nn.Module:
        return self.model

    def forward(self, *args, **kwargs) -> Union[torch.Tensor, BaseNavitOutputWithPast]:
        if self.training:
            return self.forward_train(*args, **kwargs)
        return self.forward_inference(*args, **kwargs)

    def forward_train(
        self,
        packed_sequence: torch.Tensor,
        sample_lens: List[int],
        attention_mask,
        packed_position_ids: torch.Tensor,
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
        mot_perm: Optional[torch.LongTensor] = None,
        mot_inv: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # BAGEL's forward_train returns the raw hidden states; the upstream
        # training loop applies lm_head only on text-token positions to save
        # memory. Callers that want CE over the full packed sequence should
        # apply ``self.lm_head`` themselves.
        return self.model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            packed_position_ids=packed_position_ids,
            attention_mask=attention_mask,
            packed_und_token_indexes=packed_und_token_indexes,
            packed_gen_token_indexes=packed_gen_token_indexes,
            mot_perm=mot_perm,
            mot_inv=mot_inv,
        )

    def forward_inference(
        self,
        packed_query_sequence: torch.Tensor,
        query_lens: torch.Tensor,
        packed_query_position_ids: torch.Tensor,
        packed_query_indexes: torch.Tensor,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.Tensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_vae_token_indexes: Optional[torch.Tensor] = None,
        packed_text_indexes: Optional[torch.Tensor] = None,
    ) -> BaseNavitOutputWithPast:
        return self.model(
            packed_query_sequence=packed_query_sequence,
            query_lens=query_lens,
            packed_query_position_ids=packed_query_position_ids,
            packed_query_indexes=packed_query_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=update_past_key_values,
            is_causal=is_causal,
            mode=mode,
            packed_vae_token_indexes=packed_vae_token_indexes,
            packed_text_indexes=packed_text_indexes,
        )

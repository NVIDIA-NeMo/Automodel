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

"""Rotary Position Embedding utilities for Llama and Qwen2 models.

This module provides RoPE implementation following HuggingFace's architecture.

API:
    rotary_emb = RotaryEmbedding(config)
    cos, sin = rotary_emb(x, position_ids)  # Returns (cos, sin) tuple
    q, k = apply_rotary_pos_emb(q, k, cos, sin)  # Applies RoPE

Supports both:
- LlamaConfig: uses config.rope_theta and config.rope_scaling
- Qwen2Config: uses config.rope_parameters["rope_theta"] and config.rope_parameters

Note: gpt_oss and deepseek_v3 have their own specialized rope_utils.py
with model-specific optimizations (YaRN, MLA, etc.).
"""

import math
from typing import Optional

import torch
from torch import nn


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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim]
        k: Key tensor [batch, num_kv_heads, seq_len, head_dim]
        cos: Cosine embeddings [batch, seq_len, head_dim]
        sin: Sine embeddings [batch, seq_len, head_dim]

    Returns:
        Rotated (q, k) tensors
    """
    cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _get_rope_config(config) -> tuple[float, dict]:
    """Extract rope parameters from config (handles both Llama and Qwen2 formats).

    Returns:
        Tuple of (rope_theta, rope_scaling_dict)
    """
    # Qwen2 uses config.rope_parameters, Llama uses config.rope_theta + config.rope_scaling
    if hasattr(config, "rope_parameters") and config.rope_parameters:
        rope_params = config.rope_parameters
        base = rope_params.get("rope_theta", 10000.0)
        rope_scaling = rope_params  # rope_parameters contains scaling info too
    else:
        base = getattr(config, "rope_theta", 10000.0)
        rope_scaling = getattr(config, "rope_scaling", {}) or {}
    return base, rope_scaling


def _compute_default_inv_freq(config, device: Optional[torch.device] = None) -> tuple[torch.Tensor, float]:
    """Computes inverse frequencies for standard RoPE."""
    base, _ = _get_rope_config(config)
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    dim = int(dim * partial_rotary_factor)

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32) / dim))
    return inv_freq, 1.0


def _compute_llama3_inv_freq(config, device: Optional[torch.device] = None) -> tuple[torch.Tensor, float]:
    """Computes inverse frequencies for Llama3-style RoPE with smooth interpolation."""
    inv_freq, _ = _compute_default_inv_freq(config, device)

    _, rope_scaling = _get_rope_config(config)
    factor = rope_scaling.get("factor", 1.0)
    low_freq_factor = rope_scaling.get("low_freq_factor", 1.0)
    high_freq_factor = rope_scaling.get("high_freq_factor", 4.0)
    original_max_pos = rope_scaling.get("original_max_position_embeddings", config.max_position_embeddings)

    low_freq_wavelen = original_max_pos / low_freq_factor
    high_freq_wavelen = original_max_pos / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    smooth = (original_max_pos / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smooth = smooth.clamp(0, 1)

    scaled_inv_freq = inv_freq / factor
    inv_freq = torch.where(
        wavelen > low_freq_wavelen,
        inv_freq,
        torch.where(wavelen < high_freq_wavelen, scaled_inv_freq, (1 - smooth) * scaled_inv_freq + smooth * inv_freq),
    )
    return inv_freq, 1.0


class LlamaRotaryEmbedding(nn.Module):
    """Rotary Position Embedding module for Llama and Qwen2 models.

    Returns (cos, sin) tuple for use with apply_rotary_pos_emb.

    Usage:
        rotary_emb = RotaryEmbedding(config)
        cos, sin = rotary_emb(x, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    """

    inv_freq: torch.Tensor

    def __init__(self, config, device: Optional[torch.device] = None):
        super().__init__()
        self.max_seq_len_cached = 0
        self.dtype = getattr(config, "torch_dtype", None) or torch.float32

        # Map rope types to their respective computation functions
        rope_functions = {
            "default": _compute_default_inv_freq,
            "llama3": _compute_llama3_inv_freq,
        }

        # Determine rope_type and compute inv_freq
        _, rope_scaling = _get_rope_config(config)
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))

        # Fallback to llama3 as the robust default if type is not in our map
        compute_fn = rope_functions.get(rope_type, _compute_llama3_inv_freq)
        inv_freq, self.attention_scaling = compute_fn(config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("_cos_cache", None, persistent=False)
        self.register_buffer("_sin_cache", None, persistent=False)

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        """Build cos/sin cache in config dtype for positions [0, seq_len)."""
        self.max_seq_len_cached = seq_len

        # Compute in float32 for precision, then convert to target dtype
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # [seq, head_dim]

        self._cos_cache = (emb.cos() * self.attention_scaling).to(self.dtype)
        self._sin_cache = (emb.sin() * self.attention_scaling).to(self.dtype)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) for the given positions.

        Args:
            x: Input tensor (used for device and dtype)
            position_ids: Position IDs tensor [batch, seq_len]

        Returns:
            (cos, sin) tensors [batch, seq_len, head_dim]
        """
        seq_len = position_ids.shape[-1]

        # Build cache if needed
        if self._cos_cache is None or seq_len > self.max_seq_len_cached:
            self._build_cache(seq_len, x.device)

        # Slice cache and expand for batch
        cos = self._cos_cache[:seq_len].unsqueeze(0).expand(position_ids.shape[0], -1, -1)
        sin = self._sin_cache[:seq_len].unsqueeze(0).expand(position_ids.shape[0], -1, -1)

        return cos, sin


# Aliases for HuggingFace compatibility
RotaryEmbedding = LlamaRotaryEmbedding
Qwen2RotaryEmbedding = LlamaRotaryEmbedding


__all__ = [
    "RotaryEmbedding",
    "LlamaRotaryEmbedding",
    "Qwen2RotaryEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",
]

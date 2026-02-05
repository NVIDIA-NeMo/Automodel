# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_linear_module,
)
from nemo_automodel.components.models.gpt_oss.rope_utils import apply_rotary_emb_qk


class Step3p5RMSNorm(nn.Module):
    """RMSNorm with (weight + 1) scaling used by Step3p5.

    Unlike standard RMSNorm which uses `x_normed * weight`, Step3p5 uses
    `x_normed * (weight + 1)`. The weight is initialized to zeros,
    so initially the scaling factor is 1.

    Note: Cannot use TE's fused RMSNorm because the (weight + 1) adjustment
    cannot be intercepted.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def reset_parameters(self) -> None:
        """Reset parameters to initial state (zeros)."""
        nn.init.zeros_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(variance + self.variance_epsilon)
        normed = normed * (self.weight.float() + 1)
        return normed.to(dtype)


class Step3p5RotaryEmbedding(nn.Module):
    """Rotary embedding for Step3p5 with per-layer theta and partial rotary factor support."""

    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        # Get per-layer rope_theta
        rope_theta = config.rope_theta
        if isinstance(rope_theta, list):
            self.base = rope_theta[layer_idx]
        else:
            self.base = rope_theta

        # Get per-layer partial_rotary_factor
        partial_rotary_factors = getattr(config, "partial_rotary_factors", None)
        if partial_rotary_factors is not None:
            self.partial_rotary_factor = partial_rotary_factors[layer_idx]
        else:
            self.partial_rotary_factor = 1.0

        # Compute head_dim for this layer based on attention settings
        layer_types = getattr(config, "layer_types", [])
        is_sliding = layer_types and layer_types[layer_idx] == "sliding_attention"

        if is_sliding:
            num_heads = config.attention_other_setting.get("num_attention_heads", config.num_attention_heads)
        else:
            num_heads = config.num_attention_heads

        self.head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        self.max_position_embeddings = config.max_position_embeddings

        # Register the inverse frequency buffer
        inv_freq = self._compute_inv_freq()
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_inv_freq(self) -> torch.Tensor:
        """Compute inverse frequencies for rotary embeddings."""
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim))
        return inv_freq

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin for rotary embeddings.

        Args:
            x: Input tensor (used for dtype and device).
            position_ids: Position indices [batch_size, seq_len].

        Returns:
            Tuple of (cos, sin) tensors.
        """
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)
        inv_freq_expanded = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Step3p5MLP(nn.Module):
    """Step3p5 MLP with SwiGLU activation and optional clamping."""

    def __init__(
        self,
        config: Any,
        backend: BackendConfig,
        intermediate_size: int | None = None,
        swiglu_limit: float | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size
        self.swiglu_limit = swiglu_limit

        self.gate_proj = initialize_linear_module(backend.linear, self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = initialize_linear_module(backend.linear, self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = initialize_linear_module(backend.linear, self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.up_proj(x)
        gate = F.silu(self.gate_proj(x))

        if self.swiglu_limit is not None:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)

        return self.down_proj(gate * up)

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        for linear in [self.gate_proj, self.up_proj, self.down_proj]:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class Step3p5Attention(nn.Module):
    """Step3p5 attention with Q/K per-head RMSNorm, optional head-wise gate, and alternating attention patterns.

    Key features:
    - Q/K per-head normalization using Step3p5RMSNorm
    - Optional head-wise attention gate (g_proj + sigmoid)
    - Per-layer RoPE theta and partial_rotary_factors
    - Sliding window based on layer_types config
    """

    def __init__(self, config: Any, layer_idx: int, backend: BackendConfig) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.backend = backend

        # Determine attention configuration based on layer_types
        layer_types = getattr(config, "layer_types", [])
        is_sliding = layer_types and layer_types[layer_idx] == "sliding_attention"

        if is_sliding:
            self.num_heads = config.attention_other_setting.get("num_attention_heads", config.num_attention_heads)
            self.num_kv_heads = config.attention_other_setting.get("num_attention_groups", config.num_attention_groups)
            self.sliding_window = config.sliding_window
        else:
            self.num_heads = config.num_attention_heads
            self.num_kv_heads = config.num_attention_groups
            self.sliding_window = None

        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        # Projections
        attention_bias = getattr(config, "attention_bias", False)
        self.q_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_heads * self.head_dim, attention_bias
        )
        self.k_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_kv_heads * self.head_dim, attention_bias
        )
        self.v_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_kv_heads * self.head_dim, attention_bias
        )
        self.o_proj = initialize_linear_module(
            backend.linear, self.num_heads * self.head_dim, config.hidden_size, attention_bias
        )

        # Per-head Q/K normalization using Step3p5RMSNorm
        self.q_norm = Step3p5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Step3p5RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Optional head-wise attention gate
        self.use_head_wise_attn_gate = getattr(config, "use_head_wise_attn_gate", False)
        if self.use_head_wise_attn_gate:
            self.g_proj = initialize_linear_module(backend.linear, config.hidden_size, self.num_heads, bias=False)
        else:
            self.g_proj = None

        # Per-layer rotary embedding
        self.rotary_emb = Step3p5RotaryEmbedding(config, layer_idx)

        # Check if RoPE should be applied for this layer
        # Empty list or None means all layers use RoPE
        use_rope_layers = getattr(config, "use_rope_layers", None)
        if use_rope_layers is not None and len(use_rope_layers) > layer_idx:
            self.use_rope = use_rope_layers[layer_idx]
        else:
            self.use_rope = True

        # Attention implementation
        softmax_scale = self.head_dim**-0.5
        self.attn_module, self.attn_func = initialize_attn_module_and_func(
            attn_impl=backend.attn,
            num_attention_heads=self.num_heads,
            num_qk_channels=self.head_dim,
            num_v_channels=self.head_dim,
            softmax_scale=softmax_scale,
            num_gqa_groups=self.num_kv_heads,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if len(x.shape) == 2:
            qkv_format = "thd"
            num_tokens = x.shape[0]
        else:
            qkv_format = "bshd"
            bsz, seqlen, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if qkv_format == "thd":
            q = q.view(num_tokens, self.num_heads, self.head_dim)
            k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
            v = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        else:
            q = q.view(bsz, seqlen, self.num_heads, self.head_dim)
            k = k.view(bsz, seqlen, self.num_kv_heads, self.head_dim)
            v = v.view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        # Per-head RMSNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply rotary embeddings if enabled
        if self.use_rope:
            q, k = apply_rotary_emb_qk(
                q,
                k,
                freqs_cis,
                format=qkv_format,
                rope_fusion=self.backend.rope_fusion,
                cu_seqlens=attn_kwargs.get("cu_seqlens", None),
                cp_size=attn_kwargs.get("cp_size", 1),
                cp_rank=attn_kwargs.get("cp_rank", 0),
            )

        # Compute head-wise gate if enabled
        if self.use_head_wise_attn_gate:
            gate_states = self.g_proj(x)

        # Backend-specific attention
        window_size = (-1, 0) if self.sliding_window is None else (self.sliding_window, 0)
        q, k, v, _attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q, k, v, attention_mask, self.backend.attn, window_size=window_size, **attn_kwargs
        )
        out = self.attn_func(q, k, v, **_attn_kwargs)
        out = postprocess_output_for_attn(out, self.backend.attn)

        # Apply head-wise gate
        if self.use_head_wise_attn_gate:
            if qkv_format == "thd":
                out = out.view(num_tokens, self.num_heads, self.head_dim)
                out = out * gate_states.unsqueeze(-1).sigmoid()
                out = out.view(num_tokens, -1)
            else:
                out = out.view(bsz, seqlen, self.num_heads, self.head_dim)
                out = out * gate_states.unsqueeze(-1).sigmoid()
                out = out.view(bsz, seqlen, -1)
        else:
            flatten_dim = 2 if qkv_format == "bshd" else 1
            out = out.flatten(flatten_dim)

        out = self.o_proj(out)
        return out

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        linear_list = [self.q_proj, self.k_proj, self.v_proj, self.o_proj]
        if self.g_proj is not None:
            linear_list.append(self.g_proj)

        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
            if hasattr(linear, "bias") and linear.bias is not None:
                nn.init.zeros_(linear.bias)

        for norm in (self.q_norm, self.k_norm):
            norm.reset_parameters()

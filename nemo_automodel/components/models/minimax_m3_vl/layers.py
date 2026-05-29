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

"""MiniMax M3 VL text-backbone layers.

Stage 1 covers the dense + MoE text path (no sparse-attention index branch and
no MTP).  Mirrors the canonical sglang reference
``sglang.srt.models.minimax_m3`` (``MiniMaxM3Attention`` / ``MiniMaxM3MLP`` /
``MiniMaxM3MoE`` / ``MiniMaxM3DecoderLayer``):

* per-head **Gemma** RMSNorm on Q/K (``qk_norm_type='per_head'``,
  ``use_gemma_norm=True``),
* partial RoPE (``rotary_dim=64`` of ``head_dim=128``) reusing the gpt_oss
  rotary utilities (as the existing ``minimax_m2`` backbone does),
* SwiGLU-OAI activation ``gate * sigmoid(alpha * gate) * (up + 1)`` with gate
  clamped ``max=limit`` and up clamped ``+/-limit`` for dense and shared experts,
* per-layer dense-vs-MoE selection from ``moe_layer_freq``.
"""

from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.gpt_oss.rope_utils import apply_rotary_emb_qk
from nemo_automodel.components.moe.layers import MoE, MoEConfig


class MiniMaxM3RMSNorm(nn.Module):
    """RMSNorm with optional Gemma-style zero-centered gamma (``x_normed * (1 + w)``).

    When ``gemma=True`` the learnable weight is centered at 0 and the effective
    scale is ``1 + weight`` (matching HF ``GemmaRMSNorm`` and the sglang M3
    reference). Used both for hidden-size norms and, with ``dim=head_dim``, for
    per-head Q/K normalization (the input is normalized over its last dim, so a
    ``[..., num_heads, head_dim]`` tensor is normalized independently per head).
    """

    def __init__(self, dim: int, eps: float = 1e-6, gemma: bool = True):
        super().__init__()
        self.eps = eps
        self.gemma = gemma
        self.weight = nn.Parameter(torch.zeros(dim) if gemma else torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        weight = self.weight.float()
        if self.gemma:
            weight = weight + 1.0
        return (x * weight).to(dtype)

    def reset_parameters(self) -> None:
        if self.gemma:
            nn.init.zeros_(self.weight)
        else:
            nn.init.ones_(self.weight)


def swiglu_oai(gate: torch.Tensor, up: torch.Tensor, alpha: float, limit: float) -> torch.Tensor:
    """GPT-OSS / MiniMax-M3 SwiGLU-OAI: ``gate * sigmoid(alpha * gate) * (up + 1)``.

    Gate is clamped ``max=limit`` and up is clamped ``+/-limit`` (when
    ``limit > 0``), computed in fp32 and cast back. Equivalent to sglang's
    ``swiglu_no_interleaved_with_alpha_and_limit``.
    """
    dtype = gate.dtype
    gate = gate.float()
    up = up.float()
    if limit > 0.0:
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
    out = gate * torch.sigmoid(alpha * gate) * (up + 1.0)
    return out.to(dtype)


class MiniMaxM3MLP(nn.Module):
    """Dense / shared-expert MLP with SwiGLU-OAI activation (separate gate/up/down)."""

    def __init__(self, config: Any, intermediate_size: int, backend: BackendConfig):
        super().__init__()
        self.alpha = float(getattr(config, "swiglu_alpha", 1.702))
        self.limit = float(getattr(config, "swiglu_limit", 7.0))
        self.gate_proj = initialize_linear_module(backend.linear, config.hidden_size, intermediate_size, bias=False)
        self.up_proj = initialize_linear_module(backend.linear, config.hidden_size, intermediate_size, bias=False)
        self.down_proj = initialize_linear_module(backend.linear, intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(swiglu_oai(self.gate_proj(x), self.up_proj(x), self.alpha, self.limit))

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        for linear in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class MiniMaxM3Attention(nn.Module):
    """MiniMax M3 GQA attention with per-head Gemma Q/K norm and partial RoPE.

    Stage 1: dense attention only. The sparse-attention index branch
    (``index_q/k_proj`` + block-level top-k selection) is added in Stage 2.
    """

    def __init__(self, config: Any, backend: BackendConfig):
        super().__init__()
        self.backend = backend
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or config.hidden_size // self.num_heads
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        gemma = getattr(config, "use_gemma_norm", False)

        self.q_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = initialize_linear_module(
            backend.linear, self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

        if self.use_qk_norm:
            assert getattr(config, "qk_norm_type", "per_head") == "per_head", "M3 only supports per_head QK norm"
            self.q_norm = MiniMaxM3RMSNorm(self.head_dim, eps=config.rms_norm_eps, gemma=gemma)
            self.k_norm = MiniMaxM3RMSNorm(self.head_dim, eps=config.rms_norm_eps, gemma=gemma)
        else:
            self.q_norm = None
            self.k_norm = None

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
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if len(x.shape) == 2:
            qkv_format = "thd"
            num_tokens = x.shape[0]
            q = self.q_proj(x).view(num_tokens, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(num_tokens, self.num_kv_heads, self.head_dim)
            v = self.v_proj(x).view(num_tokens, self.num_kv_heads, self.head_dim)
        else:
            qkv_format = "bshd"
            bsz, seqlen, _ = x.size()
            q = self.q_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim)
            v = self.v_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        # Per-head QK norm (over head_dim) is applied before RoPE, matching the
        # sglang reference (``_qk_norm`` then ``rotary_emb``).
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

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

        q, k, v, _attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q, k, v, attention_mask, self.backend.attn, **attn_kwargs
        )
        out = self.attn_func(q, k, v, **_attn_kwargs)
        out = postprocess_output_for_attn(out, self.backend.attn)

        flatten_dim = 2 if qkv_format == "bshd" else 1
        return self.o_proj(out.flatten(flatten_dim))

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02):
        for linear in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
            self.k_norm.reset_parameters()


class Block(nn.Module):
    """MiniMax M3 decoder block: attention + (dense MLP or MoE) with Gemma norms.

    ``moe_layer_freq[layer_idx] == 0`` -> dense ``MiniMaxM3MLP`` (with
    ``dense_intermediate_size``); otherwise a routed ``MoE`` plus a separate
    SwiGLU-OAI shared expert (kept M3-local rather than using ``MoE``'s built-in
    shared expert, whose generic ``MLP`` does not implement SwiGLU-OAI).
    """

    def __init__(self, layer_idx: int, config: Any, moe_config: MoEConfig, backend: BackendConfig):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = MiniMaxM3Attention(config, backend)

        moe_layer_freq = getattr(config, "moe_layer_freq", None)
        self.is_moe_layer = True if moe_layer_freq is None else moe_layer_freq[layer_idx] != 0

        if self.is_moe_layer:
            self.mlp = MoE(moe_config, backend)
            n_shared = getattr(config, "n_shared_experts", 0) or 0
            if n_shared > 0:
                shared_inter = getattr(config, "shared_intermediate_size", config.intermediate_size) * n_shared
                self.shared_experts = MiniMaxM3MLP(config, shared_inter, backend)
            else:
                self.shared_experts = None
        else:
            self.mlp = MiniMaxM3MLP(config, config.dense_intermediate_size, backend)
            self.shared_experts = None

        gemma = getattr(config, "use_gemma_norm", False)
        self.input_layernorm = MiniMaxM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma=gemma)
        self.post_attention_layernorm = MiniMaxM3RMSNorm(config.hidden_size, eps=config.rms_norm_eps, gemma=gemma)

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if attention_mask is not None and padding_mask is None:
            padding_mask = attention_mask.bool().logical_not()

        attn_out = self.self_attn(
            x=self.input_layernorm(x),
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            **attn_kwargs,
        )
        x = x + attn_out

        normed = self.post_attention_layernorm(x)
        if self.is_moe_layer:
            mlp_out = self.mlp(normed, padding_mask)
            if self.shared_experts is not None:
                mlp_out = mlp_out + self.shared_experts(normed)
        else:
            mlp_out = self.mlp(normed)
        x = x + mlp_out
        return x

    def init_weights(self, buffer_device: torch.device):
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
        self.self_attn.init_weights(buffer_device)
        self.mlp.init_weights(buffer_device)
        if self.shared_experts is not None:
            self.shared_experts.init_weights(buffer_device)

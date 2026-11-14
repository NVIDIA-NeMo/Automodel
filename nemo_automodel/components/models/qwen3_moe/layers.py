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

import logging
from typing import Any

import torch
from torch import nn
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.gpt_oss.rope_utils import apply_rotary_emb_qk
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

logger = logging.getLogger(__name__)


class Qwen3MoeAttention(nn.Module):
    """Qwen3 MoE attention with query/key RMSNorm, RoPE, and TE/SDPA/FA4 backends.

    Shapes:
      - Input: x -> [B, S, H]
      - Projections:
          q: [B, S, n_heads, head_dim]
          k/v: [B, S, n_kv_heads, head_dim] -> repeated to n_heads via groups
      - Output: [B, S, H]
    """

    def __init__(self, config: Qwen3MoeConfig, backend: BackendConfig):
        super().__init__()
        self.backend = backend

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)

        attention_bias = getattr(config, "attention_bias", False)

        # Thread dtype explicitly from config.torch_dtype so fp32 master
        # weights work even when construction is not wrapped in
        # local_torch_dtype().
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        self.q_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_heads * self.head_dim, attention_bias, dtype=dtype
        )
        self.k_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_kv_heads * self.head_dim, attention_bias, dtype=dtype
        )
        self.v_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_kv_heads * self.head_dim, attention_bias, dtype=dtype
        )
        self.o_proj = initialize_linear_module(
            backend.linear, self.num_heads * self.head_dim, config.hidden_size, attention_bias, dtype=dtype
        )

        # Per-head RMSNorm
        self.q_norm = initialize_rms_norm_module(backend.rms_norm, self.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        self.k_norm = initialize_rms_norm_module(backend.rms_norm, self.head_dim, eps=config.rms_norm_eps, dtype=dtype)

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

        # SDPA can compile the whole attention. Packed FA4 keeps variable-length
        # gathers, offsets and kernel dispatch outside the fixed-shape projections.
        self._compiled_forward = None
        self._compiled_qkv = None
        self._compiled_output = None
        if backend.compile_attn:
            if backend.attn == "fa4":
                if backend.linear != "torch" or backend.rms_norm != "torch" or backend.rope_fusion:
                    raise ValueError(
                        "Qwen3-MoE compiled FA4 requires linear='torch', rms_norm='torch', and rope_fusion=False"
                    )
                self._compiled_qkv = torch.compile(self._project_qkv, fullgraph=True, dynamic=False)
                self._compiled_output = torch.compile(self._project_output, fullgraph=True, dynamic=False)
            elif backend.attn != "sdpa":
                logger.warning(
                    "backend.compile_attn=True ignored: requires attn='sdpa' or 'fa4' (got attn='%s'); "
                    "TE fused attention is not fullgraph-compilable.",
                    backend.attn,
                )
            else:  # pragma: no cover - torch.compile path exercised on GPU benchmark runs only
                logger.warning(
                    "backend.compile_attn=True: torch.compile(fullgraph=True) the Qwen3-MoE attention (attn=sdpa)."
                )
                self._compiled_forward = torch.compile(self._forward_impl, fullgraph=True, dynamic=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """Apply attention to batched or THD hidden states.

        Args:
            x: Hidden states [batch, sequence, hidden] or [tokens, hidden].
            freqs_cis: RoPE tables [batch, sequence, rotary_dim] or [tokens,
                rotary_dim]; fused TE RoPE uses [sequence, 1, 1, rotary_dim].
            attention_mask: Optional [batch, sequence] padding/document IDs or
                [batch, 1, sequence, sequence] dense mask accepted by the backend.
            **attn_kwargs: Backend metadata, including int32 cu_seqlens [documents + 1]
                and int64 packed_token_indices [valid_tokens] for packed FA4.

        Returns:
            Hidden states with the same shape and dtype as x.
        """
        if self._compiled_forward is not None:  # pragma: no cover - compiled path only on GPU benchmark runs
            return self._compiled_forward(x, freqs_cis=freqs_cis, attention_mask=attention_mask, **attn_kwargs)
        return self._forward_impl(x, freqs_cis=freqs_cis, attention_mask=attention_mask, **attn_kwargs)

    def _forward_impl(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """Project hidden states and execute the explicitly selected attention backend.

        Args:
            x: Hidden states [batch, sequence, hidden] or [tokens, hidden].
            freqs_cis: RoPE tables matching x's token axes; fused TE tables use
                [sequence, 1, 1, rotary_dim].
            attention_mask: Optional [batch, sequence] padding/document IDs or
                [batch, 1, sequence, sequence] dense mask.
            **attn_kwargs: Backend metadata, including int32 cu_seqlens [documents + 1]
                and int64 packed_token_indices [valid_tokens].

        Returns:
            Hidden states with the same shape and dtype as x.
        """
        if self._compiled_qkv is not None:
            # FA4's document offsets never enter the fixed-shape compiled graph.
            q, k, v = self._compiled_qkv(x, freqs_cis=freqs_cis)
        else:
            q, k, v = self._project_qkv(
                x,
                freqs_cis=freqs_cis,
                cu_seqlens=attn_kwargs.get("cu_seqlens"),
                cp_size=attn_kwargs.get("cp_size", 1),
                cp_rank=attn_kwargs.get("cp_rank", 0),
            )

        q, k, v, backend_kwargs = preprocess_args_and_kwargs_for_attn(
            q, k, v, attention_mask, self.backend.attn, **attn_kwargs
        )
        out = self.attn_func(q, k, v, **backend_kwargs)
        out = postprocess_output_for_attn(out, self.backend.attn)
        # TE already flattens heads. Use the original token axes so BSHD and
        # THD TE outputs retain their batch/sequence dimensions.
        out = out.flatten(x.ndim - 1)
        if self._compiled_output is not None:
            return self._compiled_output(out)
        return self._project_output(out)

    def _project_qkv(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        cp_size: int = 1,
        cp_rank: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project, normalize and rotate Q/K while retaining the padded token axes.

        Args:
            x: Hidden states [batch, sequence, hidden] or [tokens, hidden].
            freqs_cis: RoPE tables matching x's token axes; fused TE tables use
                [sequence, 1, 1, rotary_dim].
            cu_seqlens: Optional int32 [documents + 1] offsets for fused THD RoPE.
            cp_size: Context-parallel group size.
            cp_rank: Context-parallel rank.

        Returns:
            Q, K and V with x's leading token axes followed by [heads, head_dim];
            Q uses num_heads, while K/V use num_kv_heads.
        """
        qkv_format = "thd" if x.ndim == 2 else "bshd"

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(*x.shape[:-1], self.num_heads, self.head_dim)
        k = k.view(*x.shape[:-1], self.num_kv_heads, self.head_dim)
        v = v.view(*x.shape[:-1], self.num_kv_heads, self.head_dim)

        # Per-head RMSNorm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE (complex rotation)
        q, k = apply_rotary_emb_qk(
            q,
            k,
            freqs_cis,
            format=qkv_format,
            rope_fusion=self.backend.rope_fusion,
            cu_seqlens=cu_seqlens,
            cp_size=cp_size,
            cp_rank=cp_rank,
        )
        return q, k, v

    def _project_output(self, out: torch.Tensor) -> torch.Tensor:
        """Project attention output after FA4 has restored padded token positions.

        Args:
            out: Attention output [batch, sequence, heads * head_dim] or
                [tokens, heads * head_dim].

        Returns:
            Hidden states [batch, sequence, hidden] or [tokens, hidden].
        """
        return self.o_proj(out)

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02):
        linear_list = [self.q_proj, self.k_proj, self.v_proj, self.o_proj]
        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
            if hasattr(linear, "bias") and linear.bias is not None:
                nn.init.zeros_(linear.bias)
        for norm in (self.q_norm, self.k_norm):
            norm.reset_parameters()

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

"""DeepSeek V3.2 Layers.

Contains the DeepseekV32Indexer for top-k sparse attention selection
and DeepseekV32MLA which integrates the indexer with Multi-head Latent Attention.
"""

from typing import Any

import torch
from torch import nn

# Try to import fast_hadamard_transform, fall back to torch implementation
try:
    from fast_hadamard_transform import hadamard_transform

    _FAST_HADAMARD_AVAILABLE = True
except ImportError:
    _FAST_HADAMARD_AVAILABLE = False

    # Taken from https://github.com/HazyResearch/structured-nets/blob/master/pytorch/structure/hadamard.py#L26
    def hadamard_transform_torch(u, scale: float, normalize=False):
        """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
        n must be a power of 2.
        Parameters:
            u: Tensor of shape (..., n)
            normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
        Returns:
            product: Tensor of shape (..., n)
        """
        import math

        n = u.shape[-1]
        m = int(math.log2(n))
        assert n == 1 << m, "n must be a power of 2"
        x = u.unsqueeze(-1)
        for _ in range(m):
            x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
        x = x.squeeze(-2) / 2 ** (m / 2) if normalize else x.squeeze(-2)
        return x * scale

    def hadamard_transform(x: torch.Tensor, scale: float) -> torch.Tensor:
        """Fallback hadamard_transform when fast_hadamard_transform is not available."""
        return hadamard_transform_torch(x, scale)


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
from nemo_automodel.components.models.deepseek_v3.rope_utils import (
    apply_rotary_emb,
    yarn_get_mscale,
)
from nemo_automodel.components.models.deepseek_v32.config import DeepseekV32Config


def _rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation activation.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L424-L428

    Args:
        x: Input tensor (must be bfloat16).

    Returns:
        Rotated tensor.
    """
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size**-0.5)


class DeepseekV32Indexer(nn.Module):
    """Indexer for top-k sparse attention selection.

    Based on the official DeepSeek V3.2 training implementation. Computes attention
    scores between queries and keys with per-head weights, applies ReLU activation,
    then selects the top-k positions to attend to.

    Key features:
    - Uses LayerNorm (not RMSNorm) for key normalization
    - Has a weights_proj that learns per-head importance weights
    - Optional Hadamard transform (rotate_activation) on Q and K
    - ReLU activation on attention scores before weighting
    """

    def __init__(self, config: DeepseekV32Config, backend: BackendConfig):
        super().__init__()

        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.hidden_size = config.hidden_size
        self.softmax_scale = self.head_dim**-0.5

        self.backend = backend
        linear_impl = backend.linear

        # Project Q from q_lora residual -> num_heads * head_dim
        self.wq_b = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=self.q_lora_rank,
            out_features=self.num_heads * self.head_dim,
            bias=False,
        )

        # Project K from hidden states -> single head_dim (shared across heads)
        self.wk = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=self.hidden_size,
            out_features=self.head_dim,
            bias=False,
        )

        # LayerNorm for K (official uses LayerNorm, not RMSNorm)
        self.k_norm = nn.LayerNorm(self.head_dim)

        # Per-head weight projection from hidden states
        self.weights_proj = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=self.hidden_size,
            out_features=self.num_heads,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        q_resid: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """Compute top-k indices for sparse attention.

        Args:
            x: Hidden states [B, S, hidden] or [T, hidden] for thd format
            q_resid: Q lora residual from MLA [B, S, q_lora_rank] or [T, q_lora_rank]
            freqs_cis: RoPE frequencies
            attention_mask: Optional attention mask
            **attn_kwargs: Additional attention kwargs (cu_seqlens, etc.)

        Returns:
            topk_indices: Indices of top-k positions [B, S, topk] or [T, topk]
        """
        if len(x.shape) == 2:
            qkv_format = "thd"
            num_tokens = x.shape[0]
            bsz = 1
            seq_len = num_tokens
        else:
            qkv_format = "bshd"
            bsz, seq_len, _ = x.size()

        # Project Q from q_lora residual
        q = self.wq_b(q_resid)
        if qkv_format == "thd":
            q = q.view(num_tokens, self.num_heads, self.head_dim)
        else:
            q = q.view(bsz, seq_len, self.num_heads, self.head_dim)

        # Split Q into nope and pe parts (nope first, then pe - matching training code)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Project K from hidden states
        k = self.k_norm(self.wk(x))

        # Split K into nope and pe parts
        k_nope, k_pe = torch.split(k, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Apply RoPE to the pe parts
        head_unsqueeze_dim = 2 if qkv_format == "bshd" else 1
        q_pe = apply_rotary_emb(q_pe, freqs_cis, qkv_format=qkv_format)
        k_pe = apply_rotary_emb(k_pe, freqs_cis, qkv_format=qkv_format, unsqueeze_dim=head_unsqueeze_dim)
        k_pe = k_pe.squeeze(head_unsqueeze_dim)

        # Combine nope and pe parts (nope first, matching training code)
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)

        # Apply optional Hadamard rotation (if fast_hadamard_transform is available)
        q = _rotate_activation(q)
        k = _rotate_activation(k)

        # Compute per-head weights from hidden states
        # weights: [B, S, H] or [T, H]
        # Scale: (n_heads ** -0.5) * softmax_scale
        weights = self.weights_proj(x).float() * (self.num_heads**-0.5) * self.softmax_scale

        # Expand K to all heads
        if qkv_format == "thd":
            k = k.unsqueeze(1).expand(num_tokens, self.num_heads, self.head_dim)
        else:
            k = k.unsqueeze(2).expand(bsz, seq_len, self.num_heads, self.head_dim)

        # Compute attention scores: Q @ K^T with ReLU activation
        if qkv_format == "thd":
            # [T, H, D] -> [H, T, D]
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            # scores: [H, T, T]
            scores = torch.bmm(q.float(), k.float().transpose(-2, -1))
            # Apply ReLU activation (per training implementation)
            scores = torch.relu(scores)
            # Apply per-head weights: [T, H] -> [H, T, 1]
            weights = weights.transpose(0, 1).unsqueeze(-1)
            scores = scores * weights  # [H, T, T]
            # Sum over heads
            scores = scores.sum(dim=0)  # [T, T]
        else:
            # [B, S, H, D] -> [B, H, S, D]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            # scores: [B, H, S, S]
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
            # Apply ReLU activation (per training implementation)
            scores = torch.relu(scores)
            # Apply per-head weights: [B, S, H] -> [B, H, S, 1]
            weights = weights.transpose(1, 2).unsqueeze(-1)
            scores = scores * weights  # [B, H, S, S]
            # Sum over heads
            scores = scores.sum(dim=1)  # [B, S, S]

        # Apply attention mask if provided
        if attention_mask is not None:
            if qkv_format == "bshd":
                scores = scores + attention_mask.squeeze(1)
            else:
                if attention_mask.dim() == 4:
                    scores = scores + attention_mask.squeeze(0).squeeze(0)
                else:
                    scores = scores + attention_mask

        # Select top-k indices
        actual_topk = min(self.index_topk, seq_len)
        topk_indices = scores.topk(actual_topk, dim=-1).indices

        return topk_indices

    def init_weights(self, init_std: float = 0.02):
        for module in [self.wq_b, self.wk, self.weights_proj]:
            if hasattr(module, "weight"):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=init_std)
        self.k_norm.reset_parameters()


class DeepseekV32MLA(nn.Module):
    """Multi-head Latent Attention with Indexer for sparse attention.

    This extends the V3 MLA with an Indexer module that performs
    top-k selection for sparse attention. The indexer uses the
    q_lora residual and hidden states to compute which positions
    to attend to.
    """

    def __init__(self, config: DeepseekV32Config, backend: BackendConfig):
        super().__init__()

        self.n_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = (
            config.qk_head_dim if hasattr(config, "qk_head_dim") else (self.qk_nope_head_dim + self.qk_rope_head_dim)
        )
        self.v_head_dim = config.v_head_dim
        self.index_topk = config.index_topk

        self.backend = backend
        self.rope_fusion = backend.rope_fusion
        attn_impl = backend.attn
        linear_impl = backend.linear
        rms_norm_impl = backend.rms_norm

        hidden_size = config.hidden_size

        # V3.2 always uses q_lora (q_lora_rank is not None)
        self.q_a_proj = initialize_linear_module(
            linear_impl=linear_impl, in_features=hidden_size, out_features=self.q_lora_rank, bias=False
        )
        self.q_a_layernorm = initialize_rms_norm_module(rms_norm_impl=rms_norm_impl, dim=self.q_lora_rank)
        self.q_b_proj = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=self.q_lora_rank,
            out_features=self.n_heads * self.qk_head_dim,
            bias=False,
        )

        self.kv_a_proj_with_mqa = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=hidden_size,
            out_features=self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = initialize_rms_norm_module(rms_norm_impl=rms_norm_impl, dim=self.kv_lora_rank)
        self.kv_b_proj = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=self.kv_lora_rank,
            out_features=self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=self.n_heads * self.v_head_dim,
            out_features=hidden_size,
            bias=False,
        )
        self.softmax_scale = self.qk_head_dim**-0.5

        rope_parameters = config.rope_parameters if hasattr(config, "rope_parameters") else config.rope_scaling
        if rope_parameters and all(
            map(lambda x: x in rope_parameters, ["factor", "mscale", "original_max_position_embeddings"])
        ):
            factor = rope_parameters["factor"]
            mscale = rope_parameters["mscale"]
            original_seq_len = rope_parameters["original_max_position_embeddings"]
            if config.max_position_embeddings > original_seq_len:
                mscale = yarn_get_mscale(factor, mscale)
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.attn_module, self.attn_func = initialize_attn_module_and_func(
            attn_impl=attn_impl,
            num_attention_heads=self.n_heads,
            num_qk_channels=self.qk_head_dim,
            num_v_channels=self.v_head_dim,
            softmax_scale=self.softmax_scale,
        )

        # Initialize the Indexer
        self.indexer = DeepseekV32Indexer(config, backend)

    def _build_sparse_mask(
        self,
        topk_indices: torch.Tensor,
        seq_len: int,
        qkv_format: str,
        bsz: int = 1,
        n_heads: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        attention_mask: torch.Tensor | None = None,
        union_across_batches: bool = False,
    ) -> torch.Tensor:
        """Build a sparse attention mask/bias from top-k indices.

        Creates a mask tensor where non-top-k positions are set to -inf.
        Works for both TE (core_attention_bias) and SDPA (attn_mask).

        Uses the same efficient pattern as the official DeepSeek inference code:
        `torch.full(..., -inf).scatter_(-1, topk_indices, 0)`

        Args:
            topk_indices: Indices of top-k positions [B, S, topk] or [T, topk]
            seq_len: Sequence length
            qkv_format: 'bshd' or 'thd'
            bsz: Batch size (only used for bshd format)
            n_heads: Number of attention heads to expand to
            dtype: Data type for the output tensor
            attention_mask: Optional attention mask to combine with (for SDPA)
            union_across_batches: If True, union top-k across batches (for TE);
                                  if False, keep per-batch masks (for SDPA)

        Returns:
            sparse_mask: Mask tensor with shape:
                - [1, n_heads, S, S] if union_across_batches=True
                - [B, n_heads, S, S] if union_across_batches=False (bshd)
                - [1, n_heads, T, T] for thd format
        """
        device = topk_indices.device

        if qkv_format == "thd":
            num_tokens = topk_indices.shape[0]
            # Create mask directly in final shape [1, n_heads, T, T]
            # All heads share the same mask, so we create [T, T] and expand
            sparse_mask = torch.full((num_tokens, num_tokens), float("-inf"), device=device, dtype=dtype).scatter_(
                -1, topk_indices, 0.0
            )
            # expand creates a view, contiguous makes a copy
            sparse_mask = sparse_mask.view(1, 1, num_tokens, num_tokens).expand(1, n_heads, -1, -1).contiguous()
        else:
            if union_across_batches:
                # For TE: create [B, S, S], scatter, union via max, then expand
                sparse_mask = torch.full((bsz, seq_len, seq_len), float("-inf"), device=device, dtype=dtype).scatter_(
                    -1, topk_indices, 0.0
                )
                # Union: max(0, -inf) = 0 for any position selected in any batch
                sparse_mask = sparse_mask.max(dim=0, keepdim=True).values
                sparse_mask = sparse_mask.view(1, 1, seq_len, seq_len).expand(1, n_heads, -1, -1).contiguous()
            else:
                # For SDPA: create [B, S, S], scatter, expand (no contiguous needed)
                sparse_mask = torch.full((bsz, seq_len, seq_len), float("-inf"), device=device, dtype=dtype).scatter_(
                    -1, topk_indices, 0.0
                )
                sparse_mask = sparse_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)

        # Combine with existing attention mask if provided
        if attention_mask is not None:
            sparse_mask = attention_mask + sparse_mask

        return sparse_mask

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ):
        if len(x.shape) == 2:
            qkv_format = "thd"
            num_tokens = x.shape[0]
            bsz = 1
            seq_len = num_tokens
        else:
            qkv_format = "bshd"
            bsz, seq_len, _ = x.size()

        # Compute q_resid for indexer and main attention path
        q_resid = self.q_a_layernorm(self.q_a_proj(x))

        # Get top-k indices from indexer
        topk_indices = self.indexer(x, q_resid, freqs_cis, attention_mask, **attn_kwargs)

        # Build sparse bias/mask from top-k indices based on backend
        if self.backend.attn == "te":
            # For TE: build sparse bias for core_attention_bias (must match Q/K/V dtype)
            # Union across batches since TE expects [1, n_heads, S, S]
            sparse_mask = self._build_sparse_mask(
                topk_indices,
                seq_len,
                qkv_format,
                bsz,
                n_heads=self.n_heads,
                dtype=x.dtype,
                attention_mask=None,
                union_across_batches=True,
            )
        else:
            # For SDPA: build sparse mask, keep per-batch masks
            sparse_mask = self._build_sparse_mask(
                topk_indices,
                seq_len,
                qkv_format,
                bsz,
                n_heads=1,
                dtype=torch.float32,
                attention_mask=attention_mask,
                union_across_batches=False,
            )

        # Compute Q from q_resid
        q = self.q_b_proj(q_resid)

        if qkv_format == "thd":
            q = q.view(num_tokens, self.n_heads, self.qk_head_dim)
        else:
            q = q.view(bsz, seq_len, self.n_heads, self.qk_head_dim)

        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv = self.kv_a_proj_with_mqa(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_a_layernorm(kv)

        # For MLA, k_pe needs an extra head dimension for apply_rotary_emb
        head_unsqueeze_dim = 2 if qkv_format == "bshd" else 1
        k_pe = k_pe.unsqueeze(head_unsqueeze_dim)

        # Apply rotary embeddings to q_pe and k_pe
        q_pe = apply_rotary_emb(q_pe, freqs_cis, qkv_format=qkv_format)
        k_pe = apply_rotary_emb(k_pe, freqs_cis, qkv_format=qkv_format)

        # Remove the head dimension we added to k_pe
        k_pe = k_pe.squeeze(head_unsqueeze_dim)

        q = torch.cat([q_nope, q_pe], dim=-1)

        kv = self.kv_b_proj(kv)
        if qkv_format == "thd":
            kv = kv.view(num_tokens, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_pe = k_pe.unsqueeze(1).expand([num_tokens, self.n_heads, self.qk_rope_head_dim])
        else:
            kv = kv.view(bsz, seq_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_pe = k_pe.unsqueeze(2).expand([bsz, seq_len, self.n_heads, self.qk_rope_head_dim])

        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)

        # Handle attention based on backend
        if self.backend.attn == "te":
            # For TE: use core_attention_bias for sparse attention
            q, k, v, _attn_kwargs = preprocess_args_and_kwargs_for_attn(
                q, k, v, attention_mask, self.backend.attn, **attn_kwargs
            )
            # Add sparse mask as core_attention_bias
            _attn_kwargs["core_attention_bias_type"] = "post_scale_bias"
            _attn_kwargs["core_attention_bias"] = sparse_mask
        else:
            # For SDPA: use sparse mask (already combined with attention_mask)
            q, k, v, _attn_kwargs = preprocess_args_and_kwargs_for_attn(
                q, k, v, sparse_mask, self.backend.attn, **attn_kwargs
            )

        x = self.attn_func(q, k, v, **_attn_kwargs)
        x = postprocess_output_for_attn(x, self.backend.attn)

        flatten_dim = 2 if qkv_format == "bshd" else 1
        x = self.o_proj(x.flatten(flatten_dim))
        return x

    def init_weights(self, _buffer_device: torch.device, init_std: float = 0.02):
        linear_list = [
            self.q_a_proj,
            self.q_b_proj,
            self.kv_a_proj_with_mqa,
            self.kv_b_proj,
            self.o_proj,
        ]

        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)

        norms = [self.kv_a_layernorm, self.q_a_layernorm]
        for norm in norms:
            norm.reset_parameters()

        # Initialize indexer weights
        self.indexer.init_weights(init_std)

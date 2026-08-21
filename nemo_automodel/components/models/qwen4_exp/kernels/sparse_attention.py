# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Autograd adapter for native Qwen4-Exp TileLang sparse GQA kernels."""

from __future__ import annotations

import math

import torch

from nemo_automodel.components.models.qwen4_exp.kernels.tilelang_sparse_gqa_bwd import (
    sparse_gqa_bwd_interface,
)
from nemo_automodel.components.models.qwen4_exp.kernels.tilelang_sparse_gqa_fwd import (
    sparse_gqa_fwd_interface,
)


class Qwen4SparseGQAAttention(torch.autograd.Function):
    """Autograd bridge for one pseudo-batched sparse-GQA kernel invocation."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_ids: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        """Run fused forward and save the tensors needed to recompute probabilities."""
        output, lse = sparse_gqa_fwd_interface(
            query,
            key,
            value,
            token_ids,
            softmax_scale=softmax_scale,
        )
        ctx.save_for_backward(query, key, value, token_ids, output, lse)
        ctx.softmax_scale = softmax_scale
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Run fused backward and return gradients for independent Q, K, and V."""
        query, key, value, token_ids, output, lse = ctx.saved_tensors
        grad_query, grad_key, grad_value = sparse_gqa_bwd_interface(
            query,
            key,
            value,
            output,
            grad_output.contiguous(),
            token_ids,
            lse,
            softmax_scale=ctx.softmax_scale,
        )
        return grad_query, grad_key, grad_value, None, None


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def tilelang_sparse_gqa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_ids: torch.Tensor,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run differentiable native TileLang sparse GQA.

    The public tensors retain the model's BSHD layout. Each of the ``Hkv`` KV
    heads becomes a pseudo-batch row containing its ``Hq / Hkv`` query heads.
    Qwen4-Exp therefore maps ``Hq=24, Hkv=2`` to ``Bp=B*2, H=12`` and pads the
    small head group to 16 only for the kernel. K and V remain independent
    ``D=256`` tensors throughout.

    Args:
        query: Local queries ``[B, S, Hq, 256]``.
        key: Keys ``[B, S_kv, Hkv, 256]``. With CP these are already globally
            gathered before this function is called.
        value: Values with the same shape as ``key``.
        token_ids: Global selected-token IDs ``[B, S, K]``. ``-1`` marks an
            invalid fixed-width slot.
        softmax_scale: QK score scale, defaulting to ``1 / sqrt(256)``.

    Returns:
        Sparse-attention output ``[B, S, Hq, 256]``. Query rows without any
        valid selected token are exactly zero and generate zero Q/K/V gradient.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must use [B, S, H, D] layout")
    if key.shape != value.shape:
        raise ValueError(f"key and value shapes must match, got {tuple(key.shape)} and {tuple(value.shape)}")

    batch, seq_len, num_query_heads, head_dim = query.shape
    if key.shape[0] != batch or key.shape[-1] != head_dim:
        raise ValueError("query, key, and value batch/head dimensions must match")
    if batch < 1 or seq_len < 1 or num_query_heads < 1:
        raise ValueError("batch size, query length, and query-head count must be positive")
    if head_dim != 256:
        raise ValueError(f"Qwen4-Exp sparse GQA requires head_dim=256, got {head_dim}")
    if key.shape[1] == 0:
        raise ValueError("sparse GQA requires at least one key/value token")

    num_kv_heads = key.shape[2]
    if num_kv_heads <= 0 or num_query_heads % num_kv_heads != 0:
        raise ValueError(f"Hq must be divisible by Hkv, got Hq={num_query_heads}, Hkv={num_kv_heads}")
    if token_ids.ndim != 3 or token_ids.shape[:2] != (batch, seq_len) or token_ids.shape[-1] == 0:
        raise ValueError(
            f"token_ids must have nonempty shape [B, S, K] matching {(batch, seq_len)}, got {tuple(token_ids.shape)}"
        )
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"token_ids must be int32 or int64, got {token_ids.dtype}")
    if any(tensor.dtype != torch.bfloat16 for tensor in (query, key, value)):
        raise TypeError("Qwen4-Exp TileLang sparse GQA requires BF16 query, key, and value")
    if any(tensor.device != query.device for tensor in (key, value, token_ids)):
        raise ValueError("query, key, value, and token_ids must be on the same device")

    if softmax_scale is not None and (isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (float, int))):
        raise TypeError(f"softmax_scale must be a finite positive number, got {softmax_scale!r}")
    scale = head_dim**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"softmax_scale must be a positive finite value, got {softmax_scale}")

    group_heads = num_query_heads // num_kv_heads
    padded_group_heads = max(16, _next_power_of_two(group_heads))
    if padded_group_heads > 64:
        raise NotImplementedError(
            f"Qwen4-Exp TileLang sparse GQA supports at most 64 query heads per KV head, got {group_heads}"
        )

    # [B,S,Hkv,G,D] -> [B,Hkv,S,G,D] -> [B*Hkv,S,G,D].
    pseudo_query = (
        query.unflatten(2, (num_kv_heads, group_heads))
        .permute(0, 2, 1, 3, 4)
        .reshape(batch * num_kv_heads, seq_len, group_heads, head_dim)
        .contiguous()
    )
    if padded_group_heads != group_heads:
        head_padding = pseudo_query.new_zeros(
            batch * num_kv_heads,
            seq_len,
            padded_group_heads - group_heads,
            head_dim,
        )
        pseudo_query = torch.cat((pseudo_query, head_padding), dim=2).contiguous()

    seq_len_kv = key.shape[1]
    pseudo_key = key.permute(0, 2, 1, 3).reshape(batch * num_kv_heads, seq_len_kv, head_dim).contiguous()
    pseudo_value = value.permute(0, 2, 1, 3).reshape(batch * num_kv_heads, seq_len_kv, head_dim).contiguous()

    # Every KV head uses the same token route. Keep one safe token on otherwise
    # empty rows so forward LSE remains finite, then erase those rows outside the
    # custom Function. The outer multiplication also masks grad_output before
    # the custom backward, making Q/K/V gradients for padding rows exactly zero.
    valid_tokens = (token_ids >= 0) & (token_ids < seq_len_kv)
    valid_rows = valid_tokens.any(dim=-1)
    pseudo_ids = (
        token_ids[:, None]
        .expand(batch, num_kv_heads, seq_len, token_ids.shape[-1])
        .reshape(batch * num_kv_heads, seq_len, token_ids.shape[-1])
        .to(torch.int32)
        .contiguous()
    )
    pseudo_valid_rows = valid_rows[:, None].expand(batch, num_kv_heads, seq_len).reshape(batch * num_kv_heads, seq_len)
    safe_first_id = torch.where(
        pseudo_valid_rows,
        pseudo_ids[..., 0],
        torch.zeros_like(pseudo_ids[..., 0]),
    )
    safe_ids = torch.cat((safe_first_id.unsqueeze(-1), pseudo_ids[..., 1:]), dim=-1).contiguous()

    pseudo_output = Qwen4SparseGQAAttention.apply(
        pseudo_query,
        pseudo_key,
        pseudo_value,
        safe_ids,
        scale,
    )
    pseudo_output = pseudo_output * pseudo_valid_rows[..., None, None].to(pseudo_output.dtype)
    pseudo_output = pseudo_output[:, :, :group_heads]

    # [B*Hkv,S,G,D] -> [B,S,Hkv,G,D] -> [B,S,Hq,D].
    return (
        pseudo_output.unflatten(0, (batch, num_kv_heads))
        .permute(0, 2, 1, 3, 4)
        .reshape(batch, seq_len, num_query_heads, head_dim)
        .contiguous()
    )


__all__ = ["Qwen4SparseGQAAttention", "tilelang_sparse_gqa_attention"]

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

"""Autograd adapter for native Qwen4-Exp direct-THD sparse GQA kernels."""

from __future__ import annotations

import math

import torch

from .tilelang_sparse_gqa_thd_bwd import sparse_gqa_thd_bwd_interface
from .tilelang_sparse_gqa_thd_fwd import sparse_gqa_thd_fwd_interface

_HEAD_DIM = 256
_MAX_TOPK = 2051


def _normalize_token_ids(token_ids: torch.Tensor, tokens: int) -> torch.Tensor:
    """Normalize a nonempty shared THD route to ``[T, K]``.

    Args:
        token_ids: Route IDs shaped ``[T, K]`` or ``[T, 1, K]``.
        tokens: Expected physical token count ``T``.

    Returns:
        The rank-two route view with ``1 <= K <= 2051``.
    """
    if token_ids.ndim == 3:
        if token_ids.shape[1] != 1:
            raise ValueError(f"rank-3 THD token_ids must be [T, 1, K], got {tuple(token_ids.shape)}")
        token_ids = token_ids[:, 0, :]
    if token_ids.ndim != 2 or token_ids.shape[0] != tokens or token_ids.shape[1] < 1:
        raise ValueError(f"token_ids must have nonempty shape [T, K] matching T={tokens}, got {tuple(token_ids.shape)}")
    if token_ids.shape[1] > _MAX_TOPK:
        raise ValueError(f"Qwen4 THD sparse GQA supports at most {_MAX_TOPK} selected slots, got {token_ids.shape[1]}")
    return token_ids


def _validate_scale(softmax_scale: float | None) -> float:
    """Return a finite positive QK scale."""
    if softmax_scale is not None and (isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (float, int))):
        raise TypeError(f"softmax_scale must be a finite positive number, got {softmax_scale!r}")
    scale = _HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"softmax_scale must be finite and positive, got {softmax_scale!r}")
    return scale


class _Qwen4SparseGQAThdAttention(torch.autograd.Function):
    """Bridge direct ``[T,H,D]`` tensors to fused TileLang forward/backward."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        token_ids: torch.Tensor,
        softmax_scale: float,
    ) -> torch.Tensor:
        """Run fused forward and save the tensors needed by backward."""
        output, lse = sparse_gqa_thd_fwd_interface(
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
        """Run fused backward and return gradients for Q, K, and V."""
        query, key, value, token_ids, output, lse = ctx.saved_tensors
        grad_query, grad_key, grad_value = sparse_gqa_thd_bwd_interface(
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


def tilelang_sparse_gqa_thd_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_ids: torch.Tensor,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run direct packed-THD Qwen4 sparse GQA.

    Args:
        query: BF16 CUDA queries shaped ``[T, 24, 256]`` with ``T >= 1``.
        key: BF16 CUDA keys shaped ``[T, 2, 256]``.
        value: BF16 CUDA values shaped ``[T, 2, 256]``.
        token_ids: Global flattened route IDs shaped ``[T, K]`` or
            ``[T, 1, K]``, where ``1 <= K <= 2051``. Negative and
            out-of-range entries are padding. All inputs must share one CUDA
            device; strided tensors are made contiguous before kernel launch.
            Every valid ID must already belong to the query's packed document;
            the math kernel intentionally does not receive ``cu_seqlens``.
        softmax_scale: Positive QK scale, defaulting to ``1 / sqrt(256)``.

    Returns:
        BF16 output shaped ``[T, 24, 256]``. The custom backward returns Q/K/V
        gradients in the corresponding direct THD layouts and accumulates dK
        and dV atomics in FP32 before casting.
    """
    if query.ndim != 3:
        raise ValueError(f"query must be [T, Hq, D], got {tuple(query.shape)}")
    if token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"token_ids must be int32 or int64, got {token_ids.dtype}")
    normalized_ids = _normalize_token_ids(token_ids, query.shape[0]).contiguous()
    scale = _validate_scale(softmax_scale)
    return _Qwen4SparseGQAThdAttention.apply(query, key, value, normalized_ids, scale)


__all__ = ["tilelang_sparse_gqa_thd_attention"]

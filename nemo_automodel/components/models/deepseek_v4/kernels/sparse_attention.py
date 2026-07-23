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

"""Autograd wrapper for vendored Miles DeepSeek V4 sparse-attention kernels.

Attribution:
* Upstream project: Miles, https://github.com/yueming-yuan/miles
* Upstream revision: e561465d0b9bbf06188b7a5e2020dc7fd691f732, deepseek-v4 branch
* Upstream license: Apache-2.0, copyright 2025 Zhipu AI
* Original source:
  https://github.com/yueming-yuan/miles/blob/e561465d0b9bbf06188b7a5e2020dc7fd691f732/miles_plugins/models/deepseek_v4/ops/attention_core.py
"""

from __future__ import annotations

import torch

from nemo_automodel.components.models.deepseek_v4.kernels import tilelang_sparse_mla_bwd as sparse_mla_bwd
from nemo_automodel.components.models.deepseek_v4.kernels import tilelang_sparse_mla_fwd as sparse_mla_fwd


class DeepSeekV4SparseAttention(torch.autograd.Function):
    """TileLang sparse MQA attention with custom backward."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        sm_scale: float | None = None,
    ) -> torch.Tensor:
        """Run the vendored sparse attention forward kernel."""
        output, lse = sparse_mla_fwd.sparse_mqa_fwd_interface(q, kv, attn_sink, topk_idxs, sm_scale=sm_scale)
        ctx.save_for_backward(q, kv, attn_sink, topk_idxs, output, lse)
        ctx.sm_scale = sm_scale
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """Run the vendored sparse attention backward kernel."""
        q, kv, attn_sink, topk_idxs, output, lse = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_q, grad_kv, grad_attn_sink = sparse_mla_bwd.sparse_mqa_bwd_interface(
            q,
            kv,
            attn_sink,
            output.contiguous(),
            grad_output,
            topk_idxs,
            lse,
            sm_scale=ctx.sm_scale,
        )
        return grad_q, grad_kv, grad_attn_sink, None, None


def sparse_attn_tilelang(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Run vendored Miles DeepSeek V4 TileLang sparse attention."""
    return DeepSeekV4SparseAttention.apply(q, kv, attn_sink, topk_idxs, sm_scale)


class DeepSeekV4SparseAttentionHeadChunked(torch.autograd.Function):
    """TileLang sparse attention with smaller head groups and fp32 KV-grad accumulation."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        max_heads_per_kernel: int,
        sm_scale: float | None = None,
    ) -> torch.Tensor:
        """Run the vendored sparse attention forward kernel over head chunks."""
        output = q.new_empty(q.shape)
        lse = torch.empty(q.shape[:3], dtype=torch.float32, device=q.device)
        for start in range(0, q.shape[2], max_heads_per_kernel):
            end = min(start + max_heads_per_kernel, q.shape[2])
            chunk_output, chunk_lse = sparse_mla_fwd.sparse_mqa_fwd_interface(
                q[:, :, start:end, :].contiguous(),
                kv,
                attn_sink[start:end].contiguous(),
                topk_idxs,
                sm_scale=sm_scale,
            )
            output[:, :, start:end, :].copy_(chunk_output)
            lse[:, :, start:end].copy_(chunk_lse)
        ctx.save_for_backward(q, kv, attn_sink, topk_idxs, output, lse)
        ctx.max_heads_per_kernel = max_heads_per_kernel
        ctx.sm_scale = sm_scale
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        """Run chunked backward and accumulate shared KV gradients in fp32."""
        q, kv, attn_sink, topk_idxs, output, lse = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_q_full = torch.empty_like(q)
        grad_kv = torch.zeros_like(kv, dtype=torch.float32)
        grad_attn_sink_full = torch.empty_like(attn_sink)
        max_heads = ctx.max_heads_per_kernel
        for start in range(0, q.shape[2], max_heads):
            end = min(start + max_heads, q.shape[2])
            grad_q_chunk, grad_kv_chunk, grad_attn_sink = sparse_mla_bwd.sparse_mqa_bwd_interface(
                q[:, :, start:end, :].contiguous(),
                kv,
                attn_sink[start:end].contiguous(),
                output[:, :, start:end, :].contiguous(),
                grad_output[:, :, start:end, :].contiguous(),
                topk_idxs,
                lse[:, :, start:end].contiguous(),
                sm_scale=ctx.sm_scale,
                return_dkv_accum_dtype=True,
            )
            grad_q_full[:, :, start:end, :].copy_(grad_q_chunk)
            grad_kv += grad_kv_chunk
            grad_attn_sink_full[start:end].copy_(grad_attn_sink)
        return (
            grad_q_full,
            grad_kv.to(kv.dtype),
            grad_attn_sink_full,
            None,
            None,
            None,
        )


def sparse_attn_tilelang_head_chunked(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    max_heads_per_kernel: int,
    sm_scale: float | None = None,
) -> torch.Tensor:
    """Run vendored Miles sparse attention in TileLang head chunks."""
    return DeepSeekV4SparseAttentionHeadChunked.apply(q, kv, attn_sink, topk_idxs, max_heads_per_kernel, sm_scale)

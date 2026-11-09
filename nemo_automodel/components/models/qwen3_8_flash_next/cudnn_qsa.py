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

"""cuDNN Frontend indexed QSA training with the model's existing routes."""

from __future__ import annotations

import functools
import math
from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from nemo_automodel.shared.import_utils import safe_import_from


@functools.cache
def _load_cudnn() -> tuple[Callable, Callable]:
    """Load optional public entry points lazily; retain no tensor state."""
    forward_ok, forward = safe_import_from("cudnn.ops", "qsa_indexed_forward")
    backward_ok, backward = safe_import_from("cudnn.ops", "qsa_indexed_backward")
    if not forward_ok or not backward_ok:
        raise ImportError(
            "cuDNN QSA requires nvidia-cudnn-frontend>=1.32.1 and its CuTeDSL dependencies. "
            'Install with uv pip install "nvidia-cudnn-frontend[cutedsl]>=1.32.1".'
        )
    return forward, backward


def _pack_indices(routes: torch.Tensor, sequence_length: int) -> torch.Tensor:
    """Translate batch-local QSA IDs to packed coordinates without changing selection.

    Args:
        routes: Signed integer tensor [batch, sequence, width], produced by the
            model's compression-ratio-four indexer. Four-slot groups contain
            distinct complete blocks or a consecutive causal tail and -1 padding.
        sequence_length: Physical sequence extent shared by query and K/V.

    Returns:
        Independent contiguous int32 tensor [batch * sequence, ceil(width/4)*4].
        Invalid IDs stay -1, and valid IDs gain their batch's token offset.
        No input is mutated and no device value is read on the host.
    """
    valid = (routes >= 0) & (routes < sequence_length)
    offsets = torch.arange(routes.shape[0], device=routes.device, dtype=torch.int64)[:, None, None] * sequence_length
    # Mask before narrowing to int32, so large invalid int64 IDs cannot wrap.
    indices = torch.where(valid, routes.to(torch.int64) + offsets, -1).to(torch.int32)
    indices = F.pad(indices, (0, (-routes.shape[-1]) % 4), value=-1)
    return indices.flatten(0, 1).contiguous()


class _CudnnQSA(torch.autograd.Function):
    """Own saved FP32 output and LSE for one independent GQA head group."""

    @staticmethod
    def forward(
        ctx: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, indices: torch.Tensor, scale: float
    ) -> torch.Tensor:
        """Run forward on contiguous packed Q [T,H,256], K/V [T,1,256].

        Args:
            ctx: PyTorch autograd context.
            q: BF16 CUDA query [T,12,256].
            k: BF16 CUDA keys [T,1,256].
            v: BF16 CUDA values [T,1,256].
            indices: CUDA int32 selection [T,W], in packed token coordinates.
            scale: Finite positive score multiplier.

        Returns:
            Independent BF16 CUDA output [T,H,256]. Saves native FP32 output
            [T,H,256] and natural-log LSE [H,T], never a cast of BF16 output.
        """
        forward, _ = _load_cudnn()
        out, out_fp32, lse = forward(q, k, v, indices, scale=scale)
        ctx.save_for_backward(q, k, v, indices, out_fp32, lse)
        ctx.scale = scale
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Return BF16 dQ [T,H,256], dK/dV [T,1,256], and two None entries.

        Args:
            ctx: Context holding immutable forward inputs and saved state.
            grad_output: BF16 CUDA upstream gradient [T,H,256]; may be strided.

        Returns:
            Gradients for the five forward arguments. Discrete indices and
            scale receive no gradient. Only first-order autograd is supported.
        """
        q, k, v, indices, out_fp32, lse = ctx.saved_tensors
        _, backward = _load_cudnn()
        dq, dk, dv = backward(
            grad_output.contiguous(), q, k, v, out_fp32, lse, indices, scale=ctx.scale, selection_mode="general"
        )
        return dq, dk, dv, None, None


def cudnn_sparse_gqa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_token_ids: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Evaluate model-produced four-token QSA routes with cuDNN on Blackwell.

    Selection is unchanged: this adapter does not run the cuDNN indexer. Each
    four-slot group must be a distinct complete block or a consecutive causal
    tail padded with -1, as emitted by the model's ratio-four indexer. Blocks
    restart at each packed document origin. Arbitrary/duplicate token sets are
    not supported by this backend. These device-value invariants are the
    indexer's contract, not host-synchronized checks in the attention hot path.

    Args:
        query: BF16 CUDA tensor [batch, sequence, query_heads, 256].
        key: BF16 CUDA tensor [batch, sequence, kv_heads, 256]. Supported
            (query_heads, kv_heads) are (24,2) and (12,1), matching the published
            cuDNN Frontend 1.32.1 indexed API.
        value: BF16 CUDA tensor with key's shape and device. Strided Q/K/V
            are accepted and copied to contiguous per-KV-head packed buffers.
        selected_token_ids: Int32/int64 CUDA tensor [batch, sequence, width],
            with 1 <= width <= 2052. IDs address the physical sequence within
            each batch row; invalid IDs are padding. Packed documents are
            supported through their existing absolute-within-row route IDs.
        softmax_scale: Finite positive host scalar; defaults to 1/sqrt(256).

    Returns:
        Independent BF16 CUDA output [batch, sequence, query_heads, 256].
        Empty route rows produce zero. K/V gradients include all GQA heads.

    Raises:
        ValueError: For unsupported shapes, devices, heads, scale or CP layout.
        TypeError: For non-BF16 compute tensors or unsigned/noninteger routes.
        RuntimeError: For unsupported GPU or deterministic backward requests.
        ImportError: When the optional cuDNN Frontend QSA runtime is absent.

    CP with local Q and gathered global K/V is unsupported. First-order
    backward uses cuDNN FP32 atomic accumulation and is not bitwise deterministic.
    """
    if any(t.ndim != 4 for t in (query, key, value)):
        raise ValueError("cuDNN QSA requires [batch, sequence, heads, head_dim] layout")
    if key.shape != value.shape or query.shape[:2] != key.shape[:2]:
        raise ValueError("cuDNN QSA requires matching Q/K/V batch and sequence extents; CP local-Q is unsupported")
    batch, sequence, heads, dim = query.shape
    kv_heads = key.shape[2]
    if min(batch, sequence) <= 0 or dim != 256 or key.shape[-1] != dim:
        raise ValueError("cuDNN QSA requires nonempty batch/sequence and head_dim=256")
    if (heads, kv_heads) not in ((24, 2), (12, 1)):
        raise ValueError("cuDNN QSA supports query/KV head counts 24/2 or 12/1")
    if batch * sequence * (heads // kv_heads) * dim > torch.iinfo(torch.int32).max:
        raise ValueError("cuDNN QSA packed per-group query size must fit in int32")
    if selected_token_ids.ndim != 3 or selected_token_ids.shape[:2] != (batch, sequence):
        raise ValueError("cuDNN QSA routes must have shape [batch, sequence, width]")
    if not 1 <= selected_token_ids.shape[-1] <= 2052:
        raise ValueError("cuDNN QSA route width must be in [1, 2052]")
    if selected_token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("cuDNN QSA routes must use signed int32 or int64")
    if any(t.dtype != torch.bfloat16 for t in (query, key, value)):
        raise TypeError("cuDNN QSA requires BF16 query, key and value")
    if any(not t.is_cuda or t.device != query.device for t in (query, key, value, selected_token_ids)):
        raise ValueError("cuDNN QSA inputs must be on the same CUDA device")
    if torch.cuda.get_device_capability(query.device) not in ((10, 0), (10, 3)):
        raise RuntimeError("cuDNN QSA integration supports Blackwell SM100/SM103")
    scale = dim**-0.5 if softmax_scale is None else softmax_scale
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("cuDNN QSA softmax_scale must be a finite positive host scalar")
    if (
        torch.are_deterministic_algorithms_enabled()
        and torch.is_grad_enabled()
        and any(t.requires_grad for t in (query, key, value))
    ):
        raise RuntimeError("cuDNN QSA backward uses atomic accumulation and is not deterministic")
    _load_cudnn()
    indices = _pack_indices(selected_token_ids, sequence)
    group_heads = heads // kv_heads
    outputs = []
    for group in range(kv_heads):
        q = query[:, :, group * group_heads : (group + 1) * group_heads].reshape(-1, group_heads, dim).contiguous()
        k = key[:, :, group : group + 1].reshape(-1, 1, dim).contiguous()
        v = value[:, :, group : group + 1].reshape(-1, 1, dim).contiguous()
        outputs.append(_CudnnQSA.apply(q, k, v, indices, float(scale)).view(batch, sequence, group_heads, dim))
    return torch.cat(outputs, dim=2) if kv_heads > 1 else outputs[0]

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

"""Optional SM90 CuTe backend for Qwen3.8-Flash-Next route-sparse GQA.

Kernel dependencies are loaded only when this backend is executed on CUDA.
The model dispatcher retains its existing CPU numerical oracle.
"""

from __future__ import annotations

import functools
import math
from types import ModuleType

import torch

from nemo_automodel.shared.import_utils import safe_import


@functools.cache
def _load_kernels() -> ModuleType:
    """Load the optional CuTe/FlashAttention implementation on first use."""
    available, kernels = safe_import("nemo_automodel.components.models.qwen3_8_flash_next._cute_qsa")
    if not available:
        raise ImportError(
            "CuTe QSA requires FlashAttention's flash_attn.cute SM90 kernels, "
            "nvidia-cutlass-dsl==4.6.2 and compatible TVM FFI. "
            "See nemo_automodel/components/models/qwen3_8_flash_next/CUTE_QSA.md."
        )
    return kernels


def cute_sparse_gqa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_token_ids: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Evaluate exact route-set sparse GQA on SM90, including first-order gradients.

    Duplicate routes select a token once. Negative and out-of-range IDs are
    ignored before narrowing int64 IDs. Empty route rows have zero output and
    zero query gradients. Routes alone encode causality and packed documents;
    no additional triangular mask is imposed. Backward uses atomic reductions,
    so deterministic-algorithm mode and higher-order gradients are unsupported.

    Args:
        query: BF16 CUDA tensor [batch, query_sequence, query_heads, 256].
            Arbitrary strides are accepted; non-unit final strides are copied.
        key: BF16 CUDA tensor [batch, kv_sequence, kv_heads, 256]. K/V can be
            gathered global tensors while query contains a local CP slice.
            query_heads must be a positive multiple of kv_heads.
        value: BF16 CUDA tensor with the same shape and device as key.
        selected_token_ids: Signed int32/int64 tensor
            [batch, query_sequence, routes] on query's device. IDs index the
            physical kv_sequence dimension within each batch row. All dimensions
            must be nonempty. Noncontiguous routes are copied.
        softmax_scale: Finite positive score multiplier; defaults to 1/sqrt(256).

    Returns:
        Independently allocated BF16 tensor [batch, query_sequence, query_heads,
        256], on query's device. Inputs are not mutated.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("CuTe QSA expects Q/K/V in [batch, sequence, heads, head_dim] layout")
    if key.shape != value.shape or key.shape[0] != query.shape[0]:
        raise ValueError("CuTe QSA requires matching K/V shapes and Q/K/V batch sizes")
    if any(size <= 0 for tensor in (query, key, value) for size in tensor.shape):
        raise ValueError("CuTe QSA requires nonempty Q/K/V dimensions")
    if query.shape[-1] != 256 or key.shape[-1] != 256:
        raise ValueError("CuTe QSA requires head_dim=256")
    if query.shape[2] % key.shape[2] != 0:
        raise ValueError("CuTe QSA requires query_heads divisible by kv_heads")
    if selected_token_ids.ndim != 3 or selected_token_ids.shape[:2] != query.shape[:2]:
        raise ValueError("CuTe QSA routes must have shape [batch, query_sequence, routes]")
    if selected_token_ids.shape[-1] == 0:
        raise ValueError("CuTe QSA requires a nonempty route dimension; use -1 for empty rows")
    if selected_token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("CuTe QSA routes must be signed int32 or int64")
    if any(t.dtype != torch.bfloat16 for t in (query, key, value)):
        raise TypeError("CuTe QSA requires BF16 query, key, and value")
    if not query.is_cuda or any(t.device != query.device for t in (key, value, selected_token_ids)):
        raise ValueError("CuTe QSA requires Q/K/V/routes on the same CUDA device")
    if torch.cuda.get_device_capability(query.device) != (9, 0):
        raise RuntimeError("CuTe QSA currently supports SM90 GPUs only")
    # One shared-memory bitmap per query; bound the int32 indexing contract too.
    words = (key.shape[1] + 31) // 32
    if words * 4 > 48 * 1024 or query.shape[0] * query.shape[1] * words >= 2**31:
        raise ValueError("CuTe QSA route bitmap exceeds shared-memory or int32 indexing limits")
    scale = 256**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("CuTe QSA softmax_scale must be finite and positive")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("CuTe QSA backward uses atomic reductions and does not support deterministic algorithms")
    with torch.cuda.device(query.device):
        return _load_kernels().qsa_cutedsl(query, key, value, selected_token_ids, softmax_scale=scale)

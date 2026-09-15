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

"""Fused multi-tensor gradient clipping kernels in Triton.

Accelerates gradient norm calculation across hundreds of model parameters in
distributed and MoE training by replacing sequential per-tensor CPU-GPU kernel
launches with parallelized 2D multi-tensor grid reduction kernels.
"""

from collections.abc import Sequence

import torch

from nemo_automodel.shared.import_utils import null_decorator, safe_import

_HAVE_TRITON, triton = safe_import("triton")
if _HAVE_TRITON:
    import triton.language as tl
else:
    tl = None

_BLOCK_SIZE = 4096
_NUM_CHUNKS = 32

_DTYPE_MAP = {
    torch.bfloat16: 0,
    torch.float32: 1,
    torch.float16: 2,
    torch.float64: 3,
}

if _HAVE_TRITON:

    @triton.jit
    def _multi_tensor_max_2d_kernel(
        ptrs_ptr,
        numels_ptr,
        out_partial_max_ptr,
        NUM_CHUNKS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        DTYPE_CODE: tl.constexpr,
    ):
        chunk_id = tl.program_id(0).to(tl.int64)
        tensor_id = tl.program_id(1).to(tl.int64)

        raw_ptr = tl.load(ptrs_ptr + tensor_id)
        if DTYPE_CODE == 0:
            t_ptr = raw_ptr.to(tl.pointer_type(tl.bfloat16))
        elif DTYPE_CODE == 1:
            t_ptr = raw_ptr.to(tl.pointer_type(tl.float32))
        elif DTYPE_CODE == 2:
            t_ptr = raw_ptr.to(tl.pointer_type(tl.float16))
        else:
            t_ptr = raw_ptr.to(tl.pointer_type(tl.float64))

        n_elements = tl.load(numels_ptr + tensor_id)
        chunk_stride = NUM_CHUNKS * BLOCK_SIZE
        local_max = 0.0

        for off in range(chunk_id * BLOCK_SIZE, n_elements, chunk_stride):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < n_elements
            vals = tl.load(t_ptr + cols, mask=mask, other=0.0).to(tl.float32)
            block_max = tl.max(tl.abs(vals))
            local_max = tl.maximum(local_max, block_max)

        tl.store(out_partial_max_ptr + tensor_id * NUM_CHUNKS + chunk_id, local_max.to(tl.float64))

    @triton.jit
    def _multi_tensor_scaled_l2_2d_kernel(
        ptrs_ptr,
        numels_ptr,
        scale,
        out_partial_sq_ptr,
        NUM_CHUNKS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        DTYPE_CODE: tl.constexpr,
    ):
        chunk_id = tl.program_id(0).to(tl.int64)
        tensor_id = tl.program_id(1).to(tl.int64)

        raw_ptr = tl.load(ptrs_ptr + tensor_id)
        if DTYPE_CODE == 0:
            t_ptr = raw_ptr.to(tl.pointer_type(tl.bfloat16))
        elif DTYPE_CODE == 1:
            t_ptr = raw_ptr.to(tl.pointer_type(tl.float32))
        elif DTYPE_CODE == 2:
            t_ptr = raw_ptr.to(tl.pointer_type(tl.float16))
        else:
            t_ptr = raw_ptr.to(tl.pointer_type(tl.float64))

        n_elements = tl.load(numels_ptr + tensor_id)
        chunk_stride = NUM_CHUNKS * BLOCK_SIZE
        accum_sq = tl.zeros((), dtype=tl.float64)

        for off in range(chunk_id * BLOCK_SIZE, n_elements, chunk_stride):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < n_elements
            vals = tl.load(t_ptr + cols, mask=mask, other=0.0).to(tl.float64)
            scaled_vals = vals / scale
            accum_sq += tl.sum(scaled_vals * scaled_vals)

        tl.store(out_partial_sq_ptr + tensor_id * NUM_CHUNKS + chunk_id, accum_sq)

else:
    _multi_tensor_max_2d_kernel = null_decorator
    _multi_tensor_scaled_l2_2d_kernel = null_decorator


def can_use_fused_grad_norm(target_device: torch.device) -> bool:
    """Return True if fused Triton kernels can be executed on target_device."""
    return _HAVE_TRITON and target_device.type == "cuda" and torch.cuda.is_available()


def fused_multi_tensor_max(grads: Sequence[torch.Tensor], target_device: torch.device) -> torch.Tensor:
    """Compute global maximum absolute gradient value across multiple tensors using Triton.

    Args:
        grads: Sequence of gradient tensors of shape [*, ...], with arbitrary ranks
            and dimensions, residing on target_device.
        target_device: CUDA device where tensors reside.

    Returns:
        Scalar torch.float64 tensor of shape [] containing the global maximum absolute gradient value.
    """
    if not grads:
        return torch.zeros((), dtype=torch.float64, device=target_device)

    grads_by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
    for g in grads:
        if not g.is_contiguous():
            g = g.contiguous()
        grads_by_dtype.setdefault(g.dtype, []).append(g)

    all_maxes: list[torch.Tensor] = []
    for dtype, dtype_grads in grads_by_dtype.items():
        if dtype not in _DTYPE_MAP:
            # Fallback for unsupported dtype
            for g in dtype_grads:
                all_maxes.append(g.detach().abs().max().to(device=target_device, dtype=torch.float64))
            continue

        n = len(dtype_grads)
        ptrs = torch.tensor([g.data_ptr() for g in dtype_grads], device=target_device, dtype=torch.int64)
        numels = torch.tensor([g.numel() for g in dtype_grads], device=target_device, dtype=torch.int64)
        out_partial_max = torch.empty((n, _NUM_CHUNKS), device=target_device, dtype=torch.float64)

        _multi_tensor_max_2d_kernel[(_NUM_CHUNKS, n)](
            ptrs,
            numels,
            out_partial_max,
            NUM_CHUNKS=_NUM_CHUNKS,
            BLOCK_SIZE=_BLOCK_SIZE,
            DTYPE_CODE=_DTYPE_MAP[dtype],
            num_warps=4,
        )
        all_maxes.append(out_partial_max.max())

    if not all_maxes:
        return torch.zeros((), dtype=torch.float64, device=target_device)
    return torch.stack(all_maxes).max()


def fused_multi_tensor_scaled_l2(
    grads: Sequence[torch.Tensor], scale: torch.Tensor, target_device: torch.device
) -> torch.Tensor:
    """Compute scaled sum of squares across multiple tensors using Triton.

    Computes sum((g / scale)^2) in float64 accumulation across all gradients.

    Args:
        grads: Sequence of gradient tensors of shape [*, ...], with arbitrary ranks
            and dimensions, residing on target_device.
        scale: Scalar tensor of shape [] (typically global max abs norm).
        target_device: CUDA device where tensors reside.

    Returns:
        Scalar torch.float64 tensor of shape [] containing the scaled sum of squares.
    """

    if not grads:
        return torch.zeros((), dtype=torch.float64, device=target_device)

    scale_val = float(scale)
    grads_by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
    for g in grads:
        if not g.is_contiguous():
            g = g.contiguous()
        grads_by_dtype.setdefault(g.dtype, []).append(g)

    all_sqs: list[torch.Tensor] = []
    for dtype, dtype_grads in grads_by_dtype.items():
        if dtype not in _DTYPE_MAP:
            # Fallback for unsupported dtype
            for g in dtype_grads:
                g_scaled = g.detach().abs().div(scale)
                all_sqs.append(g_scaled.square().sum(dtype=torch.float64))
            continue

        n = len(dtype_grads)
        ptrs = torch.tensor([g.data_ptr() for g in dtype_grads], device=target_device, dtype=torch.int64)
        numels = torch.tensor([g.numel() for g in dtype_grads], device=target_device, dtype=torch.int64)
        out_partial_sq = torch.empty((n, _NUM_CHUNKS), device=target_device, dtype=torch.float64)

        _multi_tensor_scaled_l2_2d_kernel[(_NUM_CHUNKS, n)](
            ptrs,
            numels,
            scale_val,
            out_partial_sq,
            NUM_CHUNKS=_NUM_CHUNKS,
            BLOCK_SIZE=_BLOCK_SIZE,
            DTYPE_CODE=_DTYPE_MAP[dtype],
            num_warps=4,
        )
        all_sqs.append(out_partial_sq.sum())

    if not all_sqs:
        return torch.zeros((), dtype=torch.float64, device=target_device)
    return torch.stack(all_sqs).sum()

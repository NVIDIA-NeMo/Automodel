# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2025, Tri Dao.
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
#
# The Triton kernel is vendored from flash-attention's rotary kernel via the
# Untied Ulysses reference implementation. Requires triton >= 3.0.

"""In-place rotary embedding for the UPipe fused attention.

UPipe applies RoPE inside its staged loop, so it needs an in-place kernel that
accepts the ``(cos, sin)`` tables directly rather than a complex ``freqs_cis``
tensor. Two conventions matter:

* ``interleaved=False`` -- GPT-NeoX / Llama half-split (``rotate_half``).
* ``interleaved=True`` -- pairwise interleaved (DeepSeek / Meta-Llama).

The backward pass inverts the rotation with ``conjugate=True``, which negates
``sin``; RoPE is unitary, so this recovers the gradient with respect to the
pre-rotary tensor without saving it.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_kernel(
    OUT,  # Pointers to matrices
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,  # this could be int or a pointer
    # Matrix dimensions
    seqlen,
    nheads,
    seqlen_ro,
    # strides
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    # Meta-parameters
    # We want ROTARY_DIM to be constexpr, otherwise the triton compiler doesn't know that
    # the mask is constant every 8 elements, and it will generate LDG.16 instead of LDG.128
    ROTARY_DIM: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    BLOCK_K: tl.constexpr = triton.next_power_of_2(ROTARY_DIM)
    ROTARY_DIM_HALF = ROTARY_DIM // 2
    pid_head = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)

    # Cast pid_batch to int64 to avoid overflow in stride multiplication for large sequences
    pid_batch_i64 = pid_batch.to(tl.int64)

    if not IS_VARLEN:
        # Use int64 pid to force int64 multiplication with strides
        X = X + pid_batch_i64 * stride_x_batch
        OUT = OUT + pid_batch_i64 * stride_out_batch
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch).to(tl.int64)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - tl.load(CU_SEQLENS + pid_batch)
        X = X + start_idx * stride_x_seqlen
        OUT = OUT + start_idx * stride_out_seqlen

    if pid_m * BLOCK_M >= seqlen:
        return

    # Cast indices to int64 for stride multiplications
    rh = (pid_head * BLOCK_H + tl.arange(0, BLOCK_H)).to(tl.int64)
    rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)

    rk_half = tl.arange(0, BLOCK_K // 2).to(tl.int64)
    COS = COS + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
    SIN = SIN + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
    mask_cs = (rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < ROTARY_DIM_HALF)
    cos = tl.load(COS, mask=mask_cs, other=1.0).to(tl.float32)
    sin = tl.load(SIN, mask=mask_cs, other=0.0).to(tl.float32)
    if CONJUGATE:
        sin = -sin

    if not INTERLEAVED:
        # Load the 1st and 2nd halves of X, do calculation, then store to 1st and 2nd halves of OUT
        X = X + (
            rh[:, None, None] * stride_x_nheads
            + rm[None, :, None] * stride_x_seqlen
            + rk_half[None, None, :] * stride_x_headdim
        )
        OUT = OUT + (
            rh[:, None, None] * stride_out_nheads
            + rm[None, :, None] * stride_out_seqlen
            + rk_half[None, None, :] * stride_out_headdim
        )
        mask = (rh[:, None, None] < nheads) & (rm[None, :, None] < seqlen) & (rk_half[None, None, :] < ROTARY_DIM_HALF)
        x0 = tl.load(X, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(
            X + ROTARY_DIM_HALF * stride_x_headdim,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        tl.store(OUT, o0, mask=mask)
        tl.store(OUT + ROTARY_DIM_HALF * stride_out_headdim, o1, mask=mask)
    else:
        rk = tl.arange(0, BLOCK_K).to(tl.int64)
        X = X + (
            rh[:, None, None] * stride_x_nheads
            + rm[None, :, None] * stride_x_seqlen
            + rk[None, None, :] * stride_x_headdim
        )
        OUT = OUT + (
            rh[:, None, None] * stride_out_nheads
            + rm[None, :, None] * stride_out_seqlen
            + rk[None, None, :] * stride_out_headdim
        )
        mask = (rh[:, None, None] < nheads) & (rm[None, :, None] < seqlen) & (rk[None, None, :] < ROTARY_DIM)
        x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
        x0, x1 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_M, BLOCK_K // 2, 2]))
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        o = tl.reshape(tl.join(o0, o1), [BLOCK_H, BLOCK_M, BLOCK_K])
        tl.store(OUT, o, mask=mask)


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    """Apply rotary embeddings with a Triton kernel.

    Args:
        x: ``[batch, seqlen, nheads, headdim]`` if ``cu_seqlens`` is None, else
            ``[total_seqlen, nheads, headdim]``.
        cos: ``[seqlen_ro, rotary_dim / 2]``.
        sin: ``[seqlen_ro, rotary_dim / 2]``.
        seqlen_offsets: Integer or integer tensor of size ``[batch]``.
        cu_seqlens: ``[batch + 1]`` cumulative lengths, or None.
        max_seqlen: Required when ``cu_seqlens`` is given.
        interleaved: Pairwise-interleaved rotation instead of half-split.
        inplace: Write into ``x`` instead of allocating an output.
        conjugate: Negate ``sin``, i.e. apply the inverse rotation.

    Returns:
        The rotated tensor, shaped like ``x``.
    """
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None, "If cu_seqlens is passed in, then max_seqlen must be passed"
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    output = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        output[..., rotary_dim:].copy_(x[..., rotary_dim:])

    grid = lambda META: (  # noqa: E731
        triton.cdiv(nheads, META["BLOCK_H"]),
        triton.cdiv(seqlen, META["BLOCK_M"]),
        batch,
    )
    # The CUDA grid Y dimension caps at 65535, so BLOCK_M must be at least
    # seqlen / 65535 for the long sequences UPipe targets.
    cuda_max_grid_y = 65535
    block_m = 8 if rotary_dim <= 128 else 4
    min_block_m_for_grid = triton.cdiv(seqlen, cuda_max_grid_y)
    if min_block_m_for_grid > block_m:
        block_m = 1 << (min_block_m_for_grid - 1).bit_length()

    # Without this Triton launches from cuda:0 and rejects pointers on other devices.
    with torch.cuda.device(x.device.index):
        torch.library.wrap_triton(_rotary_kernel)[grid](
            output,  # data ptrs
            x,
            cos,
            sin,
            cu_seqlens,
            seqlen_offsets,
            seqlen,  # shapes
            nheads,
            seqlen_ro,
            output.stride(0) if not is_varlen else 0,  # batch stride if not varlen else 0
            output.stride(-3),  # seqlen stride or total_seqlen stride
            output.stride(-2),  # nheads stride
            output.stride(-1),  # headdim stride
            x.stride(0) if not is_varlen else 0,
            x.stride(-3),
            x.stride(-2),
            x.stride(-1),
            rotary_dim,
            isinstance(seqlen_offsets, torch.Tensor),
            is_varlen,
            interleaved,
            conjugate,
            BLOCK_M=block_m,
            BLOCK_H=2,
        )
    return output


def rope_tables_from_position_embeddings(
    cos: torch.Tensor,
    sin: torch.Tensor,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert HuggingFace ``(cos, sin)`` tables into the layout the kernel wants.

    HuggingFace rotary modules emit ``[batch, seq, head_dim]`` (or ``[seq,
    head_dim]``) tables whose two halves are duplicates, because ``rotate_half``
    consumes the full width. The Triton kernel instead indexes
    ``[seq, head_dim / 2]`` and reconstructs the halves itself, so take the
    first half of the last axis.

    All context-parallel ranks share the same batch, and HuggingFace builds
    these tables from ``position_ids``, so row ``i`` already carries the phase
    for this rank's ``i``-th local token. That is why the kernel is always
    called with ``seqlen_offsets=0``: the offset is baked into the table.

    Args:
        cos: Cosine table, ``[batch, seq, head_dim]`` or ``[seq, head_dim]``.
        sin: Sine table, same shape as ``cos``.
        head_dim: Attention head dimension.

    Returns:
        ``(cos, sin)`` each shaped ``[seq, head_dim / 2]`` and contiguous.

    Raises:
        ValueError: If the tables are not 2D/3D or the last axis is too narrow.
    """
    if cos.dim() == 3:
        cos, sin = cos[0], sin[0]
    elif cos.dim() != 2:
        raise ValueError(f"expected 2D or 3D rotary tables, got {cos.dim()}D")

    half = head_dim // 2
    if cos.shape[-1] < half:
        raise ValueError(f"rotary tables have width {cos.shape[-1]}, need at least {half} for head_dim={head_dim}")

    return cos[:, :half].contiguous(), sin[:, :half].contiguous()

# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Per-expert row padding driven by device-side token counts, without host synchronization.

Grouped expert GEMMs on block-scaled FP8 inputs need every expert's rows to start on an aligned boundary
(MXFP8: 32 rows; Transformer Engine's native grouped-tensor MXFP8 path: 128). Transformer Engine's
``Fp8Padding``/``Fp8Unpadding`` compute that layout from a Python list of per-expert counts, which costs a
``tokens_per_expert.tolist()`` (a device-to-host copy plus a stream synchronization) in every MoE layer.

The kernel here reads the counts from a CUDA tensor. Every program derives the layout itself (an
``experts``-wide cumulative sum), so the host only supplies static sizes:

* compact layout: expert ``e`` owns rows ``[s_e, s_e + c_e)`` with ``s_e = c_0 + ... + c_{e-1}``; rows from
  ``c_0 + ... + c_{E-1}`` to the end of the buffer are ignored (for example the unused tail of a
  capacity-sized dispatch buffer);
* padded layout: expert ``e`` owns rows ``[p_e, p_e + c_e)`` with ``p_e`` the sum of the earlier counts
  each rounded up to ``align``; padding rows and the tail are zero.

:func:`pad_expert_rows` and :func:`unpad_expert_rows` are exact adjoints (a row permutation plus zero fill),
so each is the other's backward and neither accumulates: gradients are bit-reproducible. A padded buffer
never needs more than :func:`padded_rows_bound` rows, a host-side constant when the compact row count is.
"""

from __future__ import annotations

import torch

from nemo_automodel.shared.import_utils import safe_import

_HAVE_TRITON, triton = safe_import("triton")
_HAVE_TRITON_LANGUAGE, tl = safe_import("triton.language")
HAVE_TRITON: bool = bool(_HAVE_TRITON and _HAVE_TRITON_LANGUAGE)

# The row-to-expert lookup is a [rows, experts] comparison per program; local expert counts are far below this.
_MAX_EXPERTS = 4096
# Elements per program tile: 32 per thread at 4 warps keeps the copy in registers (larger tiles spill and take
# minutes to compile).
_TILE_ELEMENTS = 4096
# The kernel moves raw bits: every dtype is viewed as the integer type of its element size.
_BITS_DTYPES = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _check_align(align: int) -> None:
    """Reject alignments below one row.

    Args:
        align: Row alignment of every expert segment.
    """
    if align < 1:
        raise ValueError(f"align must be >= 1, got {align}")


if HAVE_TRITON:

    @triton.jit
    def _move_expert_rows_kernel(
        src_ptr,
        dst_ptr,
        counts_ptr,
        n_experts,
        align,
        n_src_rows,
        n_dst_rows,
        hidden,
        stride_src,
        stride_dst,
        TO_PADDED: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):  # pragma: no cover - Triton JIT executes on the GPU, outside Python tracing.
        # One [BLOCK_R, BLOCK_H] tile of destination rows, copied from the other layout (zeros elsewhere).
        pid_r = tl.program_id(0).to(tl.int64)
        pid_h = tl.program_id(1)

        experts = tl.arange(0, BLOCK_E)
        in_range = experts < n_experts
        counts = tl.maximum(tl.load(counts_ptr + experts, mask=in_range, other=0).to(tl.int64), 0)
        padded = (counts + align - 1) // align * align
        compact_end = tl.cumsum(counts, axis=0)
        padded_end = tl.cumsum(padded, axis=0)
        compact_start = compact_end - counts
        padded_start = padded_end - padded

        # Owner of each destination row: the number of experts whose segment ends at or before it.
        rows = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
        never = 0x7FFFFFFFFFFFFFFF
        if TO_PADDED:
            ends = tl.where(in_range, padded_end, never)
        else:
            ends = tl.where(in_range, compact_end, never)
        owner = tl.sum((ends[None, :] <= rows[:, None]).to(tl.int32), axis=1)
        hit = experts[None, :] == owner[:, None]
        row_count = tl.sum(tl.where(hit, counts[None, :], 0), axis=1)
        row_compact = tl.sum(tl.where(hit, compact_start[None, :], 0), axis=1)
        row_padded = tl.sum(tl.where(hit, padded_start[None, :], 0), axis=1)

        if TO_PADDED:
            local = rows - row_padded
            valid = (owner < n_experts) & (local < row_count)
            src_rows = row_compact + local
        else:
            local = rows - row_compact
            valid = owner < n_experts
            src_rows = row_padded + local
        in_dst = rows < n_dst_rows
        valid = valid & in_dst & (src_rows < n_src_rows)

        cols = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        col_ok = cols < hidden
        values = tl.load(
            src_ptr + src_rows[:, None] * stride_src + cols[None, :],
            mask=valid[:, None] & col_ok[None, :],
            other=0,
        )
        tl.store(dst_ptr + rows[:, None] * stride_dst + cols[None, :], values, mask=in_dst[:, None] & col_ok[None, :])


def padded_rows_bound(rows: int, *, num_experts: int, align: int) -> int:
    """Rows a padded buffer needs for any split of ``rows`` compact rows over ``num_experts`` experts.

    Args:
        rows: Compact row count (for example a dispatch capacity).
        num_experts: Number of expert segments.
        align: Row alignment of every segment.

    Returns:
        ``rows + num_experts * (align - 1)`` rounded up to a multiple of ``align``.
    """
    _check_align(align)
    bound = rows + num_experts * (align - 1)
    return -(-bound // align) * align


def padded_splits(tokens_per_expert: torch.Tensor, *, align: int) -> torch.Tensor:
    """Per-expert row counts rounded up to ``align``, computed where the counts live.

    Args:
        tokens_per_expert: ``[experts]`` integer tensor of rows per expert, on any device.
        align: Row alignment of every segment.

    Returns:
        ``[experts]`` int64 tensor on ``tokens_per_expert.device``: each count rounded up to ``align``.
    """
    _check_align(align)
    counts = tokens_per_expert.to(torch.int64)
    return (counts + (align - 1)) // align * align


def _move_on_host(src: torch.Tensor, counts: torch.Tensor, align: int, dst_rows: int, to_padded: bool) -> torch.Tensor:
    """Host-driven layout change for CPU tensors (reads the counts with ``tolist()``).

    Args:
        src: ``[src_rows, hidden]`` CPU tensor in the source layout.
        counts: ``[experts]`` integer CPU tensor of rows per expert.
        align: Row alignment of the padded layout.
        dst_rows: Rows of the result.
        to_padded: True moves compact rows to the padded layout, False the reverse.

    Returns:
        ``[dst_rows, hidden]`` tensor of ``src.dtype`` in the other layout, zeros where no row maps.
    """
    dst = src.new_zeros((dst_rows, src.shape[1]))
    compact = padded = 0
    for count in counts.tolist():
        count = max(int(count), 0)
        if to_padded:
            take = max(0, min(count, src.shape[0] - compact, dst_rows - padded))
            dst[padded : padded + take] = src[compact : compact + take]
        else:
            take = max(0, min(count, src.shape[0] - padded, dst_rows - compact))
            dst[compact : compact + take] = src[padded : padded + take]
        compact += count
        padded += -(-count // align) * align
    return dst


def _move(src: torch.Tensor, counts: torch.Tensor, align: int, dst_rows: int, to_padded: bool) -> torch.Tensor:
    """Move rows between the compact and padded expert layouts (see the module docstring).

    Args:
        src: ``[src_rows, *feature]`` tensor in the source layout; CUDA tensors use the Triton kernel and are
            never synchronized, CPU tensors use the host-driven path.
        counts: ``[experts]`` integer tensor of rows per expert, on ``src.device``.
        align: Row alignment of the padded layout, at least 1.
        dst_rows: Rows of the result.
        to_padded: True moves compact rows to the padded layout, False the reverse.

    Returns:
        ``[dst_rows, *feature]`` tensor of ``src.dtype`` on ``src.device``; rows no source row maps to are zero.
    """
    _check_align(align)
    if dst_rows < 0:
        raise ValueError(f"dst_rows must be >= 0, got {dst_rows}")
    if counts.dim() != 1 or counts.dtype.is_floating_point or counts.dtype.is_complex or counts.dtype == torch.bool:
        raise ValueError(f"tokens_per_expert must be a 1-D integer tensor, got {counts.dtype} {tuple(counts.shape)}")
    if counts.device != src.device:
        raise ValueError(f"tokens_per_expert must be on the rows' device {src.device}, got device {counts.device}")
    if counts.numel() > _MAX_EXPERTS:
        raise ValueError(f"at most {_MAX_EXPERTS} experts are supported, got {counts.numel()}")
    feature_shape = tuple(src.shape[1:])
    src2d = src.reshape(src.shape[0], -1)
    if src2d.shape[1] and src2d.stride(1) != 1:
        src2d = src2d.contiguous()
    if not src.is_cuda:
        return _move_on_host(src2d, counts, align, dst_rows, to_padded).reshape((dst_rows,) + feature_shape)
    if not HAVE_TRITON:
        raise RuntimeError("Device-side expert padding of CUDA tensors requires Triton.")
    bits = _BITS_DTYPES.get(src2d.element_size())
    if bits is None:
        raise ValueError(f"unsupported element size {src2d.element_size()} of {src.dtype}")
    hidden = src2d.shape[1]
    dst = torch.empty((dst_rows, hidden), dtype=bits, device=src2d.device)
    if dst_rows and hidden:
        block_h = min(1024, max(16, triton.next_power_of_2(hidden)))
        block_r = min(64, _TILE_ELEMENTS // block_h)
        block_e = max(16, triton.next_power_of_2(counts.numel()))
        grid = (triton.cdiv(dst_rows, block_r), triton.cdiv(hidden, block_h))
        _move_expert_rows_kernel[grid](
            src2d.view(bits),
            dst,
            counts.contiguous(),
            counts.numel(),
            align,
            src2d.shape[0],
            dst_rows,
            hidden,
            src2d.stride(0),
            dst.stride(0),
            TO_PADDED=to_padded,
            BLOCK_R=block_r,
            BLOCK_H=block_h,
            BLOCK_E=block_e,
            num_warps=4,
        )
    return dst.view(src2d.dtype).reshape((dst_rows,) + feature_shape)


class _PadExpertRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, tokens_per_expert: torch.Tensor, align: int, padded_rows: int) -> torch.Tensor:
        """Compact to padded layout.

        Args:
            ctx: Autograd context.
            x: ``[rows, *feature]`` expert-grouped rows.
            tokens_per_expert: ``[experts]`` integer tensor of rows per expert, on ``x.device``.
            align: Row alignment of every expert segment.
            padded_rows: Rows of the result.

        Returns:
            ``[padded_rows, *feature]`` tensor of ``x.dtype`` in the padded layout.
        """
        ctx.save_for_backward(tokens_per_expert)
        ctx.align = align
        ctx.rows = x.shape[0]
        return _move(x, tokens_per_expert, align, padded_rows, to_padded=True)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        """Gradient of the padding: gather the padded gradient back into the compact layout.

        Args:
            ctx: Autograd context.
            grad: ``[padded_rows, *feature]`` gradient of the padded output.

        Returns:
            ``[rows, *feature]`` gradient for ``x`` (zeros for rows past the counted total); no gradients for
            the counts, alignment, or size.
        """
        (tokens_per_expert,) = ctx.saved_tensors
        return _move(grad, tokens_per_expert, ctx.align, ctx.rows, to_padded=False), None, None, None


class _UnpadExpertRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y: torch.Tensor, tokens_per_expert: torch.Tensor, align: int, rows: int) -> torch.Tensor:
        """Padded to compact layout.

        Args:
            ctx: Autograd context.
            y: ``[padded_rows, *feature]`` rows in the padded layout.
            tokens_per_expert: ``[experts]`` integer tensor of rows per expert, on ``y.device``.
            align: Row alignment of every expert segment.
            rows: Rows of the result.

        Returns:
            ``[rows, *feature]`` tensor of ``y.dtype`` in the compact layout.
        """
        ctx.save_for_backward(tokens_per_expert)
        ctx.align = align
        ctx.padded_rows = y.shape[0]
        return _move(y, tokens_per_expert, align, rows, to_padded=False)

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor, None, None, None]:
        """Gradient of the unpadding: spread the compact gradient back into the padded layout.

        Args:
            ctx: Autograd context.
            grad: ``[rows, *feature]`` gradient of the compact output.

        Returns:
            ``[padded_rows, *feature]`` gradient for ``y`` (zeros on padding rows); no gradients for the
            counts, alignment, or size.
        """
        (tokens_per_expert,) = ctx.saved_tensors
        return _move(grad, tokens_per_expert, ctx.align, ctx.padded_rows, to_padded=True), None, None, None


def pad_expert_rows(
    x: torch.Tensor, tokens_per_expert: torch.Tensor, *, align: int, padded_rows: int | None = None
) -> torch.Tensor:
    """Spread expert-grouped rows so that every expert segment starts on an ``align``-row boundary.

    Args:
        x: ``[rows, *feature]`` rows grouped by expert in expert order (any dtype, CUDA or CPU); rows past
            ``tokens_per_expert.sum()`` are ignored.
        tokens_per_expert: ``[experts]`` integer tensor of rows per expert on ``x.device``. On CUDA it is
            read only by the kernel, never on the host.
        align: Row alignment of every expert segment.
        padded_rows: Rows of the result; defaults to ``padded_rows_bound(rows, num_experts=experts,
            align=align)``, enough for any split.

    Returns:
        ``[padded_rows, *feature]`` tensor of ``x.dtype`` on ``x.device``: expert ``e``'s rows at offset
        ``sum(padded_splits(tokens_per_expert, align=align)[:e])``, zeros on padding and tail rows.
    """
    if padded_rows is None:
        padded_rows = padded_rows_bound(x.shape[0], num_experts=tokens_per_expert.numel(), align=align)
    return _PadExpertRows.apply(x, tokens_per_expert, align, padded_rows)


def unpad_expert_rows(y: torch.Tensor, tokens_per_expert: torch.Tensor, *, align: int, rows: int) -> torch.Tensor:
    """Gather every expert's rows from the padded layout back into the compact layout.

    Inverse of :func:`pad_expert_rows` for the same ``tokens_per_expert`` and ``align``.

    Args:
        y: ``[padded_rows, *feature]`` rows in the padded layout (any dtype, CUDA or CPU).
        tokens_per_expert: ``[experts]`` integer tensor of rows per expert on ``y.device``. On CUDA it is
            read only by the kernel, never on the host.
        align: Row alignment the padded layout was built with.
        rows: Rows of the result.

    Returns:
        ``[rows, *feature]`` tensor of ``y.dtype`` on ``y.device``; rows past ``tokens_per_expert.sum()``
        are zero.
    """
    return _UnpadExpertRows.apply(y, tokens_per_expert, align, rows)

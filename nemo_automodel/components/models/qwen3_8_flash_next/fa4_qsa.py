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

"""SM90 QSA using Triton preprocessing and FlashAttention 4.

Optional FA4/CuTe dependencies are loaded on the first CUDA call. The only
CuTe code owned here is the mask callback compiled into FA4's kernels.
Three Triton kernels build FA4's byte membership table and block lists from
the routes: they need no recompilation when the route width changes, and one
read of the table classifies both the forward and the backward tiles.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable

import torch

from nemo_automodel.components.attention.fa4 import import_fa4_modules
from nemo_automodel.shared.import_utils import safe_import

_HAVE_TRITON, triton = safe_import("triton")
_HAVE_TRITON_LANGUAGE, tl = safe_import("triton.language")
HAVE_TRITON: bool = bool(_HAVE_TRITON and _HAVE_TRITON_LANGUAGE)

# FA4 QSA query blocks hold 128 rows.
_QUERY_BLOCK = 128
# Keys each chunk-classification program reads per row (a 128 x 256-byte tile).
_KEYS_PER_CHUNK_PROGRAM = 256


@functools.cache
def _load_fa4() -> tuple[Callable, type, Callable]:
    """Cache optional FA4 entry points and the mask callback, never tensors."""
    interface, sparsity, cutlass, cute, utils = import_fa4_modules(
        "flash_attn.cute.interface",
        "flash_attn.cute.block_sparsity",
        "cutlass",
        "cutlass.cute",
        "flash_attn.cute.utils",
    )

    @cute.jit
    def mask_mod(batch, head, query_idx, key_idx, seqlen_info, aux_tensors):
        """Read QSA membership inside FA4; this is not a standalone kernel.

        Args:
            batch: Scalar SSA batch coordinate.
            head: Scalar SSA query-head coordinate; membership is head-independent.
            query_idx: Scalar SSA query position.
            key_idx: Scalar SSA physical K/V position.
            seqlen_info: FA4 query/key sequence lengths.
            aux_tensors: One uint8 tensor [batch, queries, keys], with padded
                row strides; one means selected and zero means excluded.

        Returns:
            Scalar boolean SSA membership, false outside the logical sequences.
        """
        membership = aux_tensors[0]
        qi = cutlass.min(query_idx[0], seqlen_info.seqlen_q - 1)
        ki = cutlass.min(key_idx[0], seqlen_info.seqlen_k - 1)
        selected = membership[batch[0], qi, ki] != 0
        valid = selected & (query_idx[0] < seqlen_info.seqlen_q) & (key_idx[0] < seqlen_info.seqlen_k)
        return utils.scalar_to_ssa(valid, cutlass.Boolean)

    return interface.flash_attn_func, sparsity.BlockSparseTensorsTorch, mask_mod


if HAVE_TRITON:
    # The route width follows each packed batch's longest document; keep it a runtime value so a
    # new width never compiles a new kernel variant.
    @triton.jit(do_not_specialize=["width", "route_batch_stride", "route_query_stride"])
    def _scatter_routes_kernel(
        routes,
        membership,
        queries,
        width,
        kv_length,
        route_batch_stride,
        route_query_stride,
        route_width_stride,
        membership_row_stride,
        BLOCK: tl.constexpr,
        ROW_BLOCK: tl.constexpr,
    ):
        """Write one query row of the membership table: zeros, then ones at every valid route.

        The row is zeroed by this program right before its routes are marked, so its cache lines are still in
        L2 when the scattered stores land; they merge there instead of becoming partial-sector writes to DRAM.
        The indexer emits each selected compressed block as four consecutive keys, and packed documents start
        at multiples of 64. A route group that covers one aligned 4-key block is stored as a single 32-bit word
        of ones; every other valid ID is stored as one byte.

        Args:
            routes: Signed int32/int64 tensor [batch, queries, width] with arbitrary strides.
            membership: Uninitialized uint8 tensor [batch * queries, membership_row_stride] with a row stride
                that is a multiple of four; one program writes every byte of one row. IDs outside
                [0, kv_length) are skipped, and duplicates store one twice.
        """
        row = tl.program_id(0).to(tl.int64)
        source = routes + (row // queries) * route_batch_stride + (row % queries) * route_query_stride
        target = membership + row * membership_row_stride
        for start in range(0, membership_row_stride, ROW_BLOCK):
            cols = start + tl.arange(0, ROW_BLOCK)
            tl.store(target + cols, tl.zeros((ROW_BLOCK,), tl.uint8), mask=cols < membership_row_stride)
        # The zeros must land before the ones that overwrite them; only this program writes this row.
        tl.debug_barrier()
        lanes = tl.arange(0, 4)
        for start in range(0, width, BLOCK):
            offsets = start + tl.arange(0, BLOCK)
            ids = tl.load(source + offsets.to(tl.int64) * route_width_stride, mask=offsets < width, other=-1)
            valid = (ids >= 0) & (ids < kv_length)
            groups = tl.reshape(ids, (BLOCK // 4, 4))
            first = tl.sum(tl.where(lanes[None, :] == 0, groups, 0), axis=1)
            consecutive = tl.min((groups == first[:, None] + lanes[None, :]).to(tl.int32), axis=1) == 1
            word = consecutive & (first % 4 == 0) & (first >= 0) & (first + 3 < kv_length)
            word_target = (target + tl.where(word, first, 0)).to(tl.pointer_type(tl.uint32))
            tl.store(word_target, tl.full((BLOCK // 4,), 0x01010101, tl.uint32), mask=word)
            in_word = tl.reshape(tl.broadcast_to(word[:, None], (BLOCK // 4, 4)), (BLOCK,))
            single = valid & ~in_word
            tl.store(target + tl.where(single, ids, 0), tl.full((BLOCK,), 1, tl.uint8), mask=single)

    @triton.jit
    def _chunk_kinds_kernel(
        membership,
        kinds,
        queries,
        membership_row_stride,
        row_blocks,
        chunks,
        CHUNK_KEYS: tl.constexpr,
        CHUNKS_PER_PROGRAM: tl.constexpr,
        ROWS: tl.constexpr,
    ):
        """Classify 128-query x CHUNK_KEYS-key chunks of the membership table.

        Args:
            membership: Uint8 tensor [batch * queries, membership_row_stride] of zeros and ones whose row
                padding past the last key is zero.
            kinds: Int8 tensor [batch, row_blocks, chunks] receiving absent (0), partial (1) or
                full (2). Chunks that reach past the last query or key are never full.
        """
        tile = tl.program_id(0)
        chunk_group = tl.program_id(2) * tl.num_programs(1) + tl.program_id(1)
        rows = (tile % row_blocks) * ROWS + tl.arange(0, ROWS)
        cols = chunk_group.to(tl.int64) * (CHUNKS_PER_PROGRAM * CHUNK_KEYS) + tl.arange(
            0, CHUNKS_PER_PROGRAM * CHUNK_KEYS
        )
        # Bound columns by the 32-byte-aligned row stride, not kv_length, so loads stay vectorized: the
        # row padding past kv_length is zero, which keeps a chunk that reaches past the last key from being full.
        in_bounds = (rows < queries)[:, None] & (cols < membership_row_stride)[None, :]
        offsets = ((tile // row_blocks) * queries + rows).to(tl.int64)[:, None] * membership_row_stride + cols[None, :]
        values = tl.load(membership + offsets, mask=in_bounds, other=0).to(tl.int32)
        counts = tl.sum(tl.sum(tl.reshape(values, (ROWS, CHUNKS_PER_PROGRAM, CHUNK_KEYS)), axis=2), axis=0)
        chunk_kinds = (counts > 0).to(tl.int8) + (counts == ROWS * CHUNK_KEYS).to(tl.int8)
        chunk_ids = chunk_group * CHUNKS_PER_PROGRAM + tl.arange(0, CHUNKS_PER_PROGRAM)
        tl.store(kinds + tile.to(tl.int64) * chunks + chunk_ids, chunk_kinds, mask=chunk_ids < chunks)

    @triton.jit
    def _compact_block_lists_kernel(
        kinds,
        partial_counts,
        partial_ids,
        full_counts,
        full_ids,
        lists,
        columns,
        row_blocks,
        chunks,
        LIST_ROW_BLOCK: tl.constexpr,
        COLUMN_ROW_BLOCK: tl.constexpr,
        LIST_CHUNK: tl.constexpr,
        COLUMN_CHUNK: tl.constexpr,
        SPAN: tl.constexpr,
        BLOCK_COLUMNS: tl.constexpr,
        COLUMN_TILES: tl.constexpr,
    ):
        """Merge chunk kinds into FA4 block kinds and write one sorted block list per program.

        A list's column c covers query block ``list * LIST_ROW_BLOCK + c * COLUMN_ROW_BLOCK`` and
        chunks ``list * LIST_CHUNK + c * COLUMN_CHUNK + [0, SPAN)``: lists are query blocks and
        columns key blocks in the forward direction, and the reverse in the backward direction.

        Args:
            kinds: Int8 tensor [batch, row_blocks, chunks] from _chunk_kinds_kernel.
            partial_counts: Int32 tensor [batch, 1, lists] receiving the partial-block counts.
            partial_ids: Int32 tensor [batch, 1, lists, columns] receiving the partial columns in
                ascending order, then ``columns`` as padding.
            full_counts: Int32 tensor [batch, 1, lists] receiving the full-block counts.
            full_ids: Int32 tensor [batch, 1, lists, columns] laid out like partial_ids.
        """
        program = tl.program_id(0)
        item = program % lists
        partial_count = 0
        full_count = 0
        output = program.to(tl.int64) * columns
        for tile in range(COLUMN_TILES):
            cols = tile * BLOCK_COLUMNS + tl.arange(0, BLOCK_COLUMNS)
            col_ok = cols < columns
            row_block = item * LIST_ROW_BLOCK + cols * COLUMN_ROW_BLOCK
            first_chunk = item * LIST_CHUNK + cols * COLUMN_CHUNK
            base = kinds + ((program // lists) * row_blocks + row_block).to(tl.int64) * chunks
            present = tl.zeros((BLOCK_COLUMNS,), tl.int1)
            full = col_ok
            for j in tl.static_range(SPAN):
                chunk = first_chunk + j
                kind = tl.load(base + chunk, mask=col_ok & (chunk < chunks), other=0)
                present = present | (kind > 0)
                full = full & (kind == 2)
            partial = present & ~full
            # Append selected columns in ascending order. Write padding from the opposite
            # end, so each column claims one slot without knowing the final counts.
            partial_prefix = partial_count + tl.cumsum(partial.to(tl.int32), axis=0)
            full_prefix = full_count + tl.cumsum(full.to(tl.int32), axis=0)
            partial_slot = tl.where(partial, partial_prefix - 1, columns - 1 - cols + partial_prefix)
            full_slot = tl.where(full, full_prefix - 1, columns - 1 - cols + full_prefix)
            tl.store(partial_ids + output + partial_slot, tl.where(partial, cols, columns), mask=col_ok)
            tl.store(full_ids + output + full_slot, tl.where(full, cols, columns), mask=col_ok)
            partial_count += tl.sum(partial.to(tl.int32), axis=0)
            full_count += tl.sum(full.to(tl.int32), axis=0)
        tl.store(partial_counts + program, partial_count)
        tl.store(full_counts + program, full_count)


def _preprocess(
    routes: torch.Tensor, kv_length: int, forward_block_keys: int, backward_block_keys: int
) -> tuple[torch.Tensor, ...]:
    """Build a byte membership table and FA4 forward/reverse block lists with Triton.

    Args:
        routes: Signed int32/int64 CUDA tensor [batch, queries, routes_per_query]
            with arbitrary strides. Duplicate IDs collapse; invalid IDs, including
            large int64 values, are discarded. routes_per_query may differ between
            calls without compiling new kernels.
        kv_length: Positive number of physical key/value positions.
        forward_block_keys: Keys per forward block, FA4's forward KV tile width.
        backward_block_keys: Keys per backward block, FA4's backward KV tile width.

    Returns:
        Uint8 membership [batch, queries, kv_length], with each row padded to a
        multiple of 32 bytes (one more than kv_length, rounded up). Then the four
        forward tensors and the four reverse tensors, each set in this layout:
        int32 partial counts [batch, 1, block_rows], partial column indices
        [batch, 1, block_rows, block_columns] in ascending order padded with
        block_columns, then full counts and indices with the same layouts. Only
        each count's index prefix is consumed. Forward rows are ceil(queries/128)
        and columns ceil(kv_length/forward_block_keys); reverse rows are
        ceil(kv_length/backward_block_keys) and columns ceil(queries/128). All
        tensors are independently allocated and contiguous apart from the
        membership row padding. No input is mutated, and no device scalar is
        read on the host.

    Raises:
        ImportError: When Triton is not installed.
    """
    if not HAVE_TRITON:
        raise ImportError("FA4 QSA preprocessing requires Triton")
    batch, queries, width = routes.shape
    row_stride = ((kv_length + 1 + 31) // 32) * 32
    storage = torch.empty((batch, queries, row_stride), device=routes.device, dtype=torch.uint8)
    _scatter_routes_kernel[(batch * queries,)](
        routes,
        storage,
        queries,
        width,
        kv_length,
        *routes.stride(),
        row_stride,
        BLOCK=2048,
        ROW_BLOCK=min(4096, triton.next_power_of_2(row_stride)),
        # Fewer, wider programs keep fewer rows in flight, so a zeroed row is still in L2 when its routes land:
        # on H100 (50 MiB L2) the scatter of 24K-32K-token rows runs up to 1.9x faster than with 4 warps, and
        # on GB200 (129 MiB L2) the warp count changes it by under 5%.
        num_warps=16,
    )
    # One pass over the table classifies chunks that tile both the forward and the backward blocks.
    chunk_keys = math.gcd(forward_block_keys, backward_block_keys)
    chunks = triton.cdiv(kv_length, chunk_keys)
    row_blocks = triton.cdiv(queries, _QUERY_BLOCK)
    chunks_per_program = max(1, _KEYS_PER_CHUNK_PROGRAM // chunk_keys)
    kinds = torch.empty((batch, row_blocks, chunks), device=routes.device, dtype=torch.int8)
    # CUDA limits each of grid.y and grid.z to 65535 programs.
    groups = triton.cdiv(chunks, chunks_per_program)
    grid_z = triton.cdiv(groups, 65535)
    _chunk_kinds_kernel[(batch * row_blocks, triton.cdiv(groups, grid_z), grid_z)](
        storage,
        kinds,
        queries,
        row_stride,
        row_blocks,
        chunks,
        CHUNK_KEYS=chunk_keys,
        CHUNKS_PER_PROGRAM=chunks_per_program,
        ROWS=_QUERY_BLOCK,
        num_warps=8,
    )
    block_lists = []
    forward_span, backward_span = forward_block_keys // chunk_keys, backward_block_keys // chunk_keys
    # Forward lists are query blocks over key blocks; backward lists are key blocks over query blocks.
    for lists, columns, list_row_block, column_row_block, list_chunk, column_chunk, span in (
        (row_blocks, triton.cdiv(kv_length, forward_block_keys), 1, 0, 0, forward_span, forward_span),
        (triton.cdiv(kv_length, backward_block_keys), row_blocks, 0, 1, backward_span, 0, backward_span),
    ):
        counts = torch.empty((batch, 1, lists), device=routes.device, dtype=torch.int32)
        ids = torch.empty((batch, 1, lists, columns), device=routes.device, dtype=torch.int32)
        full_counts, full_ids = torch.empty_like(counts), torch.empty_like(ids)
        block_columns = min(4096, triton.next_power_of_2(columns))
        _compact_block_lists_kernel[(batch * lists,)](
            kinds,
            counts,
            ids,
            full_counts,
            full_ids,
            lists,
            columns,
            row_blocks,
            chunks,
            LIST_ROW_BLOCK=list_row_block,
            COLUMN_ROW_BLOCK=column_row_block,
            LIST_CHUNK=list_chunk,
            COLUMN_CHUNK=column_chunk,
            SPAN=span,
            BLOCK_COLUMNS=block_columns,
            COLUMN_TILES=triton.cdiv(columns, block_columns),
        )
        block_lists += [counts, ids, full_counts, full_ids]
    return storage[..., :kv_length], *block_lists


def fa4_sparse_gqa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_token_ids: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Evaluate exact route-set GQA using FA4 on SM90.

    Duplicate routes select a token once. Invalid IDs are ignored. Empty query
    rows have zero output and query gradients. Routes encode causality,
    documents and padding; no additional triangular mask is imposed.
    FA4 owns first-order autograd; higher-order gradients and deterministic
    backward are unsupported. Discrete preprocessing runs in Triton kernels.

    Args:
        query: BF16 CUDA tensor [batch, queries, query_heads, 256].
            Arbitrary strides are accepted; non-unit final strides are copied.
        key: BF16 CUDA tensor [batch, keys, kv_heads, 256]. May contain gathered
            global K/V while query is a local CP slice. query_heads must be a
            positive multiple of kv_heads.
        value: BF16 CUDA tensor with key's shape and device.
        selected_token_ids: Signed int32/int64 CUDA tensor
            [batch, queries, routes_per_query] in physical K/V coordinates.
            Noncontiguous inputs are accepted. Dimensions must be nonempty.
        softmax_scale: Finite positive score multiplier; defaults to 1/sqrt(256).

    Returns:
        Independent BF16 CUDA tensor [batch, queries, query_heads, 256].
        Inputs are not mutated.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("FA4 QSA expects Q/K/V in [batch, sequence, heads, head_dim] layout")
    if key.shape != value.shape or key.shape[0] != query.shape[0]:
        raise ValueError("FA4 QSA requires matching K/V shapes and Q/K/V batch sizes")
    if any(size <= 0 for tensor in (query, key, value) for size in tensor.shape):
        raise ValueError("FA4 QSA requires nonempty Q/K/V dimensions")
    if query.shape[-1] != 256 or key.shape[-1] != 256:
        raise ValueError("FA4 QSA requires head_dim=256")
    if query.shape[2] % key.shape[2] != 0:
        raise ValueError("FA4 QSA requires query_heads divisible by kv_heads")
    if selected_token_ids.ndim != 3 or selected_token_ids.shape[:2] != query.shape[:2]:
        raise ValueError("FA4 QSA routes must have shape [batch, query_sequence, routes]")
    if selected_token_ids.shape[-1] == 0:
        raise ValueError("FA4 QSA requires a nonempty route dimension; use -1 for empty rows")
    if selected_token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("FA4 QSA routes must be signed int32 or int64")
    if any(t.dtype != torch.bfloat16 for t in (query, key, value)):
        raise TypeError("FA4 QSA requires BF16 query, key, and value")
    if not query.is_cuda or any(t.device != query.device for t in (key, value, selected_token_ids)):
        raise ValueError("FA4 QSA requires Q/K/V/routes on the same CUDA device")
    if torch.cuda.get_device_capability(query.device) != (9, 0):
        raise RuntimeError("FA4 QSA currently supports SM90 GPUs only")
    # SM90's head_dim=256 FA4 kernels run 128x80 forward and 64x64 backward tiles.
    forward_keys, backward_keys = 80, 64
    padded_keys = ((key.shape[1] + 32) // 32) * 32
    if query.shape[0] * query.shape[1] * padded_keys >= 2**31:
        raise ValueError("FA4 QSA byte mask exceeds int32 indexing limits")
    scale = 256**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("FA4 QSA softmax_scale must be finite and positive")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("FA4 QSA backward uses atomic reductions and does not support deterministic algorithms")
    with torch.cuda.device(query.device):
        attention, block_lists, mask_mod = _load_fa4()
        raw = _preprocess(selected_token_ids, key.shape[1], forward_keys, backward_keys)
        forward_blocks = block_lists(*raw[1:5], block_size=(128, forward_keys))
        backward_blocks = block_lists(*raw[5:9], block_size=(128, backward_keys))
        # No tensor Python attributes: FA4 can also consume checkpoint-restored
        # auxiliary tensors, whose storage/layout survive without custom tags.
        output, _ = attention(
            *(t.contiguous() if t.stride(-1) != 1 else t for t in (query, key, value)),
            softmax_scale=scale,
            pack_gqa=False,
            mask_mod=mask_mod,
            aux_tensors=[raw[0]],
            block_sparse_tensors=forward_blocks,
            block_sparse_tensors_bwd=backward_blocks,
        )
        return output

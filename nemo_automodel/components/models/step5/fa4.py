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

"""Step-5 attention through FA4, with compact region membership metadata."""

import functools
from collections.abc import Callable
from typing import NamedTuple

import torch
from torch.nn import functional as F

from nemo_automodel.components.models.step5.sparse import CausalDocumentLayout, RegionSelection, pack_region_bitmap
from nemo_automodel.shared.import_utils import safe_import


@functools.cache
def _load_fa4(
    window: int | None,
    bitmap: bool,
    slots: int,
    provider_groups: int,
    *,
    heads_per_group: int = 0,
    canonical_region_size: int = 0,
) -> tuple[Callable, type, Callable]:
    """Load FA4 and cache a mask, optionally specializing a proven document.

    canonical_region_size>0 requires one unpadded B1 CP1 document with Q=K,
    zero query_start and positions exactly 0..K-1. Its token_regions are
    physical_key // canonical_region_size; learned membership is unchanged.
    """
    imported = [
        safe_import(name)
        for name in (
            "flash_attn.cute.interface",
            "flash_attn.cute.block_sparsity",
            "cutlass",
            "cutlass.cute",
            "flash_attn.cute.utils",
        )
    ]
    if not all(available for available, _ in imported):
        raise ImportError("Step5 FA4 requires the AutoModel INSTALL_FA4 image (CuTe DSL 4.6.2 and compatible TVM FFI)")
    interface, sparsity, cutlass, cute, utils = [module for _, module in imported]

    @cute.jit
    def mask_mod(batch, head, query_idx, key_idx, seqlen_info, aux_tensors):
        """Evaluate Step-5 membership inside FA4.

        Args:
            batch: Scalar SSA batch coordinate.
            head: Scalar SSA query-head coordinate.
            query_idx: Scalar SSA local query coordinate.
            key_idx: Scalar SSA global key coordinate.
            seqlen_info: FA4 sequence-length metadata.
            aux_tensors: Local positions/documents [batch, queries], global
                positions/documents/padding [batch, keys], membership
                [batch, groups, queries, slots_or_words], and token-to-region
                indices [batch, keys]. Integer metadata is read-only.

        Returns:
            Scalar boolean SSA selecting valid causal keys in the same document
            and the sliding window or learned region set.
        """
        positions, key_positions, documents, key_documents, padding, membership, token_regions = aux_tensors
        b = batch[0] if cutlass.const_expr(heads_per_group > 0) else batch[0] // provider_groups
        q = cutlass.min(query_idx[0], seqlen_info.seqlen_q - 1)
        k = cutlass.min(key_idx[0], seqlen_info.seqlen_k - 1)
        if cutlass.const_expr(canonical_region_size > 0):
            distance = q - k
            valid = (query_idx[0] < seqlen_info.seqlen_q) & (key_idx[0] < seqlen_info.seqlen_k) & (distance >= 0)
        else:
            distance = positions[b, q] - key_positions[b, k]
            valid = (
                (query_idx[0] < seqlen_info.seqlen_q)
                & (key_idx[0] < seqlen_info.seqlen_k)
                & (distance >= 0)
                & (padding[b, k] == 0)
                & (documents[b, q] == key_documents[b, k])
            )
        if cutlass.const_expr(window is not None):
            valid = valid & (distance < window)
        elif cutlass.const_expr(slots > 0):
            target = (
                k // canonical_region_size if cutlass.const_expr(canonical_region_size > 0) else token_regions[b, k]
            )
            group = (
                head[0] // heads_per_group if cutlass.const_expr(heads_per_group > 0) else batch[0] % provider_groups
            )
            if cutlass.const_expr(bitmap):
                word = membership[b, group, q, target // 32]
                valid = valid & (((word >> (target % 32)) & 1) != 0)
            else:
                index = cutlass.Int32(-1)
                for shift in cutlass.range_constexpr(slots.bit_length() - 1, -1, -1):
                    candidate = cutlass.min(index + (1 << shift), slots - 1)
                    if membership[b, group, q, candidate] <= target:
                        index = candidate
                found = membership[b, group, q, cutlass.max(index, 0)] == target
                valid = valid & (index >= 0) & found
        return utils.scalar_to_ssa(valid, cutlass.Boolean)

    return interface.flash_attn_func, sparsity.BlockSparseTensorsTorch, mask_mod


@functools.cache
def _load_fa4_varlen() -> Callable:
    """Load native FA4 causal varlen attention without a custom mask."""
    available, interface = safe_import("flash_attn.cute.interface")
    if not available:
        raise ImportError("Step5 FA4 requires the AutoModel INSTALL_FA4 image")
    return interface.flash_attn_varlen_func


def _block_lists(
    queries: int,
    keys: int,
    query_start: int,
    window: int | None,
    *,
    block_q: int,
    block_k: int,
    reverse: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build block lists without allocating a dense block grid.

    Args:
        queries: Local query length.
        keys: Global key length.
        query_start: Physical global offset of the query shard.
        window: Sliding window size, or None for all causal blocks.
        block_q: Query block size.
        block_k: Key block size.
        reverse: Return key-to-query lists used by backward.
        device: CUDA metadata device.

    Returns:
        Int32 counts [1, 1, block_rows] and indices [1, 1, block_rows, width].
        For CSA the indices broadcast a single column vector across all rows,
        so physical storage is linear in sequence length. SWA stores only its
        bounded window. Tensor strides, including zero strides, are intentional.
    """
    nq, nk = (queries + block_q - 1) // block_q, (keys + block_k - 1) // block_k
    if reverse:
        rows = torch.arange(nk, device=device, dtype=torch.int32)
        first = ((rows * block_k - query_start) // block_q).clamp(min=0, max=nq)
        if window is None:
            count = nq - first
            indices = torch.arange(nq - 1, -1, -1, device=device, dtype=torch.int32).expand(nk, -1)
        else:
            last = (((rows + 1) * block_k + window - 2 - query_start) // block_q).clamp(max=nq - 1)
            width = min(nq, (window + block_k + block_q - 2) // block_q + 1)
            count = (last - first + 1).clamp(min=0, max=width)
            indices = (first[:, None] + torch.arange(width, device=device, dtype=torch.int32)).clamp(max=nq - 1)
    else:
        rows = torch.arange(nq, device=device, dtype=torch.int32)
        last = ((query_start + (rows + 1) * block_q - 1) // block_k).clamp(max=nk - 1)
        if window is None:
            count = last + 1
            indices = torch.arange(nk, device=device, dtype=torch.int32).expand(nq, -1)
        else:
            first = ((query_start + rows * block_q - window + 1) // block_k).clamp(min=0)
            width = min(nk, (window + block_q + block_k - 2) // block_k + 1)
            count = (last - first + 1).clamp(min=0, max=width)
            indices = (first[:, None] + torch.arange(width, device=device, dtype=torch.int32)).clamp(max=nk - 1)
    return count[None, None], indices[None, None]


# Packed document boundaries change the number of regions between microbatches.
@torch.compile(dynamic=True)
def _region_tiles(
    ids: torch.Tensor,
    token_regions: torch.Tensor,
    regions: int,
    fwd_k: int,
    bwd_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Union selected regions for bounded chunks of 128-query tiles.

    Args:
        ids: Sorted region IDs [batch, groups, queries, slots].
        token_regions: Monotonic region IDs [batch, keys].
        regions: Number of global regions; also the unused-route sentinel.
        fwd_k: Forward key tile width.
        bwd_k: Backward key tile width.

    Returns:
        Forward and backward tile-presence grids [batch, groups, query_tiles,
        key_tiles]. Temporary storage is query-tile by region, never query by
        token. Groups remain separate.
    """
    batch, groups, queries, slots = ids.shape
    rows = (queries + 127) // 128
    # F.pad specializes its scalar fill value; tensor arithmetic keeps regions symbolic.
    padding = ids.new_zeros((batch, groups, rows * 128 - queries, slots)) + regions
    ids = torch.cat((ids, padding), dim=2)
    union = torch.zeros((batch, groups, rows, regions + 1), dtype=torch.bool, device=ids.device)
    union.scatter_(-1, ids.reshape(batch, groups, rows, 128 * slots).long(), True)
    prefix = F.pad(union.cumsum(-1, dtype=torch.int32), (1, 0))
    grids = []
    for width in (fwd_k, bwd_k):
        starts = torch.arange(0, token_regions.shape[1], width, device=ids.device)
        ends = (starts + width - 1).clamp(max=token_regions.shape[1] - 1)
        first = token_regions[:, starts, None].transpose(1, 2)[:, None].expand(-1, groups, rows, -1)
        last = (token_regions[:, ends, None].transpose(1, 2)[:, None] + 1).expand(-1, groups, rows, -1)
        grids.append(prefix.gather(-1, last) > prefix.gather(-1, first))
    return grids[0], grids[1]


@torch.compile(dynamic=False)
def _compact_selected_tiles(present: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compact a boolean block grid into FA4 partial-block counts and indices."""
    columns = torch.arange(present.shape[-1], device=present.device, dtype=torch.int32)
    counts = present.sum(-1, dtype=torch.int32)
    indices = torch.where(present, columns, present.shape[-1]).sort(-1).values
    # Provider groups become independent attention batches, so block metadata is
    # shared across that group's query heads without copying it for every head.
    counts = counts.flatten(0, 1).unsqueeze(1).contiguous()
    indices = indices.flatten(0, 1).unsqueeze(1).contiguous()
    if indices.shape[-1] == 1:
        # A contiguous singleton axis may retain its pre-sort stride. CuTe
        # requires the explicitly selected last axis to have unit stride.
        # Changing a size-one axis stride preserves every storage address.
        indices = indices.as_strided(indices.shape, (*indices.stride()[:-1], 1))
    return counts, indices


def _provider_heads(tensor: torch.Tensor, *, batch: int, heads_per_group: int) -> torch.Tensor:
    """Copy provider-specific block metadata into FA4's query-head axis.

    Args:
        tensor: Read-only int32 tensor [batch * groups, 1, query_tiles] for
            counts, or [batch * groups, 1, query_tiles, key_tiles] for indices.
            Backward metadata swaps query_tiles and key_tiles. Each provider
            owns one KV group, with a common number of query heads per group.
        batch: Original attention batch size before provider folding.
        heads_per_group: Number of consecutive query heads sharing a KV group.

    Returns:
        Independent contiguous int32 metadata [batch, query_heads, ...],
        where query_heads = groups * heads_per_group. No token mask is created.
    """
    return tensor.unflatten(0, (batch, -1)).squeeze(2).repeat_interleave(heads_per_group, dim=1)


@torch.compile(dynamic=False)
def _split_forward_full_blocks(
    counts: torch.Tensor, indices: torch.Tensor, *, prefix_length: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Separate a low contiguous full-block prefix without changing K order.

    Args:
        counts: Int32 selected K-block counts [provider_batches, heads, query_tiles].
            Provider-folded layout uses provider_batches=batch*groups and heads=1;
            native head layout uses provider_batches=batch and heads=query_heads.
        indices: Sorted unique nonnegative K-block indices
            [provider_batches, heads, query_tiles, key_tiles], with ignored trailing slots.
            Both tensors use SM90 D192 tiles: 128 queries and 112 keys.
        prefix_length: Proven all-selected initial query count in a single
            unpadded CP1 document with physical positions starting at zero.

    Returns:
        Partial-block counts, shifted partial-block indices, and full-block
        counts, shaped like counts, indices, and counts respectively. Inputs
        are never modified. The original indices alias the full-block list:
        FA4 processes partial indices descending, then full indices descending.
    """
    starts = torch.arange(counts.shape[-1], dtype=torch.int32, device=counts.device) * 128
    proposed = torch.where(starts + 128 <= prefix_length, starts // 112, 0).expand_as(counts)
    # Sorted unique nonnegative indices end at n-1 iff their first n values
    # are exactly 0..n-1. This device-only check preserves a low prefix.
    endpoint = indices.gather(-1, (proposed - 1).clamp_min(0).long().unsqueeze(-1)).squeeze(-1)
    full_counts = torch.where((proposed <= counts) & (endpoint == proposed - 1), proposed, 0)
    columns = torch.arange(indices.shape[-1], device=indices.device)
    shifted = (columns + full_counts.long().unsqueeze(-1)).clamp_max(indices.shape[-1] - 1)
    partial_indices = indices.gather(-1, shifted)
    return counts - full_counts, partial_indices, full_counts


class _BackwardBlockPartition(NamedTuple):
    """Independent int32 lists in the original provider/head layout.

    Attributes:
        partial_counts: Tensor [provider_batches, heads, key_tiles].
        partial_indices: Tensor [provider_batches, heads, key_tiles, query_tiles].
        full_counts: Tensor with the partial_counts layout.
        full_indices: Tensor with the partial_indices layout.
    """

    partial_counts: torch.Tensor
    partial_indices: torch.Tensor
    full_counts: torch.Tensor
    full_indices: torch.Tensor


@torch.compile(dynamic=False)
def _split_backward_full_blocks(
    counts: torch.Tensor, indices: torch.Tensor, *, prefix_length: int
) -> _BackwardBlockPartition:
    """Classify complete causal rectangles using SM90 D192 backward tiles.

    Args:
        counts: Int32 counts [provider_batches, heads, key_tiles].
        indices: Ascending unique nonnegative query-block IDs of shape
            [provider_batches, heads, key_tiles, query_tiles]. Entries beyond
            counts are ignored. Provider-folded tensors have heads=1; native
            layout has provider_batches=batch and heads=query_heads.
        prefix_length: Proven all-selected query prefix in one unpadded B1
            CP1 document whose positions equal physical token indices.

    Returns:
        Independent partial/full counts and index lists with the input layouts.
        Each list preserves its original ascending order and their disjoint
        union equals the input list. FA4 processes partial then full, changing
        dK/dV FP32 accumulation order when full blocks were inside that list.
        Inputs are not modified. Only existing block-grid-sized integer storage
        is used; no query-token by key-token mask is created.
    """
    columns = torch.arange(indices.shape[-1], device=indices.device)
    valid = columns < counts.unsqueeze(-1)
    key_end = (torch.arange(counts.shape[-1], device=counts.device) + 1) * 96
    # Entire Q128 tile must be inside the proven prefix. Every K96 token
    # must precede its first query; the diagonal/mixed-prefix tiles stay partial.
    first_full = (key_end - 1 + 127) // 128
    past_full = prefix_length // 128
    before = valid & (indices < first_full[:, None])
    eligible = valid & (indices >= first_full[:, None]) & (indices < past_full)
    pre_count = before.sum(-1, dtype=torch.int32)
    full_count = eligible.sum(-1, dtype=torch.int32)
    partial_source = columns + torch.where(columns >= pre_count.unsqueeze(-1), full_count.unsqueeze(-1), 0)
    full_source = columns + pre_count.unsqueeze(-1)
    partial_indices = indices.gather(-1, partial_source.clamp_max(indices.shape[-1] - 1).long())
    full_indices = indices.gather(-1, full_source.clamp_max(indices.shape[-1] - 1).long())
    return _BackwardBlockPartition(counts - full_count, partial_indices, full_count, full_indices)


def _selected_block_lists(
    selection: RegionSelection,
    regions: int,
    query_start: int,
    fwd_k: int,
    bwd_k: int,
    fwd_q: int = 128,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create route-derived forward/reverse FA4 lists with bounded scratch space.

    Args:
        selection: Sorted region IDs [batch, groups, queries, slots] and
            token-region IDs [batch, keys] with monotonic key-region order.
        regions: Number of global regions.
        query_start: Global token offset of the local query shard.
        fwd_k: Key tokens represented by each forward metadata block.
        bwd_k: Key tokens represented by each backward metadata block.
        fwd_q: Query tokens per forward metadata block, either 128 or 256.

    Returns:
        Forward counts [batch * groups, 1, ceil(queries / fwd_q)] and
        indices [batch * groups, 1, ceil(queries / fwd_q), ceil(keys / fwd_k)],
        followed by backward counts [batch * groups, 1, ceil(keys / bwd_k)]
        and indices [batch * groups, 1, ceil(keys / bwd_k), ceil(queries / 128)].
        All four tensors contain int32 values on the selection device.
    """
    ids, token_regions = selection.region_ids, selection.token_regions
    batch, groups, queries, _ = ids.shape
    rows = (queries + 127) // 128
    keys = token_regions.shape[1]
    forward = torch.empty((batch, groups, rows, (keys + fwd_k - 1) // fwd_k), dtype=torch.bool, device=ids.device)
    backward = torch.empty((batch, groups, rows, (keys + bwd_k - 1) // bwd_k), dtype=torch.bool, device=ids.device)
    # At most eight query tiles enter the region-union workspace at a time.
    for start in range(0, queries, 1024):
        stop = min(start + 1024, queries)
        fg, bg = _region_tiles(ids[:, :, start:stop], token_regions, regions, fwd_k, bwd_k)
        forward[:, :, start // 128 : (stop + 127) // 128] = fg
        backward[:, :, start // 128 : (stop + 127) // 128] = bg
    if fwd_q == 256:
        forward = F.pad(forward, (0, 0, 0, rows % 2))
        forward = forward.unflatten(-2, (-1, 2)).any(-2)
    for grid, width, query_tile in ((forward, fwd_k, fwd_q), (backward, bwd_k, 128)):
        last_query = query_start + torch.arange(1, grid.shape[-2] + 1, device=ids.device) * query_tile
        grid &= torch.arange(grid.shape[-1], device=ids.device) * width < last_query[:, None]
    fc, fi = _compact_selected_tiles(forward)
    bc, bi = _compact_selected_tiles(backward.transpose(-1, -2))
    return fc, fi, bc, bi


def _all_padding_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, padding_key: int
) -> torch.Tensor:
    """Copy singleton-key attention while retaining explicit zero Q/K gradients.

    Args:
        query: Tensor [batch, local_queries, query_heads, head_dim].
        key: Tensor [batch, global_keys, kv_heads, head_dim].
        value: Tensor shaped like key, potentially a noncontiguous KV-unbind view.
        padding_key: The only allowed global key, proven by shared metadata.
            query_heads must be divisible by kv_heads.

    Returns:
        Materialized tensor [batch, local_queries, query_heads, head_dim].
        Each grouped query head copies the corresponding value head exactly.
        Empty Q/K slices preserve backward hooks and CP collective dependencies
        without reading their contents or changing signed zeros in V.
    """
    batch, queries, heads, dim = query.shape
    groups = value.shape[2]
    heads_per_group = heads // groups
    padding_value = value[:, padding_key : padding_key + 1, :, None]
    output = padding_value.expand(batch, queries, groups, heads_per_group, dim)
    empty_query = query[:, :0].unflatten(2, (groups, heads_per_group))
    empty_key = key[:, :0, :, None].expand(batch, 0, groups, heads_per_group, dim)
    return torch.cat((output, empty_query, empty_key), dim=1).flatten(2, 3)


def _causal_document_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, layout: CausalDocumentLayout
) -> torch.Tensor:
    """Run the existing B1 causal prefix or proven singleton-padding operation.

    Args:
        query: BF16 tensor [1, local_queries, query_heads, head_dim].
        key: BF16 tensor [1, global_keys, kv_heads, head_dim], possibly strided.
        value: Tensor with the same shape/dtype as key; global storage is shared.
        layout: Proven prefix metadata for this one batch row. A zero query
            count denotes a fully padded row with a valid singleton padding key.

    Returns:
        Tensor [1, local_queries, query_heads, head_dim], preserving B1 arithmetic.
    """
    queries, dim = query.shape[1], query.shape[-1]
    if layout.valid_queries == 0:
        if layout.padding_key is None:
            raise ValueError("A fully padded Step5 causal layout requires its singleton key")
        return _all_padding_attention(query, key, value, layout.padding_key)
    # Each query segment is the suffix of its document's K prefix. Native
    # bottom-right causality therefore matches document-local positions.
    # In particular, the final K segment ends at the final local query,
    # even if that document continues on the next CP rank.
    value_prefix = value[0, layout.key_start : layout.key_stop]
    output, _ = _load_fa4_varlen()(
        query[0, : layout.valid_queries],
        key[0, layout.key_start : layout.key_stop],
        value_prefix,
        cu_seqlens_q=layout.cu_seqlens_q,
        cu_seqlens_k=layout.cu_seqlens_k,
        # These are scheduling upper bounds, not document boundaries.
        # Keep short-document packs on the shard's ordinary kernel variant;
        # cu_seqlens still defines every valid query/key and causal offset.
        max_seqlen_q=queries,
        max_seqlen_k=max(layout.max_seqlen_k, queries),
        softmax_scale=dim**-0.5,
        causal=True,
        pack_gqa=False,
    )
    if layout.padding_key is None:
        return output.unsqueeze(0)
    # A padding query has position zero but retains the final document ID.
    # The old mask therefore selects that document's first V, not zeros.
    # Keep GQA expansion as a view and concatenate into one output buffer.
    # The valid Q/K slices retain zero gradients for unused padding entries.
    groups = value.shape[2]
    heads_per_group = query.shape[2] // groups
    # Share the compact V view so both branches accumulate before the
    # single global slice backward allocates a full-length V gradient.
    padding_offset = layout.padding_key - layout.key_start
    padding_value = value_prefix[padding_offset : padding_offset + 1][None, :, :, None]
    padding_output = padding_value.expand(1, queries - layout.valid_queries, groups, heads_per_group, dim)
    return torch.cat(
        (output.reshape(1, layout.valid_queries, groups, heads_per_group, dim), padding_output), dim=1
    ).flatten(2, 3)


def fa4_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    key_positions: torch.Tensor,
    key_padding: torch.Tensor,
    documents: torch.Tensor,
    *,
    query_start: int,
    window: int | None,
    selection: RegionSelection | None,
    window_layout: CausalDocumentLayout | None = None,
) -> torch.Tensor:
    """Run FA4 forward/backward with compact CSA metadata on SM90 or SM100.

    Args:
        query: BF16 CUDA tensor [batch, local_queries, query_heads, head_dim].
        key: BF16 CUDA tensor [batch, global_keys, kv_heads, head_dim].
        value: Same shape/dtype/device as key.
        positions: Int tensor [batch, local_queries], document-local positions.
        key_positions: Int tensor [batch, global_keys], document-local positions.
        key_padding: Boolean tensor [batch, global_keys], true for padding.
        documents: Int tensor [batch, global_keys], document IDs.
        query_start: Physical global offset of the contiguous query shard.
        window: Sliding window size, or None for CSA.
        selection: Compact sorted region metadata, or None for sliding attention.
        window_layout: Opt-in valid SWA document segments in global token
            coordinates. The existing halo may contain local alignment padding.

    Returns:
        BF16 tensor [batch, local_queries, query_heads, head_dim].
        FA4 owns first-order autograd. No dense token membership tensor is built.
        SM100 keeps Q/K at 192 dimensions and calls the native (192, 128)
        kernel twice for the two V-channel slices; only the 64-channel V tail
        is padded to 128. This repeats QK work but avoids 256-dimensional Q/K.
    """
    if not query.is_cuda or any(x.dtype != torch.bfloat16 for x in (query, key, value)):
        raise ValueError("Step5 FA4 requires CUDA BF16 Q/K/V")
    capability = torch.cuda.get_device_capability(query.device)
    if capability not in ((9, 0), (10, 0)):
        raise RuntimeError("Step5 FA4 supports SM90 and SM100")
    split_values = capability == (10, 0)
    if split_values and query.shape[-1] != 192:
        raise ValueError("Step5 SM100 FA4 requires head_dim=192")
    if query.shape[-1] not in (192, 256):
        raise ValueError("Step5 FA4 supports head_dim=192 or 256")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Step5 FA4 uses nondeterministic attention backward")
    queries, keys, dim = query.shape[1], key.shape[1], query.shape[-1]
    if (
        capability == (9, 0)
        and dim == 192
        and query.shape[0] == 1
        and window is None
        and selection is not None
        and selection.all_regions_selected
        and selection.all_padding_key is not None
    ):
        return _all_padding_attention(query, key, value, selection.all_padding_key)
    if (
        capability == (9, 0)
        and dim == 192
        and query.shape[0] == 1
        and window is not None
        and selection is None
        and window_layout is not None
    ):
        # Translate global layout coordinates into the existing, locally padded
        # halo. Transport is unchanged; only irrelevant document/alignment keys
        # are omitted from this native window-attention call.
        offset = window_layout.key_stop - window_layout.valid_queries - query_start
        first_key = window_layout.key_start - offset
        last_key = window_layout.key_stop - offset
        if not 0 <= first_key < last_key <= keys:
            raise ValueError("Step5 native SWA layout extends outside the received halo")
        output, _ = _load_fa4_varlen()(
            query[0, : window_layout.valid_queries],
            key[0, first_key:last_key],
            value[0, first_key:last_key],
            cu_seqlens_q=window_layout.cu_seqlens_q,
            cu_seqlens_k=window_layout.cu_seqlens_k,
            # Scheduling bounds stay on the ordinary shard variant even for
            # short-document packs; cu_seqlens retains exact window boundaries.
            max_seqlen_q=queries,
            max_seqlen_k=max(window_layout.max_seqlen_k, queries),
            softmax_scale=dim**-0.5,
            causal=True,
            window_size=(window - 1, 0),
            pack_gqa=False,
        )
        if window_layout.padding_key is None:
            return output.unsqueeze(0)
        # Keep padded-query semantics from the original custom-mask kernel.
        # Its physical block lists can prune the document's first V even
        # though padding positions reset to zero, so a singleton-V shortcut
        # is not equivalent. Replay only the final partial Q tile and padding,
        # with the old halo origin; outputs for repeated valid Q are discarded.
        global_start = window_layout.key_stop - window_layout.valid_queries
        halo_first = max(0, global_start - window + 1)
        original_padding = halo_first % 112
        original_query_start = global_start - halo_first + original_padding
        trim = query_start - original_query_start
        tail_start = (window_layout.valid_queries // 128) * 128
        padding_output = fa4_attention(
            query[:, tail_start:],
            key[:, trim:],
            value[:, trim:],
            positions[:, tail_start:],
            key_positions[:, trim:],
            key_padding[:, trim:],
            documents[:, trim:],
            query_start=original_query_start + tail_start,
            window=window,
            selection=None,
        )
        return torch.cat((output.unsqueeze(0), padding_output[:, window_layout.valid_queries - tail_start :]), dim=1)
    if (
        capability == (9, 0)
        and dim == 192
        and query.shape[0] == 1
        and window is None
        and selection is not None
        and selection.all_regions_selected
        and selection.causal_layout is not None
    ):
        return _causal_document_attention(query, key, value, selection.causal_layout)
    if (
        capability == (9, 0)
        and dim == 192
        and query.shape[0] == 2
        and window is None
        and selection is not None
        and selection.all_regions_selected
        and selection.causal_batch_layouts is not None
    ):
        # Unbind once: repeated key[b] would create a full-B select_backward
        # allocation for every row. Prefix slices share the existing CP buffers.
        # CP's existing gradient-layout reorder remains unchanged.
        outputs = [
            _causal_document_attention(q[None], k[None], v[None], layout).squeeze(0)
            for q, k, v, layout in zip(
                query.unbind(0), key.unbind(0), value.unbind(0), selection.causal_batch_layouts, strict=True
            )
        ]
        return torch.stack(outputs, dim=0)
    bitmap = False
    slots = 0
    regions = 0 if selection is None else selection.region_count
    if selection is None or selection.all_regions_selected:
        # Fully selected prefixes are exactly document-causal attention. Keep
        # their selected block lists, but skip bitmap creation and membership.
        membership = torch.empty((1, 1, 1, 2), dtype=torch.int32, device=query.device)
        token_regions = torch.empty((1, 2), dtype=torch.int64, device=query.device)
    else:
        membership, token_regions = selection.region_ids, selection.token_regions
        slots = membership.shape[-1]
        bitmap = (regions + 31) // 32 <= slots
        if bitmap:
            membership = pack_region_bitmap(membership, regions)
    groups = 1 if selection is None else selection.region_ids.shape[1]
    heads_per_group = query.shape[2] // groups
    batch = query.shape[0]
    # Requires FA4 #2888 (3c2c6b3): native 192-dimensional GQA backward.
    fwd_k, bwd_k = (128, 256) if split_values else ((112, 96) if dim == 192 else (80, 64))
    fwd_q = 256 if split_values and queries > 128 else 128
    head_layout = False
    backward_full_blocks = (
        capability == (9, 0)
        and dim == 192
        and batch == 1
        and window is None
        and selection is not None
        and selection.backward_full_block_prefix >= 256
        and query_start == 0
        and queries == keys
    )
    if selection is None:
        fc, fi = _block_lists(
            queries, keys, query_start, window, block_q=fwd_q, block_k=fwd_k, reverse=False, device=query.device
        )
        bc, bi = _block_lists(
            queries, keys, query_start, window, block_q=128, block_k=bwd_k, reverse=True, device=query.device
        )
    else:
        fc, fi, bc, bi = _selected_block_lists(selection, regions, query_start, fwd_k, bwd_k, fwd_q)
        # Native head layout avoids copying the full attention output back
        # from provider-major order. Bound the complete expanded sparse-list
        # storage (including full-block counts and a second forward index
        # table, plus a second backward table when enabled) by the eliminated
        # BF16 output copy. This is a metadata
        # payload bound, not a bound on total CUDA peak memory.
        metadata_bytes = (
            heads_per_group
            * (2 * fc.numel() + 2 * fi.numel() + 2 * bc.numel() + (2 if backward_full_blocks else 1) * bi.numel())
            * fc.element_size()
        )
        head_layout = (
            capability == (9, 0) and dim == 192 and groups > 1 and metadata_bytes < query.numel() * query.element_size()
        )
        if head_layout:
            fc, fi, bc, bi = (
                _provider_heads(tensor, batch=batch, heads_per_group=heads_per_group) for tensor in (fc, fi, bc, bi)
            )
        else:
            # Provider folding keeps metadata compact for large key histories.
            # For batch=1 these input transforms are strided views.
            query = query.unflatten(2, (groups, heads_per_group)).permute(0, 2, 1, 3, 4).flatten(0, 1)
            key = key.unflatten(2, (groups, key.shape[2] // groups)).permute(0, 2, 1, 3, 4).flatten(0, 1)
            value = value.unflatten(2, (groups, value.shape[2] // groups)).permute(0, 2, 1, 3, 4).flatten(0, 1)
    canonical_region_size = (
        selection.canonical_region_size
        if (
            capability == (9, 0)
            and dim == 192
            and batch == 1
            and window is None
            and selection is not None
            and query_start == 0
            and queries == keys
        )
        else 0
    )
    if head_layout:
        attention, block_lists, mask_mod = _load_fa4(
            window,
            bitmap,
            slots,
            groups,
            heads_per_group=heads_per_group,
            canonical_region_size=canonical_region_size,
        )
    else:
        attention, block_lists, mask_mod = _load_fa4(
            window,
            bitmap,
            slots,
            groups,
            canonical_region_size=canonical_region_size,
        )
    # This FA4 revision requires full-block tensors even for the default path.
    # Zero counts leave its aliased indices unused.
    full_indices = fi
    if (
        capability == (9, 0)
        and dim == 192
        and batch == 1
        and window is None
        and selection is not None
        and selection.forward_full_block_prefix >= 256
        and query_start == 0
        and queries == keys
    ):
        fc, fi, full_counts = _split_forward_full_blocks(fc, fi, prefix_length=selection.forward_full_block_prefix)
    else:
        full_counts = torch.zeros_like(fc)
    forward = block_lists(fc, fi, full_counts, full_indices, block_size=(fwd_q, fwd_k))
    if backward_full_blocks:
        partition = _split_backward_full_blocks(bc, bi, prefix_length=selection.backward_full_block_prefix)
        backward = block_lists(*partition, block_size=(128, bwd_k))
    else:
        backward = block_lists(bc, bi, torch.zeros_like(bc), bi, block_size=(128, bwd_k))
    # SM100 provides native (192, 128) kernels. V-channel splitting preserves
    # the 192-dimensional Q/K scale and avoids padding Q/K to 256. Autograd
    # sums both contributions to dQ/dK. The second call repeats QK work.
    values = value.split(128, dim=-1) if split_values else (value,)
    aux_tensors = [
        positions,
        key_positions,
        documents[:, query_start : query_start + queries],
        documents,
        key_padding,
        membership,
        token_regions,
    ]
    for index, tensor in enumerate(aux_tensors):
        if tensor.numel() == 1 and tensor.stride(-1) == 1:
            # FA4 cannot infer a unique leading dimension for [1, 1] metadata.
            # A view carries its supported layout annotation without changing
            # the caller's tensor or allocating/padding any token storage.
            annotated = tensor.view_as(tensor)
            annotated.__leading_dim__ = tensor.ndim - 1
            aux_tensors[index] = annotated
    outputs = []
    for part in values:
        value_width = part.shape[-1]
        if split_values and value_width < 128:
            part = F.pad(part, (0, 128 - value_width))
        output, _ = attention(
            query,
            key,
            part,
            softmax_scale=dim**-0.5,
            pack_gqa=False,
            mask_mod=mask_mod,
            aux_tensors=aux_tensors,
            block_sparse_tensors=forward,
            block_sparse_tensors_bwd=backward,
        )
        outputs.append(output[..., :value_width])
    output = torch.cat(outputs, dim=-1) if split_values else outputs[0]
    if selection is not None and not head_layout:
        output = output.unflatten(0, (batch, groups)).permute(0, 2, 1, 3, 4).flatten(2, 3)
    return output

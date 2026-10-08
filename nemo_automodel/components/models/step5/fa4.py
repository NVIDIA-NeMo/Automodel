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

import torch
from torch.nn import functional as F

from nemo_automodel.components.models.step5.kernels import pack_region_bitmap
from nemo_automodel.components.models.step5.sparse import RegionSelection
from nemo_automodel.shared.import_utils import safe_import


@functools.cache
def _load_fa4(window: int | None, bitmap: bool, slots: int, provider_groups: int) -> tuple[Callable, type, Callable]:
    """Load optional FA4 dependencies and cache a shape-independent CuTe mask."""
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
        b = batch[0] // provider_groups
        q = cutlass.min(query_idx[0], seqlen_info.seqlen_q - 1)
        k = cutlass.min(key_idx[0], seqlen_info.seqlen_k - 1)
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
            target = token_regions[b, k]
            group = batch[0] % provider_groups
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


@torch.compile(dynamic=False)
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
    ids = F.pad(ids, (0, 0, 0, rows * 128 - queries), value=regions)
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
    return counts.flatten(0, 1).unsqueeze(1).contiguous(), indices.flatten(0, 1).unsqueeze(1).contiguous()


def _selected_block_lists(
    selection: RegionSelection,
    regions: int,
    query_start: int,
    fwd_k: int,
    bwd_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create route-derived forward/reverse FA4 lists with bounded scratch space."""
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
    last_query = query_start + torch.arange(1, rows + 1, device=ids.device) * 128
    for grid, width in ((forward, fwd_k), (backward, bwd_k)):
        grid &= torch.arange(grid.shape[-1], device=ids.device) * width < last_query[:, None]
    fc, fi = _compact_selected_tiles(forward)
    bc, bi = _compact_selected_tiles(backward.transpose(-1, -2))
    return fc, fi, bc, bi


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
) -> torch.Tensor:
    """Run FA4 forward/backward with compact CSA metadata on SM90.

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

    Returns:
        BF16 tensor [batch, local_queries, query_heads, head_dim].
        FA4 owns first-order autograd. No dense token membership tensor is built.
    """
    if not query.is_cuda or any(x.dtype != torch.bfloat16 for x in (query, key, value)):
        raise ValueError("Step5 FA4 requires CUDA BF16 Q/K/V")
    if torch.cuda.get_device_capability(query.device) != (9, 0):
        raise RuntimeError("Step5 FA4 block geometry is currently validated on SM90")
    if query.shape[-1] not in (192, 256):
        raise ValueError("Step5 FA4 supports head_dim=192 or 256")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Step5 FA4 uses nondeterministic attention backward")
    queries, keys, dim = query.shape[1], key.shape[1], query.shape[-1]
    bitmap = False
    slots = 0
    if selection is None:
        membership = torch.empty((1, 1, 1, 2), dtype=torch.int32, device=query.device)
        token_regions = torch.empty((1, 2), dtype=torch.int64, device=query.device)
    else:
        membership, token_regions = selection.region_ids, selection.token_regions
        regions = int(token_regions.max().item()) + 1
        slots = membership.shape[-1]
        bitmap = (regions + 31) // 32 <= slots
        if bitmap:
            membership = pack_region_bitmap(membership, regions)
    groups = 1 if selection is None else selection.region_ids.shape[1]
    heads_per_group = query.shape[2] // groups
    batch = query.shape[0]
    attention, block_lists, mask_mod = _load_fa4(window, bitmap, slots, groups)
    # Requires FA4 #2888 (3c2c6b3): native 192-dimensional GQA backward.
    fwd_k, bwd_k = (112, 96) if dim == 192 else (80, 64)
    if selection is None:
        fc, fi = _block_lists(
            queries, keys, query_start, window, block_q=128, block_k=fwd_k, reverse=False, device=query.device
        )
        bc, bi = _block_lists(
            queries, keys, query_start, window, block_q=128, block_k=bwd_k, reverse=True, device=query.device
        )
    else:
        fc, fi, bc, bi = _selected_block_lists(selection, regions, query_start, fwd_k, bwd_k)
        # Grouping providers into the batch avoids a 16x metadata replication.
        # For the usual batch=1 these input transforms are strided views.
        query = query.unflatten(2, (groups, heads_per_group)).permute(0, 2, 1, 3, 4).flatten(0, 1)
        key = key.unflatten(2, (groups, key.shape[2] // groups)).permute(0, 2, 1, 3, 4).flatten(0, 1)
        value = value.unflatten(2, (groups, value.shape[2] // groups)).permute(0, 2, 1, 3, 4).flatten(0, 1)
    # This FA4 revision requires tensor full-block lists even when every tile
    # uses the custom predicate. Zero counts make the aliased indices unused.
    forward = block_lists(fc, fi, torch.zeros_like(fc), fi, block_size=(128, fwd_k))
    backward = block_lists(bc, bi, torch.zeros_like(bc), bi, block_size=(128, bwd_k))
    output, _ = attention(
        query,
        key,
        value,
        softmax_scale=dim**-0.5,
        pack_gqa=False,
        mask_mod=mask_mod,
        aux_tensors=[
            positions,
            key_positions,
            documents[:, query_start : query_start + queries],
            documents,
            key_padding,
            membership,
            token_regions,
        ],
        block_sparse_tensors=forward,
        block_sparse_tensors_bwd=backward,
    )
    if selection is not None:
        output = output.unflatten(0, (batch, groups)).permute(0, 2, 1, 3, 4).flatten(2, 3)
    return output

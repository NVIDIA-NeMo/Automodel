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

"""Pack sparse attention route sets without a dense token or block mask."""

from __future__ import annotations

import torch

from nemo_automodel.shared.import_utils import null_decorator, safe_import

_HAVE_TRITON, triton = safe_import("triton")
_HAVE_LANGUAGE, tl = safe_import("triton.language")
HAVE_TRITON = _HAVE_TRITON and _HAVE_LANGUAGE


@(triton.jit if HAVE_TRITON else null_decorator)
def _scatter_bits(
    routes,
    membership,
    queries: tl.constexpr,
    route_count: tl.constexpr,
    kv_length: tl.constexpr,
    batch_stride: tl.constexpr,
    query_stride: tl.constexpr,
    route_stride: tl.constexpr,
    row_words: tl.constexpr,
    tile: tl.constexpr,
):
    """Set bits for a tile of one [batch, query] row.

    Reads signed int32/int64 routes [batch, queries, route_count] with explicit
    element strides. Atomically ORs into zero-initialized int32 membership
    [batch, queries, row_words], which cannot alias routes. No return value.
    """
    row = tl.program_id(0)
    ri = tl.program_id(1) * tile + tl.arange(0, tile)
    offsets = (
        (row // queries).to(tl.int64) * batch_stride
        + (row % queries).to(tl.int64) * query_stride
        + ri.to(tl.int64) * route_stride
    )
    token = tl.load(routes + offsets, ri < route_count, other=-1)
    valid = (ri < route_count) & (token >= 0) & (token < kv_length)
    # Check int64 IDs before narrowing, including invalid IDs above 2**32.
    token = tl.where(valid, token, 0).to(tl.int32)
    bits = tl.full((tile,), 1, tl.uint32) << (token % 32)
    # OR is idempotent: duplicates cannot carry into a neighbouring key's bit.
    tl.atomic_or(membership + row.to(tl.int64) * row_words + token // 32, bits, valid, sem="relaxed")


@torch.library.triton_op("nemo_automodel::route_bitset", mutates_args={})
def _build_cuda_bitset(routes: torch.Tensor, num_keys: int, row_alignment: int) -> torch.Tensor:
    """Pack strided signed CUDA routes [batch, queries, routes_per_query].

    Returns freshly allocated uint8 [batch, queries, ceil(num_keys/8)] with
    row_alignment-byte padded strides; see build_route_bitset for bit ordering.
    Inputs are read-only; zeroing and atomic OR run on the current CUDA stream.
    """
    if not HAVE_TRITON:
        raise ImportError("CUDA route bitset preprocessing requires Triton.")
    batch, queries, count = routes.shape
    width = ((num_keys + 7) // 8 + row_alignment - 1) // row_alignment * row_alignment
    row_words = width // 4
    storage = torch.zeros((batch, queries, row_words), device=routes.device, dtype=torch.int32)
    if batch * queries * count:
        tile = 256
        torch.library.wrap_triton(_scatter_bits)[(batch * queries, triton.cdiv(count, tile))](
            routes, storage, queries, count, num_keys, *routes.stride(), row_words, tile, num_warps=4
        )
    return storage.view(torch.uint8)[..., : (num_keys + 7) // 8]


def build_route_bitset(routes: torch.Tensor, num_keys: int, *, row_alignment: int = 4) -> torch.Tensor:
    """Pack route sets into bits, without materializing a dense membership table.

    Duplicate IDs select a key once; negative and out-of-range IDs are ignored.
    IDs may denote tokens or key blocks: this helper imposes no visibility rules.
    CUDA uses atomic OR on the current stream; CPU uses a sorted reference.

    Args:
        routes: Signed int32/int64 tensor [batch, queries, routes_per_query].
            Arbitrary strides, including expanded zero strides, are supported.
        num_keys: Positive number of addressable keys, smaller than 2**31.
        row_alignment: Row stride alignment in bytes, a positive multiple of
            four for atomic OR. FA4 auxiliary tensors require 32.

    Returns:
        Uint8 tensor [batch, queries, ceil(num_keys/8)]. Bit k%8 of byte k//8
        selects key k. Unused bits are zero. Rows have the requested padded
        stride. The output owns fresh storage and does not alias the input.
    """
    if routes.ndim != 3 or routes.dtype not in (torch.int32, torch.int64):
        raise ValueError("routes must be a signed int32/int64 [batch, queries, routes] tensor")
    if not 0 < num_keys < 2**31:
        raise ValueError("num_keys must be positive and smaller than 2**31")
    if row_alignment <= 0 or row_alignment % 4:
        raise ValueError("row_alignment must be a positive multiple of four")
    if routes.is_cuda:
        with torch.cuda.device(routes.device):
            return _build_cuda_bitset(routes, num_keys, row_alignment)
    batch, queries, count = routes.shape
    width = ((num_keys + 7) // 8 + row_alignment - 1) // row_alignment * row_alignment
    storage = torch.zeros((batch, queries, width), device=routes.device, dtype=torch.int32)
    if count:
        valid = (routes >= 0) & (routes < num_keys)
        indices = torch.where(valid, routes, num_keys).sort(-1).values
        first = torch.nn.functional.pad(indices[..., 1:] != indices[..., :-1], (1, 0), value=True)
        bits = torch.where(first & (indices < num_keys), 1 << (indices % 8), 0).to(torch.int32)
        storage.scatter_add_(-1, (indices.clamp(max=num_keys - 1) // 8).long(), bits)
    return storage.to(torch.uint8)[..., : (num_keys + 7) // 8]

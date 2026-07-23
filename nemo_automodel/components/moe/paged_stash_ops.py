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

"""Triton kernels for graph-safe, device-counted MoE activation stashing."""

from typing import Any

from nemo_automodel.shared.import_utils import null_decorator, safe_import

_has_triton, triton = safe_import("triton")
_has_tl, tl = safe_import("triton.language")
HAVE_TRITON = _has_triton and _has_tl

# Keep this optional module importable on CPU-only installations. Active paged
# stash checks HAVE_TRITON before it can launch either undecorated function.
_triton_jit = triton.jit if HAVE_TRITON else null_decorator
_triton_constexpr = tl.constexpr if HAVE_TRITON else Any

GLOBAL_BLOCK_SIZE = 1024


@_triton_jit
def paged_stash_copy_kernel(
    src_ptr,
    dst_ptr,
    num_tokens_ptr,
    live_token_mask_ptr,
    live_token_offsets_ptr,
    free_list_ptr,
    free_list_head_ptr,
    free_list_tail_ptr,
    free_list_capacity_ptr,
    page_record_ptr,
    overflow_ptr,
    new_free_list_head_ptr,
    PAGE_SIZE: _triton_constexpr,
    HIDDEN_SIZE: _triton_constexpr,
    MAX_NUM_TOKENS: _triton_constexpr,
    BLOCK_SIZE: _triton_constexpr,
):
    """Copy a runtime number of live rows into pages without a host sync.

    Args:
        src_ptr: Contiguous source with shape ``[MAX_NUM_TOKENS, HIDDEN_SIZE]``.
        dst_ptr: Page storage with shape ``[capacity * PAGE_SIZE, HIDDEN_SIZE]``; live rows are written in packed
            order.
        num_tokens_ptr: Integer device scalar with shape ``[1]``.
        live_token_mask_ptr: Boolean live-row mask with shape ``[MAX_NUM_TOKENS]``.
        live_token_offsets_ptr: Packed row offsets with shape ``[MAX_NUM_TOKENS]``; only live-row values are read.
        free_list_ptr: Circular array of physical page ids with shape ``[capacity]``.
        free_list_head_ptr: Integer device scalar with shape ``[1]``; advanced by the allocated page count.
        free_list_tail_ptr: Integer device scalar with shape ``[1]`` containing the current release position.
        free_list_capacity_ptr: Integer device scalar with shape ``[1]``.
        page_record_ptr: Output page ids with shape ``[ceil(MAX_NUM_TOKENS / PAGE_SIZE)]``.
        overflow_ptr: Shared integer device scalar with shape ``[1]``; set on invalid counts or insufficient pages.
        new_free_list_head_ptr: Integer output scalar with shape ``[1]`` copied into the persistent head by caller.
        PAGE_SIZE: Compile-time number of rows per page.
        HIDDEN_SIZE: Compile-time flattened elements per row.
        MAX_NUM_TOKENS: Compile-time physical source-row count.
        BLOCK_SIZE: Compile-time number of elements copied per inner block.
    """
    program_id = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    num_tokens = tl.load(num_tokens_ptr)
    head = tl.load(free_list_head_ptr)
    tail = tl.load(free_list_tail_ptr)
    capacity = tl.load(free_list_capacity_ptr)
    required_pages = tl.cdiv(num_tokens, PAGE_SIZE)
    overflow = tl.load(overflow_ptr)

    invalid_count = (num_tokens < 0) | (num_tokens > MAX_NUM_TOKENS)
    insufficient_pages = tail - head < required_pages
    if (overflow != 0) | invalid_count | insufficient_pages:
        if program_id == 0:
            if invalid_count | insufficient_pages:
                tl.store(overflow_ptr, 1)
            tl.store(new_free_list_head_ptr, head)
        return

    physical_token_index = program_id
    while physical_token_index < MAX_NUM_TOKENS:
        live = tl.load(live_token_mask_ptr + physical_token_index)
        live_token_index = tl.load(live_token_offsets_ptr + physical_token_index)
        page_slot = live_token_index // PAGE_SIZE
        token_in_page = live_token_index % PAGE_SIZE
        free_list_index = (head + page_slot) % capacity
        page_id = tl.load(free_list_ptr + free_list_index, mask=live, other=0)
        if live & (token_in_page == 0):
            tl.store(page_record_ptr + page_slot, page_id)

        destination_token_index = page_id * PAGE_SIZE + token_in_page
        source_base = src_ptr + physical_token_index.to(tl.int64) * HIDDEN_SIZE
        destination_base = dst_ptr + destination_token_index.to(tl.int64) * HIDDEN_SIZE
        for block_index in range(tl.cdiv(HIDDEN_SIZE, BLOCK_SIZE)):
            offsets = tl.arange(0, BLOCK_SIZE) + block_index * BLOCK_SIZE
            mask = offsets < HIDDEN_SIZE
            values = tl.load(source_base + offsets, mask=live & mask, other=0)
            tl.store(destination_base + offsets, values, mask=live & mask)
        physical_token_index += num_programs

    if program_id == 0:
        tl.store(new_free_list_head_ptr, head + required_pages)


@_triton_jit
def paged_stash_pop_kernel(
    src_ptr,
    dst_ptr,
    num_tokens_ptr,
    live_token_mask_ptr,
    live_token_offsets_ptr,
    page_record_ptr,
    overflow_ptr,
    free_list_ptr,
    free_list_tail_ptr,
    free_list_capacity_ptr,
    new_free_list_tail_ptr,
    PAGE_SIZE: _triton_constexpr,
    HIDDEN_SIZE: _triton_constexpr,
    MAX_NUM_TOKENS: _triton_constexpr,
    BLOCK_SIZE: _triton_constexpr,
):
    """Restore runtime-counted rows and recycle their pages on the GPU.

    Args:
        src_ptr: Page storage with shape ``[capacity * PAGE_SIZE, HIDDEN_SIZE]``.
        dst_ptr: Zero-initialized destination with shape ``[MAX_NUM_TOKENS, HIDDEN_SIZE]``; live rows are restored
            to their original physical positions.
        num_tokens_ptr: Integer device scalar with shape ``[1]``.
        live_token_mask_ptr: Boolean live-row mask with shape ``[MAX_NUM_TOKENS]``.
        live_token_offsets_ptr: Packed row offsets with shape ``[MAX_NUM_TOKENS]``; only live-row values are read.
        page_record_ptr: Physical page ids with shape ``[ceil(MAX_NUM_TOKENS / PAGE_SIZE)]``.
        overflow_ptr: Shared integer device scalar with shape ``[1]``; set on invalid counts.
        free_list_ptr: Circular array of physical page ids with shape ``[capacity]``; released ids are appended.
        free_list_tail_ptr: Integer device scalar with shape ``[1]`` containing the current release position.
        free_list_capacity_ptr: Integer device scalar with shape ``[1]``.
        new_free_list_tail_ptr: Integer output scalar with shape ``[1]`` copied into the persistent tail by caller.
        PAGE_SIZE: Compile-time number of rows per page.
        HIDDEN_SIZE: Compile-time flattened elements per row.
        MAX_NUM_TOKENS: Compile-time physical destination-row count.
        BLOCK_SIZE: Compile-time number of elements copied per inner block.
    """
    program_id = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    num_tokens = tl.load(num_tokens_ptr)
    tail = tl.load(free_list_tail_ptr)
    capacity = tl.load(free_list_capacity_ptr)
    required_pages = tl.cdiv(num_tokens, PAGE_SIZE)
    overflow = tl.load(overflow_ptr)

    invalid_count = (num_tokens < 0) | (num_tokens > MAX_NUM_TOKENS)
    if (overflow != 0) | invalid_count:
        if program_id == 0:
            if invalid_count:
                tl.store(overflow_ptr, 1)
            tl.store(new_free_list_tail_ptr, tail)
        return

    physical_token_index = program_id
    while physical_token_index < MAX_NUM_TOKENS:
        live = tl.load(live_token_mask_ptr + physical_token_index)
        live_token_index = tl.load(live_token_offsets_ptr + physical_token_index)
        page_slot = live_token_index // PAGE_SIZE
        token_in_page = live_token_index % PAGE_SIZE
        page_id = tl.load(page_record_ptr + page_slot, mask=live, other=0)
        source_token_index = page_id * PAGE_SIZE + token_in_page

        source_base = src_ptr + source_token_index.to(tl.int64) * HIDDEN_SIZE
        destination_base = dst_ptr + physical_token_index.to(tl.int64) * HIDDEN_SIZE
        for block_index in range(tl.cdiv(HIDDEN_SIZE, BLOCK_SIZE)):
            offsets = tl.arange(0, BLOCK_SIZE) + block_index * BLOCK_SIZE
            mask = offsets < HIDDEN_SIZE
            values = tl.load(source_base + offsets, mask=live & mask, other=0)
            tl.store(destination_base + offsets, values, mask=live & mask)

        if live & (token_in_page == 0):
            free_list_index = (tail + page_slot) % capacity
            tl.store(free_list_ptr + free_list_index, page_id)
        physical_token_index += num_programs

    if program_id == 0:
        tl.store(new_free_list_tail_ptr, tail + required_pages)


__all__ = ["GLOBAL_BLOCK_SIZE", "HAVE_TRITON", "paged_stash_copy_kernel", "paged_stash_pop_kernel"]

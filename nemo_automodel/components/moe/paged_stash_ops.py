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

from unittest.mock import MagicMock

from nemo_automodel.shared.import_utils import null_decorator, safe_import

_missing_triton = MagicMock()
_missing_triton.jit = null_decorator
HAVE_TRITON, triton = safe_import("triton", alt=_missing_triton)
_, tl = safe_import("triton.language", alt=MagicMock())

GLOBAL_BLOCK_SIZE = 1024


@triton.jit
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
    PAGE_SIZE: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    MAX_NUM_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy a runtime number of rows into pages without a device-to-host sync."""
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


@triton.jit
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
    PAGE_SIZE: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    MAX_NUM_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Restore runtime-counted rows and recycle their pages on the GPU."""
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

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

"""Packed token permutations, checked against independent labeled-row oracles."""

import itertools
from collections.abc import Callable

import pytest
import torch

from nemo_automodel.components.distributed.context_parallel.mamba import (
    _deinterleave_packed_seqs,
    _redo_attention_load_balancing,
    _reinterleave_packed_seqs,
    _reorder_chunks,
    _undo_attention_load_balancing,
)


def _boundaries(lengths: list[int]) -> torch.Tensor:
    return torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32)


def _rank_and_sequence_rows(
    cp_size: int, local_lengths: list[int]
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    rank_major = [
        (rank, document, token)
        for rank in range(cp_size)
        for document, length in enumerate(local_lengths)
        for token in range(length)
    ]
    sequence_major = [
        (rank, document, token)
        for document, length in enumerate(local_lengths)
        for rank in range(cp_size)
        for token in range(length)
    ]
    return rank_major, sequence_major


def _assert_permutation_and_gradient(
    operation: Callable[[torch.Tensor], torch.Tensor],
    source_rows: list[tuple[int, int, int]],
    target_rows: list[tuple[int, int, int]],
    dtype: torch.dtype,
    noncontiguous: bool,
) -> None:
    lookup = {row: index for index, row in enumerate(source_rows)}
    permutation = [lookup[row] for row in target_rows]
    assert sorted(permutation) == list(range(len(source_rows)))
    torch.manual_seed(92)
    storage = torch.randn(len(source_rows), 10, dtype=dtype)
    value = storage[:, ::2] if noncontiguous else storage[:, :5].contiguous()
    value.requires_grad_()
    before = value.detach().clone()
    upstream = torch.randn_like(value)
    expected = before[permutation]
    expected_grad = torch.empty_like(value)
    for target, source in enumerate(permutation):
        expected_grad[source] = upstream[target]

    output = operation(value)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    output.backward(upstream)
    torch.testing.assert_close(value.grad, expected_grad, rtol=0, atol=0)
    torch.testing.assert_close(value, before, rtol=0, atol=0)


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("local_lengths", [[2], [2, 10, 4, 18], [2] * 17])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_packed_rank_sequence_permutations(
    cp_size: int, local_lengths: list[int], dtype: torch.dtype, noncontiguous: bool
) -> None:
    rank_rows, sequence_rows = _rank_and_sequence_rows(cp_size, local_lengths)
    cu_seqlens = _boundaries(local_lengths)
    _assert_permutation_and_gradient(
        lambda value: _deinterleave_packed_seqs(value, cu_seqlens, cp_size),
        rank_rows,
        sequence_rows,
        dtype,
        noncontiguous,
    )
    _assert_permutation_and_gradient(
        lambda value: _reinterleave_packed_seqs(value, cu_seqlens, cp_size),
        sequence_rows,
        rank_rows,
        dtype,
        noncontiguous,
    )


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("chunks_per_document", [[1], [1, 5, 2, 9]])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_chunk_permutation(cp_size: int, chunks_per_document: list[int], dtype: torch.dtype) -> None:
    num_chunks = 2 * cp_size
    order = list(reversed(range(num_chunks)))
    source_rows = [
        (document, chunk, offset)
        for document, chunk_size in enumerate(chunks_per_document)
        for chunk in range(num_chunks)
        for offset in range(chunk_size)
    ]
    target_rows = [
        (document, chunk, offset)
        for document, chunk_size in enumerate(chunks_per_document)
        for chunk in order
        for offset in range(chunk_size)
    ]
    cu_seqlens = _boundaries([n * num_chunks for n in chunks_per_document])
    _assert_permutation_and_gradient(
        lambda value: _reorder_chunks(value, order, cu_seqlens), source_rows, target_rows, dtype, True
    )


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
def test_packed_dual_chunk_swap_matches_document_order(cp_size: int) -> None:
    # Source labels express the CP contract directly: rank r owns chunk r and
    # chunk 2*cp-r-1 of every document. The oracle never calls a production helper.
    chunk_sizes = [1, 7, 3, 2]
    rank_rows = [
        (document, chunk, offset)
        for rank in range(cp_size)
        for document, size in enumerate(chunk_sizes)
        for chunk in [rank, 2 * cp_size - rank - 1]
        for offset in range(size)
    ]
    sequential_rows = [
        (document, chunk, offset)
        for document, size in enumerate(chunk_sizes)
        for chunk in range(2 * cp_size)
        for offset in range(size)
    ]
    cu_local = _boundaries([2 * n for n in chunk_sizes])
    cu_global = cu_local * cp_size

    def to_sequential(value: torch.Tensor) -> torch.Tensor:
        """Map rank blocks to document token order.

        Args:
            value: Tensor of shape [global_tokens, local_hidden] in rank-major
                DualChunkSwap order, with each rank's two chunks per document.

        Returns:
            Tensor of shape [global_tokens, local_hidden] in document-major
            sequential order across all CP ranks.
        """
        value = _deinterleave_packed_seqs(value, cu_local, cp_size)
        return _undo_attention_load_balancing(value, cp_size, cu_global)

    def to_rank_major(value: torch.Tensor) -> torch.Tensor:
        """Map document tokens back to rank blocks.

        Args:
            value: Tensor of shape [global_tokens, local_hidden] in document-major
                sequential order across all CP ranks.

        Returns:
            Tensor of shape [global_tokens, local_hidden] in rank-major
            DualChunkSwap order, with each rank's two chunks per document.
        """
        value = _redo_attention_load_balancing(value, cp_size, cu_global)
        return _reinterleave_packed_seqs(value, cu_local, cp_size)

    _assert_permutation_and_gradient(to_sequential, rank_rows, sequential_rows, torch.float32, False)
    _assert_permutation_and_gradient(to_rank_major, sequential_rows, rank_rows, torch.float32, True)


def test_packed_reorder_has_no_host_scalar_reads() -> None:
    # CUDA .item() uses this same ATen operation and forces a device/host sync.
    # This regression fails against the old per-document implementation on CPU.
    cu_local = _boundaries([2, 6, 4, 10])
    cp_size = 8
    value = torch.randn(22 * cp_size, 3, requires_grad=True)
    upstream = torch.randn_like(value)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        output = _deinterleave_packed_seqs(value, cu_local, cp_size)
        output = _undo_attention_load_balancing(output, cp_size, cu_local * cp_size)
        output = _redo_attention_load_balancing(output, cp_size, cu_local * cp_size)
        output = _reinterleave_packed_seqs(output, cu_local, cp_size)
        output.backward(upstream)
    scalar_reads = sum(event.count for event in profile.key_averages() if event.key == "aten::_local_scalar_dense")
    assert scalar_reads == 0
    torch.testing.assert_close(output, value, rtol=0, atol=0)
    torch.testing.assert_close(value.grad, upstream, rtol=0, atol=0)


def test_packed_permutation_gradcheck() -> None:
    cu_local = _boundaries([2, 4, 2])
    value = torch.randn(16, 2, dtype=torch.float64, requires_grad=True)
    operation = lambda x: _deinterleave_packed_seqs(x, cu_local, 2)
    assert torch.autograd.gradcheck(operation, (value,))
    assert torch.autograd.gradgradcheck(operation, (value,))


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
def test_unpacked_control_roundtrip(cp_size: int) -> None:
    torch.manual_seed(17)
    value = torch.randn(2, cp_size * 16, 5, requires_grad=True)
    upstream = torch.randn_like(value)
    output = _redo_attention_load_balancing(_undo_attention_load_balancing(value, cp_size), cp_size)
    torch.testing.assert_close(output, value, rtol=0, atol=0)
    output.backward(upstream)
    torch.testing.assert_close(value.grad, upstream, rtol=0, atol=0)

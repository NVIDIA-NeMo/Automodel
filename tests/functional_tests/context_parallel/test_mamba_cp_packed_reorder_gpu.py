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

"""Real CUDA permutation/gradient checks; collectives use run_mamba_cp.py."""

import itertools

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.distributed.context_parallel.mamba import (
    _deinterleave_packed_seqs,
    _redo_attention_load_balancing,
    _reinterleave_packed_seqs,
    _undo_attention_load_balancing,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _labeled_rows(cp_size: int, lengths: list[int]) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    rank_rows = [(r, s, j) for r in range(cp_size) for s, length in enumerate(lengths) for j in range(length)]
    sequence_rows = [(r, s, j) for s, length in enumerate(lengths) for r in range(cp_size) for j in range(length)]
    return rank_rows, sequence_rows


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("metadata_device", ["cpu", "cuda"])
@pytest.mark.parametrize("deterministic", [False, True])
def test_cuda_packed_permutation_gradients_and_update(
    cp_size: int, dtype: torch.dtype, metadata_device: str, deterministic: bool
) -> None:
    lengths = [2, 18, 4, 6, 2]
    rank_rows, sequence_rows = _labeled_rows(cp_size, lengths)
    cu_seqlens = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32, device=metadata_device)
    previous_mode = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(deterministic)
    try:
        for operation, source, target in [
            (_deinterleave_packed_seqs, rank_rows, sequence_rows),
            (_reinterleave_packed_seqs, sequence_rows, rank_rows),
        ]:
            lookup = {row: index for index, row in enumerate(source)}
            order = [lookup[row] for row in target]
            assert sorted(order) == list(range(len(source)))
            inverse = [0] * len(order)
            for dst, src in enumerate(order):
                inverse[src] = dst
            order = torch.tensor(order, device="cuda")
            inverse = torch.tensor(inverse, device="cuda")
            torch.manual_seed(831)
            # A strided trainable leaf exercises non-contiguous reads and
            # gradients that actually reach a parameter, not just an activation.
            value = torch.nn.Parameter(torch.randn(len(source), 14, device="cuda", dtype=dtype)[:, ::2])
            initial = value.detach().clone()
            upstream = torch.randn_like(value)
            expected = initial.index_select(0, order)
            expected_grad = upstream.index_select(0, inverse)
            previous_grad = None
            for _ in range(2):
                value.grad = None
                output = operation(value, cu_seqlens, cp_size)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                output.backward(upstream)
                torch.testing.assert_close(value.grad, expected_grad, rtol=0, atol=0)
                if previous_grad is not None:
                    assert torch.equal(value.grad, previous_grad)
                previous_grad = value.grad.clone()
            optimizer = torch.optim.SGD([value], lr=0.125, foreach=False)
            optimizer.step()
            torch.testing.assert_close(value, torch.add(initial, expected_grad, alpha=-0.125), rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(previous_mode)


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize(
    "chunk_lengths", [[1, 9, 2, 3], [4096], [1] * 4096], ids=["uneven", "one_long_doc", "many_minimal_docs"]
)
def test_cuda_packed_checkpoint_gradient_parity(cp_size: int, chunk_lengths: list[int]) -> None:
    # The source labels independently encode the TE DualChunkSwap layout.
    source = [
        (s, chunk, j)
        for rank in range(cp_size)
        for s, length in enumerate(chunk_lengths)
        for chunk in [rank, 2 * cp_size - rank - 1]
        for j in range(length)
    ]
    target = [
        (s, chunk, j) for s, length in enumerate(chunk_lengths) for chunk in range(2 * cp_size) for j in range(length)
    ]
    lookup = {row: index for index, row in enumerate(source)}
    order = torch.tensor([lookup[row] for row in target], device="cuda")
    cu_local = torch.tensor([0, *itertools.accumulate([2 * n for n in chunk_lengths])], device="cuda")
    torch.manual_seed(49)
    initial = torch.randn(len(source), 7, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn_like(initial)
    reference = initial.detach().clone().requires_grad_()
    reference_output = reference.index_select(0, order)
    reference_output.backward(upstream)

    def reordered(value: torch.Tensor) -> torch.Tensor:
        """Restore original document order from rank-major packed tokens.

        Args:
            value: Tensor of shape [global_tokens, local_hidden].

        Returns:
            Tensor of shape [global_tokens, local_hidden] in document token order.
        """
        value = _deinterleave_packed_seqs(value, cu_local, cp_size)
        return _undo_attention_load_balancing(value, cp_size, cu_local * cp_size)

    for use_checkpoint in [False, True]:
        value = initial.detach().clone().requires_grad_()
        output = checkpoint(reordered, value, use_reentrant=False) if use_checkpoint else reordered(value)
        output.backward(upstream)
        torch.testing.assert_close(output, reference_output, rtol=0, atol=0)
        torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)


@pytest.mark.parametrize("cp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("redo", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_chunk_reorder_avoids_host_synchronization(cp_size: int, redo: bool, dtype: torch.dtype) -> None:
    """GPU-resident metadata must not introduce pageable host-to-device copies."""
    lengths = [1, 9, 2, 3]
    balanced = [
        (doc, chunk, token)
        for doc, length in enumerate(lengths)
        for rank in range(cp_size)
        for chunk in (rank, 2 * cp_size - rank - 1)
        for token in range(length)
    ]
    sequential = [
        (doc, chunk, token)
        for doc, length in enumerate(lengths)
        for chunk in range(2 * cp_size)
        for token in range(length)
    ]
    source, target = (sequential, balanced) if redo else (balanced, sequential)
    lookup = {row: index for index, row in enumerate(source)}
    order = torch.tensor([lookup[row] for row in target], device="cuda")
    boundaries = torch.tensor([0, *itertools.accumulate(n * 2 * cp_size for n in lengths)], device="cuda")
    value = torch.randn(len(source), 7, dtype=dtype, device="cuda", requires_grad=True)
    reference = value.detach().clone().requires_grad_()
    upstream = torch.randn_like(value)
    expected = reference.index_select(0, order)
    expected.backward(upstream)
    operation = _redo_attention_load_balancing if redo else _undo_attention_load_balancing
    previous_mode = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        actual = operation(value, cp_size, boundaries)
        actual.backward(upstream)
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(value.grad, reference.grad, rtol=0, atol=0)

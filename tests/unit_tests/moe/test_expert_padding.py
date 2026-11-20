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

import itertools

import pytest
import torch

from nemo_automodel.components.moe import expert_padding
from nemo_automodel.components.moe.expert_padding import (
    pad_expert_rows,
    padded_rows_bound,
    padded_splits,
    unpad_expert_rows,
)

pytestmark = pytest.mark.timeout(600)
requires_gpu_triton = pytest.mark.skipif(
    not torch.cuda.is_available() or not expert_padding.HAVE_TRITON, reason="CUDA and Triton are required"
)


def _row_maps(counts: list[int], align: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Matching compact and padded row indices, built independently of the module under test."""
    compact_rows, padded_rows = [], []
    compact = padded = 0
    for count in counts:
        compact_rows.extend(range(compact, compact + count))
        padded_rows.extend(range(padded, padded + count))
        compact += count
        padded += (count + align - 1) // align * align
    return torch.tensor(compact_rows, dtype=torch.long), torch.tensor(padded_rows, dtype=torch.long)


def _reference_pad(x: torch.Tensor, counts: list[int], align: int, padded_rows: int) -> torch.Tensor:
    compact, padded = _row_maps(counts, align)
    keep = (compact < x.shape[0]) & (padded < padded_rows)
    compact, padded = compact[keep].to(x.device), padded[keep].to(x.device)
    return x.new_zeros((padded_rows,) + tuple(x.shape[1:])).index_put((padded,), x[compact])


def _reference_unpad(y: torch.Tensor, counts: list[int], align: int, rows: int) -> torch.Tensor:
    compact, padded = _row_maps(counts, align)
    keep = (compact < rows) & (padded < y.shape[0])
    compact, padded = compact[keep].to(y.device), padded[keep].to(y.device)
    return y.new_zeros((rows,) + tuple(y.shape[1:])).index_put((compact,), y[padded])


CASES = [
    # (counts, align, extra tail rows, feature shape)
    ([5, 0, 33, 1], 32, 0, (96,)),
    ([5, 0, 33, 1], 32, 11, (96,)),
    ([0, 0, 0, 0], 32, 7, (64,)),
    ([128, 256], 128, 0, (130,)),
    ([1] * 9, 128, 3, (1,)),
    ([3, 17, 0, 40], 16, 5, ()),
    ([7, 9], 4, 2, (2, 33)),
    ([2, 3, 4], 1, 1, (48,)),
]


@pytest.mark.parametrize("counts,align,tail,feature", CASES)
def test_cpu_layout_matches_the_reference(counts, align, tail, feature):
    torch.manual_seed(0)
    rows = sum(counts) + tail
    x = torch.randn((rows,) + feature)
    tokens_per_expert = torch.tensor(counts, dtype=torch.int32)
    bound = padded_rows_bound(rows, num_experts=len(counts), align=align)

    padded = pad_expert_rows(x, tokens_per_expert, align=align)
    assert padded.shape == (bound,) + feature
    torch.testing.assert_close(padded, _reference_pad(x, counts, align, bound), rtol=0, atol=0)

    compact = unpad_expert_rows(padded, tokens_per_expert, align=align, rows=rows)
    expected = x.clone()
    expected[sum(counts) :] = 0
    torch.testing.assert_close(compact, expected, rtol=0, atol=0)


@requires_gpu_triton
@pytest.mark.parametrize("counts,align,tail,feature", CASES)
@pytest.mark.parametrize(
    "dtype,count_dtype", [(torch.bfloat16, torch.int32), (torch.float32, torch.int64), (torch.float16, torch.int64)]
)
def test_triton_forward_and_backward_match_the_reference(counts, align, tail, feature, dtype, count_dtype):
    torch.manual_seed(0)
    rows = sum(counts) + tail
    bound = padded_rows_bound(rows, num_experts=len(counts), align=align)
    tokens_per_expert = torch.tensor(counts, dtype=count_dtype, device="cuda")
    x = torch.randn((rows,) + feature, device="cuda").to(dtype).requires_grad_(True)
    x_ref = x.detach().clone().requires_grad_(True)
    grad = torch.randn((bound,) + feature, device="cuda").to(dtype)

    padded = pad_expert_rows(x, tokens_per_expert, align=align)
    padded_ref = _reference_pad(x_ref, counts, align, bound)
    assert torch.equal(padded, padded_ref)
    padded.backward(grad)
    padded_ref.backward(grad)
    assert torch.equal(x.grad, x_ref.grad)

    y = padded.detach().clone().requires_grad_(True)
    y_ref = y.detach().clone().requires_grad_(True)
    grad_rows = torch.randn((rows,) + feature, device="cuda").to(dtype)
    compact = unpad_expert_rows(y, tokens_per_expert, align=align, rows=rows)
    compact_ref = _reference_unpad(y_ref, counts, align, rows)
    assert torch.equal(compact, compact_ref)
    compact.backward(grad_rows)
    compact_ref.backward(grad_rows)
    assert torch.equal(y.grad, y_ref.grad)


@requires_gpu_triton
def test_large_expert_count_and_hidden_size_match_the_reference():
    torch.manual_seed(0)
    counts = torch.randint(0, 300, (64,)).tolist()
    counts[3] = 0
    counts[40] = 128
    rows = sum(counts) + 100
    x = torch.randn(rows, 7168, device="cuda", dtype=torch.bfloat16)
    tokens_per_expert = torch.tensor(counts, device="cuda")
    padded = pad_expert_rows(x, tokens_per_expert, align=128)
    assert torch.equal(padded, _reference_pad(x, counts, 128, padded.shape[0]))
    assert torch.equal(
        unpad_expert_rows(padded, tokens_per_expert, align=128, rows=rows),
        _reference_unpad(padded, counts, 128, rows),
    )


@requires_gpu_triton
def test_float8_and_bool_rows_are_moved_bit_for_bit():
    counts = torch.tensor([3, 0, 6], device="cuda")
    x = torch.randn(11, 32, device="cuda").to(torch.float8_e4m3fn)
    padded = pad_expert_rows(x, counts, align=4)
    assert padded.dtype == torch.float8_e4m3fn
    assert torch.equal(padded.view(torch.uint8), _reference_pad(x.view(torch.uint8), [3, 0, 6], 4, padded.shape[0]))
    mask = torch.rand(11, 5, device="cuda") > 0.5
    assert torch.equal(pad_expert_rows(mask, counts, align=4), _reference_pad(mask, [3, 0, 6], 4, padded.shape[0]))


@requires_gpu_triton
def test_gradients_are_bitwise_reproducible():
    torch.manual_seed(0)
    counts = torch.tensor([31, 0, 77, 5, 128, 2], device="cuda")
    x = torch.randn(300, 512, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn(300, 512, device="cuda", dtype=torch.bfloat16)
    grads = []
    for _ in range(2):
        leaf = x.clone().requires_grad_(True)
        out = unpad_expert_rows(pad_expert_rows(leaf, counts, align=32), counts, align=32, rows=300)
        out.backward(grad)
        grads.append(leaf.grad)
    assert torch.equal(grads[0], grads[1])


@requires_gpu_triton
def test_cuda_path_never_synchronizes_with_the_host():
    counts = torch.tensor([31, 0, 77, 5], device="cuda")
    x = torch.randn(120, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn(120, 256, device="cuda", dtype=torch.bfloat16)
    unpad_expert_rows(pad_expert_rows(x, counts, align=32), counts, align=32, rows=120).backward(grad)  # compile
    x.grad = None
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        splits = padded_splits(counts, align=32)
        padded = pad_expert_rows(x, counts, align=32)
        out = unpad_expert_rows(padded, counts, align=32, rows=120)
        out.backward(grad)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert splits.tolist() == [32, 0, 96, 32]
    torch.testing.assert_close(x.grad[:113], grad[:113], rtol=0, atol=0)
    assert not x.grad[113:].any()


@requires_gpu_triton
def test_cuda_graph_replay_reads_the_counts_on_the_device():
    counts = torch.tensor([3, 0, 5], device="cuda")
    x = torch.randn(10, 64, device="cuda")
    pad_expert_rows(x, counts, align=4)  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        padded = pad_expert_rows(x, counts, align=4, padded_rows=16)
    for new_counts in ([3, 0, 5], [1, 6, 2], [0, 0, 10]):
        counts.copy_(torch.tensor(new_counts, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(padded, _reference_pad(x, new_counts, 4, 16))


@pytest.mark.parametrize("experts,align", [(1, 32), (3, 4), (4, 128), (5, 1)])
def test_bound_and_splits_cover_every_split(experts, align):
    rows = 9
    for counts in itertools.product(range(rows + 1), repeat=experts):
        if sum(counts) > rows:
            continue
        splits = padded_splits(torch.tensor(counts), align=align)
        assert all(int(s) % align == 0 and 0 <= int(s) - c < align for s, c in zip(splits, counts))
        assert int(splits.sum()) <= padded_rows_bound(rows, num_experts=experts, align=align)
        if experts > 3:
            break  # the cartesian product is large; the first experts cover the worst case


def test_padded_rows_override_truncates_and_extends():
    x = torch.arange(6.0).reshape(6, 1)
    counts = torch.tensor([2, 4])
    assert pad_expert_rows(x, counts, align=4, padded_rows=5).flatten().tolist() == [0, 1, 0, 0, 2]
    assert pad_expert_rows(x, counts, align=4, padded_rows=10).flatten().tolist() == [0, 1, 0, 0, 2, 3, 4, 5, 0, 0]


@pytest.mark.parametrize(
    "counts,match",
    [
        (torch.tensor([1.0, 2.0]), "integer"),
        (torch.tensor([[1, 2]]), "1-D"),
        (torch.tensor([True, False]), "integer"),
    ],
)
def test_rejects_malformed_counts(counts, match):
    with pytest.raises(ValueError, match=match):
        pad_expert_rows(torch.zeros(3, 4), counts, align=4)


def test_rejects_bad_alignment_and_sizes():
    counts = torch.tensor([1, 2])
    with pytest.raises(ValueError, match="align"):
        pad_expert_rows(torch.zeros(3, 4), counts, align=0)
    with pytest.raises(ValueError, match="dst_rows"):
        unpad_expert_rows(torch.zeros(8, 4), counts, align=4, rows=-1)
    with pytest.raises(ValueError, match="experts"):
        pad_expert_rows(torch.zeros(3, 4), torch.zeros(expert_padding._MAX_EXPERTS + 1, dtype=torch.long), align=4)


@requires_gpu_triton
def test_rejects_counts_on_another_device():
    with pytest.raises(ValueError, match="device"):
        pad_expert_rows(torch.zeros(3, 4, device="cuda"), torch.tensor([1, 2]), align=4)

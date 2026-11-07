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
"""A sparse-index plan prepares the cuDNN / TileLang index work of ``dsv4_sparse_attention`` once per selection."""

from __future__ import annotations

import pytest
import torch

import nemo_automodel.components.models.deepseek_v4.optimized_kernels as kernels
from nemo_automodel.components.models.common.cudnn_sparse_attention import (
    _compact_and_sort_indices,
    plan_sparse_indices,
)


def test_plan_sparse_indices_matches_the_per_call_canonicalisation() -> None:
    idx = torch.tensor([[3, -1, 0, 7], [9, 9, -1, -1]])  # 9 is outside key_count 8 and therefore invalid
    indices, lengths = plan_sparse_indices(idx, 8)
    sorted_indices, expected_lengths = _compact_and_sort_indices(idx, 8)
    assert indices.shape == (2, 512) and indices.dtype == torch.int32 and indices.is_contiguous()
    assert torch.equal(indices[:, :4], sorted_indices) and bool((indices[:, 4:] == -1).all())
    assert torch.equal(lengths, expected_lengths) and lengths.tolist() == [3, 0]


def test_cudnn_plan_reaches_the_shared_helper_with_precomputed_lengths(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_helper(q, kv_latent, topk_indices, softmax_scale, *, attn_sink, **kwargs):
        seen.update(idx=topk_indices, kwargs=kwargs)
        return torch.zeros(q.shape[0], q.shape[1], 512, dtype=q.dtype)

    import nemo_automodel.components.models.common.cudnn_sparse_attention as shared_cudnn

    monkeypatch.setattr(shared_cudnn, "cudnn_sparse_attention", fake_helper)
    monkeypatch.setattr(shared_cudnn, "is_cudnn_sparse_attention_available", lambda: True)
    batch, sequence, heads, dim, kv_sequence = 2, 3, 4, 512, 5
    q = torch.randn(batch, sequence, heads, dim).to(torch.bfloat16)
    kv = torch.randn(batch, kv_sequence, dim).to(torch.bfloat16)
    sinks = torch.zeros(heads)
    idx = torch.tensor([[[1, 0, -1], [2, -1, 1], [-1, -1, -1]], [[4, -1, 0], [3, 2, -1], [1, 4, 3]]], dtype=torch.int32)
    plan = kernels.plan_dsv4_sparse_indices(idx, kv_sequence, backend="cudnn", all_rows_nonempty=False)
    expected = torch.full((batch * sequence, 512), -1, dtype=torch.int32)
    for row, values in enumerate([[0, 1], [1, 2], [], [5, 9], [7, 8], [6, 8, 9]]):
        expected[row, : len(values)] = torch.tensor(values, dtype=torch.int32)
    assert torch.equal(plan.indices, expected.unsqueeze(1)) and plan.indices.is_contiguous()
    assert torch.equal(plan.topk_length, torch.tensor([2, 2, 0, 2, 2, 3], dtype=torch.int32))
    assert torch.equal(plan.valid_rows, torch.tensor([0, 1, 3, 4, 5]))
    assert (
        kernels.plan_dsv4_sparse_indices(idx, kv_sequence, backend="cudnn", all_rows_nonempty=True).valid_rows is None
    )
    kernels.dsv4_sparse_attention(q, kv, sinks, None, 0.25, backend="cudnn", plan=plan)
    assert seen["idx"] is plan.indices and seen["kwargs"]["topk_length"] is plan.topk_length
    assert seen["kwargs"]["valid_row_indices"] is plan.valid_rows and seen["kwargs"]["all_rows_nonempty"] is False
    kernels.dsv4_sparse_attention(q, kv, sinks, None, 0.25, backend="cudnn", plan=plan, all_rows_nonempty=True)
    assert seen["kwargs"]["valid_row_indices"] is None  # rows are not forwarded when the caller vouches for every row
    kernels.dsv4_sparse_attention(q, kv, sinks, idx, 0.25, backend="cudnn")
    assert seen["kwargs"]["topk_length"] is None and seen["kwargs"]["valid_row_indices"] is None  # unplanned: as before
    unplanned = torch.tensor(
        [[1, 0, -1], [2, -1, 1], [-1, -1, -1], [9, -1, 5], [8, 7, -1], [6, 9, 8]], dtype=torch.int32
    )
    assert torch.equal(seen["idx"].squeeze(1), unplanned)
    with pytest.raises(ValueError, match="sparse-index plan"):
        kernels.dsv4_sparse_attention(q, kv[:, :4], sinks, None, 0.25, backend="cudnn", plan=plan)
    with pytest.raises(ValueError, match="sparse-index plan"):
        kernels.dsv4_sparse_attention(q, kv, sinks, None, 0.25, backend="tilelang", plan=plan)
    with pytest.raises(ValueError, match="needs topk_idxs or a prepared plan"):
        kernels.dsv4_sparse_attention(q, kv, sinks, None, 0.25, backend="cudnn")


def test_tilelang_plan_is_the_int32_slot_tensor_the_kernel_receives(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[torch.Tensor] = []

    def fake_kernel(q, kv, sinks, topk_idxs, sm_scale, **kwargs):
        seen.append(topk_idxs)
        return torch.zeros(q.shape[0], q.shape[1], q.shape[2], 8)

    monkeypatch.setattr(kernels, "_should_use_tilelang", lambda *args, **kwargs: True)
    monkeypatch.setattr(kernels, "_miles_sparse_attn_tilelang", fake_kernel)
    idx = torch.tensor([[[1, 0, -1], [2, -1, 1]]])
    plan = kernels.plan_dsv4_sparse_indices(idx, 4, backend="tilelang")
    assert plan.backend == "tilelang" and plan.topk_length is None and plan.valid_rows is None
    assert plan.indices.dtype == torch.int32 and plan.indices.is_contiguous() and torch.equal(plan.indices, idx.int())
    q = torch.randn(1, 2, 2, 8)
    kv = torch.randn(1, 4, 8)
    sinks = torch.zeros(2)
    kernels.dsv4_sparse_attention(q, kv, sinks, None, 0.5, backend="tilelang", plan=plan)
    kernels.dsv4_sparse_attention(q, kv, sinks, idx, 0.5, backend="tilelang")
    assert seen[0] is plan.indices  # the plan's tensor goes to the kernel as is
    assert torch.equal(seen[1], plan.indices) and seen[1].dtype == torch.int32  # unplanned: converted per call

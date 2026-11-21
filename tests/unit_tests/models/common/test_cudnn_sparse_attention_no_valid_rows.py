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

"""cuDNN sparse attention on a shard whose queries have no valid key: zero output, zero gradients, no backward kernel."""

import pytest
import torch

from nemo_automodel.components.models.common import cudnn_sparse_attention as csa

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not csa.is_cudnn_sparse_attention_available(),
    reason="requires the cuDNN DSA and FlashMLA kernels",
)

_HEAD_DIM, _VALUE_DIM, _HEADS, _TOPK = 576, 512, 8, 64


def _inputs(query_tokens: int, key_tokens: int):
    device = torch.device("cuda")
    torch.manual_seed(0)
    q = torch.randn(query_tokens, _HEADS, _HEAD_DIM, device=device, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(key_tokens, 1, _HEAD_DIM, device=device, dtype=torch.bfloat16, requires_grad=True)
    return q, kv


def test_shard_without_valid_queries_returns_zeros_and_skips_the_backward_kernel(monkeypatch):
    """A context-parallel shard made of pack padding has every top-k row empty (``topk_length == 0``)."""
    q, kv = _inputs(6, 10)
    topk_indices = torch.full((6, 1, _TOPK), -1, dtype=torch.int32, device=q.device)
    topk_length = torch.zeros(6, dtype=torch.int32, device=q.device)
    valid_rows = torch.empty(0, dtype=torch.int64, device=q.device)

    def no_backward(*args, **kwargs):
        raise AssertionError("the cuDNN backward kernel must not run for a shard without valid queries")

    monkeypatch.setattr(csa._CUDNN_DSA, "sparse_attention_backward_wrapper", no_backward)
    out = csa.cudnn_sparse_attention(
        q, kv, topk_indices, 0.1, topk_length=topk_length, all_rows_nonempty=False, valid_row_indices=valid_rows
    )
    assert out.shape == (6, _HEADS, _VALUE_DIM) and out.dtype == torch.bfloat16
    assert torch.equal(out, torch.zeros_like(out))

    (out.float() * torch.randn_like(out, dtype=torch.float32)).sum().backward()
    assert q.grad is not None and kv.grad is not None
    assert q.grad.shape == q.shape and kv.grad.shape == kv.shape
    assert torch.equal(q.grad, torch.zeros_like(q)) and torch.equal(kv.grad, torch.zeros_like(kv))


def test_shard_with_some_valid_queries_still_runs_the_backward_kernel(monkeypatch):
    """Control: one valid row keeps the kernel path (the gradient of that row is not identically zero)."""
    q, kv = _inputs(6, 10)
    topk_indices = torch.full((6, 1, _TOPK), -1, dtype=torch.int32, device=q.device)
    topk_indices[3, 0, :4] = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=q.device)
    topk_length = torch.zeros(6, dtype=torch.int32, device=q.device)
    topk_length[3] = 4
    valid_rows = torch.tensor([3], dtype=torch.int64, device=q.device)
    calls: list[int] = []
    original = csa._CUDNN_DSA.sparse_attention_backward_wrapper

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(csa._CUDNN_DSA, "sparse_attention_backward_wrapper", counting)
    out = csa.cudnn_sparse_attention(
        q, kv, topk_indices, 0.1, topk_length=topk_length, all_rows_nonempty=False, valid_row_indices=valid_rows
    )
    assert torch.equal(out[[0, 1, 2, 4, 5]], torch.zeros_like(out[[0, 1, 2, 4, 5]]))
    (out.float() * torch.randn_like(out, dtype=torch.float32)).sum().backward()
    assert calls == [1]
    assert not torch.equal(q.grad[3], torch.zeros_like(q.grad[3]))
    assert torch.equal(q.grad[[0, 1, 2, 4, 5]], torch.zeros_like(q.grad[[0, 1, 2, 4, 5]]))
    assert not torch.equal(kv.grad, torch.zeros_like(kv.grad))

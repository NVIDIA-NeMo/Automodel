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

"""Learned-prefix layout and gradient contracts, independent of CUDA kernels."""

import pytest
import torch

from nemo_automodel.components.models.nemotron_flash.packing import _expand_documents, _PackedInput

pytestmark = pytest.mark.cpu_only


def test_memory_prefix_per_document_and_multirow_scatter_gradients() -> None:
    embeddings = torch.arange(24, dtype=torch.float32).reshape(2, 6, 2).requires_grad_()
    memory = torch.tensor([[100.0, 101.0], [102.0, 103.0]], requires_grad=True)
    # Row-local gathered indices and cumulative lengths: two documents in each
    # row, with different lengths and trailing padding in both physical rows.
    indices = torch.tensor([[0, 1, 2, 3, -1, -1], [0, 1, 2, 3, 4, -1]])
    cu = torch.tensor([[0, 1, 4, -1], [0, 2, 5, -1]], dtype=torch.int32)
    hidden, positions, destinations, source, physical = _expand_documents(
        embeddings, memory, _PackedInput(indices, cu, 3)
    )
    expected_sources = [0, 1, 2, 3, 6, 7, 8, 9, 10]
    assert source.tolist() == expected_sources
    assert physical.cu.tolist() == [0, 3, 8, 12, 17]
    assert physical.seq_idx.tolist() == [[0] * 3 + [1] * 5 + [2] * 4 + [3] * 5]
    assert positions.tolist() == [[0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 3, 4]]
    for start in [0, 3, 8, 12]:
        torch.testing.assert_close(hidden[0, start : start + 2], memory)
    torch.testing.assert_close(hidden[0, destinations], embeddings.flatten(0, 1)[expected_sources])
    hidden.sum().backward()
    torch.testing.assert_close(memory.grad, torch.full_like(memory, 4))
    expected_gradient = torch.zeros_like(embeddings).flatten(0, 1)
    expected_gradient[expected_sources] = 1
    torch.testing.assert_close(embeddings.grad, expected_gradient.reshape_as(embeddings))


def test_no_padding_still_gets_independent_memory_prefixes() -> None:
    embeddings = torch.randn(1, 5, 3)
    memory = torch.randn(1, 3)
    packed = _PackedInput(torch.arange(5), torch.tensor([0, 2, 5], dtype=torch.int32), 3)
    hidden, positions, destinations, source, physical = _expand_documents(embeddings, memory, packed)
    assert hidden.shape == (1, 7, 3)
    assert physical.maximum == 4
    assert physical.cu.tolist() == [0, 3, 7]
    assert positions.tolist() == [[0, 1, 2, 0, 1, 2, 3]]
    restored = torch.zeros_like(embeddings).flatten(0, 1).index_copy(0, source, hidden[0, destinations])
    torch.testing.assert_close(restored.reshape_as(embeddings), embeddings)

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

"""Sparse metadata ownership for native FA4 query-head layout."""

import math

import pytest
import torch

from nemo_automodel.components.models.step5.fa4 import _provider_heads


@pytest.mark.parametrize("batch,groups,heads_per_group", [(1, 4, 16), (2, 3, 2), (2, 1, 1)])
@pytest.mark.parametrize("indices", [False, True])
def test_provider_heads_preserve_distinct_batch_and_group_routes(
    batch: int, groups: int, heads_per_group: int, indices: bool
) -> None:
    shape = (batch * groups, 1, 5, 7) if indices else (batch * groups, 1, 5)
    backing = torch.arange(2 * math.prod(shape), dtype=torch.int32).view(*shape, 2)
    tensor = backing[..., 1]
    original = tensor.clone()
    result = _provider_heads(tensor, batch=batch, heads_per_group=heads_per_group)
    assert result.shape == (batch, groups * heads_per_group, *shape[2:])
    assert result.dtype == torch.int32 and result.is_contiguous()
    for b in range(batch):
        for h in range(groups * heads_per_group):
            assert torch.equal(result[b, h], original[b * groups + h // heads_per_group, 0])
    assert torch.equal(tensor, original)
    result.fill_(-1)
    assert torch.equal(tensor, original), "Converted metadata must not alias its provider source"

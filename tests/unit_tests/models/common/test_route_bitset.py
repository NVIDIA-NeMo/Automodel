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

"""Exact set semantics of the shared packed route representation."""

import pytest
import torch

from nemo_automodel.components.models.common.route_bitset import build_route_bitset


@pytest.mark.parametrize("keys", [1, 7, 8, 31, 32, 33, 257])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_bitset_matches_independent_membership(keys, dtype):
    torch.manual_seed(7)
    backing = torch.randint(-5, keys + 3, (4, 10, 36), dtype=dtype)
    routes = backing[::2, ::2, ::2]
    routes[:, 0] = -1
    routes[:, 1] = 0
    routes[:, 2, -1] = 2**40 if dtype == torch.int64 else 2**30
    packed = build_route_bitset(routes, keys)
    ids = torch.arange(keys)
    expected = (routes[..., None] == ids).any(-2)
    actual = ((packed[..., ids // 8].int() >> (ids % 8)) & 1).bool()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert packed.dtype == torch.uint8
    assert packed.stride(-2) % 4 == 0
    assert packed.untyped_storage().data_ptr() != routes.untyped_storage().data_ptr()


def test_empty_route_width_and_expanded_strides():
    for routes in (torch.empty(2, 3, 0, dtype=torch.int64), torch.tensor([[[31, 32, -1]]]).expand(2, 3, 3)):
        result = build_route_bitset(routes, 33)
        ids = torch.arange(33)
        actual = ((result[..., ids // 8].int() >> (ids % 8)) & 1).bool()
        assert torch.equal(actual, (routes[..., None] == ids).any(-2))

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

"""The weighted Engram norms are the reference's ``torch.nn.RMSNorm`` modules, bit for bit."""

import pytest
import torch

from nemo_automodel.components.models.hy_v4_flash.engram import HY4FlashWeightOnlyNorm, weighted_rms_norm


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("scale", [1.0, 1e-3, 40.0])
def test_weighted_rms_norm_equals_torch_rms_norm_bitwise(dtype: torch.dtype, scale: float) -> None:
    """One ``nn.RMSNorm`` per slice with a non-unit weight, on inputs of ordinary, small and large magnitude.

    In BF16, normalizing and then multiplying a BF16 weight rounds twice and differs from ``nn.RMSNorm`` on about a
    quarter of the elements; ``nn.RMSNorm`` multiplies the weight in fp32 and casts once.
    """
    torch.manual_seed(0)
    slices, features = 4, 96
    x = (scale * torch.randn(2, 37, slices, features)).to(dtype)
    weights = [(1 + 0.1 * torch.randn(features)).to(dtype) for _ in range(slices)]
    ours = torch.nn.ModuleList(HY4FlashWeightOnlyNorm(features, dtype) for _ in range(slices))
    reference = [torch.nn.RMSNorm(features).to(dtype) for _ in range(slices)]
    with torch.no_grad():
        for norm, module, weight in zip(ours, reference, weights):
            norm.weight.copy_(weight)
            module.weight.copy_(weight)
    expected = torch.stack([module(x[..., index, :]) for index, module in enumerate(reference)], dim=-2)
    got = weighted_rms_norm(x, ours)
    assert got.dtype == dtype
    assert torch.equal(got, expected)

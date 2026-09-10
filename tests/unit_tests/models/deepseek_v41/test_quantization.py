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

"""Analytical representation and gradient contracts for cache QAT."""

import pytest
import torch

from nemo_automodel.components.models.deepseek_v41.quantization import quantize_cache


@pytest.mark.parametrize("format", ["mxfp4", "nvfp4"])
def test_fp4_ties_use_even_encoding_and_preserve_sign(format: str) -> None:
    # A maximum of 6 fixes the scale to 1 for both scale formats.
    values = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0]])
    expected = torch.tensor([[0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0]])
    positive = quantize_cache(values, format=format, block_size=8)
    negative = quantize_cache(-values, format=format, block_size=8)
    torch.testing.assert_close(positive, expected, rtol=0, atol=0)
    torch.testing.assert_close(negative, -expected, rtol=0, atol=0)


@pytest.mark.parametrize("format,block", [("fp8", 32), ("mxfp4", 32), ("nvfp4", 16)])
def test_cache_quantization_zero_partial_group_and_straight_through_gradient(format: str, block: int) -> None:
    torch.manual_seed(38)
    values = torch.randn(2, 3, block + 5, dtype=torch.bfloat16, requires_grad=True)
    with torch.no_grad():
        values[0, 0].zero_()
    original = values.detach().clone()
    upstream = torch.randn_like(values)
    quantized = quantize_cache(values, format=format, block_size=block)
    assert quantized.dtype == values.dtype
    assert quantized.shape == values.shape
    assert quantized.data_ptr() != values.data_ptr()
    torch.testing.assert_close(values, original, atol=0, rtol=0)
    assert torch.count_nonzero(quantized[0, 0]) == 0
    assert torch.isfinite(quantized).all()
    quantized.backward(upstream)
    torch.testing.assert_close(values.grad, upstream, rtol=0, atol=0)

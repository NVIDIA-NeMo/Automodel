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

"""Cache quantization encodings, boundary values and straight-through gradients."""

import math

import pytest
import torch

from nemo_automodel.components.models.deepseek_v41.quantization import quantize_cache, quantize_native


@pytest.mark.parametrize("format,block_size", [("int8", 32), ("fp8", 0), ("nvfp4", -1)])
def test_cache_quantization_rejects_unsupported_format_or_group_size(format, block_size):
    with pytest.raises(ValueError, match="positive block_size"):
        quantize_cache(torch.zeros(2, 20), format=format, block_size=block_size)


def _independent_mx_scale(amax, max_value):
    # frexp observes the exact FP32 reciprocal-product value, without the
    # production IEEE754 bit reinterpretation or the old inaccurate log2 path.
    scaled = float(torch.tensor(amax, dtype=torch.float32) * (1 / max_value))
    mantissa, exponent = math.frexp(scaled)
    return math.ldexp(1.0, exponent - (mantissa == 0.5))


@pytest.mark.parametrize("format", ["fp8", "mxfp4", "nvfp4"])
@pytest.mark.parametrize("exponent", [-8, 0, 8])
def test_cache_quantization_boundary_values_signed_zero_and_ste(format, exponent):
    max_value = 448.0 if format == "fp8" else 6.0
    unit = 2.0**exponent
    boundary = torch.tensor(max_value * unit)
    maxima = [
        torch.nextafter(boundary, torch.tensor(-torch.inf)),
        boundary,
        torch.nextafter(boundary, torch.tensor(torch.inf)),
        torch.tensor(0.0),
    ]
    block_size = 16 if format == "nvfp4" else 32
    rows = []
    for maximum in maxima:
        row = torch.zeros(block_size)
        row[:10] = torch.tensor([-0.0, 0.0, -0.25, 0.5, 0.75, 1.75, 3.5, -5.0, 1 / 512, 1.5]) * unit
        if maximum == 0:
            row.zero_()
            row[0] = -0.0
        else:
            row[-1] = maximum
        rows.append(row)
    x = torch.stack(rows).requires_grad_()
    actual = quantize_cache(x, format=format, block_size=block_size)
    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    expected_rows = []
    for row in x.detach():
        amax = float(row.abs().max())
        if format == "nvfp4":
            scale = float(torch.tensor(max(amax, 6 * 2.0**-9) / 6).to(torch.float8_e4m3fn).float())
        else:
            floor = 1e-4 if format == "fp8" else 6 * 2.0**-126
            scale = _independent_mx_scale(max(amax, floor), max_value)
        if format == "fp8":
            expected_rows.append((row / scale).to(torch.float8_e4m3fn).float() * scale)
        else:
            values = []
            for value in (row / scale).tolist():
                index = min(range(len(grid)), key=lambda i: (abs(abs(value) - grid[i]), i % 2))
                values.append(math.copysign(grid[index] * scale, value))
            expected_rows.append(torch.tensor(values))
    expected = torch.stack(expected_rows)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.signbit(actual), torch.signbit(expected))
    if format in ("fp8", "mxfp4") and exponent == 8:
        # Negative control: at large exponents, float32 log2 can round the
        # adjacent-above power back down, producing the old wrong MX scale.
        old_scale = torch.exp2(torch.ceil(torch.log2(maxima[2] / max_value)))
        assert _independent_mx_scale(float(maxima[2]), max_value) != old_scale.item()
    upstream = torch.arange(actual.numel()).reshape_as(actual).float() / 7
    actual.backward(upstream)
    torch.testing.assert_close(x.grad, upstream, rtol=0, atol=0)


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


@pytest.mark.parametrize("fp4,weight", [(True, True), (False, True), (False, False)])
def test_native_qat_ste_does_not_modify_master_weights(fp4, weight):
    torch.manual_seed(271)
    values = torch.randn(2, 64, 96, requires_grad=True)
    original = values.detach().clone()
    output = quantize_native(values, fp4=fp4, weight=weight)
    assert output.data_ptr() != values.data_ptr()
    assert torch.isfinite(output).all()
    assert not torch.equal(output, values)
    upstream = torch.randn_like(output)
    output.backward(upstream)
    torch.testing.assert_close(values.grad, upstream, rtol=0, atol=0)
    torch.testing.assert_close(values, original, rtol=0, atol=0)


def test_native_expert_quantizes_input_channels_not_output_channels():
    checkpoint = torch.zeros(32, 64)
    checkpoint[0, :4] = torch.tensor([6., .25, .75, 1.25])
    checkpoint[1, :4] = torch.tensor([12., .5, 1.5, 2.5])
    actual = quantize_native(checkpoint, fp4=True, weight=True)
    torch.testing.assert_close(actual[0, :4], torch.tensor([6., 0., 1., 1.]), rtol=0, atol=0)
    torch.testing.assert_close(actual[1, :4], torch.tensor([12., 0., 2., 2.]), rtol=0, atol=0)
    # AM uses [input, output], so omitting the transpose changes the result.
    wrong = quantize_native(checkpoint.T, fp4=True, weight=True).T
    assert not torch.equal(actual, wrong)


@pytest.mark.parametrize("fp4", [False, True])
def test_native_quantization_chunk_boundaries_preserve_scale_groups(fp4, monkeypatch):
    from nemo_automodel.components.models.deepseek_v41 import quantization

    torch.manual_seed(130)
    # Two chunks, with the split exactly on a 32-row scale boundary.
    monkeypatch.setattr(quantization, "_NATIVE_CHUNK_ELEMENTS", 1 << 20)
    values = torch.randn(64, 32768, dtype=torch.bfloat16)
    actual = quantize_native(values, fp4=fp4, weight=True)
    for start in (0, 32):
        expected = quantize_native(values[start:start + 32], fp4=fp4, weight=True)
        torch.testing.assert_close(actual[start:start + 32], expected, rtol=0, atol=0)


def test_native_fp4_rounds_to_nearest_e2m1_level_with_ties_to_even_mantissa():
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    midpoints = (levels[1:] + levels[:-1]) / 2
    # At a midpoint the neighbour with the even E2M1 mantissa (even level index) wins.
    tie_targets = torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])
    off = 2.0**-5  # still exact in bf16, which the quantizer reads
    padding = torch.zeros(3)
    magnitudes = torch.cat([levels, midpoints, midpoints * (1 + off), midpoints * (1 - off), padding])
    expected = torch.cat([levels, tie_targets, levels[1:], levels[:-1], padding])
    for exponent in (-12, 0, 9):
        for sign in (1.0, -1.0):
            # One row and one 32-channel block; the 6.0 entry fixes the scale at 2**exponent.
            values = (sign * magnitudes * 2.0**exponent).reshape(1, 32)
            actual = quantize_native(values, fp4=True, weight=True)
            torch.testing.assert_close(actual, (sign * expected * 2.0**exponent).reshape(1, 32), rtol=0, atol=0)
            assert torch.equal(torch.signbit(actual), torch.signbit(values))

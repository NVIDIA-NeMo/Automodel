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

"""Stochastic rounding to four significant bits: neighbours, absence of bias, values that must not change."""

import math

import pytest
import torch

from nemo_automodel.components.quantization.stochastic_rounding import (
    HAVE_TRITON,
    _stochastic_round_torch,
    stochastic_round_to_e4m3,
)

# The CUDA case runs the Triton kernel, the CPU case the PyTorch implementation.
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() and HAVE_TRITON else [])
DTYPES = [torch.bfloat16, torch.float32]


def _neighbours(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Bracket every value by numbers with four significant bits, computed one value at a time with ``math``.

    Args:
        values: Tensor of shape [count], finite.

    Returns:
        FP64 tensors ``(low, high)`` of shape [count] on the CPU with ``low <= values <= high``; equal where the
        value has four significant bits already.
    """
    low, high = [], []
    for value in values.double().cpu().tolist():
        if value == 0.0:
            low.append(0.0)
            high.append(0.0)
            continue
        _, exponent = math.frexp(abs(value))  # abs(value) is in [2**(exponent - 1), 2**exponent)
        step = 2.0 ** (exponent - 4)
        below = math.floor(abs(value) / step) * step
        above = below if below == abs(value) else below + step
        low.append(below if value > 0 else -above)
        high.append(above if value > 0 else -below)
    return torch.tensor(low, dtype=torch.float64), torch.tensor(high, dtype=torch.float64)


def _sample(count: int, dtype: torch.dtype, device: str) -> torch.Tensor:
    """Values of both signs spread over 24 powers of two, shape [count]."""
    generator = torch.Generator().manual_seed(7)
    magnitude = torch.exp2(torch.rand(count, generator=generator) * 24 - 12)
    sign = torch.where(torch.rand(count, generator=generator) < 0.5, -1.0, 1.0)
    return (magnitude * sign).to(dtype).to(device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_result_is_a_neighbour_with_four_significant_bits(device: str, dtype: torch.dtype):
    values = _sample(4096, dtype, device)
    low, high = _neighbours(values)
    torch.manual_seed(0)
    rounded = stochastic_round_to_e4m3(values)
    assert rounded.shape == values.shape and rounded.dtype == dtype and rounded.device == values.device
    result = rounded.double().cpu()
    assert bool(((result == low) | (result == high)).all())
    # Both directions occur, so this is not rounding down, up, or to nearest in disguise.
    inexact = low != high
    assert 0.2 < float((result == high)[inexact].double().mean()) < 0.8


@pytest.mark.parametrize("device", DEVICES)
def test_mean_of_many_draws_is_the_input_and_nearest_rounding_is_not(device: str):
    values = _sample(2048, torch.float32, device)
    low, high = _neighbours(values)
    exact = values.double().cpu()
    draws = 4000
    torch.manual_seed(0)
    total = torch.zeros_like(exact)
    for _ in range(draws):
        total += stochastic_round_to_e4m3(values).double().cpu()
    # One draw deviates by at most the neighbour spacing with a standard deviation of at most half of it.
    tolerance = 6 * (high - low) / (2 * math.sqrt(draws))
    assert bool(((total / draws - exact).abs() <= tolerance).all())
    # Control: rounding to nearest has a fixed error far outside that tolerance for most values.
    nearest = torch.where(exact - low < high - exact, low, high)
    assert float(((nearest - exact).abs() > tolerance).double().mean()) > 0.9


@pytest.mark.parametrize("device", DEVICES)
def test_upward_probability_is_the_position_between_the_neighbours(device: str):
    # 1.0 + k/64 for k in 0..7 lies between 1.0 and 1.125, at k/8 of the way.
    values = (1.0 + torch.arange(8, dtype=torch.float32) / 64).to(device).repeat_interleave(20000)
    torch.manual_seed(0)
    went_up = (stochastic_round_to_e4m3(values) == 1.125).view(8, 20000).double().mean(dim=1).cpu()
    expected = torch.arange(8, dtype=torch.float64) / 8
    assert bool(((went_up - expected).abs() < 0.02).all())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_values_that_must_not_change(device: str, dtype: torch.dtype):
    values = torch.tensor(
        [0.0, -0.0, 1.0, -1.5, 448.0, 0.875 * 2.0**-20, 15 * 2.0**10, float("inf"), float("-inf"), 2.0**-124],
        dtype=dtype,
        device=device,
    )
    for _ in range(4):
        assert torch.equal(stochastic_round_to_e4m3(values), values)
    assert bool(torch.isnan(stochastic_round_to_e4m3(torch.tensor([float("nan")], dtype=dtype, device=device))).all())


@pytest.mark.parametrize("device", DEVICES)
def test_draws_follow_the_global_generator(device: str):
    values = _sample(4096, torch.float32, device)
    torch.manual_seed(3)
    first = stochastic_round_to_e4m3(values)
    second = stochastic_round_to_e4m3(values)
    torch.manual_seed(3)
    assert torch.equal(stochastic_round_to_e4m3(values), first)
    assert not torch.equal(first, second)


@pytest.mark.parametrize("device", DEVICES)
def test_shapes_and_strides(device: str):
    values = _sample(6 * 32 * 64, torch.bfloat16, device).view(6, 32, 64)
    assert stochastic_round_to_e4m3(values).shape == values.shape
    strided = values.transpose(1, 2)  # not contiguous: takes the PyTorch implementation on every device
    low, high = _neighbours(strided.reshape(-1))
    result = stochastic_round_to_e4m3(strided)
    assert result.shape == strided.shape
    flat = result.reshape(-1).double().cpu()
    assert bool(((flat == low) | (flat == high)).all())


def test_pytorch_implementation_matches_the_contract():
    values = _sample(4096, torch.bfloat16, "cpu")
    low, high = _neighbours(values)
    torch.manual_seed(0)
    result = _stochastic_round_torch(values).double()
    assert bool(((result == low) | (result == high)).all())


@pytest.mark.parametrize("dtype", [torch.float64, torch.int32])
def test_unsupported_dtypes_are_rejected(dtype: torch.dtype):
    with pytest.raises(TypeError, match="BF16, FP16 and FP32"):
        stochastic_round_to_e4m3(torch.ones(4, dtype=dtype))

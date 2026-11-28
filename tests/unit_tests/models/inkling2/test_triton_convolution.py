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

"""The tiled static-convolution kernels against a per-position reference written with Python loops."""

import pytest
import torch

from nemo_automodel.components.models.inkling2 import layers, triton_convolution
from nemo_automodel.components.models.inkling2.layers import ShortConvolution

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="The kernels run on CUDA")


def _reference(
    x: torch.Tensor, valid: torch.Tensor, positions: torch.Tensor, weight: torch.Tensor, prefix: torch.Tensor
) -> torch.Tensor:
    """Static convolution of [batch, sequence, channels], one position and one tap at a time, in FP64."""
    batch, sequence, _ = x.shape
    width = weight.shape[-1]
    halo = width - 1
    masked = x.double() * valid.unsqueeze(-1)
    taps = weight.double()[:, 0]
    rows = []
    for t in range(sequence):
        total = masked[:, t].clone()
        for lag in range(width):
            earlier = masked[:, t - lag] if t - lag >= 0 else prefix.double()[:, halo + t - lag]
            reaches = (positions[:, t] >= lag).unsqueeze(-1)
            total = total + torch.where(reaches, earlier * taps[:, halo - lag], 0.0)
        rows.append(total * (width + 1) ** -0.5 * valid[:, t, None])
    return torch.stack(rows, dim=1)


def _case(batch: int, sequence: int, channels: int, width: int, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    """Inputs with a document start in every row, two padded positions and a nonzero history."""
    torch.manual_seed(batch * 1000 + sequence * 10 + channels)
    x = torch.randn(batch, sequence, channels, device="cuda").to(dtype).requires_grad_(True)
    weight = (0.3 * torch.randn(channels, 1, width, device="cuda")).to(dtype).requires_grad_(True)
    prefix = torch.randn(batch, width - 1, channels, device="cuda").requires_grad_(True)
    valid = torch.ones(batch, sequence, dtype=torch.bool, device="cuda")
    positions = torch.arange(sequence, device="cuda").expand(batch, sequence).clone() + 5
    for row in range(batch):
        start = (7 * row + sequence // 2) % sequence
        positions[row, start:] = torch.arange(sequence - start, device="cuda")
    if sequence > 4:
        valid[:, -2:] = False
        valid[0, 3] = False
    return x, valid, positions, weight, prefix


@needs_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 37, 70, 4), (1, 5, 3, 4), (3, 64, 129, 4), (2, 2, 9, 4), (2, 40, 16, 2)])
def test_tiled_convolution_matches_the_per_position_reference(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    """Output and the gradients of the input, the taps and the history, on shapes that do not fill a tile."""
    batch, sequence, channels, width = shape
    x, valid, positions, weight, prefix = _case(batch, sequence, channels, width, dtype)
    probe = torch.randn(batch, sequence, channels, device="cuda", dtype=torch.float64)
    out = triton_convolution.static_convolution(x, valid, positions, weight, prefix, width)
    (out.double() * probe).sum().backward()
    got = (out.detach().double(), x.grad.double(), weight.grad.double(), prefix.grad.double())

    leaves = [leaf.detach().clone().requires_grad_(True) for leaf in (x, weight, prefix)]
    expected_out = _reference(leaves[0], valid, positions, leaves[1], leaves[2])
    (expected_out * probe).sum().backward()
    expected = (expected_out.detach(), *(leaf.grad.double() for leaf in leaves))
    # BF16 keeps eight mantissa bits; FP32 differs from FP64 by its own rounding.
    tolerance = 2e-5 if dtype == torch.float32 else 2e-2
    for name, value, reference in zip(("output", "input", "taps", "history"), got, expected):
        scale = reference.abs().max().clamp_min(1e-6)
        assert (value - reference).abs().max() <= tolerance * scale, name


@needs_gpu
def test_tiled_convolution_stops_at_document_starts_and_reads_the_history() -> None:
    """Negative controls: without document starts, and without the history, the output is another one."""
    x, valid, positions, weight, prefix = _case(2, 37, 70, 4, torch.float32)
    out = triton_convolution.static_convolution(x, valid, positions, weight, prefix, 4)
    no_starts = triton_convolution.static_convolution(x, valid, positions + 100, weight, prefix, 4)
    no_history = triton_convolution.static_convolution(x, valid, positions + 100, weight, torch.zeros_like(prefix), 4)
    assert (out - no_starts).abs().max() > 1e-2
    assert (no_starts - no_history).abs().max() > 1e-2
    # The history only reaches the first width - 1 positions.
    torch.testing.assert_close(no_starts[:, 3:], no_history[:, 3:], rtol=0, atol=0)


@needs_gpu
def test_layer_uses_the_tiled_kernels_only_with_compiled_chains(monkeypatch: pytest.MonkeyPatch) -> None:
    """ShortConvolution takes the kernels on CUDA when the chains are compiled and the chain otherwise."""
    calls = []
    real = triton_convolution.static_convolution
    monkeypatch.setattr(layers, "_tiled_static_convolution", lambda *args: (calls.append(1), real(*args))[1])
    module = ShortConvolution(16, 4, dtype=torch.float32).cuda()
    module.reset_parameters(0.3)
    x = torch.randn(2, 9, 16, device="cuda")
    valid = torch.ones(2, 9, dtype=torch.bool, device="cuda")
    positions = torch.arange(9, device="cuda").expand(2, 9).contiguous()
    monkeypatch.setattr(layers, "_CORES_COMPILED", False)
    chain = module(x, valid, positions=positions)
    assert calls == []
    monkeypatch.setattr(layers, "_CORES_COMPILED", True)
    tiled = module(x, valid, positions=positions)
    assert calls == [1]
    torch.testing.assert_close(tiled, chain, rtol=1e-5, atol=1e-6)

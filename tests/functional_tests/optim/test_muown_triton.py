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

"""GPU regression coverage for fused Muown rows and fallback layouts."""

import pytest
import torch
from torch import nn

pytest.importorskip("dion")
pytest.importorskip("triton")

from nemo_automodel.components.optim.muown import Muown

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("shape,transposed", [((67, 89), False), ((267, 89), False), ((3, 89, 67), True)])
@pytest.mark.parametrize("nesterov,decay", [(True, 0.0), (False, 0.1)])
def test_triton_rows_match_torch_trajectory(shape, transposed, nesterov, decay):
    torch.manual_seed(73)
    baseline = nn.Parameter(torch.randn(shape, device="cuda") * 0.02)
    actual = nn.Parameter(baseline.detach().clone())
    unused = nn.Parameter(torch.zeros_like(actual))
    reference = Muown(
        [{"params": [baseline], "matrix_transposed": transposed}],
        nesterov=nesterov,
        weight_decay=decay,
    )
    optimizer = Muown(
        [{"params": [actual, unused], "matrix_transposed": transposed}],
        nesterov=nesterov,
        weight_decay=decay,
        use_triton=True,
    )
    # Signed magnitudes and exact zero rows must preserve the reference guards.
    with torch.no_grad():
        for weight, opt in ((baseline, reference), (actual, optimizer)):
            opt.state[weight]["g"].neg_()
            if not transposed:
                weight[..., 0, :].zero_()
                opt.state[weight]["g"][..., 0, :].zero_()
            else:
                weight[..., :, 0].zero_()
                opt.state[weight]["g"][..., :, 0].zero_()
    buffers = {
        key: value.data_ptr() for key, value in optimizer.state[actual].items() if isinstance(value, torch.Tensor)
    }
    with torch._dynamo.config.patch(disable=True):
        for step in range(8):
            grad = torch.randn_like(baseline) * 0.01
            baseline.grad = grad
            actual.grad = grad.clone()
            saved_grad = actual.grad
            before = actual._version
            for opt in (reference, optimizer):
                opt.param_groups[0]["lr"] = 3e-4 * (1 - step / 9)
                opt.step()
            assert actual._version > before
            assert actual.grad is saved_grad
            torch.testing.assert_close(actual.grad, grad, atol=0, rtol=0)
            # BF16 NS can amplify the last-bit difference of an FP32 row
            # reduction. This bound also covers the independent 30-step check.
            torch.testing.assert_close(actual, baseline, atol=3e-5, rtol=5e-4)
            for key in ("g", "v_norm", "m_g", "v_g", "momentum"):
                left, right = optimizer.state[actual][key], reference.state[baseline][key]
                assert torch.isfinite(left).all()
                if step == 0:
                    torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
                else:
                    # Near-zero magnitudes amplify the last-bit differences
                    # over repeated BF16 NS updates. Bound the whole state by
                    # the same 5e-4 relative-L2 budget as the 30-step validation.
                    relative_error = (left - right).norm() / right.norm().clamp_min(1e-30)
                    assert relative_error < 5e-4, (step, key, relative_error.item())
                assert optimizer.state[actual][key].data_ptr() == buffers[key]
    assert optimizer.state[unused]["muown_step"] == 0
    torch.testing.assert_close(unused, torch.zeros_like(unused), atol=0, rtol=0)


@pytest.mark.parametrize("layout", ["strided", "float64", "wide_transposed", "many_batches"])
def test_triton_unsupported_layout_falls_back(layout):
    torch.manual_seed(17)
    value = torch.randn((8193, 16) if layout == "wide_transposed" else (16, 24), device="cuda")
    if layout == "many_batches":
        value = torch.randn(65536, 1, 1, device="cuda")
    if layout == "strided":
        value = value.t()
    if layout == "float64":
        value = value.double()
    reference = nn.Parameter(value.clone())
    actual = nn.Parameter(value.clone())
    transposed = layout == "wide_transposed"
    opts = [
        Muown([{"params": [reference], "matrix_transposed": transposed}]),
        Muown([{"params": [actual], "matrix_transposed": transposed}], use_triton=True),
    ]
    grad = torch.randn_like(reference)
    with torch._dynamo.config.patch(disable=True):
        for weight, opt in zip((reference, actual), opts):
            weight.grad = grad.clone()
            opt.step()
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


@pytest.mark.parametrize(
    "shape,transposed",
    [
        ((1, 4096, 4096), True),
        ((1, 6144, 4096), True),
        ((1, 2048, 6144), True),
        ((6144, 16384), False),
    ],
)
def test_mimo_shapes_use_triton_and_match_torch(shape, transposed, monkeypatch):
    from nemo_automodel.components.optim import muown_triton

    torch.manual_seed(91)
    expected = nn.Parameter(torch.randn(shape, device="cuda") * 0.02)
    actual = nn.Parameter(expected.detach().clone())
    reference = Muown([{"params": [expected], "matrix_transposed": transposed}], weight_decay=0.1)
    optimizer = Muown(
        [{"params": [actual], "matrix_transposed": transposed}],
        weight_decay=0.1,
        use_triton=True,
    )
    calls = 0
    prepare = muown_triton._prepare

    def tracked_prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        return prepare(*args, **kwargs)

    monkeypatch.setattr(muown_triton, "_prepare", tracked_prepare)
    with torch._dynamo.config.patch(disable=True):
        for _ in range(3):
            expected.grad = torch.randn_like(expected) * 0.01
            actual.grad = expected.grad.clone()
            reference.step()
            optimizer.step()
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=5e-4)
            for key in ("g", "v_norm", "m_g", "v_g", "momentum"):
                torch.testing.assert_close(
                    optimizer.state[actual][key], reference.state[expected][key], atol=3e-5, rtol=5e-4
                )
    assert calls == 3

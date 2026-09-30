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
@pytest.mark.parametrize("update_scale", [0.0, 0.5, 1.0])
def test_triton_rows_match_torch_trajectory(shape, transposed, nesterov, decay, update_scale):
    torch.manual_seed(73)
    baseline = nn.Parameter(torch.randn(shape, device="cuda") * 0.02)
    actual = nn.Parameter(baseline.detach().clone())
    unused = nn.Parameter(torch.zeros_like(actual))
    reference = Muown(
        [{"params": [baseline], "matrix_transposed": transposed}],
        nesterov=nesterov,
        weight_decay=decay,
        muon_update_scale=update_scale,
    )
    optimizer = Muown(
        [{"params": [actual, unused], "matrix_transposed": transposed}],
        nesterov=nesterov,
        weight_decay=decay,
        muon_update_scale=update_scale,
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
                    # BF16 NS can amplify last-bit reduction differences.
                    # Bound total state error, with an absolute term for small
                    # or cancelling magnitude moments. Shared-NS tests below
                    # separately check the FP32 row kernels at tighter bounds.
                    error, reference_norm = (left - right).norm(), right.norm()
                    assert error <= 3e-5 + 5e-4 * reference_norm, (step, key, error.item(), reference_norm.item())
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
    reference = Muown(
        [{"params": [expected], "matrix_transposed": transposed}], weight_decay=0.1, muon_update_scale=0.5
    )
    optimizer = Muown(
        [{"params": [actual], "matrix_transposed": transposed}],
        weight_decay=0.1,
        muon_update_scale=0.5,
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


@pytest.mark.parametrize("use_triton", [False, True])
@pytest.mark.parametrize("update_scale", [0.0, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("transposed", [False, True])
def test_direction_scale_has_analytic_update_without_scaling_magnitude(
    use_triton, update_scale, transposed, monkeypatch
):
    import nemo_automodel.components.optim.muown as implementation

    update = torch.tensor([[0.0, 0.75]], device="cuda", dtype=torch.bfloat16)
    value = torch.tensor([[1.0, 0.0]], device="cuda")
    grad = torch.tensor([[0.25, 1.0]], device="cuda")
    if transposed:
        value, grad, update = value.mT.contiguous(), grad.mT.contiguous(), update.mT.contiguous()
    monkeypatch.setattr(implementation, "_newton_schulz", lambda *args, **kwargs: update)
    weight = nn.Parameter(value)
    optimizer = Muown(
        [{"params": [weight], "matrix_transposed": transposed}],
        lr=0.0033,
        mu=0.0,
        nesterov=False,
        muon_update_scale=update_scale,
        use_triton=use_triton,
    )
    weight.grad = grad
    with torch._dynamo.config.patch(disable=True):
        optimizer.step()
    result = weight.mT if transposed else weight
    # This angle is independent of row normalization and the learned magnitude.
    expected_ratio = -0.0033 * update_scale * (0.2 * 2**0.5) * 0.75
    assert abs((result[0, 1] / result[0, 0]).item() - expected_ratio) < 1e-9
    torch.testing.assert_close(
        optimizer.state[weight]["g"],
        torch.full_like(optimizer.state[weight]["g"], 1.0 - 0.0033),
        rtol=0,
        atol=1e-7,
    )


def test_scaled_triton_rows_with_shared_newton_schulz_updates(monkeypatch):
    """Isolate local FP32 kernels from BF16 NS rounding across repeated steps."""
    import nemo_automodel.components.optim.muown as implementation

    torch.manual_seed(73)
    expected = nn.Parameter(torch.randn(3, 89, 67, device="cuda") * 0.02)
    actual = nn.Parameter(expected.detach().clone())
    reference = Muown(
        [{"params": [expected], "matrix_transposed": True}], nesterov=False, weight_decay=0.1, muon_update_scale=0.5
    )
    optimizer = Muown(
        [{"params": [actual], "matrix_transposed": True}],
        nesterov=False,
        weight_decay=0.1,
        muon_update_scale=0.5,
        use_triton=True,
    )
    with torch.no_grad():
        for weight, opt in ((expected, reference), (actual, optimizer)):
            opt.state[weight]["g"].neg_()
            weight[..., :, 0].zero_()
            opt.state[weight]["g"][..., :, 0].zero_()
    original = implementation._newton_schulz
    updates = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        updates.append(result.clone())
        return result

    with torch._dynamo.config.patch(disable=True):
        for step in range(8):
            expected.grad = torch.randn_like(expected) * 0.01
            actual.grad = expected.grad.clone()
            for opt in (reference, optimizer):
                opt.param_groups[0]["lr"] = 3e-4 * (1 - step / 9)
            monkeypatch.setattr(implementation, "_newton_schulz", capture)
            reference.step()
            monkeypatch.setattr(implementation, "_newton_schulz", lambda *args, **kwargs: updates.pop(0))
            optimizer.step()
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            for key in ("g", "v_norm", "m_g", "v_g", "momentum"):
                torch.testing.assert_close(
                    optimizer.state[actual][key], reference.state[expected][key], atol=2e-6, rtol=2e-5
                )

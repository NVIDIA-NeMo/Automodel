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

import pytest
import torch

import nemo_automodel.components.models.kimi_k3.situ as situ_mod
from nemo_automodel.components.models.kimi_k3.situ import (
    _dense_situ_core,
    _enable_situ_triton,
    _situ_triton_applies,
    _WeightedSiTUFunction,
    dense_situ,
)
from nemo_automodel.components.models.kimi_k3.situ_triton import HAVE_TRITON, situ_bwd_triton, situ_fwd_triton

BETA = 4.0
LINEAR_BETA = 25.0

needs_gpu_triton = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAVE_TRITON), reason="Triton SiTU kernels need a GPU and Triton"
)


def _eager_reference(gate_up, routing_weights, beta, linear_beta):
    """Plain-autograd fp32 weighted SiTU (routing_weights may be None for the dense form)."""
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    activated = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    out = activated * up
    if routing_weights is not None:
        out = out * routing_weights.float()
    return out.to(gate_up.dtype)


@pytest.fixture
def triton_enabled(monkeypatch):
    monkeypatch.setattr(situ_mod, "_SITU_TRITON_ENABLED", True)


def test_applies_requires_enabled_cuda_even_rows():
    x = torch.randn(4, 8)
    assert not _situ_triton_applies(x, None)  # disabled by default
    situ_mod_flag = situ_mod._SITU_TRITON_ENABLED
    try:
        situ_mod._SITU_TRITON_ENABLED = True
        assert not _situ_triton_applies(x, None)  # CPU tensor
        if torch.cuda.is_available():
            xc = x.cuda()
            assert _situ_triton_applies(xc, None)
            assert _situ_triton_applies(xc, torch.rand(4, 1, device="cuda"))
            assert not _situ_triton_applies(xc, torch.rand(4, 2, device="cuda"))  # k != 1
            assert not _situ_triton_applies(xc[:0], None)  # zero rows
            assert not _situ_triton_applies(torch.randn(4, 7, device="cuda"), None)  # odd last axis
    finally:
        situ_mod._SITU_TRITON_ENABLED = situ_mod_flag


def test_enable_is_noop_without_cuda(monkeypatch):
    monkeypatch.setattr(situ_mod, "_SITU_TRITON_ENABLED", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _enable_situ_triton()
    assert situ_mod._SITU_TRITON_ENABLED is False


def test_enable_sets_flag_once(monkeypatch):
    monkeypatch.setattr(situ_mod, "_SITU_TRITON_ENABLED", False)
    monkeypatch.setattr(situ_mod, "_HAVE_SITU_TRITON", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _enable_situ_triton()
    assert situ_mod._SITU_TRITON_ENABLED is True
    _enable_situ_triton()  # idempotent: the second call returns early and keeps the flag
    assert situ_mod._SITU_TRITON_ENABLED is True


@needs_gpu_triton
@pytest.mark.parametrize("linear_beta", [None, LINEAR_BETA])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows,half", [(1, 8), (37, 100), (4099, 3072)])
@pytest.mark.runtime_budget(
    60,
    hard_timeout=300,
    reason="Compiles and autotunes the Triton SiTU kernels on a cold cache (up to 5.6 s measured on GB200).",
)
def test_weighted_forward_matches_eager(dtype, linear_beta, rows, half):
    torch.manual_seed(rows + half)
    gate_up = torch.randn(rows, 2 * half, dtype=dtype, device="cuda") * 3
    rw = torch.rand(rows, 1, dtype=torch.float32, device="cuda")
    out = situ_fwd_triton(gate_up, rw, BETA, linear_beta)
    expected = _eager_reference(gate_up, rw, BETA, linear_beta)
    assert out.shape == expected.shape and out.dtype == dtype
    torch.testing.assert_close(out, expected)


@needs_gpu_triton
@pytest.mark.parametrize("linear_beta", [None, LINEAR_BETA])
@pytest.mark.parametrize("rows,half", [(1, 8), (37, 100), (4099, 3072)])
@pytest.mark.runtime_budget(
    60,
    hard_timeout=300,
    reason="Compiles and autotunes the Triton SiTU kernels on a cold cache (up to 5.6 s measured on GB200).",
)
def test_weighted_backward_matches_autograd(linear_beta, rows, half):
    torch.manual_seed(100 + rows)
    gate_up = torch.randn(rows, 2 * half, dtype=torch.float32, device="cuda") * 3
    rw = torch.rand(rows, 1, dtype=torch.float32, device="cuda")
    grad_out = torch.randn(rows, half, dtype=torch.float32, device="cuda")
    d_gu, d_rw = situ_bwd_triton(gate_up, rw, grad_out, BETA, linear_beta, True)
    gu_e = gate_up.clone().requires_grad_()
    rw_e = rw.clone().requires_grad_()
    _eager_reference(gu_e, rw_e, BETA, linear_beta).backward(grad_out)
    torch.testing.assert_close(d_gu, gu_e.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(d_rw, rw_e.grad, rtol=1e-4, atol=1e-4)
    d_gu_only, none = situ_bwd_triton(gate_up, rw, grad_out, BETA, linear_beta, False)
    assert none is None
    torch.testing.assert_close(d_gu_only, d_gu)


@needs_gpu_triton
@pytest.mark.runtime_budget(
    60,
    hard_timeout=300,
    reason="Compiles and autotunes the Triton SiTU kernels on a cold cache (up to 5.6 s measured on GB200).",
)
def test_bf16_backward_matches_eager_chunked_function():
    torch.manual_seed(7)
    gate_up = torch.randn(2048, 128, dtype=torch.bfloat16, device="cuda") * 3
    rw = torch.rand(2048, 1, dtype=torch.float32, device="cuda")
    grad_out = torch.randn(2048, 64, dtype=torch.bfloat16, device="cuda")
    d_gu, d_rw = situ_bwd_triton(gate_up, rw, grad_out, BETA, LINEAR_BETA, True)
    gu_e = gate_up.clone().requires_grad_()
    rw_e = rw.clone().requires_grad_()
    _WeightedSiTUFunction.apply(gu_e, rw_e, BETA, LINEAR_BETA).backward(grad_out)  # eager chunk loop
    torch.testing.assert_close(d_gu, gu_e.grad)
    torch.testing.assert_close(d_rw, rw_e.grad, rtol=1e-4, atol=1e-4)


@needs_gpu_triton
@pytest.mark.parametrize("linear_beta", [None, LINEAR_BETA])
@pytest.mark.runtime_budget(
    60,
    hard_timeout=300,
    reason="Compiles and autotunes the Triton SiTU kernels on a cold cache (up to 5.6 s measured on GB200).",
)
def test_dense_kernels_match_core(linear_beta):
    torch.manual_seed(9)
    x = torch.randn(3, 517, 2 * 96, dtype=torch.bfloat16, device="cuda") * 3
    x2 = x.reshape(-1, x.shape[-1])
    out = situ_fwd_triton(x2, None, BETA, linear_beta).reshape(3, 517, 96)
    torch.testing.assert_close(out, _dense_situ_core(x, BETA, linear_beta))
    grad_out = torch.randn(3, 517, 96, dtype=torch.bfloat16, device="cuda")
    d_x, d_rw = situ_bwd_triton(x2, None, grad_out.reshape(-1, 96), BETA, linear_beta, True)
    assert d_rw is None  # no weights -> no weight gradient even when asked
    x_e = x.float().clone().requires_grad_()
    _dense_situ_core(x_e, BETA, linear_beta).backward(grad_out.float())
    torch.testing.assert_close(d_x.reshape(x.shape).float(), x_e.grad, rtol=2e-2, atol=2e-2)


@needs_gpu_triton
@pytest.mark.runtime_budget(
    60,
    hard_timeout=300,
    reason="Compiles and autotunes the Triton SiTU kernels on a cold cache (up to 5.6 s measured on GB200).",
)
def test_zero_rows_and_noncontiguous_grad():
    empty = torch.empty(0, 64, dtype=torch.bfloat16, device="cuda")
    rw0 = torch.empty(0, 1, dtype=torch.float32, device="cuda")
    assert situ_fwd_triton(empty, rw0, BETA, LINEAR_BETA).shape == (0, 32)
    d_gu, d_rw = situ_bwd_triton(empty, rw0, torch.empty(0, 32, device="cuda", dtype=torch.bfloat16), BETA, None, True)
    assert d_gu.shape == (0, 64) and d_rw.shape == (0, 1)
    gate_up = torch.randn(64, 32, dtype=torch.float32, device="cuda")
    rw = torch.rand(64, 1, dtype=torch.float32, device="cuda")
    wide = torch.randn(64, 40, dtype=torch.float32, device="cuda")
    # A column slice keeps unit stride along the last axis: the kernel follows the row stride.
    d_strided, _ = situ_bwd_triton(gate_up, rw, wide[:, :16], BETA, None, False)
    d_dense, _ = situ_bwd_triton(gate_up, rw, wide[:, :16].contiguous(), BETA, None, False)
    torch.testing.assert_close(d_strided, d_dense)
    with pytest.raises(ValueError):
        situ_bwd_triton(gate_up, rw, torch.randn(16, 64, device="cuda").t(), BETA, None, False)  # last stride != 1
    with pytest.raises(ValueError):
        situ_fwd_triton(gate_up.t(), None, BETA, None)
    with pytest.raises(ValueError):
        situ_bwd_triton(gate_up, rw, torch.randn(64, 8, device="cuda"), BETA, None, False)  # wrong grad shape


@needs_gpu_triton
@pytest.mark.runtime_budget(
    60,
    hard_timeout=300,
    reason="Compiles and autotunes the Triton SiTU kernels on a cold cache (up to 5.6 s measured on GB200).",
)
def test_function_and_dense_entry_route_through_triton(triton_enabled, monkeypatch):
    calls = {"fwd": 0, "bwd": 0}
    real_fwd, real_bwd = situ_mod.situ_fwd_triton, situ_mod.situ_bwd_triton

    def _fwd(*a, **k):
        calls["fwd"] += 1
        return real_fwd(*a, **k)

    def _bwd(*a, **k):
        calls["bwd"] += 1
        return real_bwd(*a, **k)

    monkeypatch.setattr(situ_mod, "situ_fwd_triton", _fwd)
    monkeypatch.setattr(situ_mod, "situ_bwd_triton", _bwd)
    torch.manual_seed(25)
    gate_up = torch.randn(4096, 64, dtype=torch.bfloat16, device="cuda")
    rw = torch.rand(4096, 1, dtype=torch.float32, device="cuda")
    grad_out = torch.randn(4096, 32, dtype=torch.bfloat16, device="cuda")
    gu_c = gate_up.clone().requires_grad_()
    rw_c = rw.clone().requires_grad_()
    out = _WeightedSiTUFunction.apply(gu_c, rw_c, BETA, LINEAR_BETA)
    gu_e = gate_up.clone().requires_grad_()
    rw_e = rw.clone().requires_grad_()
    expected = _eager_reference(gu_e, rw_e, BETA, LINEAR_BETA)
    torch.testing.assert_close(out, expected)
    out.backward(grad_out)
    expected.backward(grad_out)
    torch.testing.assert_close(gu_c.grad, gu_e.grad)
    torch.testing.assert_close(rw_c.grad, rw_e.grad, rtol=1e-3, atol=1e-3)
    # 3-D row-aligned weights with k == 1 also take the kernels; broadcast [1, 1] weights do not.
    gu3 = torch.randn(2, 8, 64, dtype=torch.bfloat16, device="cuda")
    rw3 = torch.rand(2, 8, 1, dtype=torch.float32, device="cuda")
    torch.testing.assert_close(
        _WeightedSiTUFunction.apply(gu3, rw3, BETA, None), _eager_reference(gu3, rw3, BETA, None)
    )
    n_before = calls["fwd"]
    rw11 = torch.rand(1, 1, dtype=torch.float32, device="cuda")
    _WeightedSiTUFunction.apply(gate_up, rw11, BETA, None)
    assert calls["fwd"] == n_before
    # Dense entry (SituAndMul) saves only the input and returns the input dtype.
    x = torch.randn(8, 4096, 64, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    y = dense_situ(x, BETA, LINEAR_BETA)
    torch.testing.assert_close(y, _dense_situ_core(x, BETA, LINEAR_BETA))
    y.sum().backward()
    x_e = x.detach().float().requires_grad_()
    _dense_situ_core(x_e, BETA, LINEAR_BETA).sum().backward()
    torch.testing.assert_close(x.grad.float(), x_e.grad, rtol=2e-2, atol=2e-2)
    assert calls["fwd"] >= 3 and calls["bwd"] >= 2

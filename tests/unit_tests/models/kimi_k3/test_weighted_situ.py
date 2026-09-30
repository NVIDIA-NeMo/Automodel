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

import pytest
import torch

import nemo_automodel.components.models.kimi_k3.situ as kimi_k3_model
from nemo_automodel.components.models.kimi_k3.situ import (
    _SITU_CHUNK_THRESHOLD,
    _compile_situ_cores,
    _dense_situ_core,
    _weighted_situ,
    _weighted_situ_bwd_fused,
    _weighted_situ_fwd_fused,
    _WeightedSiTUFunction,
    dense_situ,
)

BETA = 1.702
LINEAR_BETA = 0.75


def _eager_reference(
    gate_up: torch.Tensor,
    routing_weights: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    """Plain-autograd fp32 weighted SiTU, the trusted reference.

    Args:
        gate_up: Gate+up projections of shape [..., 2 * intermediate].
        routing_weights: Routing weights broadcastable against
            ``gate_up.shape[:-1] + [intermediate]``, typically [tokens, 1].
        beta: SiTU beta applied to the gate branch.
        linear_beta: Optional bounded-linear beta applied to the up branch.

    Returns:
        Tensor of the broadcast output shape in ``gate_up``'s dtype.
    """
    input_dtype = gate_up.dtype
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    activated = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (activated * up * routing_weights.float()).to(input_dtype)


@pytest.mark.parametrize("linear_beta", [None, LINEAR_BETA])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_is_bitwise_identical_to_eager(dtype, linear_beta):
    torch.manual_seed(0)
    gate_up = torch.randn(64, 32, dtype=dtype)
    routing_weights = torch.rand(64, 1, dtype=torch.float32)

    actual = _WeightedSiTUFunction.apply(gate_up, routing_weights, BETA, linear_beta)
    expected = _eager_reference(gate_up, routing_weights, BETA, linear_beta)

    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("linear_beta", [None, LINEAR_BETA])
def test_backward_matches_autograd_reference(linear_beta):
    torch.manual_seed(1)
    gate_up = torch.randn(64, 32, dtype=torch.float32)
    routing_weights = torch.rand(64, 1, dtype=torch.float32)
    grad_out = torch.randn(64, 16, dtype=torch.float32)

    gu_c = gate_up.clone().requires_grad_()
    rw_c = routing_weights.clone().requires_grad_()
    _WeightedSiTUFunction.apply(gu_c, rw_c, BETA, linear_beta).backward(grad_out)

    gu_e = gate_up.clone().requires_grad_()
    rw_e = routing_weights.clone().requires_grad_()
    _eager_reference(gu_e, rw_e, BETA, linear_beta).backward(grad_out)

    torch.testing.assert_close(gu_c.grad, gu_e.grad, rtol=0.0, atol=5e-7)
    torch.testing.assert_close(rw_c.grad, rw_e.grad, rtol=0.0, atol=5e-7)


def test_multi_chunk_forward_and_backward_match_eager(monkeypatch):
    """Force several chunks per call so chunk-boundary handling is exercised."""
    monkeypatch.setattr(kimi_k3_model, "_SITU_CHUNK_ROWS", 7)
    torch.manual_seed(2)
    gate_up = torch.randn(30, 16, dtype=torch.float32)
    routing_weights = torch.rand(30, 1, dtype=torch.float32)
    grad_out = torch.randn(30, 8, dtype=torch.float32)

    gu_c = gate_up.clone().requires_grad_()
    rw_c = routing_weights.clone().requires_grad_()
    out = _WeightedSiTUFunction.apply(gu_c, rw_c, BETA, LINEAR_BETA)

    gu_e = gate_up.clone().requires_grad_()
    rw_e = routing_weights.clone().requires_grad_()
    expected = _eager_reference(gu_e, rw_e, BETA, LINEAR_BETA)

    assert torch.equal(out, expected)
    out.backward(grad_out)
    expected.backward(grad_out)
    torch.testing.assert_close(gu_c.grad, gu_e.grad, rtol=0.0, atol=5e-7)
    torch.testing.assert_close(rw_c.grad, rw_e.grad, rtol=0.0, atol=5e-7)


def test_dispatch_routes_large_inputs_through_chunked_function():
    torch.manual_seed(3)
    rows = _SITU_CHUNK_THRESHOLD + 1
    gate_up = torch.randn(rows, 8, dtype=torch.bfloat16)
    routing_weights = torch.rand(rows, 1, dtype=torch.float32)

    saved: list[torch.Tensor] = []

    def _pack(tensor: torch.Tensor) -> torch.Tensor:
        saved.append(tensor)
        return tensor

    gu = gate_up.clone().requires_grad_()
    with torch.autograd.graph.saved_tensors_hooks(_pack, lambda t: t):
        out = _weighted_situ(gu, routing_weights, beta=BETA, linear_beta=None)

    # The chunked Function saves exactly the two inputs, in their original
    # dtypes -- no full-size fp32 intermediates are kept for backward.
    assert len(saved) == 2
    assert {t.dtype for t in saved} == {torch.bfloat16, torch.float32}

    expected = _eager_reference(gate_up, routing_weights, BETA, None)
    assert torch.equal(out, expected)

    out.backward(torch.randn_like(out))
    assert gu.grad is not None
    assert torch.isfinite(gu.grad).all()


def test_small_inputs_keep_eager_path():
    torch.manual_seed(4)
    gate_up = torch.randn(8, 8, dtype=torch.float32, requires_grad=True)
    routing_weights = torch.rand(8, 1, dtype=torch.float32)

    out = _weighted_situ(gate_up, routing_weights, beta=BETA, linear_beta=None)

    assert torch.equal(out, _eager_reference(gate_up.detach(), routing_weights, BETA, None))
    assert out.grad_fn is not None
    assert "WeightedSiTU" not in type(out.grad_fn).__name__


def test_one_dimensional_dummy_path_broadcast():
    """experts.py's zero-token dummy path passes 1-D gate_up x [1, 1] weights."""
    torch.manual_seed(5)
    gate_up = torch.randn(16, dtype=torch.float32)
    routing_weights = torch.ones(1, 1, dtype=torch.float32)

    out = _WeightedSiTUFunction.apply(gate_up, routing_weights, BETA, LINEAR_BETA)
    expected = _eager_reference(gate_up, routing_weights, BETA, LINEAR_BETA)

    assert out.shape == (1, 8)
    assert torch.equal(out, expected)


def test_zero_row_input():
    gate_up = torch.zeros(0, 16, dtype=torch.float32, requires_grad=True)
    routing_weights = torch.zeros(0, 1, dtype=torch.float32)

    out = _WeightedSiTUFunction.apply(gate_up, routing_weights, BETA, None)

    assert out.shape == (0, 8)
    out.backward(torch.zeros_like(out))
    assert gate_up.grad.shape == gate_up.shape


def test_three_dimensional_row_aligned_input():
    torch.manual_seed(6)
    gate_up = torch.randn(2, 5, 16, dtype=torch.float32)
    routing_weights = torch.rand(2, 5, 1, dtype=torch.float32)
    grad_out = torch.randn(2, 5, 8, dtype=torch.float32)

    gu_c = gate_up.clone().requires_grad_()
    rw_c = routing_weights.clone().requires_grad_()
    out = _WeightedSiTUFunction.apply(gu_c, rw_c, BETA, None)

    gu_e = gate_up.clone().requires_grad_()
    rw_e = routing_weights.clone().requires_grad_()
    expected = _eager_reference(gu_e, rw_e, BETA, None)

    assert torch.equal(out, expected)
    out.backward(grad_out)
    expected.backward(grad_out)
    torch.testing.assert_close(gu_c.grad, gu_e.grad, rtol=0.0, atol=5e-7)
    torch.testing.assert_close(rw_c.grad, rw_e.grad, rtol=0.0, atol=5e-7)


def test_broadcast_non_row_aligned_weights_gradient():
    """Non-row-aligned weights take the fp32 accumulator branch in backward."""
    torch.manual_seed(7)
    gate_up = torch.randn(6, 16, dtype=torch.float32)
    routing_weights = torch.rand(1, 1, dtype=torch.float32)
    grad_out = torch.randn(6, 8, dtype=torch.float32)

    gu_c = gate_up.clone().requires_grad_()
    rw_c = routing_weights.clone().requires_grad_()
    out = _WeightedSiTUFunction.apply(gu_c, rw_c, BETA, LINEAR_BETA)

    gu_e = gate_up.clone().requires_grad_()
    rw_e = routing_weights.clone().requires_grad_()
    expected = _eager_reference(gu_e, rw_e, BETA, LINEAR_BETA)

    assert torch.equal(out, expected)
    out.backward(grad_out)
    expected.backward(grad_out)
    torch.testing.assert_close(gu_c.grad, gu_e.grad, rtol=0.0, atol=5e-7)
    torch.testing.assert_close(rw_c.grad, rw_e.grad, rtol=1e-6, atol=5e-6)


@pytest.mark.parametrize(
    ("gate_up_grad", "weights_grad"),
    [(True, False), (False, True), (True, True)],
)
def test_needs_input_grad_combinations(gate_up_grad, weights_grad):
    torch.manual_seed(8)
    gate_up = torch.randn(10, 16, dtype=torch.float32, requires_grad=gate_up_grad)
    routing_weights = torch.rand(10, 1, dtype=torch.float32, requires_grad=weights_grad)

    out = _WeightedSiTUFunction.apply(gate_up, routing_weights, BETA, None)
    out.backward(torch.randn_like(out))

    assert (gate_up.grad is not None) == gate_up_grad
    assert (routing_weights.grad is not None) == weights_grad


# --------------------------------------------------------------------------------------
# Compiled (fused whole-tensor) path of _WeightedSiTUFunction and the dense SiTU core
# --------------------------------------------------------------------------------------

_SITU_STATE_ATTRS = (
    "_situ_fwd_core",
    "_situ_bwd_core",
    "_attn_res_core",
    "_weighted_situ_fwd_fused_dispatch",
    "_weighted_situ_bwd_fused_dispatch",
    "_dense_situ_dispatch",
    "_SITU_CORES_COMPILED",
)


@pytest.fixture
def restore_situ_state():
    """Snapshot and restore every module-level SiTU dispatch target around a test."""
    snapshot = {name: getattr(kimi_k3_model, name) for name in _SITU_STATE_ATTRS}
    yield
    for name, value in snapshot.items():
        setattr(kimi_k3_model, name, value)


def _mark_compiled_with_eager_fused(monkeypatch) -> dict[str, int]:
    """Route the Function through the fused whole-tensor passes without invoking inductor."""
    calls = {"fwd": 0, "bwd": 0}

    def _fwd(*args, **kwargs):
        calls["fwd"] += 1
        return _weighted_situ_fwd_fused(*args, **kwargs)

    def _bwd(*args, **kwargs):
        calls["bwd"] += 1
        return _weighted_situ_bwd_fused(*args, **kwargs)

    monkeypatch.setattr(kimi_k3_model, "_weighted_situ_fwd_fused_dispatch", _fwd)
    monkeypatch.setattr(kimi_k3_model, "_weighted_situ_bwd_fused_dispatch", _bwd)
    monkeypatch.setattr(kimi_k3_model, "_SITU_CORES_COMPILED", True)
    monkeypatch.setattr(kimi_k3_model, "_SITU_CHUNK_ROWS", 7)  # the eager path would chunk
    return calls


@pytest.mark.parametrize("linear_beta", [None, LINEAR_BETA])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_path_forward_matches_eager(restore_situ_state, monkeypatch, dtype, linear_beta):
    calls = _mark_compiled_with_eager_fused(monkeypatch)
    torch.manual_seed(20)
    gate_up = torch.randn(30, 32, dtype=dtype)
    routing_weights = torch.rand(30, 1, dtype=torch.float32)
    actual = _WeightedSiTUFunction.apply(gate_up, routing_weights, BETA, linear_beta)
    expected = _eager_reference(gate_up, routing_weights, BETA, linear_beta)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)
    assert calls == {"fwd": 1, "bwd": 0}


@pytest.mark.parametrize("linear_beta", [None, LINEAR_BETA])
def test_fused_path_backward_matches_autograd_reference(restore_situ_state, monkeypatch, linear_beta):
    calls = _mark_compiled_with_eager_fused(monkeypatch)
    torch.manual_seed(21)
    gate_up = torch.randn(30, 32, dtype=torch.float32)
    routing_weights = torch.rand(30, 1, dtype=torch.float32)
    grad_out = torch.randn(30, 16, dtype=torch.float32)
    gu_c = gate_up.clone().requires_grad_()
    rw_c = routing_weights.clone().requires_grad_()
    _WeightedSiTUFunction.apply(gu_c, rw_c, BETA, linear_beta).backward(grad_out)
    gu_e = gate_up.clone().requires_grad_()
    rw_e = routing_weights.clone().requires_grad_()
    _eager_reference(gu_e, rw_e, BETA, linear_beta).backward(grad_out)
    torch.testing.assert_close(gu_c.grad, gu_e.grad, rtol=0.0, atol=5e-7)
    torch.testing.assert_close(rw_c.grad, rw_e.grad, rtol=0.0, atol=5e-7)
    assert calls == {"fwd": 1, "bwd": 1}


def test_fused_path_broadcast_weights_and_three_dimensional_input(restore_situ_state, monkeypatch):
    _mark_compiled_with_eager_fused(monkeypatch)
    torch.manual_seed(22)
    # Non-row-aligned [1, 1] weights (fp32 reduction to the weight shape).
    gate_up = torch.randn(6, 16, dtype=torch.float32)
    routing_weights = torch.rand(1, 1, dtype=torch.float32)
    grad_out = torch.randn(6, 8, dtype=torch.float32)
    gu_c = gate_up.clone().requires_grad_()
    rw_c = routing_weights.clone().requires_grad_()
    out = _WeightedSiTUFunction.apply(gu_c, rw_c, BETA, LINEAR_BETA)
    gu_e = gate_up.clone().requires_grad_()
    rw_e = routing_weights.clone().requires_grad_()
    expected = _eager_reference(gu_e, rw_e, BETA, LINEAR_BETA)
    assert torch.equal(out, expected)
    out.backward(grad_out)
    expected.backward(grad_out)
    torch.testing.assert_close(gu_c.grad, gu_e.grad, rtol=0.0, atol=5e-7)
    torch.testing.assert_close(rw_c.grad, rw_e.grad, rtol=1e-6, atol=5e-6)
    # Row-aligned 3-D input keeps its leading dimensions.
    gate_up = torch.randn(2, 5, 16, dtype=torch.float32)
    routing_weights = torch.rand(2, 5, 1, dtype=torch.float32)
    grad_out = torch.randn(2, 5, 8, dtype=torch.float32)
    gu_c = gate_up.clone().requires_grad_()
    rw_c = routing_weights.clone().requires_grad_()
    out = _WeightedSiTUFunction.apply(gu_c, rw_c, BETA, None)
    gu_e = gate_up.clone().requires_grad_()
    rw_e = routing_weights.clone().requires_grad_()
    expected = _eager_reference(gu_e, rw_e, BETA, None)
    assert out.shape == (2, 5, 8)
    assert torch.equal(out, expected)
    out.backward(grad_out)
    expected.backward(grad_out)
    torch.testing.assert_close(gu_c.grad, gu_e.grad, rtol=0.0, atol=5e-7)
    torch.testing.assert_close(rw_c.grad, rw_e.grad, rtol=0.0, atol=5e-7)


def test_fused_path_one_dimensional_dummy_and_zero_rows(restore_situ_state, monkeypatch):
    _mark_compiled_with_eager_fused(monkeypatch)
    torch.manual_seed(23)
    gate_up = torch.randn(16, dtype=torch.float32)
    routing_weights = torch.ones(1, 1, dtype=torch.float32)
    out = _WeightedSiTUFunction.apply(gate_up, routing_weights, BETA, LINEAR_BETA)
    assert out.shape == (1, 8)
    assert torch.equal(out, _eager_reference(gate_up, routing_weights, BETA, LINEAR_BETA))
    # Zero-row inputs stay on the eager path (dynamic-shape compile is skipped).
    gate_up = torch.zeros(0, 16, dtype=torch.float32, requires_grad=True)
    routing_weights = torch.zeros(0, 1, dtype=torch.float32)
    out = _WeightedSiTUFunction.apply(gate_up, routing_weights, BETA, None)
    assert out.shape == (0, 8)
    out.backward(torch.zeros_like(out))
    assert gate_up.grad.shape == gate_up.shape


@pytest.mark.parametrize("linear_beta", [None, LINEAR_BETA])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dense_situ_core_matches_module_math(dtype, linear_beta):
    """The dense core is the SituAndMul math verbatim (chunk, fp32 chain, cast back)."""
    torch.manual_seed(24)
    x = torch.randn(4, 6, 32, dtype=dtype)
    gate, up = x.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    activated = BETA * torch.tanh(gate / BETA) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    expected = (activated * up).to(x.dtype)
    assert torch.equal(_dense_situ_core(x, BETA, linear_beta), expected)
    assert torch.equal(dense_situ(x, BETA, linear_beta), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled execution needs a GPU and inductor")
@pytest.mark.runtime_budget(
    60,
    hard_timeout=300,
    reason="torch.compile of the SiTU cores on first use.",
)
def test_compiled_execution_matches_eager_on_gpu(restore_situ_state):
    kimi_k3_model._SITU_CORES_COMPILED = False
    _compile_situ_cores()
    torch.manual_seed(25)
    gate_up = torch.randn(4096, 64, dtype=torch.bfloat16, device="cuda")
    routing_weights = torch.rand(4096, 1, dtype=torch.float32, device="cuda")
    grad_out = torch.randn(4096, 32, dtype=torch.bfloat16, device="cuda")
    gu_c = gate_up.clone().requires_grad_()
    rw_c = routing_weights.clone().requires_grad_()
    out = _WeightedSiTUFunction.apply(gu_c, rw_c, BETA, LINEAR_BETA)
    gu_e = gate_up.clone().requires_grad_()
    rw_e = routing_weights.clone().requires_grad_()
    expected = _eager_reference(gu_e, rw_e, BETA, LINEAR_BETA)
    torch.testing.assert_close(out, expected)
    out.backward(grad_out)
    expected.backward(grad_out)
    torch.testing.assert_close(gu_c.grad, gu_e.grad)
    torch.testing.assert_close(rw_c.grad, rw_e.grad, rtol=1e-3, atol=1e-3)
    x = torch.randn(8, 4096, 64, dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(dense_situ(x, BETA, LINEAR_BETA), _dense_situ_core(x, BETA, LINEAR_BETA))

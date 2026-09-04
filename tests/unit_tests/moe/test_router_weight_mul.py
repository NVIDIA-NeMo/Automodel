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

import nemo_automodel.components.moe.experts as _experts_unused  # noqa: F401
import nemo_automodel.components.moe.optimized_ops as experts_mod
from nemo_automodel.components.moe.optimized_ops import (
    _RW_CHUNK_THRESHOLD,
    _apply_router_weight_fp32,
    _RouterWeightMulFunction,
)


def _eager_reference(x: torch.Tensor, probs: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """Plain-autograd fp32 router-weight multiply, the trusted reference.

    Args:
        x: Expert outputs of shape [tokens, hidden].
        probs: Routing probabilities of shape [tokens, 1].
        out_dtype: Output dtype.

    Returns:
        Tensor of shape [tokens, hidden] and dtype ``out_dtype``.
    """
    return (x.float() * probs.float()).to(out_dtype)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float32])
def test_forward_is_bitwise_identical_to_eager(x_dtype, out_dtype):
    torch.manual_seed(0)
    x = torch.randn(64, 32, dtype=x_dtype)
    probs = torch.rand(64, 1, dtype=torch.float32)

    actual = _RouterWeightMulFunction.apply(x, probs, out_dtype, False)
    expected = _eager_reference(x, probs, out_dtype)

    assert actual.dtype == out_dtype
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_backward_matches_autograd_reference(out_dtype):
    torch.manual_seed(1)
    x = torch.randn(64, 32, dtype=torch.float32)
    probs = torch.rand(64, 1, dtype=torch.float32)
    grad_out = torch.randn(64, 32, dtype=out_dtype)

    x_c = x.clone().requires_grad_()
    p_c = probs.clone().requires_grad_()
    _RouterWeightMulFunction.apply(x_c, p_c, out_dtype, True).backward(grad_out)

    x_e = x.clone().requires_grad_()
    p_e = probs.clone().requires_grad_()
    _eager_reference(x_e, p_e, out_dtype).backward(grad_out)

    torch.testing.assert_close(x_c.grad, x_e.grad, rtol=0.0, atol=5e-7)
    torch.testing.assert_close(p_c.grad, p_e.grad, rtol=1e-6, atol=5e-6)


def test_multi_chunk_forward_and_backward_match_eager(monkeypatch):
    """Force several chunks per call so chunk-boundary handling is exercised."""
    monkeypatch.setattr(experts_mod, "_RW_CHUNK_ROWS", 7)
    torch.manual_seed(2)
    x = torch.randn(30, 16, dtype=torch.float32)
    probs = torch.rand(30, 1, dtype=torch.float32)
    grad_out = torch.randn(30, 16, dtype=torch.float32)

    x_c = x.clone().requires_grad_()
    p_c = probs.clone().requires_grad_()
    out = _RouterWeightMulFunction.apply(x_c, p_c, torch.float32, True)

    x_e = x.clone().requires_grad_()
    p_e = probs.clone().requires_grad_()
    expected = _eager_reference(x_e, p_e, torch.float32)

    assert torch.equal(out, expected)
    out.backward(grad_out)
    expected.backward(grad_out)
    torch.testing.assert_close(x_c.grad, x_e.grad, rtol=0.0, atol=5e-7)
    torch.testing.assert_close(p_c.grad, p_e.grad, rtol=1e-6, atol=5e-6)


def test_apply_router_weight_fp32_routes_large_inputs_through_function():
    torch.manual_seed(3)
    rows = _RW_CHUNK_THRESHOLD + 1
    x = torch.randn(rows, 8, dtype=torch.bfloat16, requires_grad=True)
    probs = torch.rand(rows, 1, dtype=torch.float32)

    out = _apply_router_weight_fp32(x, probs, torch.bfloat16)

    assert type(out.grad_fn).__name__.startswith(("_RouterWeightMulFunction", "_TritonRouterWeightMulFunction"))
    assert torch.equal(out, _eager_reference(x.detach(), probs, torch.bfloat16))

    out.backward(torch.randn_like(out))
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_apply_router_weight_fp32_small_inputs_keep_eager_path():
    torch.manual_seed(4)
    x = torch.randn(8, 8, dtype=torch.float32, requires_grad=True)
    probs = torch.rand(8, 1, dtype=torch.float32)

    out = _apply_router_weight_fp32(x, probs, torch.float32)

    assert not type(out.grad_fn).__name__.startswith(("_RouterWeightMulFunction", "_TritonRouterWeightMulFunction"))
    assert torch.equal(out, _eager_reference(x.detach(), probs, torch.float32))


def test_apply_router_weight_fp32_broadcast_fallback():
    """Non [tokens, 1] probs shapes must keep the eager broadcast semantics."""
    torch.manual_seed(5)
    x = torch.randn(20000, 4, dtype=torch.float32)
    probs = torch.rand(1, 4, dtype=torch.float32)

    out = _apply_router_weight_fp32(x, probs, torch.float32)

    assert torch.equal(out, _eager_reference(x, probs, torch.float32))


def test_zero_row_input():
    x = torch.zeros(0, 8, dtype=torch.float32, requires_grad=True)
    probs = torch.zeros(0, 1, dtype=torch.float32)

    out = _RouterWeightMulFunction.apply(x, probs, torch.float32, False)

    assert out.shape == (0, 8)
    out.backward(torch.zeros_like(out))
    assert x.grad.shape == x.shape


def test_saves_x_only_when_probs_needs_grad():
    torch.manual_seed(6)
    x = torch.randn(16, 8, dtype=torch.bfloat16, requires_grad=True)

    for probs_requires_grad, expected_saved in ((False, 1), (True, 2)):
        probs = torch.rand(16, 1, dtype=torch.float32, requires_grad=probs_requires_grad)
        saved: list[torch.Tensor] = []

        def _pack(tensor: torch.Tensor) -> torch.Tensor:
            saved.append(tensor)
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(_pack, lambda t: t):
            out = _RouterWeightMulFunction.apply(x, probs, torch.bfloat16, probs.requires_grad)

        # With grad-free probs (e.g. FakeBalancedGate) x is not saved, so no
        # full-size [tokens, hidden] tensor is pinned for backward.
        assert len(saved) == expected_saved, f"probs_requires_grad={probs_requires_grad}"
        out.backward(torch.randn_like(out))
        if probs_requires_grad:
            assert probs.grad is not None
        assert x.grad is not None
        x.grad = None


def test_grad_only_for_probs():
    torch.manual_seed(7)
    x = torch.randn(16, 8, dtype=torch.float32)
    probs = torch.rand(16, 1, dtype=torch.float32)
    grad_out = torch.randn(16, 8, dtype=torch.float32)

    p_e = probs.clone().requires_grad_()
    _eager_reference(x, p_e, torch.float32).backward(grad_out)

    p_c = probs.clone().requires_grad_()
    out = _RouterWeightMulFunction.apply(x, p_c, torch.float32, True)
    out.backward(grad_out)

    assert p_c.grad is not None
    torch.testing.assert_close(p_c.grad, p_e.grad, rtol=1e-6, atol=5e-6)


@pytest.mark.skipif(
    not (getattr(experts_mod, "_TRITON_ROUTER_WEIGHT_AVAILABLE", False) and torch.cuda.is_available()),
    reason="Triton + CUDA required",
)
@pytest.mark.parametrize("tokens,hidden", [(1024, 2048), (16384, 2048), (16384, 4096), (16384, 3584)])
@pytest.mark.parametrize("compute_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_triton_router_weight_forward_backward_parity(tokens: int, hidden: int, compute_dtype: torch.dtype) -> None:
    """Test parity of Triton kernel against eager reference on GPU.

    Args:
        tokens: Number of routed tokens.
        hidden: Hidden dimension.
        compute_dtype: Output computation dtype.
    """
    device = "cuda"
    x = torch.randn(tokens, hidden, dtype=compute_dtype, device=device, requires_grad=True)
    probs = torch.rand(tokens, 1, dtype=torch.float32, device=device, requires_grad=True)

    # Reference
    ref_out = _eager_reference(x, probs, compute_dtype)
    loss_ref = (ref_out.float() * 0.5).sum()
    loss_ref.backward()
    grad_x_ref = x.grad.clone()
    grad_p_ref = probs.grad.clone()

    # Triton Path
    x_tri = x.detach().clone().requires_grad_(True)
    probs_tri = probs.detach().clone().requires_grad_(True)

    tri_out = _apply_router_weight_fp32(x_tri, probs_tri, compute_dtype)
    loss_tri = (tri_out.float() * 0.5).sum()
    loss_tri.backward()

    # Check forward
    torch.testing.assert_close(tri_out, ref_out, rtol=1e-3, atol=1e-3)
    # Check backward
    torch.testing.assert_close(x_tri.grad, grad_x_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(probs_tri.grad, grad_p_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.skipif(
    not (getattr(experts_mod, "_TRITON_ROUTER_WEIGHT_AVAILABLE", False) and torch.cuda.is_available()),
    reason="Triton + CUDA required",
)
def test_triton_save_x_false() -> None:
    """Verify save_x=False optimization when probs does not require grad."""
    tokens, hidden = 16384, 2048
    device = "cuda"
    x = torch.randn(tokens, hidden, dtype=torch.bfloat16, device=device, requires_grad=True)
    probs = torch.rand(tokens, 1, dtype=torch.float32, device=device, requires_grad=False)

    tri_out = _apply_router_weight_fp32(x, probs, torch.bfloat16)
    loss = tri_out.float().sum()
    loss.backward()

    assert x.grad is not None
    assert probs.grad is None

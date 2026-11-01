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
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.moe import swiglu_triton
from nemo_automodel.components.moe.experts import (
    _swiglu_clamped_deepep_compiled,
    swiglu_clamped_deepep,
)

pytestmark = pytest.mark.timeout(600)
requires_gpu_triton = pytest.mark.skipif(
    not torch.cuda.is_available() or not swiglu_triton.HAVE_TRITON, reason="CUDA and Triton are required"
)


def _reference(x, probs, limit):
    gate, up = torch.chunk(x, 2, dim=-1)
    gate = gate.float().clamp(max=limit)
    up = up.float().clamp(min=-limit, max=limit)
    return (F.silu(gate) * up * probs).to(x.dtype)


def _inputs(rows, inter, limit, device, seed=0):
    torch.manual_seed(seed)
    x = (torch.randn(rows, 2 * inter, device=device) * limit * 1.5).to(torch.bfloat16).requires_grad_(True)
    probs = torch.rand(rows, 1, device=device, dtype=torch.float32).requires_grad_(True)
    grad = torch.randn(rows, inter, device=device).to(torch.bfloat16)
    return x, probs, grad


@requires_gpu_triton
@pytest.mark.parametrize("rows,inter,limit", [(37, 96, 2.0), (1024, 2304, 7.0), (5, 1000, 3.0)])
def test_triton_matches_the_eager_reference(rows, inter, limit):
    x, probs, grad = _inputs(rows, inter, limit, "cuda")
    out = swiglu_triton.swiglu_clamped_triton(x, probs, limit)
    out.backward(grad)
    x_ref = x.detach().clone().requires_grad_(True)
    probs_ref = probs.detach().clone().requires_grad_(True)
    ref = _reference(x_ref, probs_ref, limit)
    ref.backward(grad)
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(x.grad.float(), x_ref.grad.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(probs.grad, probs_ref.grad, rtol=1e-3, atol=1e-3)
    # clamp masks
    gate_raw, up_raw = torch.chunk(x.detach().float(), 2, dim=-1)
    grad_gate, grad_up = torch.chunk(x.grad.float(), 2, dim=-1)
    assert torch.all(grad_gate[gate_raw > limit] == 0)
    assert torch.all(grad_up[up_raw.abs() > limit] == 0)


@requires_gpu_triton
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_clamp_boundary_gradients_match_the_installed_pytorch(dtype: torch.dtype) -> None:
    """Follow the framework's scalar-clamp subgradient at both exact bounds."""
    gate = torch.tensor([[-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]], device="cuda", dtype=dtype)
    x = torch.cat((gate, gate.flip(-1)), dim=-1).requires_grad_()
    probs = torch.tensor([[0.7]], device="cuda", requires_grad=True)
    actual = swiglu_triton.swiglu_clamped_triton(x, probs, 2.0)
    expected = _reference(x, probs, 2.0)
    actual_grads = torch.autograd.grad(actual, (x, probs), torch.ones_like(actual))
    expected_grads = torch.autograd.grad(expected, (x, probs), torch.ones_like(expected))
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    for grad, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(grad, reference, rtol=1e-2, atol=1e-2)


@requires_gpu_triton
def test_dispatcher_prefers_triton_on_cuda_and_matches_the_compiled_function():
    x, probs, _ = _inputs(16, 64, 2.0, "cuda")
    out = swiglu_clamped_deepep(x, probs, 2.0)
    assert type(out.grad_fn).__name__.startswith("SwigluClampedTriton")
    ref = _swiglu_clamped_deepep_compiled(x.detach().clone(), probs.detach().clone(), 2.0)
    assert not type(ref.grad_fn).__name__.startswith("SwigluClampedTriton") if ref.grad_fn is not None else True
    torch.testing.assert_close(out.float(), ref.float(), rtol=1e-2, atol=1e-2)


@requires_gpu_triton
def test_probability_gradient_is_skipped_when_not_required():
    x, probs, grad = _inputs(8, 32, 2.0, "cuda")
    probs = probs.detach()
    out = swiglu_triton.swiglu_clamped_triton(x, probs, 2.0)
    out.backward(grad)
    assert x.grad is not None and probs.grad is None


def test_cpu_inputs_take_the_compiled_path():
    # Isolate this fixed-limit dispatch check from earlier scalar specializations.
    torch._dynamo.reset()
    x, probs, grad = _inputs(6, 8, 2.0, "cpu")
    assert not swiglu_triton.uses_triton_kernel(x, probs)
    out = swiglu_clamped_deepep(x, probs, 2.0)
    ref = _swiglu_clamped_deepep_compiled(x.detach().clone(), probs.detach().clone(), 2.0)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(out, _reference(x, probs, 2.0), rtol=0, atol=0)
    out.backward(grad)
    assert x.grad is not None and probs.grad is not None


def test_guard_rejects_unsupported_layouts():
    x = torch.zeros(4, 8, dtype=torch.bfloat16)
    probs = torch.zeros(4, 1)
    assert not swiglu_triton.uses_triton_kernel(x, probs)  # CPU
    if torch.cuda.is_available() and swiglu_triton.HAVE_TRITON:
        x = x.cuda()
        probs = probs.cuda()
        assert swiglu_triton.uses_triton_kernel(x, probs)
        assert not swiglu_triton.uses_triton_kernel(x.t(), probs)  # non-contiguous
        assert not swiglu_triton.uses_triton_kernel(x[:, :7], probs)  # odd width
        assert not swiglu_triton.uses_triton_kernel(x, probs.to(torch.bfloat16))  # probs dtype
        assert not swiglu_triton.uses_triton_kernel(x, probs[:3])  # one probability per row
    assert isinstance(swiglu_triton.HAVE_TRITON, bool)


class _Expert(nn.Module):
    """A projection followed by the routed activation: what autocast and the compiler see in training."""

    def __init__(self, act, dtype: torch.dtype, device: str, hidden: int = 32, inter: int = 48):
        super().__init__()
        self.w = nn.Linear(hidden, 2 * inter, bias=False, dtype=dtype, device=device)
        self.act = act

    def forward(self, x, probs):
        return self.act(self.w(x).contiguous(), probs, 2.0)


@requires_gpu_triton
@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
def test_autocast_with_fp32_parameters_matches_the_eager_reference(input_dtype):
    """AMP training: FP32 parameters, BF16 or FP32 activations under autocast — the Triton path must return what the
    eager chain returns (dtype, values, gradient dtypes); it is not an autocast op, so the projection's dtype rules."""
    torch.manual_seed(1)
    pr = _Expert(swiglu_clamped_deepep, torch.float32, "cuda")
    ref = _Expert(_reference, torch.float32, "cuda")
    ref.load_state_dict(pr.state_dict())
    x = (torch.randn(64, 32, device="cuda") * 2).to(input_dtype).requires_grad_(True)
    probs = torch.rand(64, 1, device="cuda").requires_grad_(True)
    x_ref, probs_ref = x.detach().clone().requires_grad_(True), probs.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = pr(x, probs)
        out_ref = ref(x_ref, probs_ref)
    assert out.dtype == out_ref.dtype == torch.bfloat16
    assert type(out.grad_fn).__name__.startswith("SwigluClampedTriton")
    g = torch.randn_like(out_ref)
    gx, gw, gp = torch.autograd.grad(out, (x, pr.w.weight, probs), g)
    gx_ref, gw_ref, gp_ref = torch.autograd.grad(out_ref, (x_ref, ref.w.weight, probs_ref), g)
    assert (
        (gx.dtype, gw.dtype, gp.dtype)
        == (gx_ref.dtype, gw_ref.dtype, gp_ref.dtype)
        == (input_dtype, torch.float32, torch.float32)
    )
    torch.testing.assert_close(out.float(), out_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(gx.float(), gx_ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(gw, gw_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(gp, gp_ref, rtol=1e-2, atol=1e-2)


@requires_gpu_triton
@pytest.mark.runtime_budget(30, hard_timeout=90, reason="inductor compiles the module with the Triton kernels")
def test_compiled_module_keeps_one_graph_and_matches_eager_on_cuda():
    """torch.compile of a module around the Triton Function: no graph break, fullgraph compiles and runs forward +
    backward through inductor (the backward kernel's loop-carried accumulator must stay fp32 when inductor passes the
    Python-float limit as fp64)."""
    torch._dynamo.reset()
    torch.manual_seed(2)
    mod = _Expert(swiglu_clamped_deepep, torch.bfloat16, "cuda")
    x = (torch.randn(64, 32, device="cuda") * 2).to(torch.bfloat16).requires_grad_(True)
    probs = torch.rand(64, 1, device="cuda").requires_grad_(True)
    explanation = torch._dynamo.explain(mod)(x, probs)
    assert explanation.graph_break_count == 0 and explanation.graph_count == 1, explanation
    out_e = mod(x, probs)
    gx_e, gp_e = torch.autograd.grad(out_e.float().sum(), (x, probs))
    torch._dynamo.reset()
    compiled = torch.compile(mod, fullgraph=True)
    out_c = compiled(x, probs)
    gx_c, gp_c = torch.autograd.grad(out_c.float().sum(), (x, probs))
    torch.testing.assert_close(out_c.float(), out_e.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(gx_c.float(), gx_e.float(), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(gp_c, gp_e, rtol=1e-2, atol=1e-2)


@pytest.mark.runtime_budget(30, hard_timeout=90, reason="Dynamo traces the dispatcher on the CPU")
def test_dispatcher_is_capturable_without_graph_breaks_on_cpu():
    """On CPU the dispatcher takes the compiled function; an outer torch.compile must see one graph (backend eager:
    capture only, no inductor build on the CI runner)."""
    torch._dynamo.reset()
    torch.manual_seed(3)
    mod = _Expert(swiglu_clamped_deepep, torch.float32, "cpu")
    x = torch.randn(6, 32, requires_grad=True)
    probs = torch.rand(6, 1, requires_grad=True)
    explanation = torch._dynamo.explain(mod)(x, probs)
    assert explanation.graph_break_count == 0 and explanation.graph_count == 1, explanation
    out_e = mod(x, probs)
    out_c = torch.compile(mod, backend="eager", fullgraph=True)(x, probs)
    torch.testing.assert_close(out_c, out_e)

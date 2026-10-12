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
    x, probs, grad = _inputs(6, 8, 2.0, "cpu")
    assert not swiglu_triton.uses_triton_kernel(x, probs)
    out = swiglu_clamped_deepep(x, probs, 2.0)
    ref = _swiglu_clamped_deepep_compiled(x.detach().clone(), probs.detach().clone(), 2.0)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
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

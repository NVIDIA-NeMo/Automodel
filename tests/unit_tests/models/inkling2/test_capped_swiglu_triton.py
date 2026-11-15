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

from nemo_automodel.components.models.inkling2 import capped_swiglu_triton
from nemo_automodel.components.models.inkling2.layers import capped_swiglu

pytestmark = pytest.mark.timeout(600)
requires_gpu_triton = pytest.mark.skipif(
    not torch.cuda.is_available() or not capped_swiglu_triton.HAVE_TRITON, reason="CUDA and Triton are required"
)


def _formula(gate_up: torch.Tensor, weights: torch.Tensor, gate_softcap: float, up_softcap: float) -> torch.Tensor:
    """The activation in FP64, written out from the model definition."""
    gate, up = gate_up.double()[:, 0::2], gate_up.double()[:, 1::2]
    capped_gate = gate_softcap * torch.tanh(gate / gate_softcap) * torch.sigmoid(gate)
    return capped_gate * (up_softcap * torch.tanh(up / up_softcap)) * weights.double()


def _inputs(
    rows: int, inter: int, scale: float, dtype: torch.dtype, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    gate_up = (torch.randn(rows, 2 * inter, device="cuda") * scale).to(dtype).requires_grad_(True)
    weights = torch.rand(rows, 1, device="cuda", dtype=torch.float32).requires_grad_(True)
    grad = torch.randn(rows, inter, device="cuda").to(dtype)
    return gate_up, weights, grad


def _reference(
    gate_up: torch.Tensor, weights: torch.Tensor, grad: torch.Tensor, gate_softcap: float, up_softcap: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gate_up = gate_up.detach().clone().requires_grad_(True)
    weights = weights.detach().clone().requires_grad_(True)
    out = _formula(gate_up, weights, gate_softcap, up_softcap)
    out.backward(grad.double())
    return out.detach(), gate_up.grad, weights.grad


@requires_gpu_triton
@pytest.mark.parametrize(
    "rows,inter,gate_softcap,up_softcap", [(37, 96, 4.0, 25.0), (1024, 2304, 4.0, 25.0), (5, 1000, 1.0, 2.0)]
)
def test_fp32_kernels_match_the_formula(rows: int, inter: int, gate_softcap: float, up_softcap: float) -> None:
    gate_up, weights, grad = _inputs(rows, inter, 3.0, torch.float32)
    out = capped_swiglu_triton.capped_swiglu_triton(gate_up, weights, gate_softcap, up_softcap)
    out.backward(grad)
    expected, expected_grad, expected_weight_grad = _reference(gate_up, weights, grad, gate_softcap, up_softcap)
    torch.testing.assert_close(out.double(), expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(gate_up.grad.double(), expected_grad.double(), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(weights.grad.double(), expected_weight_grad.double(), rtol=1e-4, atol=1e-4)


@requires_gpu_triton
def test_bf16_kernels_match_the_formula_to_bf16_rounding() -> None:
    gate_up, weights, grad = _inputs(513, 2304, 3.0, torch.bfloat16)
    out = capped_swiglu_triton.capped_swiglu_triton(gate_up, weights, 4.0, 25.0)
    out.backward(grad)
    expected, expected_grad, expected_weight_grad = _reference(gate_up, weights, grad, 4.0, 25.0)
    assert out.dtype == torch.bfloat16 and gate_up.grad.dtype == torch.bfloat16
    torch.testing.assert_close(out.double(), expected, rtol=2**-7, atol=1e-6)
    torch.testing.assert_close(gate_up.grad.double(), expected_grad.double(), rtol=2**-7, atol=1e-4)
    torch.testing.assert_close(weights.grad.double(), expected_weight_grad.double(), rtol=1e-4, atol=1e-4)


@requires_gpu_triton
def test_small_inputs_keep_their_relative_accuracy() -> None:
    """tanh of a small argument must not lose its digits."""
    gate_up, weights, grad = _inputs(64, 512, 1e-3, torch.float32)
    out = capped_swiglu_triton.capped_swiglu_triton(gate_up, weights, 4.0, 25.0)
    out.backward(grad)
    expected, expected_grad, _ = _reference(gate_up, weights, grad, 4.0, 25.0)
    torch.testing.assert_close(out.double(), expected, rtol=1e-5, atol=0.0)
    torch.testing.assert_close(gate_up.grad.double(), expected_grad.double(), rtol=1e-5, atol=0.0)


@requires_gpu_triton
def test_capped_swiglu_runs_the_kernels_on_contiguous_cuda_rows() -> None:
    gate_up, weights, grad = _inputs(130, 256, 3.0, torch.bfloat16)
    fused = capped_swiglu(gate_up, weights, gate_softcap=4.0, up_softcap=25.0)
    assert type(fused.grad_fn).__name__ == "CappedSwigluTritonBackward"
    fused.backward(grad)
    # The same rows inside a wider tensor are not contiguous, so the PyTorch operations run.
    wide = torch.zeros(130, 600, device="cuda", dtype=torch.bfloat16)
    wide[:, :512] = gate_up.detach()
    sliced = wide[:, :512].requires_grad_(True)
    plain_weights = weights.detach().clone().requires_grad_(True)
    assert not capped_swiglu_triton.uses_triton_kernel(sliced, plain_weights)
    plain = capped_swiglu(sliced, plain_weights, gate_softcap=4.0, up_softcap=25.0)
    assert type(plain.grad_fn).__name__ != "CappedSwigluTritonBackward"
    plain.backward(grad)
    torch.testing.assert_close(fused.float(), plain.float(), rtol=2**-7, atol=1e-6)
    torch.testing.assert_close(gate_up.grad.float(), sliced.grad.float(), rtol=2**-6, atol=1e-4)
    torch.testing.assert_close(weights.grad, plain_weights.grad, rtol=1e-2, atol=1e-2)


@requires_gpu_triton
def test_weight_gradient_is_skipped_when_not_required() -> None:
    gate_up, weights, grad = _inputs(16, 64, 3.0, torch.bfloat16)
    out = capped_swiglu_triton.capped_swiglu_triton(gate_up, weights.detach(), 4.0, 25.0)
    out.backward(grad)
    assert gate_up.grad is not None and weights.grad is None


def test_guard_rejects_cpu_and_grouped_layouts() -> None:
    gate_up, weights = torch.randn(4, 8), torch.rand(4, 1)
    assert not capped_swiglu_triton.uses_triton_kernel(gate_up, weights)
    if torch.cuda.is_available():
        assert not capped_swiglu_triton.uses_triton_kernel(gate_up.cuda().reshape(2, 2, 8), weights.cuda())
        assert not capped_swiglu_triton.uses_triton_kernel(gate_up.cuda(), weights.cuda()[:2])

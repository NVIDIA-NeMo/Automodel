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

"""The one-pass gradient kernel of the capped expert activation against FP64 autograd."""

from __future__ import annotations

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2 import layers, triton_activation
from nemo_automodel.components.models.inkling2.layers import Inkling2MoE, capped_swiglu
from nemo_automodel.components.models.inkling2.triton_activation import capped_swiglu_gradient

from .test_reference import tiny_config

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="The kernel runs on CUDA")

GATE_CAP, UP_CAP = 4.0, 25.0


def _reference(
    gate_up: torch.Tensor, weights: torch.Tensor, grad_output: torch.Tensor, gate_cap: float, up_cap: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP64 autograd gradients of the activation of interleaved gate/up [tokens, 2 * intermediate]."""
    pairs = gate_up.double().requires_grad_(True)
    weight = weights.double().requires_grad_(True)
    gate, up = pairs[:, 0::2], pairs[:, 1::2]
    out = gate_cap * torch.tanh(gate / gate_cap) * torch.sigmoid(gate) * (up_cap * torch.tanh(up / up_cap)) * weight
    gradients = torch.autograd.grad((out * grad_output.double()).sum(), [pairs, weight])
    return gradients[0], gradients[1]


def _case(
    tokens: int, intermediate: int, dtype: torch.dtype, weight_dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gates and ups on both sides of their caps, one weight per token and an upstream gradient."""
    generator = torch.Generator().manual_seed(1000 * tokens + intermediate)
    scale = torch.tensor([3.0, 20.0]).repeat(intermediate)
    gate_up = (scale * torch.randn(tokens, 2 * intermediate, generator=generator)).to("cuda", dtype)
    weights = (0.1 + torch.rand(tokens, 1, generator=generator)).to("cuda", weight_dtype)
    grad_output = torch.randn(tokens, intermediate, generator=generator).to("cuda", dtype)
    return gate_up, weights, grad_output


def _close(got: torch.Tensor, wanted: torch.Tensor, dtype: torch.dtype) -> bool:
    # FP32 keeps about seven digits; a BF16 result is the FP32 one rounded to eight bits.
    rtol, atol = (2e-5, 1e-6) if dtype == torch.float32 else (2**-7, 1e-6)
    return torch.allclose(got.double(), wanted, rtol=rtol, atol=atol)


@needs_gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens,intermediate", [(1, 1), (37, 50), (19, 200), (64, 64)])
def test_gradients_match_fp64_autograd(dtype: torch.dtype, tokens: int, intermediate: int) -> None:
    """Token and pair counts that leave the last program and the last step partly empty."""
    gate_up, weights, grad_output = _case(tokens, intermediate, dtype)
    wanted_gate_up, wanted_weights = _reference(gate_up, weights, grad_output, GATE_CAP, UP_CAP)
    got_gate_up, got_weights = capped_swiglu_gradient(gate_up, weights, grad_output, GATE_CAP, UP_CAP, True)
    assert got_gate_up.dtype == gate_up.dtype and got_gate_up.shape == gate_up.shape
    assert got_weights is not None and got_weights.dtype == weights.dtype and got_weights.shape == weights.shape
    assert _close(got_gate_up, wanted_gate_up, dtype)
    # The weight gradient sums the terms of a token; cancellation leaves an absolute error.
    scale = (grad_output.double().abs().sum(-1, keepdim=True) * GATE_CAP * UP_CAP).clamp_min(1.0)
    assert ((got_weights.double() - wanted_weights).abs() / scale).max() < 1e-5

    without, none = capped_swiglu_gradient(gate_up, weights, grad_output, GATE_CAP, UP_CAP, False)
    assert none is None
    assert torch.equal(without, got_gate_up)


@needs_gpu
def test_wrong_caps_and_wrong_interleaving_do_not_match() -> None:
    """The comparison separates the kernel from its two nearest mistakes."""
    gate_up, weights, grad_output = _case(37, 50, torch.float32)
    wanted, _ = _reference(gate_up, weights, grad_output, GATE_CAP, UP_CAP)
    swapped_caps, _ = capped_swiglu_gradient(gate_up, weights, grad_output, UP_CAP, GATE_CAP, False)
    assert not _close(swapped_caps, wanted, torch.float32)
    # Up before gate in every pair, undone on the way out: the kernel would then read the wrong role.
    flipped = gate_up.view(37, 50, 2).flip(-1).reshape(37, 100)
    swapped_roles, _ = capped_swiglu_gradient(flipped, weights, grad_output, GATE_CAP, UP_CAP, False)
    assert not _close(swapped_roles.view(37, 50, 2).flip(-1).reshape(37, 100), wanted, torch.float32)


@needs_gpu
def test_bf16_weights_and_strided_inputs() -> None:
    """BF16 router weights, and inputs that are views with a step."""
    gate_up, weights, grad_output = _case(23, 70, torch.bfloat16, torch.bfloat16)
    wide_gate_up = torch.zeros(23, 280, dtype=gate_up.dtype, device="cuda")
    wide_gate_up[:, ::2] = gate_up
    wide_grad = torch.zeros(23, 140, dtype=grad_output.dtype, device="cuda")
    wide_grad[:, ::2] = grad_output
    wanted_gate_up, wanted_weights = _reference(gate_up, weights, grad_output, GATE_CAP, UP_CAP)
    got_gate_up, got_weights = capped_swiglu_gradient(
        wide_gate_up[:, ::2], weights, wide_grad[:, ::2], GATE_CAP, UP_CAP, True
    )
    assert _close(got_gate_up, wanted_gate_up, torch.bfloat16)
    assert got_weights is not None and got_weights.dtype == torch.bfloat16
    scale = (grad_output.double().abs().sum(-1, keepdim=True) * GATE_CAP * UP_CAP).clamp_min(1.0)
    assert ((got_weights.double() - wanted_weights).abs() / scale).max() < 2**-7


@needs_gpu
def test_no_tokens() -> None:
    gate_up = torch.empty(0, 12, dtype=torch.bfloat16, device="cuda")
    weights = torch.empty(0, 1, device="cuda")
    grad_output = torch.empty(0, 6, dtype=torch.bfloat16, device="cuda")
    got_gate_up, got_weights = capped_swiglu_gradient(gate_up, weights, grad_output, GATE_CAP, UP_CAP, True)
    assert got_gate_up.shape == (0, 12) and got_weights is not None and got_weights.shape == (0, 1)


@needs_gpu
def test_activation_uses_the_kernel_only_with_compiled_chains(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    real = triton_activation.capped_swiglu_gradient
    monkeypatch.setattr(layers, "_one_pass_swiglu_gradient", lambda *args: (calls.append(1), real(*args))[1])
    gate_up, weights, grad_output = _case(9, 20, torch.float32)
    wanted_gate_up, wanted_weights = _reference(gate_up, weights, grad_output, GATE_CAP, UP_CAP)
    for compiled, expected_calls in ((False, 0), (True, 1)):
        monkeypatch.setattr(layers, "_ACTIVATION_COMPILED", compiled)
        pairs = gate_up.clone().requires_grad_(True)
        weight = weights.clone().requires_grad_(True)
        out = capped_swiglu(pairs, weight, gate_softcap=GATE_CAP, up_softcap=UP_CAP)
        (out * grad_output).sum().backward()
        assert len(calls) == expected_calls
        assert _close(pairs.grad, wanted_gate_up, torch.float32)
        torch.testing.assert_close(weight.grad.double(), wanted_weights, rtol=1e-4, atol=1e-5)


@needs_gpu
@pytest.mark.parametrize("compiled", [False, True])
def test_experts_without_a_routed_token_differentiate(compiled: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """A rank whose tokens are all padding still runs its expert projections, on one zero row."""
    monkeypatch.setattr(layers, "_ACTIVATION_COMPILED", compiled)
    backend = BackendConfig(attn="eager", linear="torch", rms_norm="torch", experts="torch_mm", dispatcher="torch")
    torch.manual_seed(0)
    moe = Inkling2MoE(tiny_config(), backend, dtype=torch.float32).cuda()
    with torch.no_grad():
        moe.init_weights(torch.device("cuda"), init_std=0.3)
    x = torch.randn(1, 3, 16, device="cuda", requires_grad=True)
    out = moe(x, padding_mask=torch.ones(1, 3, dtype=torch.bool, device="cuda"))
    (out * torch.randn_like(out)).sum().backward()
    # No token reaches an expert: the output is zero, and so is every gradient through the experts.
    assert torch.count_nonzero(out) == 0
    assert x.grad is not None and torch.count_nonzero(x.grad) == 0
    gradients = [parameter.grad for parameter in moe.experts.parameters()]
    assert gradients and all(gradient is not None and torch.count_nonzero(gradient) == 0 for gradient in gradients)

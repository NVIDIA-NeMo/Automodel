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

"""Unit tests for the ERNIE 5.5 layers: RMSNorm, SiTU, hyper-connection, hyper-head, MultiMax and the dense MLP."""

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.ernie5_5 import dsa, hca, layers
from nemo_automodel.components.models.ernie5_5.layers import (
    Ernie5_5DenseMLP,
    Ernie5_5HyperConnection,
    Ernie5_5HyperHead,
    Ernie5_5MultiMax,
    Ernie5_5RMSNorm,
    ernie_situ,
)

# No test of this file uses CUDA.
pytestmark = pytest.mark.cpu_only

# DENSE_ATTENTION_BACKENDS


def test_dense_attention_backends_are_defined_once_in_layers() -> None:
    """The DSA and HCA modules use the tuple that ``layers`` defines and keep no definition of their own."""
    assert layers.DENSE_ATTENTION_BACKENDS == ("eager", "torch", "sdpa")
    for module in (dsa, hca):
        assert module.DENSE_ATTENTION_BACKENDS is layers.DENSE_ATTENTION_BACKENDS
        assert not hasattr(module, "_DENSE_ATTENTION_BACKENDS")


# Ernie5_5RMSNorm


def test_fp32_input_is_rounded_to_bfloat16_before_normalization():
    torch.manual_seed(109)
    model = Ernie5_5RMSNorm(8, dtype=torch.bfloat16)
    x = torch.randn(3, 8, dtype=torch.float32, requires_grad=True)
    actual = model(x)
    rounded = x.detach().bfloat16().float()
    expected = (rounded * torch.rsqrt(rounded.square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().sum().backward()
    assert x.grad.dtype == torch.float32
    assert torch.isfinite(x.grad).all()


def test_meta_materialization_and_reset():
    with torch.device("meta"):
        model = Ernie5_5RMSNorm(4, dtype=torch.bfloat16)
    model.to_empty(device="cpu")
    model.reset_parameters()
    torch.testing.assert_close(model.weight, torch.ones(4, dtype=torch.bfloat16), rtol=0, atol=0)


@pytest.mark.parametrize("hidden,eps", [(0, 1e-6), (4, 0), (4, float("nan")), (4, float("inf"))])
def test_invalid_norm_configuration_is_rejected(hidden, eps):
    with pytest.raises(ValueError, match="positive"):
        Ernie5_5RMSNorm(hidden, eps=eps)


def test_invalid_input_is_rejected():
    model = Ernie5_5RMSNorm(4)
    with pytest.raises(ValueError, match="hidden dimension"):
        model(torch.ones(2, 3))
    with pytest.raises(ValueError, match="floating-point dtype"):
        model(torch.ones(2, 4, dtype=torch.int32))


# ernie_situ


def test_bfloat16_rounding_and_gradients_with_routing():
    generator = torch.Generator().manual_seed(55)
    packed = (torch.randn(2048, 8, generator=generator) * 3).bfloat16().requires_grad_()
    weight = torch.rand(2048, 1, generator=generator, dtype=torch.float32, requires_grad=True)
    gate, up = (part.float() for part in packed.chunk(2, dim=-1))
    unrounded = 4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * (25 * torch.tanh(up / 25))
    expected = (unrounded.bfloat16().float() * weight).bfloat16()
    actual = ernie_situ(packed, beta=4, linear_beta=25, routing_weight=weight)

    assert torch.equal(actual, expected)
    assert torch.count_nonzero(actual != (unrounded * weight).bfloat16()) > 0
    actual_grads = torch.autograd.grad(actual.float().sum(), (packed, weight), retain_graph=True)
    reference_grads = torch.autograd.grad(expected.float().sum(), (packed, weight))
    for actual_grad, reference_grad in zip(actual_grads, reference_grads, strict=True):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=0, atol=0)


def test_unclamped_up_projection():
    packed = torch.tensor([[1.0, -2.0, 3.0, -4.0]], requires_grad=True)
    gate, up = packed.chunk(2, dim=-1)
    expected = 4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * up
    torch.testing.assert_close(ernie_situ(packed, beta=4, linear_beta=None), expected)


@pytest.mark.parametrize("beta,linear_beta", [(0.0, 25.0), (4.0, 0.0), (float("inf"), None)])
def test_invalid_parameters_are_rejected(beta, linear_beta):
    with pytest.raises(ValueError, match="positive"):
        ernie_situ(torch.ones(2, 4), beta=beta, linear_beta=linear_beta)


def test_odd_packed_dimension_is_rejected():
    with pytest.raises(ValueError, match="even final dimension"):
        ernie_situ(torch.ones(2, 3), beta=4, linear_beta=25)


def test_integer_packed_input_is_rejected():
    with pytest.raises(ValueError, match="floating-point dtype"):
        ernie_situ(torch.ones(2, 4, dtype=torch.int32), beta=4, linear_beta=25)


# Ernie5_5HyperConnection


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_uniform_mix_and_gradients(dtype):
    module = Ernie5_5HyperConnection(2, 4, 8, 1e-6, 1e-6)
    with torch.no_grad():
        module.fn.zero_()
        module.base.zero_()
        module.scale.fill_(1)
    values = torch.randn(2, 3, 2, 4, dtype=dtype, requires_grad=True)
    branch = torch.randn(2, 3, 4, dtype=dtype, requires_grad=True)
    pre, post, comb = module.compute_weights(values)
    torch.testing.assert_close(pre, torch.full_like(pre, 0.500001))
    torch.testing.assert_close(post, torch.ones_like(post))
    torch.testing.assert_close(comb, torch.full_like(comb, 0.5), rtol=1e-5, atol=1e-6)
    collapsed = module.collapse(values, pre)
    expanded = module.expand(branch, values, post, comb)
    torch.testing.assert_close(collapsed, values.float().sum(dim=-2) * 0.500001)
    torch.testing.assert_close(
        expanded.float(),
        (values.float().mean(dim=-2, keepdim=True) + branch.float().unsqueeze(-2)).expand_as(values).to(dtype).float(),
        rtol=0,
        atol=0.02 if dtype == torch.bfloat16 else 1e-5,
    )
    (collapsed.square().mean() + expanded.float().square().mean()).backward()
    for tensor in (values, branch, *module.parameters()):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_meta_materialization_and_invalid_sizes():
    with pytest.raises(ValueError):
        Ernie5_5HyperConnection(0, 4, 3, 1e-6, 1e-6)
    with torch.device("meta"):
        module = Ernie5_5HyperConnection(2, 4, 3, 1e-6, 1e-6)
    module.to_empty(device="cpu")
    module.init_weights(0.02)
    assert all(torch.isfinite(parameter).all() for parameter in module.parameters())


# Ernie5_5HyperHead


def test_zero_logits_have_analytic_uniform_weights_and_backward():
    head = Ernie5_5HyperHead(2, 4)
    with torch.no_grad():
        head.hc_fn.zero_()
        head.hc_base.zero_()
    x = torch.randn(2, 3, 2, 4, requires_grad=True)
    output = head(x)
    torch.testing.assert_close(output, x.sum(dim=-2) * (0.5 + head.eps))
    output.square().sum().backward()
    for tensor in (x, *head.parameters()):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()


def test_meta_and_fail_closed():
    with torch.device("meta"):
        head = Ernie5_5HyperHead(2, 4)
    head.to_empty(device="cpu")
    head.init_weights()
    assert all(torch.isfinite(p).all() for p in head.parameters())
    with pytest.raises(ValueError):
        head(torch.empty(1, 2, 4))
    head.to(dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="FP32"):
        head(torch.empty(1, 2, 2, 4))


# Ernie5_5MultiMax


def _multimax_oracle(x: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Evaluate the scalar formula in double precision.

    Args:
        x: Logits of shape [..., vocab], with arbitrary leading dimensions.
        r: Thresholds of shape [4].
        t: Scales of shape [4].

    Returns:
        Corrected FP64 logits of shape [..., vocab].
    """
    x, r, t = x.double(), r.double(), t.double()
    below0, above1 = (r[0] - x).clamp_min(0), (x - r[1]).clamp_min(0)
    below2, above3 = (r[2] - x).clamp_min(0), (x - r[3]).clamp_min(0)
    return x + t[0] * below0 + t[1] * above1 + t[2] * below2**2 + t[3] * above3**2


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_forward_and_backward_against_fp64(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(55)
    logits = torch.randn(530, 2, 7, generator=generator, dtype=dtype).requires_grad_()
    weight = torch.randn(logits.shape, generator=generator, dtype=dtype)
    module = Ernie5_5MultiMax(dtype=dtype)
    with torch.no_grad():
        module.multimax_ranges.copy_(torch.tensor([-0.5, 0.75, -1.25, 1.5]))
        module.multimax_ts.copy_(torch.tensor([0.3, -0.2, 0.05, 0.1]))
    ref_x = logits.detach().double().requires_grad_()
    ref_r = module.multimax_ranges.detach().double().requires_grad_()
    ref_t = module.multimax_ts.detach().double().requires_grad_()
    expected = _multimax_oracle(ref_x, ref_r, ref_t)
    actual = module(logits)
    expected.backward(weight.double())
    actual.backward(weight)
    rtol, atol = (0.02, 0.02) if dtype == torch.bfloat16 else (1e-5, 1e-5)
    torch.testing.assert_close(actual.float(), expected.detach().float(), rtol=rtol, atol=atol)
    torch.testing.assert_close(logits.grad.float(), ref_x.grad.float(), rtol=rtol, atol=atol)
    torch.testing.assert_close(module.multimax_ranges.grad.float(), ref_r.grad.float(), rtol=rtol, atol=atol)
    torch.testing.assert_close(module.multimax_ts.grad.float(), ref_t.grad.float(), rtol=rtol, atol=atol)


def test_gradients_accumulate_in_fp32_across_512_row_chunks() -> None:
    module = Ernie5_5MultiMax(dtype=torch.bfloat16)
    logits = torch.zeros(1024, 1, 4, dtype=torch.bfloat16, requires_grad=True)
    incoming = torch.zeros_like(logits)
    with torch.no_grad():
        for row, grad in zip((0, 256, 512, 768), (2**24, 1, 1, -(2**24)), strict=True):
            logits[row, 0, 0] = 1
            incoming[row, 0, 0] = grad
    module(logits).backward(incoming)
    torch.testing.assert_close(module.multimax_ts.grad, torch.tensor([0, 1, 0, 1], dtype=torch.bfloat16))
    torch.testing.assert_close(module.multimax_ranges.grad, torch.zeros(4, dtype=torch.bfloat16))
    assert torch.equal(logits.grad, incoming)


def test_zero_initialized_head_is_identity_and_rejects_empty_vocabulary() -> None:
    module = Ernie5_5MultiMax()
    logits = torch.randn(2, 3, 4).transpose(0, 1)
    assert torch.equal(module(logits), logits)
    with pytest.raises(ValueError, match="vocabulary"):
        module(torch.empty(2, 0))


def test_integer_logits_are_rejected() -> None:
    with pytest.raises(ValueError, match="floating-point"):
        Ernie5_5MultiMax()(torch.ones(2, 4, dtype=torch.int32))


# Ernie5_5DenseMLP


def test_forward_and_gradient_against_explicit_formula():
    torch.manual_seed(5578)
    model = Ernie5_5DenseMLP(4, 5, dtype=torch.float32)
    x = torch.randn(2, 3, 4, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    gate = F.linear(ref_x, model.gate_proj.weight)
    up = F.linear(ref_x, model.up_proj.weight)
    activation = 4 * torch.tanh(gate / 4) * gate.sigmoid() * (25 * torch.tanh(up / 25))
    expected = F.linear(activation, model.down_proj.weight)
    actual = model(x)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x)[0], torch.autograd.grad(expected.sum(), ref_x)[0], rtol=1e-6, atol=1e-7
    )


def test_meta_initialization():
    with torch.device("meta"):
        model = Ernie5_5DenseMLP(4, 5, dtype=torch.bfloat16)
    model.to_empty(device="cpu")
    model.init_weights()
    assert all(torch.isfinite(p).all() for p in model.parameters())

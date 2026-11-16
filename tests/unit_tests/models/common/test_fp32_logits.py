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

"""FP32-output LM head: logits precision against FP64, backward contract and module surgery."""

import sys
import types

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel.components.models.common.fp32_logits import (
    Float32LogitsConvertible,
    Float32LogitsLinear,
    head_compute_dtype,
    linear_with_fp32_logits,
)

_DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"))]


def _operands(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BF16 hidden states [batch, sequence, hidden], BF16 weight [vocab, hidden] and a BF16-valued upstream gradient."""
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(2, 33, 96, generator=generator).to(device, torch.bfloat16)
    weight = (torch.randn(320, 96, generator=generator) * 0.05).to(device, torch.bfloat16)
    upstream = torch.randn(2, 33, 320, generator=generator).to(device, torch.bfloat16)
    return hidden, weight, upstream


@pytest.mark.parametrize("device", _DEVICES)
def test_logits_keep_fp32_precision(device):
    hidden, weight, _ = _operands(device)
    reference = hidden.double() @ weight.double().t()

    logits = linear_with_fp32_logits(hidden, weight)

    assert logits.dtype == torch.float32 and logits.shape == (2, 33, 320)
    fp32_error = (logits.double() - reference).abs().max()
    bf16_error = (F.linear(hidden, weight).double() - reference).abs().max()
    assert fp32_error < 1e-5
    assert bf16_error > 100 * fp32_error
    # Rounded BF16 logits upcast to FP32 would all be BF16-representable.
    assert (logits == logits.bfloat16().float()).float().mean() < 0.01


@pytest.mark.parametrize("device", _DEVICES)
def test_backward_equals_bf16_head_backward(device):
    hidden, weight, upstream = _operands(device)
    fp32_hidden, fp32_weight = hidden.clone().requires_grad_(), weight.clone().requires_grad_()
    bf16_hidden, bf16_weight = hidden.clone().requires_grad_(), weight.clone().requires_grad_()

    linear_with_fp32_logits(fp32_hidden, fp32_weight).backward(upstream.float())
    F.linear(bf16_hidden, bf16_weight).backward(upstream)

    # The FP32 logits gradient is cast to the operand dtype and fed to the BF16 head's GEMMs.
    assert fp32_hidden.grad.dtype == fp32_weight.grad.dtype == torch.bfloat16
    torch.testing.assert_close(fp32_hidden.grad, bf16_hidden.grad, rtol=0, atol=0)
    torch.testing.assert_close(fp32_weight.grad, bf16_weight.grad, rtol=0, atol=0)


def test_frozen_weight_receives_no_gradient():
    hidden, weight, upstream = _operands("cpu")
    hidden.requires_grad_()

    linear_with_fp32_logits(hidden, weight).backward(upstream.float())

    assert hidden.grad is not None and weight.grad is None


def test_fp32_weight_projects_in_fp32():
    hidden, weight, _ = _operands("cpu")
    weight = weight.float().requires_grad_()

    logits = linear_with_fp32_logits(hidden, weight)
    logits.sum().backward()

    torch.testing.assert_close(logits, F.linear(hidden.float(), weight.detach()), rtol=0, atol=0)
    assert weight.grad.dtype == torch.float32


def test_autocast_does_not_lower_logits():
    hidden, weight, _ = _operands("cpu")
    expected = linear_with_fp32_logits(hidden, weight)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = linear_with_fp32_logits(hidden.float(), weight)

    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


@pytest.mark.parametrize("bias", [False, True])
def test_from_head_owns_the_same_parameters(bias):
    head = nn.Linear(96, 320, bias=bias).eval()

    fp32_head = Float32LogitsLinear.from_head(head)

    assert fp32_head.weight is head.weight
    assert fp32_head.bias is head.bias
    assert not fp32_head.training
    assert list(fp32_head.state_dict()) == list(head.state_dict())
    hidden = torch.randn(3, 96, dtype=torch.bfloat16)
    torch.testing.assert_close(
        fp32_head(hidden), linear_with_fp32_logits(hidden, head.weight, head.bias), rtol=0, atol=0
    )


def test_bias_is_added_in_fp32_and_receives_its_gradient():
    hidden, weight, upstream = _operands("cpu")
    bias = (torch.randn(320) * 0.1).to(torch.bfloat16).requires_grad_()

    logits = linear_with_fp32_logits(hidden, weight, bias)
    logits.backward(upstream.float())

    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, linear_with_fp32_logits(hidden, weight) + bias.detach().float(), rtol=0, atol=0)
    assert bias.grad.dtype == torch.bfloat16
    torch.testing.assert_close(bias.grad, upstream.float().sum((0, 1)).bfloat16(), rtol=0, atol=0)


def test_fp32_logits_heads_are_returned_unchanged():
    head = Float32LogitsLinear(96, 320, bias=False)

    assert Float32LogitsLinear.from_head(head) is head


class _BackendLinear(nn.Module):
    """Mimics Transformer Engine's Linear: same weight layout, an empty unregistered tensor for no bias."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.Tensor()


class _QuackLinear(nn.Linear):
    """Mimics quack's Linear: an nn.Linear subclass with its own kernel."""


def test_linear_kernel_backend_heads_convert(monkeypatch):
    te_linear = types.ModuleType("transformer_engine.pytorch.module.linear")
    te_linear.Linear = _BackendLinear
    quack_linear = types.ModuleType("quack.linear")
    quack_linear.Linear = _QuackLinear
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.module.linear", te_linear)
    monkeypatch.setitem(sys.modules, "quack.linear", quack_linear)

    for head in (_BackendLinear(96, 320), _QuackLinear(96, 320, bias=True)):
        fp32_head = Float32LogitsLinear.from_head(head)
        assert type(fp32_head) is Float32LogitsLinear
        assert fp32_head.weight is head.weight
        assert fp32_head.bias is (head.bias if isinstance(head.bias, nn.Parameter) else None)


class _NormalizedHead(nn.Module, Float32LogitsConvertible):
    """A model-owned head with its own projection, like Baichuan's NormHead."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(320, 96))

    def to_fp32_logits(self) -> Float32LogitsLinear:
        head = Float32LogitsLinear(96, 320, bias=False, device="meta")
        head.weight = self.weight
        return head


def test_convertible_heads_convert_themselves():
    head = _NormalizedHead()

    fp32_head = Float32LogitsLinear.from_head(head)

    assert isinstance(fp32_head, Float32LogitsLinear) and fp32_head.weight is head.weight


class _LinearSubclass(nn.Linear):
    """Stands in for LoRA, FP8 and quantized heads, which subclass nn.Linear and own their projection."""


@pytest.mark.parametrize("head", [_LinearSubclass(4, 8, bias=False), nn.Embedding(8, 4)])
def test_from_head_rejects_heads_with_their_own_projection(head):
    with pytest.raises(ValueError, match="linear kernel backend"):
        Float32LogitsLinear.from_head(head)


def test_head_compute_dtype_follows_the_weight():
    fp32_logits_head = Float32LogitsLinear.from_head(nn.Linear(4, 8, bias=False, dtype=torch.bfloat16))

    assert head_compute_dtype(fp32_logits_head) is torch.bfloat16
    assert head_compute_dtype(nn.Identity()) is None

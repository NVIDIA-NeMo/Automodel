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

from nemo_automodel.components.models.common.utils import (
    compute_lm_head_logits,
    enable_fp32_logits,
    head_compute_dtype,
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

    logits = compute_lm_head_logits(weight, hidden).logits

    assert logits.dtype == torch.float32 and logits.shape == (2, 33, 320)
    fp32_error = (logits.double() - reference).abs().max()
    bf16_error = (F.linear(hidden, weight).double() - reference).abs().max()
    assert fp32_error < 1e-5
    assert bf16_error > 100 * fp32_error
    # Rounded BF16 logits upcast to FP32 would all be BF16-representable.
    assert (logits == logits.bfloat16().float()).float().mean() < 0.01


@pytest.mark.skipif(not torch.cuda.is_available(), reason="out_dtype GEMM requires CUDA or ROCm")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_logits_match_vllm_projection(dtype):
    generator = torch.Generator(device="cuda").manual_seed(7)
    hidden, weight, bias = [
        torch.randn(shape, device="cuda", dtype=dtype, generator=generator) for shape in ((2, 7, 64), (31, 64), (31,))
    ]
    expected = torch.mm(hidden.flatten(0, 1), weight.t(), out_dtype=torch.float32) + bias.float()
    logits = compute_lm_head_logits(weight, hidden, bias=bias).logits
    torch.testing.assert_close(logits, expected.view(2, 7, 31), rtol=0, atol=0)


@pytest.mark.parametrize("device", _DEVICES)
def test_backward_equals_bf16_head_backward(device):
    hidden, weight, upstream = _operands(device)
    fp32_hidden, fp32_weight = hidden.clone().requires_grad_(), weight.clone().requires_grad_()
    bf16_hidden, bf16_weight = hidden.clone().requires_grad_(), weight.clone().requires_grad_()

    compute_lm_head_logits(fp32_weight, fp32_hidden).logits.backward(upstream.float())
    F.linear(bf16_hidden, bf16_weight).backward(upstream)

    # The FP32 logits gradient is cast to the operand dtype and fed to the BF16 head's GEMMs.
    assert fp32_hidden.grad.dtype == fp32_weight.grad.dtype == torch.bfloat16
    torch.testing.assert_close(fp32_hidden.grad, bf16_hidden.grad, rtol=0, atol=0)
    torch.testing.assert_close(fp32_weight.grad, bf16_weight.grad, rtol=0, atol=0)


def test_frozen_weight_receives_no_gradient():
    hidden, weight, upstream = _operands("cpu")
    hidden.requires_grad_()

    compute_lm_head_logits(weight, hidden).logits.backward(upstream.float())

    assert hidden.grad is not None and weight.grad is None


def test_fp32_weight_projects_in_fp32():
    hidden, weight, _ = _operands("cpu")
    weight = weight.float().requires_grad_()

    logits = compute_lm_head_logits(weight, hidden).logits
    logits.sum().backward()

    torch.testing.assert_close(logits, F.linear(hidden.float(), weight.detach()), rtol=0, atol=0)
    assert weight.grad.dtype == torch.float32


def test_autocast_does_not_lower_logits():
    hidden, weight, _ = _operands("cpu")
    expected = compute_lm_head_logits(weight, hidden).logits

    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = compute_lm_head_logits(weight, hidden.float()).logits

    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


def test_fp32_head_backward_keeps_fp32_under_autocast():
    generator = torch.Generator().manual_seed(17)
    head = enable_fp32_logits(nn.Linear(5, 7, bias=False, dtype=torch.float32))
    hidden = torch.randn(3, 5, generator=generator, requires_grad=True)
    upstream = torch.randn(3, 7, generator=generator)
    expected = torch.autograd.grad(head(hidden), (hidden, head.weight), upstream)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = torch.autograd.grad(head(hidden), (hidden, head.weight), upstream)

    for actual_grad, expected_grad in zip(actual, expected):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


@pytest.mark.parametrize("bias", [False, True])
def test_plain_head_keeps_its_parameters_and_hooks(bias):
    head = nn.Linear(96, 320, bias=bias).eval()

    weight, original_bias = head.weight, head.bias
    keys = list(head.state_dict())
    observed = []
    head.register_forward_hook(lambda module, args, output: observed.append(output.dtype))

    fp32_head = enable_fp32_logits(head)

    assert fp32_head is head
    assert fp32_head.weight is weight
    assert fp32_head.bias is original_bias
    assert not fp32_head.training
    assert list(fp32_head.state_dict()) == keys
    hidden = torch.randn(3, 96, dtype=torch.bfloat16)
    torch.testing.assert_close(
        fp32_head(hidden), compute_lm_head_logits(head.weight, hidden, bias=head.bias).logits, rtol=0, atol=0
    )
    assert observed == [torch.float32]


def test_bias_is_added_in_fp32_and_receives_its_gradient():
    hidden, weight, upstream = _operands("cpu")
    bias = (torch.randn(320) * 0.1).to(torch.bfloat16).requires_grad_()

    logits = compute_lm_head_logits(weight, hidden, bias=bias).logits
    logits.backward(upstream.float())

    assert logits.dtype == torch.float32
    torch.testing.assert_close(
        logits, compute_lm_head_logits(weight, hidden).logits + bias.detach().float(), rtol=0, atol=0
    )
    assert bias.grad.dtype == torch.bfloat16
    torch.testing.assert_close(bias.grad, upstream.float().sum((0, 1)).bfloat16(), rtol=0, atol=0)


def test_fp32_logits_heads_are_returned_unchanged():
    head = enable_fp32_logits(nn.Linear(96, 320, bias=False))

    assert enable_fp32_logits(head) is head


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
        fp32_head = enable_fp32_logits(head)
        assert type(fp32_head) is nn.Linear
        assert fp32_head.weight is head.weight
        assert fp32_head.bias is (head.bias if isinstance(head.bias, nn.Parameter) else None)


def test_baichuan_head_keeps_its_normalized_projection():
    from nemo_automodel.components.models.baichuan.model import NormHead

    head = NormHead(8, 13)
    hidden = torch.randn(2, 8)
    fp32_head = enable_fp32_logits(head)

    assert fp32_head.weight is head.weight
    torch.testing.assert_close(fp32_head(hidden), F.linear(hidden, F.normalize(head.weight)))


class _LinearSubclass(nn.Linear):
    """Stands in for LoRA, FP8 and quantized heads, which subclass nn.Linear and own their projection."""


@pytest.mark.parametrize("head", [_LinearSubclass(4, 8, bias=False), nn.Embedding(8, 4)])
def test_rejects_heads_with_their_own_projection(head):
    with pytest.raises(ValueError, match="linear kernel backend"):
        enable_fp32_logits(head)


def test_head_compute_dtype_follows_the_weight():
    fp32_logits_head = enable_fp32_logits(nn.Linear(4, 8, bias=False, dtype=torch.bfloat16))

    assert head_compute_dtype(fp32_logits_head) is torch.bfloat16
    assert head_compute_dtype(nn.Identity()) is None

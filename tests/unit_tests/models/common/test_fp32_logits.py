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

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel.components.models.common.fp32_logits import Float32LogitsLinear, linear_with_fp32_logits

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


def test_from_linear_owns_the_same_weight():
    head = nn.Linear(96, 320, bias=False).eval()

    fp32_head = Float32LogitsLinear.from_linear(head)

    assert fp32_head.weight is head.weight
    assert not fp32_head.training
    assert list(fp32_head.state_dict()) == ["weight"]
    hidden = torch.randn(3, 96, dtype=torch.bfloat16)
    torch.testing.assert_close(fp32_head(hidden), linear_with_fp32_logits(hidden, head.weight), rtol=0, atol=0)


class _LinearSubclass(nn.Linear):
    """Stands in for LoRA, FP8 and quantized heads, which subclass nn.Linear."""


@pytest.mark.parametrize(
    ("head", "message"),
    [
        (nn.Linear(4, 8, bias=True), "bias-free"),
        (_LinearSubclass(4, 8, bias=False), "plain nn.Linear"),
        (nn.Embedding(8, 4), "plain nn.Linear"),
    ],
)
def test_from_linear_rejects_heads_with_their_own_projection(head, message):
    with pytest.raises(ValueError, match=message):
        Float32LogitsLinear.from_linear(head)

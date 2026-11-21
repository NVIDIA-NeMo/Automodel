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

"""LMHead: logits dtype against FP64, backward contract, FLOP accounting and head conversion."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode

from nemo_automodel.components.models.common.lm_head import LMHead, slice_logits_to_keep

_DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"))]


def _operands(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BF16 hidden states [batch, sequence, hidden], BF16 weight [vocab, hidden] and a BF16-valued upstream gradient."""
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(2, 33, 96, generator=generator).to(device, torch.bfloat16)
    weight = (torch.randn(320, 96, generator=generator) * 0.05).to(device, torch.bfloat16)
    upstream = torch.randn(2, 33, 320, generator=generator).to(device, torch.bfloat16)
    return hidden, weight, upstream


def _head(weight: torch.Tensor, logits_dtype: torch.dtype | None) -> LMHead:
    """Build a bias-free head holding a copy of ``weight``.

    Args:
        weight: Tensor of shape [vocab, hidden]; its dtype and device become the parameter's.
        logits_dtype: The head's logits dtype.

    Returns:
        Head with weight of shape [vocab, hidden].
    """
    head = LMHead(weight.shape[1], weight.shape[0], logits_dtype=logits_dtype, device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        head.weight.copy_(weight)
    return head


@pytest.mark.parametrize("device", _DEVICES)
def test_logits_keep_fp32_precision(device):
    hidden, weight, _ = _operands(device)
    reference = hidden.double() @ weight.double().t()

    logits = _head(weight, torch.float32)(hidden)

    assert logits.dtype == torch.float32 and logits.shape == (2, 33, 320)
    fp32_error = (logits.double() - reference).abs().max()
    bf16_error = (F.linear(hidden, weight).double() - reference).abs().max()
    assert fp32_error < 1e-5
    assert bf16_error > 100 * fp32_error
    # Rounded BF16 logits upcast to FP32 would all be BF16-representable.
    assert (logits == logits.bfloat16().float()).float().mean() < 0.01


@pytest.mark.parametrize(
    "weight_dtype,logits_dtype,expected",
    [
        (torch.bfloat16, None, lambda h, w: F.linear(h, w)),
        (torch.bfloat16, torch.float32, lambda h, w: F.linear(h.float(), w.float())),
        (torch.float32, None, lambda h, w: F.linear(h.float(), w)),
        (torch.float32, torch.float32, lambda h, w: F.linear(h.float(), w)),
        # An FP32 GEMM whose logits are rounded afterwards.
        (torch.float32, torch.bfloat16, lambda h, w: F.linear(h.float(), w).bfloat16()),
    ],
)
def test_logits_dtype_is_independent_of_the_weight_dtype(weight_dtype, logits_dtype, expected):
    hidden, weight, _ = _operands("cpu")
    weight = weight.to(weight_dtype)

    logits = _head(weight, logits_dtype)(hidden)

    reference = expected(hidden, weight)
    assert logits.dtype == (logits_dtype or weight_dtype) == reference.dtype
    torch.testing.assert_close(logits, reference, rtol=0, atol=0)


@pytest.mark.parametrize("logits_dtype", [None, torch.float32])
def test_head_uses_the_fp32_kernel_only_to_widen_and_keeps_packed_positions(logits_dtype):
    class CountProjection(TorchDispatchMode):
        calls = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.nemo_automodel.fp32_logits.default:
                self.calls += 1
            return func(*args, **(kwargs or {}))

    head = LMHead(8, 12, bias=True, logits_dtype=logits_dtype, dtype=torch.bfloat16)
    hidden = torch.randn(5, 8, dtype=torch.bfloat16)
    with CountProjection() as trace:
        logits = head(slice_logits_to_keep(hidden, 2))

    assert trace.calls == int(logits_dtype is torch.float32)
    expected = LMHead.linear(hidden[-2:], head.weight, head.bias, logits_dtype=logits_dtype)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="logits_dtype GEMM requires CUDA or ROCm")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_logits_match_vllm_projection(dtype):
    generator = torch.Generator(device="cuda").manual_seed(7)
    hidden, weight, bias = [
        torch.randn(shape, device="cuda", dtype=dtype, generator=generator) for shape in ((2, 7, 64), (31, 64), (31,))
    ]
    expected = torch.mm(hidden.flatten(0, 1), weight.t(), out_dtype=torch.float32) + bias.float()
    logits = LMHead.linear(hidden, weight, bias, logits_dtype=torch.float32)
    torch.testing.assert_close(logits, expected.view(2, 7, 31), rtol=0, atol=0)


@pytest.mark.parametrize("device", _DEVICES)
def test_backward_equals_bf16_head_backward(device):
    hidden, weight, upstream = _operands(device)
    fp32_hidden, fp32_weight = hidden.clone().requires_grad_(), weight.clone().requires_grad_()
    bf16_hidden, bf16_weight = hidden.clone().requires_grad_(), weight.clone().requires_grad_()

    LMHead.linear(fp32_hidden, fp32_weight, logits_dtype=torch.float32).backward(upstream.float())
    F.linear(bf16_hidden, bf16_weight).backward(upstream)

    # The FP32 logits gradient is cast to the operand dtype and fed to the BF16 head's GEMMs.
    assert fp32_hidden.grad.dtype == fp32_weight.grad.dtype == torch.bfloat16
    torch.testing.assert_close(fp32_hidden.grad, bf16_hidden.grad, rtol=0, atol=0)
    torch.testing.assert_close(fp32_weight.grad, bf16_weight.grad, rtol=0, atol=0)


def test_frozen_weight_receives_no_gradient():
    hidden, weight, upstream = _operands("cpu")
    hidden.requires_grad_()

    LMHead.linear(hidden, weight, logits_dtype=torch.float32).backward(upstream.float())

    assert hidden.grad is not None and weight.grad is None


def test_fp32_weight_projects_in_fp32():
    hidden, weight, _ = _operands("cpu")
    weight = weight.float().requires_grad_()

    logits = LMHead.linear(hidden, weight, logits_dtype=torch.float32)
    logits.sum().backward()

    torch.testing.assert_close(logits, F.linear(hidden.float(), weight.detach()), rtol=0, atol=0)
    assert weight.grad.dtype == torch.float32


def test_autocast_does_not_lower_logits():
    hidden, weight, _ = _operands("cpu")
    expected = LMHead.linear(hidden, weight, logits_dtype=torch.float32)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = LMHead.linear(hidden.float(), weight, logits_dtype=torch.float32)

    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


def test_fp32_head_backward_keeps_fp32_under_autocast():
    generator = torch.Generator().manual_seed(17)
    head = LMHead(5, 7, logits_dtype=torch.float32, dtype=torch.float32)
    hidden = torch.randn(3, 5, generator=generator, requires_grad=True)
    upstream = torch.randn(3, 7, generator=generator)
    expected = torch.autograd.grad(head(hidden), (hidden, head.weight), upstream)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        actual = torch.autograd.grad(head(hidden), (hidden, head.weight), upstream)

    for actual_grad, expected_grad in zip(actual, expected):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_bias_is_added_in_fp32_and_receives_its_gradient():
    hidden, weight, upstream = _operands("cpu")
    bias = (torch.randn(320) * 0.1).to(torch.bfloat16).requires_grad_()

    logits = LMHead.linear(hidden, weight, bias, logits_dtype=torch.float32)
    logits.backward(upstream.float())

    assert logits.dtype == torch.float32
    torch.testing.assert_close(
        logits, LMHead.linear(hidden, weight, logits_dtype=torch.float32) + bias.detach().float(), rtol=0, atol=0
    )
    assert bias.grad.dtype == torch.bfloat16
    torch.testing.assert_close(bias.grad, upstream.float().sum((0, 1)).bfloat16(), rtol=0, atol=0)


def test_flop_counter_counts_the_fp32_gemm():
    hidden, weight, _ = _operands("cpu")
    counts = {}
    for logits_dtype in (None, torch.float32):
        with FlopCounterMode(display=False) as counter:
            _head(weight, logits_dtype)(hidden)
        counts[logits_dtype] = counter.get_total_flops()

    assert counts[torch.float32] == counts[None] == 2 * 2 * 33 * 96 * 320


def test_project_applies_the_softcap_and_takes_a_gathered_weight():
    hidden, weight, _ = _operands("cpu")
    head = LMHead(96, 320, logits_dtype=torch.float32, logit_softcapping=2.0, dtype=torch.bfloat16)
    with torch.no_grad():
        head.weight.copy_(weight)

    logits = head(hidden)

    expected = 2.0 * torch.tanh(LMHead.linear(hidden, weight, logits_dtype=torch.float32) / 2.0)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    # A loss that gathers the weight itself calls project with that copy.
    gathered = (-weight).requires_grad_()
    overridden = head.project(hidden, gathered)
    torch.testing.assert_close(overridden, -logits, rtol=0, atol=0)
    overridden.sum().backward()
    assert gathered.grad is not None
    assert head.weight.grad is None


@pytest.mark.parametrize("bias", [False, True])
def test_from_linear_shares_the_parameters(bias):
    linear = nn.Linear(96, 320, bias=bias, dtype=torch.bfloat16).eval()
    keys = list(linear.state_dict())

    head = LMHead.from_linear(linear, logits_dtype=torch.float32)

    assert isinstance(head, LMHead) and head.logits_dtype == torch.float32
    assert head.weight is linear.weight
    assert head.bias is linear.bias
    assert not head.training
    assert list(head.state_dict()) == keys
    hidden = torch.randn(3, 96, dtype=torch.bfloat16)
    torch.testing.assert_close(
        head(hidden), LMHead.linear(hidden, linear.weight, linear.bias, logits_dtype=torch.float32), rtol=0, atol=0
    )


class _LinearSubclass(nn.Linear):
    """Stands in for LoRA, FP8 and quantized heads, which subclass nn.Linear and own their projection."""


def _linear_with_instance_forward() -> nn.Linear:
    """Build a plain linear whose instance forward was replaced, as hooks libraries do."""
    linear = nn.Linear(4, 8, bias=False)
    linear.forward = lambda input: 2 * F.linear(input, linear.weight)
    return linear


@pytest.mark.parametrize(
    "linear", [_LinearSubclass(4, 8, bias=False), nn.Embedding(8, 4), _linear_with_instance_forward()]
)
def test_from_linear_rejects_heads_with_their_own_projection(linear):
    with pytest.raises(ValueError, match="plain nn.Linear head"):
        LMHead.from_linear(linear, logits_dtype=torch.float32)


class TestSliceLogitsToKeep:
    """``slice_logits_to_keep`` picks the positions a forward projects, on batched and packed (THD) hidden states."""

    HIDDEN = 8
    VOCAB = 16

    @pytest.mark.parametrize("shape", [(2, 5, HIDDEN), (7, HIDDEN)], ids=["batched", "packed"])
    def test_zero_keeps_every_position_without_slicing(self, shape):
        """``0`` returns the hidden states themselves: a full slice of a DTensor has raised under SP."""
        hidden = torch.randn(shape)
        assert slice_logits_to_keep(hidden, 0) is hidden

    def test_int_keeps_the_last_positions(self):
        hidden = torch.randn(2, 5, self.HIDDEN)
        torch.testing.assert_close(slice_logits_to_keep(hidden, 2), hidden[:, -2:, :], rtol=0, atol=0)
        packed = torch.randn(7, self.HIDDEN)
        torch.testing.assert_close(slice_logits_to_keep(packed, 2), packed[-2:], rtol=0, atol=0)

    def test_tensor_keeps_the_given_positions(self):
        idx = torch.tensor([0, 2, 4])
        hidden = torch.randn(2, 5, self.HIDDEN)
        torch.testing.assert_close(slice_logits_to_keep(hidden, idx), hidden[:, idx, :], rtol=0, atol=0)
        packed = torch.randn(7, self.HIDDEN)
        torch.testing.assert_close(slice_logits_to_keep(packed, idx), packed[idx], rtol=0, atol=0)

    @pytest.mark.parametrize(
        ("head_dtype", "hidden_dtype"),
        [(torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)],
        ids=["bf16-head-fp32-hidden", "fp32-head-bf16-hidden"],
    )
    def test_lm_head_casts_kept_positions_to_the_weight_dtype(self, head_dtype, hidden_dtype):
        """An LMHead computes in its weight dtype whatever the hidden states' dtype, also after slicing."""
        lm_head = LMHead(self.HIDDEN, self.VOCAB, dtype=head_dtype)
        hidden = torch.randn(2, 5, self.HIDDEN, dtype=hidden_dtype)
        logits = lm_head(slice_logits_to_keep(hidden, 2))
        assert logits.shape == (2, 2, self.VOCAB)
        assert logits.dtype == head_dtype
        expected = F.linear(hidden[:, -2:].to(head_dtype), lm_head.weight)
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)

    def test_fp32_logits_head_projects_kept_packed_positions(self):
        """A bf16 LMHead with FP32 logits writes fp32 logits for the kept positions of packed input."""
        torch.manual_seed(0)
        plain = nn.Linear(self.HIDDEN, self.VOCAB, bias=False).to(torch.bfloat16)
        lm_head = LMHead.from_linear(plain, logits_dtype=torch.float32)
        hidden = torch.randn(7, self.HIDDEN, dtype=torch.bfloat16)
        logits = lm_head(slice_logits_to_keep(hidden, 3))
        assert logits.shape == (3, self.VOCAB)
        assert logits.dtype == torch.float32
        expected = hidden[-3:].float() @ lm_head.weight.float().t()
        torch.testing.assert_close(logits, expected, rtol=0, atol=0)


def test_lora_rejects_a_head_with_a_logits_dtype():
    from nemo_automodel.components._peft.lora import LinearLoRA, patch_linear_module

    # The adapter would replace the head's projection and silently drop its logits dtype.
    with pytest.raises(ValueError, match="logits_dtype"):
        patch_linear_module(LMHead(8, 16, logits_dtype=torch.float32), dim=4, use_triton=False)
    plain = patch_linear_module(LMHead(8, 16), dim=4, use_triton=False)
    assert isinstance(plain, LinearLoRA) and isinstance(plain, LMHead)

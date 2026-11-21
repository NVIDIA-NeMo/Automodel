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

"""LM head whose logits dtype is independent of its weight dtype."""

import math
from typing import Self

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.experimental import register_sharding
from torch.utils.flop_counter import register_flop_formula

from nemo_automodel.shared.tp_linear import tp_linear_forward


# aten::mm.dtype has no derivative in PyTorch 2.10, and an opaque op keeps compilers off the GEMM.
@torch.library.custom_op("nemo_automodel::fp32_logits", mutates_args=())
def _fp32_logits(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """FP32 logits [..., vocab] of input [..., hidden] and weight [vocab, hidden], one dtype, plus bias [vocab]."""
    flat = input.reshape(-1, input.shape[-1])
    with torch.autocast(device_type=flat.device.type, enabled=False):
        if flat.is_cuda and flat.dtype in (torch.float16, torch.bfloat16):
            logits = torch.mm(flat, weight.t(), out_dtype=torch.float32)
        else:
            logits = F.linear(flat.float(), weight.float())
        if bias is not None:
            logits += bias.float()
    return logits.view(*input.shape[:-1], weight.shape[0])


@_fp32_logits.register_fake
def _fp32_logits_fake(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Allocate FP32 logits [..., vocab] for input [..., hidden] and weight [vocab, hidden]."""
    return input.new_empty((*input.shape[:-1], weight.shape[0]), dtype=torch.float32)


@register_sharding(torch.ops.nemo_automodel.fp32_logits.default)
def _fp32_logits_sharding(input, weight, bias):
    """Shard logits over the vocabulary like a column-parallel linear, or over input's leading dims."""
    strategies = [(Replicate(), Replicate(), Replicate()), (Shard(input.ndim - 1), Replicate(), Shard(0))]
    strategies += [(Shard(dim), Shard(dim), Replicate()) for dim in range(input.ndim - 1)]
    # Each is (logits, input, weight); the bias [vocab] is placed like the weight's vocabulary axis.
    return [([out], [inp, w, None if bias is None else w]) for out, inp, w in strategies]


def _fp32_logits_backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
    """Run a head's GEMMs in the operand dtype on the FP32 logits gradient [..., vocab], which the bias sums."""
    input, weight, bias = ctx.saved_tensors
    grad = grad.reshape(-1, weight.shape[0])
    grad_bias = grad.sum(0).to(bias.dtype) if ctx.needs_input_grad[2] else None
    grad = grad.to(weight.dtype)
    with torch.autocast(device_type=weight.device.type, enabled=False):
        grad_input = (grad @ weight).view_as(input) if ctx.needs_input_grad[0] else None
        grad_weight = grad.t() @ input.reshape(-1, input.shape[-1]) if ctx.needs_input_grad[1] else None
    return grad_input, grad_weight, grad_bias


_fp32_logits.register_autograd(
    _fp32_logits_backward, setup_context=lambda ctx, inputs, output: ctx.save_for_backward(*inputs)
)
register_flop_formula(torch.ops.nemo_automodel.fp32_logits)(lambda x, w, *_, **__: 2 * math.prod(x) * w[0])


class LMHead(nn.Linear):
    """``nn.Linear`` vocabulary head that returns ``logits_dtype`` logits whatever its weight dtype.

    With ``logits_dtype`` set, the GEMM reads the input in the weight's dtype, accumulates in FP32 and
    writes FP32 logits (vLLM's ``head_dtype=float32``), unaffected by autocast, then casts them; the
    backward casts the gradient to the weight's dtype. Chunked losses, which gather the weight themselves, call
    ``project``, so subclasses that change the logits override it.
    Repo models take ``logits_dtype`` as a constructor argument and build their head with it; their forwards pick
    positions with ``slice_logits_to_keep`` before calling the head.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        logits_dtype: torch.dtype | None = None,
        logit_softcapping: float | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Create the head; ``logits_dtype`` None keeps the weight's dtype, and ``logit_softcapping`` ``c``
        applies ``c * tanh(logits / c)``. The other arguments are ``nn.Linear``'s."""
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.logits_dtype = logits_dtype
        self.logit_softcapping = logit_softcapping

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Project hidden states [..., hidden] to logits [..., vocab] with the head's own weight."""
        return self.project(input, self.weight)

    def project(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Project input [..., hidden] to logits [..., vocab] with ``weight`` [vocab, hidden]."""
        logits = self.linear(input, weight, self.bias, logits_dtype=self.logits_dtype)
        if self.logit_softcapping is not None:
            logits = self.logit_softcapping * torch.tanh(logits / self.logit_softcapping)
        return logits

    @staticmethod
    def linear(
        input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, *, logits_dtype: torch.dtype | None
    ) -> torch.Tensor:
        """Project input [..., hidden] to logits [..., vocab] with weight [vocab, hidden] and bias [vocab].

        ``logits_dtype`` None runs a plain linear in the weight's dtype that autocast may lower.
        """
        # The weight sets the compute dtype: inside forward, FSDP2 has already unsharded it to its param_dtype.
        input = input.to(weight.dtype)
        if logits_dtype is None:
            if isinstance(weight, DTensor):  # tensor-parallel weight: compile-safe like TPLinear
                return tp_linear_forward(input, weight, bias, mm_for_2d_compile=True)
            return F.linear(input, weight, bias)
        return torch.ops.nemo_automodel.fp32_logits(input, weight, bias).to(logits_dtype)

    @classmethod
    def from_linear(cls, head: nn.Linear, *, logits_dtype: torch.dtype | None) -> Self:
        """Wrap the plain linear head a Hugging Face parent class built, sharing its weight and bias.

        Ties, optimizer identity and checkpoint keys are unchanged.

        Raises:
            ValueError: If ``head`` is not a plain ``nn.Linear``, so it has its own projection.
        """
        if type(head) is not nn.Linear or "forward" in vars(head):
            raise ValueError(f"An LMHead wraps a plain nn.Linear head; got {type(head).__name__}.")
        new_head = cls(head.in_features, head.out_features, logits_dtype=logits_dtype, device="meta")
        new_head.weight, new_head.bias = head.weight, head.bias
        return new_head.train(head.training)


def slice_logits_to_keep(hidden_states: torch.Tensor, logits_to_keep: int | torch.Tensor) -> torch.Tensor:
    """Select the positions of hidden states whose logits a forward returns, before the head projects them.

    Hugging Face's ``slice_logits_to_keep`` (newer than the pinned transformers), for an int or a 1D index tensor.

    Args:
        hidden_states: Tensor of shape [batch, sequence, hidden], or packed (THD) [tokens, hidden].
        logits_to_keep: ``0`` keeps every position and returns ``hidden_states`` itself, since a full
            ``slice(0, None)`` of a DTensor has raised on ``aten.alias`` under sequence parallelism. A positive int
            keeps the last ``N`` positions; an integer tensor of shape [kept] keeps those positions.

    Returns:
        Tensor of shape [batch, kept, hidden], or [kept, hidden] for packed input: a view of ``hidden_states`` for
        an int, a copy for an index tensor.
    """
    if isinstance(logits_to_keep, int):
        return hidden_states if logits_to_keep == 0 else hidden_states[..., -logits_to_keep:, :]
    return hidden_states[..., logits_to_keep, :]

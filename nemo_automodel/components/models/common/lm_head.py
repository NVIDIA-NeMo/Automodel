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
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.shared.tp_linear import tp_linear_forward


# aten::mm.dtype has no derivative in PyTorch 2.10, and an opaque op keeps compilers off the GEMM.
@torch.library.custom_op("nemo_automodel::fp32_logits", mutates_args=())
def _fp32_logits(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """FP32 logits [..., vocab] of input [..., hidden] and weight [vocab, hidden], one dtype, via vLLM's GEMM."""
    flat = input.reshape(-1, input.shape[-1])
    with torch.autocast(device_type=flat.device.type, enabled=False):
        if flat.is_cuda and flat.dtype in (torch.float16, torch.bfloat16):
            logits = torch.mm(flat, weight.t(), out_dtype=torch.float32)
        else:
            logits = F.linear(flat.float(), weight.float())
    return logits.view(*input.shape[:-1], weight.shape[0])


@_fp32_logits.register_fake
def _fp32_logits_fake(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Allocate FP32 logits [..., vocab] for input [..., hidden] and weight [vocab, hidden]."""
    return input.new_empty((*input.shape[:-1], weight.shape[0]), dtype=torch.float32)


@register_sharding(torch.ops.nemo_automodel.fp32_logits.default)
def _fp32_logits_sharding(input, weight):
    """Shard logits over the vocabulary like a column-parallel linear, or over input's leading dims."""
    strategies = [([Replicate()], [Replicate(), Replicate()]), ([Shard(input.ndim - 1)], [Replicate(), Shard(0)])]
    return strategies + [([Shard(dim)], [Shard(dim), Replicate()]) for dim in range(input.ndim - 1)]


def _fp32_logits_backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Run a head's GEMMs in the operand dtype on the FP32 logits gradient [..., vocab]."""
    input, weight = ctx.saved_tensors
    grad = grad.reshape(-1, weight.shape[0]).to(weight.dtype)
    with torch.autocast(device_type=weight.device.type, enabled=False):
        grad_input = (grad @ weight).view_as(input) if ctx.needs_input_grad[0] else None
        grad_weight = grad.t() @ input.reshape(-1, input.shape[-1]) if ctx.needs_input_grad[1] else None
    return grad_input, grad_weight


_fp32_logits.register_autograd(
    _fp32_logits_backward, setup_context=lambda ctx, inputs, output: ctx.save_for_backward(*inputs)
)
register_flop_formula(torch.ops.nemo_automodel.fp32_logits)(lambda input, weight, **_: 2 * math.prod(input) * weight[0])


class LMHead(nn.Linear):
    """``nn.Linear`` vocabulary head that returns ``logits_dtype`` logits whatever its weight dtype.

    With ``logits_dtype`` set, the GEMM reads the input in the weight's dtype, accumulates in FP32 and
    writes FP32 logits (vLLM's ``head_dtype=float32``), unaffected by autocast, then casts them; the
    backward casts the gradient to the weight's dtype. Chunked losses pass their materialized weight to ``forward``.
    Repo models take ``logits_dtype`` as a constructor argument and build their head with it; their forwards call
    ``LMHead.output(self.lm_head, hidden_states, ...)``, which also takes a stage without a head.
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

    def forward(self, input: torch.Tensor, *, weight: torch.Tensor | None = None) -> torch.Tensor:
        """Project hidden states [..., hidden] with the head weight or an explicit [vocab, hidden] weight."""
        weight = self.weight if weight is None else weight
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
        if weight.element_size() > input.element_size():
            input = input.to(weight.dtype)
        if logits_dtype is None:
            if isinstance(weight, DTensor):  # tensor-parallel weight: compile-safe like TPLinear
                return tp_linear_forward(input, weight, bias, mm_for_2d_compile=True)
            return F.linear(input, weight, bias)
        logits = _fp32_logits(input, weight)
        return (logits if bias is None else logits + bias.float()).to(logits_dtype)

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

    @staticmethod
    def output(
        lm_head: nn.Module | None,
        hidden_states: torch.Tensor,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool = False,
    ) -> CausalLMOutputWithPast:
        """Project hidden states through ``lm_head`` and wrap the result, for every custom causal LM ``forward()``.

        ``lm_head`` is the model's head. An ``LMHead`` casts the input to its weight dtype and returns its
        ``logits_dtype``; any other head, such as a LoRA or quantized linear, must accept the hidden states' dtype.

        ``logits_to_keep=0`` projects every position without slicing, since ``slice(0, None)`` on a DTensor
        raises under sequence parallelism. A causal LM output is batched, so 2D (THD) logits and hidden states
        regain a leading batch dim of 1.

        Args:
            lm_head: Head module, or ``None`` on a pipeline stage without it, which passes ``hidden_states`` on.
            hidden_states: Final hidden states of shape [tokens, hidden] (THD) or [batch, sequence, hidden].
            logits_to_keep: ``0`` for every position, a positive int for the last ``N``, or a tensor of indices.
            output_hidden_states: Attach the full-sequence final hidden states, e.g. for linear losses.

        Returns:
            ``CausalLMOutputWithPast`` with logits of shape [batch, sequence, vocab] (or the passed-through
            hidden states) and, if requested, the final hidden states of shape [batch, sequence, hidden].
        """
        if lm_head is None:
            logits = hidden_states
        elif isinstance(logits_to_keep, int) and logits_to_keep == 0:
            logits = lm_head(hidden_states)
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = lm_head(hidden_states[..., slice_indices, :])
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)

        hidden_out = None
        if output_hidden_states:
            hidden_out = hidden_states.unsqueeze(0) if hidden_states.dim() == 2 else hidden_states

        return CausalLMOutputWithPast(logits=logits, hidden_states=hidden_out)

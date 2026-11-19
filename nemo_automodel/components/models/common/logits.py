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
import sys
from typing import Self

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.experimental import register_sharding
from torch.utils.flop_counter import register_flop_formula


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
    backward casts the gradient to the weight's dtype. ``forward`` calls ``project``, which losses that
    gather the weight call too, so subclasses that change the logits override ``project``.
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
        """Return ``project(input, self.weight, self.bias)``: logits [..., vocab] of input [..., hidden]."""
        return self.project(input, self.weight, self.bias)

    def project(self, input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        """Compute the head's logits [..., vocab] with ``weight``, which a chunked loss may have gathered.

        Args:
            input: Hidden states of shape [..., hidden], with arbitrary leading dimensions.
            weight: Tensor of shape [vocab, hidden]; a vocabulary-sharded DTensor shards the logits.
            bias: Optional tensor of shape [vocab].
        """
        logits = self.linear(input, weight, bias, logits_dtype=self.logits_dtype)
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
        input = input.to(weight.dtype)
        if logits_dtype is None:
            return F.linear(input, weight, bias)
        logits = _fp32_logits(input, weight)
        return (logits if bias is None else logits + bias.float()).to(logits_dtype)

    @classmethod
    def from_linear(cls, head: nn.Module, *, logits_dtype: torch.dtype | None) -> Self:
        """Wrap a plain, Transformer Engine or quack linear head's weight and bias, keeping ties and checkpoint keys.

        Raises:
            ValueError: If ``head`` has its own projection, such as a LoRA or quantized head.
        """
        backends = (
            getattr(sys.modules.get("transformer_engine.pytorch.module.linear"), "Linear", None),
            getattr(sys.modules.get("quack.linear"), "Linear", None),
        )
        if not (type(head) is nn.Linear and "forward" not in vars(head)) and type(head) not in backends:
            raise ValueError(f"logits_dtype needs a plain linear LM head; got {type(head).__name__}.")
        weight = head.get_parameter("weight")
        new_head = cls(weight.shape[1], weight.shape[0], logits_dtype=logits_dtype, device="meta")
        new_head.weight = weight
        if isinstance(bias := getattr(head, "bias", None), nn.Parameter):
            new_head.bias = bias
        return new_head.train(head.training)

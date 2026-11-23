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

import functools
import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.experimental import register_sharding
from torch.utils.flop_counter import register_flop_formula
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput

from nemo_automodel.shared.tp_linear import tp_linear_forward

if TYPE_CHECKING:
    from transformers import PreTrainedModel


# aten::mm.dtype has no derivative in PyTorch 2.10, and an opaque op keeps compilers off the GEMM.
@torch.library.custom_op("nemo_automodel::fp32_logits", mutates_args=())
def _fp32_logits(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Compute FP32 logits from operands in one dtype.

    Args:
        input: Tensor of shape [..., hidden].
        weight: Tensor of shape [vocab, hidden], in input's dtype.
        bias: Optional tensor of shape [vocab].

    Returns:
        FP32 tensor of shape [..., vocab]. CUDA BF16/FP16 operands accumulate in FP32 without an upcast copy;
        other operands are upcast first.
    """
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
    """Allocate the logits of a traced call.

    Args:
        input: Tensor of shape [..., hidden].
        weight: Tensor of shape [vocab, hidden].
        bias: Optional tensor of shape [vocab]; unused.

    Returns:
        Empty FP32 tensor of shape [..., vocab] on input's device.
    """
    return input.new_empty((*input.shape[:-1], weight.shape[0]), dtype=torch.float32)


@register_sharding(torch.ops.nemo_automodel.fp32_logits.default)
def _fp32_logits_sharding(input, weight, bias):
    """Shard the logits like a column-parallel linear.

    The backward flattens input's leading dimensions, so ``LMHead.linear`` replicates an input sharded on one of
    them before calling the op.

    Args:
        input: Tensor metadata of global shape [..., hidden].
        weight: Tensor metadata of global shape [vocab, hidden].
        bias: Tensor metadata of global shape [vocab], or None.

    Returns:
        Supported output/input placements on each mesh axis: everything replicated, or the vocabulary sharded,
        with input replicated and weight and bias sharded on dim 0.
    """
    strategies = [(Replicate(), Replicate(), Replicate()), (Shard(input.ndim - 1), Replicate(), Shard(0))]
    # Each is (logits, input, weight); the bias [vocab] is placed like the weight's vocabulary axis.
    return [([out], [inp, w, None if bias is None else w]) for out, inp, w in strategies]


def _fp32_logits_setup_context(ctx, inputs, output):
    """Save the operands for the backward GEMMs.

    Args:
        ctx: Autograd context that owns the saved tensors.
        inputs: Tuple of input [..., hidden], weight [vocab, hidden], and bias [vocab] or None.
        output: FP32 logits of shape [..., vocab]; unused.
    """
    ctx.save_for_backward(*inputs)


def _fp32_logits_backward(ctx, grad: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
    """Run the head's backward GEMMs in the weight's dtype.

    Args:
        ctx: Autograd context holding input, weight and bias.
        grad: FP32 gradient of the logits, shape [..., vocab].

    Returns:
        Gradients of input [..., hidden] and weight [vocab, hidden] in the weight's dtype and of bias [vocab] in
        its own dtype (the sum of grad over tokens); None for an operand that needs no gradient.
    """
    input, weight, bias = ctx.saved_tensors
    grad = grad.reshape(-1, weight.shape[0])
    grad_bias = grad.sum(0).to(bias.dtype) if ctx.needs_input_grad[2] else None
    grad = grad.to(weight.dtype)
    with torch.autocast(device_type=weight.device.type, enabled=False):
        grad_input = (grad @ weight).view_as(input) if ctx.needs_input_grad[0] else None
        grad_weight = grad.t() @ input.reshape(-1, input.shape[-1]) if ctx.needs_input_grad[1] else None
    return grad_input, grad_weight, grad_bias


_fp32_logits.register_autograd(_fp32_logits_backward, setup_context=_fp32_logits_setup_context)


@register_flop_formula(torch.ops.nemo_automodel.fp32_logits)
def _fp32_logits_flops(input_shape, weight_shape, bias_shape, out_shape):
    """Count the projection's FLOPs.

    Args:
        input_shape: Shape [..., hidden] of input.
        weight_shape: Shape [vocab, hidden] of weight.
        bias_shape: Shape [vocab] of bias, or None; bias additions are not counted.
        out_shape: Shape [..., vocab] of the logits; unused.

    Returns:
        Two FLOPs per multiply-accumulate: 2 * input elements * vocab.
    """
    return 2 * math.prod(input_shape) * weight_shape[0]


class LMHead(nn.Linear):
    """``nn.Linear`` vocabulary head that returns ``logits_dtype`` logits whatever its weight dtype.

    With ``logits_dtype`` set, the GEMM reads the input in the weight's dtype, accumulates in FP32 and
    writes FP32 logits (vLLM's ``head_dtype=float32``), unaffected by autocast, then casts them; the
    backward casts the gradient to the weight's dtype. Chunked losses, which gather the weight themselves, call
    ``project``, so subclasses that change the logits override it.
    Repo models take ``logits_dtype`` as a constructor argument and build their head with it; their forwards pick
    positions with ``slice_logits_to_keep`` before calling the head.
    """

    # Vocabulary rows at or beyond this index are padding, which a subclass's ``project`` masks to negative infinity;
    # None means no padding. Losses that stream the vocabulary in chunks trim the weight to it first.
    unpadded_vocab_size: int | None = None

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
        """Create the head.

        Args:
            in_features: Hidden size.
            out_features: Vocabulary size.
            bias: Whether the head has a bias of shape [vocab].
            logits_dtype: Dtype of the logits. None keeps the weight's dtype, which autocast may lower.
            logit_softcapping: Cap ``c`` that applies ``c * tanh(logits / c)`` to the logits; None applies none.
            device: Device of the parameters.
            dtype: Dtype of the parameters.
        """
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.logits_dtype = logits_dtype
        self.logit_softcapping = logit_softcapping

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Project hidden states with the head's own weight.

        Args:
            input: Tensor of shape [..., hidden].

        Returns:
            Logits of shape [..., vocab], as ``project`` returns them.
        """
        return self.project(input, self.weight)

    def project(self, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Project hidden states with a supplied vocabulary weight.

        Args:
            input: Tensor of shape [..., hidden], with arbitrary leading dimensions. It is cast to the supplied
                weight's dtype before projection.
            weight: Tensor of shape [vocab, hidden], on the input's device. Chunked losses supply their gathered
                weight here. For column-parallel DTensors these are global shapes: the weight uses Shard(0), and
                the input is replicated on the same mesh.

        Returns:
            Logits of shape [..., vocab], with leading axes unchanged and the configured softcap applied.
            Column-parallel output is sharded on the vocabulary axis. The dtype is ``logits_dtype`` when set,
            and the softcap then runs on the FP32 GEMM output, so the logits are rounded once; otherwise the
            dtype follows the linear's compute (autocast) dtype. Inputs are not mutated.
        """
        compute_dtype = None if self.logits_dtype is None else torch.float32
        logits = self.linear(input, weight, self.bias, logits_dtype=compute_dtype)
        if self.logit_softcapping is not None:
            logits = self.logit_softcapping * torch.tanh(logits / self.logit_softcapping)
        return logits if self.logits_dtype is None else logits.to(self.logits_dtype)

    @staticmethod
    def linear(
        input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None, *, logits_dtype: torch.dtype | None
    ) -> torch.Tensor:
        """Project hidden states with a supplied weight and bias, without a softcap.

        Args:
            input: Tensor of shape [..., hidden], cast to the weight's dtype. With ``logits_dtype`` set, a DTensor
                sharded on a leading dimension is replicated first.
            weight: Tensor of shape [vocab, hidden]; a column-parallel DTensor uses Shard(0), with the input
                replicated on the same mesh.
            bias: Optional tensor of shape [vocab].
            logits_dtype: Dtype of the logits. None runs a plain linear in the weight's dtype that autocast may
                lower; otherwise the GEMM accumulates and writes FP32 logits, which are cast to this dtype.

        Returns:
            Logits of shape [..., vocab], sharded on the vocabulary axis for a column-parallel weight.
        """
        # The weight sets the compute dtype: inside forward, FSDP2 has already unsharded it to its param_dtype.
        input = input.to(weight.dtype)
        if logits_dtype is None:
            if isinstance(weight, DTensor):  # tensor-parallel weight: compile-safe like TPLinear
                return tp_linear_forward(input, weight, bias, mm_for_2d_compile=True)
            return F.linear(input, weight, bias)
        if isinstance(input, DTensor):
            # The op's backward flattens the leading dimensions, which DTensor cannot do across a sharded one.
            last = input.ndim - 1
            placements = [
                Replicate() if isinstance(p, Shard) and p.dim % input.ndim < last else p for p in input.placements
            ]
            input = input.redistribute(placements=placements)
        return torch.ops.nemo_automodel.fp32_logits(input, weight, bias).to(logits_dtype)

    @classmethod
    def from_linear(cls, head: nn.Linear, *, logits_dtype: torch.dtype | None) -> Self:
        """Wrap the plain linear head a Hugging Face parent class built, sharing its weight and bias.

        Ties, optimizer identity and checkpoint keys are unchanged.

        Args:
            head: Plain ``nn.Linear`` whose weight is a Tensor of shape [vocab, hidden] and whose bias is a Tensor
                of shape [vocab] or None.
            logits_dtype: Dtype of the logits; None keeps the weight's dtype, which autocast may lower.

        Returns:
            Head that shares ``head``'s weight and bias Parameters without copying, in ``head``'s training mode.

        Raises:
            ValueError: If ``head`` is not a plain ``nn.Linear``, so it has its own projection.
        """
        if type(head) is not nn.Linear or "forward" in vars(head):
            raise ValueError(f"An LMHead wraps a plain nn.Linear head; got {type(head).__name__}.")
        new_head = cls(head.in_features, head.out_features, logits_dtype=logits_dtype, device="meta")
        new_head.weight, new_head.bias = head.weight, head.bias
        return new_head.train(head.training)


@dataclass
class CausalLMOutputWithLMHeadWeight(CausalLMOutputWithPast):
    """``CausalLMOutputWithPast`` that also carries the LM head's weight when the forward skipped the head.

    A forward called with ``compute_logits=False`` returns empty logits of shape [..., 0] and the head's weight, as
    the module holds it (a sharded DTensor under FSDP2), for a loss that projects the final hidden states itself.

    Attributes:
        lm_head_weight: Tensor of shape [vocab, hidden], or None when the forward computed logits.
    """

    lm_head_weight: torch.Tensor | None = None


def drop_logits(output: ModelOutput, lm_head: nn.Module) -> ModelOutput:
    """Empty the logits of a Hugging Face forward called without logits, and attach its head's weight.

    Hugging Face's forwards always run the head, so a forward called with ``compute_logits=False`` projects the last
    position only. A loss that projects the final hidden states itself takes the head's weight from the output.

    Args:
        output: Model output whose logits have shape [batch, kept_sequence, vocab].
        lm_head: Output embedding whose weight has shape [vocab, hidden].

    Returns:
        The same output, with empty logits of shape [batch, kept_sequence, 0] and the head's weight, as the module
        holds it, as ``lm_head_weight``.
    """
    output["logits"] = output["logits"][..., :0]
    # Item assignment registers a key the output class may not declare, and sets the attribute too.
    output["lm_head_weight"] = lm_head.weight
    return output


def with_compute_logits(forward: Callable[..., ModelOutput]) -> Callable[..., ModelOutput]:
    """Give a Hugging Face forward the ``compute_logits`` option that the repo's forwards take.

    Args:
        forward: Hugging Face ``forward``, which always runs its head.

    Returns:
        ``forward`` with a keyword-only ``compute_logits``, True by default. False projects the last position only
        and returns the output through :func:`drop_logits`. Its signature is ``forward``'s plus ``compute_logits``,
        so signature-based checks see the parameters ``forward`` takes. A forward that already takes
        ``compute_logits``, or takes no ``logits_to_keep`` to limit its head, is returned as is.
    """
    signature = inspect.signature(forward)
    if "compute_logits" in signature.parameters or "logits_to_keep" not in signature.parameters:
        return forward

    @functools.wraps(forward)
    def forward_with_compute_logits(
        self: "PreTrainedModel", *args: Any, compute_logits: bool = True, **kwargs: Any
    ) -> ModelOutput:
        """Run ``forward``, or without logits, project the last position only and drop it.

        Args:
            self: Hugging Face model.
            *args: ``forward``'s positional inputs, such as input_ids of shape [batch, sequence].
            compute_logits: Whether to return logits; False returns the head's weight instead.
            **kwargs: ``forward``'s keyword inputs.

        Returns:
            ``forward``'s output; without logits, through :func:`drop_logits`.
        """
        if compute_logits:
            return forward(self, *args, **kwargs)
        return drop_logits(forward(self, *args, **{**kwargs, "logits_to_keep": 1}), self.get_output_embeddings())

    parameters = list(signature.parameters.values())
    # A keyword-only parameter goes before **kwargs.
    end = next((i for i, p in enumerate(parameters) if p.kind is inspect.Parameter.VAR_KEYWORD), len(parameters))
    option = inspect.Parameter("compute_logits", inspect.Parameter.KEYWORD_ONLY, default=True, annotation=bool)
    forward_with_compute_logits.__signature__ = signature.replace(
        parameters=[*parameters[:end], option, *parameters[end:]]
    )
    return forward_with_compute_logits


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

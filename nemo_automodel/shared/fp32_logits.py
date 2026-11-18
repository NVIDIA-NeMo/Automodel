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

"""The FP32 LM-head logits path shared by every model.

A model returns FP32 logits exactly when its LM head is a :class:`Float32LogitsLinear`:

- ``logits_dtype: float32`` swaps any model's head for one (:meth:`Float32LogitsLinear.from_head`),
  keeping the head's parameters;
- models whose head is FP32 by design (FP32 head weights or an FP32 output stream)
  build a :class:`Float32LogitsLinear` themselves;
- linear losses reproduce the head's projection with :func:`linear_with_fp32_logits`
  when the model's head is a :class:`Float32LogitsLinear`.

Upcasting already-rounded BF16 logits does not recover their precision, and
projecting with an FP32 copy of a BF16 head weight costs a transient weight-sized
allocation and an FP32 GEMM. Instead the head GEMM reads its operands in the head
weight's compute dtype, accumulates in FP32 and stores FP32 logits, as vLLM's
``head_dtype=float32`` path and Megatron-LM's ``logit_dtype`` do. A head whose
weight computes in FP32 (FP32 storage, or an FP32 FSDP policy) runs an FP32 GEMM.

``torch.mm(..., out_dtype=torch.float32)`` has no autograd formula, so training
goes through :class:`_Float32LogitsMatmul`. Its backward follows Megatron-LM's
``output_dtype`` contract: the FP32 logits gradient is cast to the operand dtype
and the input and weight gradients use the same GEMMs as a BF16-output head.

Parameter storage, optimizer master weights and gradient reduction stay with the
model and its FSDP policy.
"""

import abc
import math
import sys

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

from nemo_automodel.shared.import_utils import safe_import_from

__all__ = ["Float32LogitsConvertible", "Float32LogitsLinear", "head_compute_dtype", "linear_with_fp32_logits"]


def _fp32_output_mm(input_2d: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Multiply local hidden states by the transposed head weight into FP32 logits.

    Args:
        input_2d: Tensor of shape [tokens, hidden] in ``weight.dtype``.
        weight: Tensor of shape [vocab, hidden] in BF16, FP16 or FP32.

    Returns:
        New FP32 tensor of shape [tokens, vocab].
    """
    if weight.dtype == torch.float32:
        return input_2d @ weight.t()
    if input_2d.is_cuda:
        return torch.mm(input_2d, weight.t(), out_dtype=torch.float32)
    # aten::mm.dtype has no CPU kernel. Products of two BF16/FP16 values are
    # exact in FP32, so upcasting the operands changes only the summation order.
    return input_2d.float() @ weight.float().t()


class _Float32LogitsMatmul(torch.autograd.Function):
    """``input @ weight.T`` with operands in ``weight.dtype`` and an FP32 result."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Project local hidden states into FP32 logits.

        Args:
            ctx: Autograd context; saves ``input`` and ``weight`` for backward.
            input: Tensor of shape [..., hidden], with arbitrary leading dimensions,
                in ``weight.dtype``.
            weight: Tensor of shape [vocab, hidden].

        Returns:
            New FP32 tensor of shape [..., vocab]; neither input is aliased or mutated.
        """
        ctx.save_for_backward(input, weight)
        logits = _fp32_output_mm(input.reshape(-1, input.shape[-1]), weight)
        return logits.view(*input.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_logits: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Cast the logits gradient to the operand dtype, then run the head's usual GEMMs.

        Args:
            ctx: Autograd context holding ``input`` of shape [..., hidden] and
                ``weight`` of shape [vocab, hidden].
            grad_logits: FP32 tensor of shape [..., vocab], with the forward's leading dimensions.

        Returns:
            Gradient of ``input`` (shape [..., hidden]) and of ``weight`` (shape
            [vocab, hidden]) in the operand dtype, each ``None`` when not required.
        """
        input, weight = ctx.saved_tensors
        grad = grad_logits.reshape(-1, grad_logits.shape[-1]).to(weight.dtype)
        grad_input = grad_weight = None
        with torch.autocast(device_type=weight.device.type, enabled=False):
            if ctx.needs_input_grad[0]:
                grad_input = (grad @ weight).view(input.shape)
            if ctx.needs_input_grad[1]:
                grad_weight = grad.t() @ input.reshape(-1, input.shape[-1])
        return grad_input, grad_weight


def _vocab_parallel_fp32_logits(input: torch.Tensor, weight: DTensor) -> DTensor:
    """Project through a column-parallel head whose weight is sharded over the vocabulary.

    Args:
        input: Tensor of shape [..., hidden], with arbitrary leading dimensions.
            A DTensor on ``weight``'s mesh with any placement is redistributed to
            ``Replicate``; a regular tensor is taken as the replicated value.
        weight: DTensor of global shape [vocab, hidden] with placements
            ``(Shard(0),)`` on a 1-D tensor-parallel mesh; each rank holds a
            [local_vocab, hidden] slice.

    Returns:
        FP32 DTensor of global shape [..., vocab] with placement ``Shard(-1)``;
        each rank's local tensor is [..., local_vocab].
    """
    mesh = weight.device_mesh
    if mesh.ndim != 1 or weight.placements != (Shard(0),):
        raise NotImplementedError(
            "FP32 logits support a vocabulary-sharded (column-parallel) LM head, whose weight has placements "
            f"(Shard(0),) on a 1-D tensor-parallel mesh; got {weight.placements} on a {mesh.ndim}-D mesh."
        )
    if not isinstance(input, DTensor):
        input = DTensor.from_local(input, mesh, [Replicate()], run_check=False)
    input = input.redistribute(mesh, [Replicate()])
    # Every rank holds a vocabulary slice of the weight, so its input gradient is a
    # partial sum; ``Partial`` makes DTensor autograd reduce it, as it does for aten.mm.
    local_logits = _Float32LogitsMatmul.apply(
        input.to_local(grad_placements=[Partial()]).to(weight.dtype),
        weight.to_local(grad_placements=[Shard(0)]),
    )
    # Pass the global shape so a vocabulary that the mesh does not divide evenly is kept exact.
    shape = (*input.shape[:-1], weight.shape[0])
    stride = tuple(math.prod(shape[dim + 1 :]) for dim in range(len(shape)))
    return DTensor.from_local(
        local_logits, mesh, [Shard(local_logits.ndim - 1)], run_check=False, shape=shape, stride=stride
    )


def linear_with_fp32_logits(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Project hidden states through an LM-head weight into FP32 logits.

    The GEMM reads ``input`` cast to ``weight.dtype`` and ``weight`` unchanged,
    accumulates in FP32 and stores FP32 logits; a bias is added in FP32. Autocast
    is disabled inside so it cannot lower the output precision.

    Args:
        input: Tensor of shape [..., hidden], with arbitrary leading dimensions, of
            any floating dtype. With a DTensor ``weight`` it may be a DTensor on the
            same mesh with any placement.
        weight: Tensor of shape [vocab, hidden] in BF16, FP16 or FP32. A DTensor must
            have global shape [vocab, hidden] and placements ``(Shard(0),)`` on a 1-D
            tensor-parallel mesh.
        bias: Optional tensor of shape [vocab] in any floating dtype; not supported
            with a DTensor ``weight``.

    Returns:
        New FP32 tensor of shape [..., vocab]. For a DTensor ``weight`` it is a
        DTensor of that global shape with placement ``Shard(-1)``. No input is
        mutated.
    """
    with torch.autocast(device_type=input.device.type, enabled=False):
        if isinstance(weight, DTensor):
            if bias is not None:
                raise NotImplementedError("FP32 logits with tensor parallelism need a bias-free LM head.")
            return _vocab_parallel_fp32_logits(input, weight)
        logits = _Float32LogitsMatmul.apply(input.to(weight.dtype), weight)
        return logits if bias is None else logits + bias.to(torch.float32)


def _linear_backend_classes() -> tuple[type[nn.Module], ...]:
    """Return the linear classes that ``initialize_linear_module`` builds.

    Each projects with its ``weight`` and ``bias`` exactly as ``nn.Linear`` does, so a
    head of one of these classes can hand its parameters to a :class:`Float32LogitsLinear`.
    A head of a kernel backend's class implies that its module is already imported,
    so these imports never load a backend that the model does not use.

    Returns:
        ``nn.Linear`` and the imported Transformer Engine and quack ``Linear`` classes.
    """
    classes: list[type[nn.Module]] = [nn.Linear]
    for module_name in ("transformer_engine.pytorch.module.linear", "quack.linear"):
        if module_name in sys.modules:
            available, linear = safe_import_from(module_name, "Linear")
            if available:
                classes.append(linear)
    return tuple(classes)


class Float32LogitsConvertible(abc.ABC):
    """Mixin for a model-owned LM head whose projection is not a plain linear map.

    :meth:`Float32LogitsLinear.from_head` asks such a head for its FP32-logits
    variant instead of rejecting it, so the model package keeps owning the projection.
    """

    @abc.abstractmethod
    def to_fp32_logits(self) -> "Float32LogitsLinear":
        """Return a head that owns this head's parameters and whose projection writes FP32 logits.

        Returns:
            A :class:`Float32LogitsLinear` subclass instance sharing this head's Parameters.
        """


class Float32LogitsLinear(nn.Linear):
    """LM head whose GEMM writes FP32 logits from operands in its weight's compute dtype.

    Models whose head is FP32 by design construct it directly; :meth:`from_head`
    wraps another head's parameters for ``logits_dtype: float32``. Only the
    projection's output dtype differs from ``nn.Linear``.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Project hidden states into FP32 logits.

        Args:
            input: Tensor of shape [..., hidden], with arbitrary leading dimensions.

        Returns:
            FP32 tensor of shape [..., vocab] (see :func:`linear_with_fp32_logits`
            for the tensor-parallel DTensor layout).
        """
        return linear_with_fp32_logits(input, self.weight, self.bias)

    @classmethod
    def from_head(cls, head: nn.Module) -> "Float32LogitsLinear":
        """Return an FP32-logits head that owns ``head``'s parameters (no copy, identity preserved).

        An FP32-logits head is returned unchanged; a :class:`Float32LogitsConvertible`
        head converts itself; a head of a linear kernel backend (``nn.Linear``,
        Transformer Engine or quack ``Linear``) becomes a :class:`Float32LogitsLinear`
        owning the same ``weight`` and ``bias`` Parameters, so parameter names, tied
        embeddings, optimizer state, checkpoints and rollout weight sync are
        unchanged. A Transformer Engine head under FP8 autocast then projects in its
        compute dtype instead.

        Args:
            head: LM head whose weight has shape [vocab, hidden] and optional bias shape [vocab].

        Returns:
            A head whose projection writes FP32 logits.

        Raises:
            ValueError: If ``head`` owns a projection that cannot write FP32 logits,
                such as LoRA, QAT or quantized linears.
        """
        if isinstance(head, Float32LogitsLinear):
            return head
        if isinstance(head, Float32LogitsConvertible):
            return head.to_fp32_logits()
        if type(head) not in _linear_backend_classes():
            raise ValueError(
                "FP32 logits require an LM head built from a linear kernel backend (torch, te or quack) or one "
                f"that converts itself (Float32LogitsConvertible); got {type(head).__name__}. LoRA, QAT and "
                "quantized linears own their projection; keep them off the LM head."
            )
        fp32_head = cls(head.in_features, head.out_features, bias=False, device="meta")
        fp32_head.weight = head.weight
        # Transformer Engine keeps an empty, unregistered tensor for a disabled bias.
        if isinstance(head.bias, nn.Parameter):
            fp32_head.bias = head.bias
        fp32_head.train(head.training)
        return fp32_head


def head_compute_dtype(head: nn.Module) -> torch.dtype | None:
    """Return the dtype in which ``head``'s GEMM reads its weight.

    A head that is its own FSDP unit computes in its mixed-precision policy's
    ``param_dtype``; its sharded weight may still expose the storage dtype before
    the pre-forward hook runs. Any other head computes in its weight's dtype.

    Args:
        head: LM-head module.

    Returns:
        The compute dtype, or None when the head has no weight tensor.
    """
    if isinstance(head, FSDPModule):
        param_dtype = head._get_fsdp_state()._mp_policy.param_dtype
        if param_dtype is not None:
            return param_dtype
    weight = getattr(head, "weight", None)
    return weight.dtype if isinstance(weight, torch.Tensor) else None

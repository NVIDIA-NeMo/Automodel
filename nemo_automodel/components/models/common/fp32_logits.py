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

"""LM-head projection that keeps BF16/FP16 operands and writes FP32 logits.

Upcasting already-rounded BF16 logits does not recover their precision, and
projecting with an FP32 copy of the head weight costs a transient weight-sized
allocation and an FP32 GEMM. Instead the head GEMM reads the low-precision
hidden states and weight, accumulates in FP32 and stores FP32 logits, as vLLM's
``head_dtype=float32`` path and Megatron-LM's ``logit_dtype`` do.

``torch.mm(..., out_dtype=torch.float32)`` has no autograd formula, so training
goes through :class:`_Float32LogitsMatmul`. Its backward follows Megatron-LM's
``output_dtype`` contract: the FP32 logits gradient is cast to the operand dtype
and the input and weight gradients use the same GEMMs as a BF16-output head.

Parameter storage, optimizer master weights and gradient reduction stay with the
model and its FSDP policy. The GEMM operands are the head's compute-dtype weight
(for example the FSDP-gathered BF16 weight) and the hidden states cast to it.
"""

import math

import torch
from torch import nn
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard

__all__ = ["Float32LogitsLinear", "linear_with_fp32_logits"]


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


def linear_with_fp32_logits(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Project hidden states through a bias-free LM-head weight into FP32 logits.

    The GEMM reads ``input`` cast to ``weight.dtype`` and ``weight`` unchanged,
    accumulates in FP32 and stores FP32 logits. Autocast is disabled inside so it
    cannot lower the output precision.

    Args:
        input: Tensor of shape [..., hidden], with arbitrary leading dimensions, of
            any floating dtype. With a DTensor ``weight`` it may be a DTensor on the
            same mesh with any placement.
        weight: Tensor of shape [vocab, hidden] in BF16, FP16 or FP32. A DTensor must
            have global shape [vocab, hidden] and placements ``(Shard(0),)`` on a 1-D
            tensor-parallel mesh.

    Returns:
        New FP32 tensor of shape [..., vocab]. For a DTensor ``weight`` it is a
        DTensor of that global shape with placement ``Shard(-1)``. Neither input is
        mutated.
    """
    with torch.autocast(device_type=input.device.type, enabled=False):
        if isinstance(weight, DTensor):
            return _vocab_parallel_fp32_logits(input, weight)
        return _Float32LogitsMatmul.apply(input.to(weight.dtype), weight)


class Float32LogitsLinear(nn.Linear):
    """Bias-free LM head whose GEMM writes FP32 logits from its compute-dtype operands.

    It owns the replaced head's ``weight`` Parameter, so parameter names, storage,
    tied embeddings, optimizer state, checkpoints and rollout weight sync are
    unchanged. Only the projection's output dtype differs from ``nn.Linear``.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Project hidden states into FP32 logits.

        Args:
            input: Tensor of shape [..., hidden], with arbitrary leading dimensions.

        Returns:
            FP32 tensor of shape [..., vocab] (see :func:`linear_with_fp32_logits`
            for the tensor-parallel DTensor layout).
        """
        return linear_with_fp32_logits(input, self.weight)

    @classmethod
    def from_linear(cls, head: nn.Linear) -> "Float32LogitsLinear":
        """Build a head that owns ``head.weight`` itself (no copy, identity preserved).

        Args:
            head: Plain bias-free ``nn.Linear`` LM head whose weight has shape [vocab, hidden].

        Returns:
            A new module whose ``weight`` is the same Parameter object as ``head.weight``.

        Raises:
            ValueError: If ``head`` has a bias or is not exactly ``nn.Linear``.
        """
        # Linear subclasses (LoRA, FP8, QAT, bitsandbytes) own their projection,
        # so its output precision cannot be changed from outside.
        if type(head) is not nn.Linear:
            raise ValueError(
                f"FP32 logits require a plain nn.Linear LM head; got {type(head).__name__}. "
                "Keep LoRA, quantization and FP8 off the LM head."
            )
        if head.bias is not None:
            raise ValueError("FP32 logits require a bias-free LM head.")
        fp32_head = cls(head.in_features, head.out_features, bias=False, device="meta")
        fp32_head.weight = head.weight
        fp32_head.train(head.training)
        return fp32_head

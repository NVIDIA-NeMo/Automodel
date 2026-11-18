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

"""Project hidden states into vocabulary logits at the requested precision."""

import math

import torch
from torch import nn
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard


class LogitsProcessor(nn.Module):
    """Apply a vocabulary projection to hidden states using the configured output dtype."""

    def __init__(self, head_dtype: torch.dtype | None = None) -> None:
        """Set the output precision.

        Args:
            head_dtype: None uses the ordinary linear projection with ambient autocast.
                torch.float32 stores FP32 logits from operands in the head weight's dtype.
        """
        super().__init__()
        if head_dtype not in (None, torch.float32):
            raise ValueError(f"head_dtype must be None or torch.float32; got {head_dtype}.")
        self.head_dtype = head_dtype

    def forward(
        self, weight: torch.Tensor, hidden_states: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Project using the weight materialized by the head or the training loss.

        Args:
            weight: Head weight shaped [vocab, hidden]. FP32 output also supports
                a vocabulary-sharded DTensor.
            hidden_states: Activations shaped [..., hidden].
            bias: Optional local head bias shaped [vocab].

        Returns:
            Logits shaped [..., vocab]. A vocabulary-sharded FP32 projection
            returns a DTensor with the vocabulary dimension sharded.
        """
        if self.head_dtype is None:
            return nn.functional.linear(hidden_states.to(weight.dtype), weight, bias)
        if isinstance(weight, DTensor):
            if bias is not None:
                raise NotImplementedError("FP32 logits with tensor parallelism need a bias-free LM head.")
            return _vocab_parallel_fp32_logits(hidden_states, weight)
        logits = _fp32_logits(hidden_states.to(weight.dtype), weight)
        return logits if bias is None else logits + bias.to(self.head_dtype)


def _fp32_logits_forward(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Project local hidden states using the vLLM FP32-output GEMM.

    Args:
        input: Tensor of shape [..., hidden] in the head's compute dtype.
        weight: Tensor of shape [vocab, hidden] in the same dtype.

    Returns:
        FP32 logits of shape [..., vocab].
    """
    flat = input.reshape(-1, input.shape[-1])
    with torch.autocast(device_type=flat.device.type, enabled=False):
        if flat.is_cuda and flat.dtype in (torch.float16, torch.bfloat16):
            logits = torch.mm(flat, weight.t(), out_dtype=torch.float32)
        else:
            logits = torch.nn.functional.linear(flat.float(), weight.float())
    return logits.view(*input.shape[:-1], weight.shape[0])


# PyTorch 2.10 has no derivative for aten::mm.dtype. Register it on our own
# projection so the forward uses the same single GEMM during training and rollout.
_fp32_logits = torch.library.custom_op("nemo_automodel::fp32_logits", _fp32_logits_forward, mutates_args=())
_fp32_logits.register_fake(_fp32_logits_forward)


def _save_projection_inputs(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def _fp32_logits_backward(ctx, grad_logits):
    """Project [..., vocab] gradients back to [..., hidden] and [vocab, hidden]."""
    input, weight = ctx.saved_tensors
    grad = grad_logits.reshape(-1, weight.shape[0]).to(weight.dtype)
    with torch.autocast(device_type=weight.device.type, enabled=False):
        grad_input = (grad @ weight).view_as(input) if ctx.needs_input_grad[0] else None
        grad_weight = grad.t() @ input.reshape(-1, input.shape[-1]) if ctx.needs_input_grad[1] else None
    return grad_input, grad_weight


_fp32_logits.register_autograd(_fp32_logits_backward, setup_context=_save_projection_inputs)


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
    local_logits = _fp32_logits(
        input.to_local(grad_placements=[Partial()]).to(weight.dtype),
        weight.to_local(grad_placements=[Shard(0)]),
    )
    # Pass the global shape so a vocabulary that the mesh does not divide evenly is kept exact.
    shape = (*input.shape[:-1], weight.shape[0])
    stride = tuple(math.prod(shape[dim + 1 :]) for dim in range(len(shape)))
    return DTensor.from_local(
        local_logits, mesh, [Shard(local_logits.ndim - 1)], run_check=False, shape=shape, stride=stride
    )

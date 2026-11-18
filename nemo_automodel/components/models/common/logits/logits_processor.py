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
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from transformers.modeling_outputs import CausalLMOutputWithPast


class LogitsProcessor(nn.Module):
    """Apply a vocabulary projection to hidden states using the configured output dtype."""

    def __init__(self, out_dtype: torch.dtype | None = None, *, head_dtype: torch.dtype | None = None) -> None:
        """Set the output precision.

        Args:
            out_dtype: None uses the ordinary linear projection with ambient autocast.
                torch.float32 stores FP32 logits from operands in the head weight's dtype.
            head_dtype: Previous keyword for out_dtype.
        """
        super().__init__()
        if head_dtype is not None:
            if out_dtype is not None and out_dtype != head_dtype:
                raise ValueError("out_dtype and head_dtype must agree when both are provided.")
            out_dtype = head_dtype
        if out_dtype not in (None, torch.float32):
            raise ValueError(f"out_dtype must be None or torch.float32; got {out_dtype}.")
        self.head_dtype = out_dtype

    @property
    def out_dtype(self) -> torch.dtype | None:
        """Requested projection output dtype."""
        return self.head_dtype

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
        if self.out_dtype is None:
            return nn.functional.linear(hidden_states.to(weight.dtype), weight, bias)
        if isinstance(weight, DTensor):
            if bias is not None:
                raise NotImplementedError("FP32 logits with tensor parallelism need a bias-free LM head.")
            return _vocab_parallel_fp32_logits(hidden_states, weight)
        logits = _FP32Logits.apply(hidden_states.to(weight.dtype), weight)
        return logits if bias is None else logits + bias.to(self.out_dtype)

    def compute_lm_head_logits(
        self,
        lm_head: nn.Module | torch.Tensor | None,
        hidden_states: torch.Tensor,
        logits_to_keep: int | torch.Tensor = 0,
        is_thd: bool = False,
        output_hidden_states: bool = False,
        *,
        bias: torch.Tensor | None = None,
    ) -> CausalLMOutputWithPast:
        """Select positions, project the LM head, and package causal model outputs.

        Configure a hooked or sharded module with ``enable_fp32_logits`` before
        requesting FP32 output so its hooks run around the projection.

        Args:
            lm_head: Head module, [vocab, hidden] weight, or None on a pipeline
                stage that passes hidden states through.
            hidden_states: Final activations shaped [tokens, hidden] or [batch, seq, hidden].
            logits_to_keep: Zero for all positions, a tail length, or explicit indices.
            is_thd: Restore the batch dimension on packed 2D outputs.
            output_hidden_states: Include the original full sequence in the result.
            bias: Optional vocabulary bias when passing a local weight tensor.

        Returns:
            CausalLMOutputWithPast with logits and optional hidden states.
        """
        if lm_head is None:
            logits = hidden_states
        else:
            if isinstance(logits_to_keep, int) and logits_to_keep == 0:
                sliced = hidden_states
            else:
                slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                if hidden_states.dim() == 2:
                    sliced = hidden_states[slice_indices, :]
                else:
                    sliced = hidden_states[:, slice_indices, :]
            if isinstance(lm_head, torch.Tensor):
                logits = self(lm_head, sliced, bias)
            elif self.out_dtype is None or getattr(lm_head, "fp32_logits", False):
                logits = lm_head(sliced.to(head_compute_dtype(lm_head)))
            elif (
                type(lm_head) is nn.Linear
                and "forward" not in vars(lm_head)
                and not (
                    lm_head._forward_pre_hooks
                    or lm_head._forward_hooks
                    or lm_head._backward_hooks
                    or lm_head._backward_pre_hooks
                )
            ):
                logits = self(lm_head.weight, sliced, lm_head.bias)
            else:
                raise ValueError("Configure the output head with enable_fp32_logits before applying FP32 projection.")
        if is_thd and logits.dim() == 2:
            logits = logits.unsqueeze(0)

        hidden_out = None
        if output_hidden_states:
            hidden_out = hidden_states.unsqueeze(0) if is_thd and hidden_states.dim() == 2 else hidden_states

        return CausalLMOutputWithPast(logits=logits, hidden_states=hidden_out)


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


_fp32_logits = torch.library.custom_op("nemo_automodel::fp32_logits", _fp32_logits_forward, mutates_args=())
_fp32_logits.register_fake(_fp32_logits_forward)


class _FP32Logits(torch.autograd.Function):
    """Trainable FP32-output GEMM. PyTorch 2.10 lacks a derivative for aten::mm.dtype."""

    @staticmethod
    def forward(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return _fp32_logits(input, weight)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(*inputs)

    @staticmethod
    def backward(ctx, grad_logits):
        """Project [..., vocab] gradients back to [..., hidden] and [vocab, hidden]."""
        input, weight = ctx.saved_tensors
        grad = grad_logits.reshape(-1, weight.shape[0]).to(weight.dtype)
        with torch.autocast(device_type=weight.device.type, enabled=False):
            grad_input = (grad @ weight).view_as(input) if ctx.needs_input_grad[0] else None
            grad_weight = grad.t() @ input.reshape(-1, input.shape[-1]) if ctx.needs_input_grad[1] else None
        return grad_input, grad_weight


# Exported graphs and direct torch.ops callers use the same gradient formula.
_fp32_logits.register_autograd(_FP32Logits.backward, setup_context=_FP32Logits.setup_context)


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
    local_logits = _FP32Logits.apply(
        input.to_local(grad_placements=[Partial()]).to(weight.dtype),
        weight.to_local(grad_placements=[Shard(0)]),
    )
    # Pass the global shape so a vocabulary that the mesh does not divide evenly is kept exact.
    shape = (*input.shape[:-1], weight.shape[0])
    stride = tuple(math.prod(shape[dim + 1 :]) for dim in range(len(shape)))
    return DTensor.from_local(
        local_logits, mesh, [Shard(local_logits.ndim - 1)], run_check=False, shape=shape, stride=stride
    )


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

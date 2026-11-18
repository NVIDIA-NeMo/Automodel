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

"""LM-head projection, FP32 output, and its training backward."""

import math
import sys
from collections.abc import Callable
from functools import partial

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from transformers.modeling_outputs import CausalLMOutputWithPast


def compute_lm_head_logits(
    lm_head: nn.Module | torch.Tensor | None,
    hidden_states: torch.Tensor,
    logits_to_keep: int | torch.Tensor = 0,
    is_thd: bool = False,
    output_hidden_states: bool = False,
    *,
    bias: torch.Tensor | None = None,
) -> CausalLMOutputWithPast:
    """Project hidden states through ``lm_head`` and wrap the result.

    Centralizes the lm_head projection and output packaging shared by every
    custom ``*ForCausalLM`` / ``*ForConditionalGeneration`` ``forward()``. The
    returned ``CausalLMOutputWithPast`` carries the projected ``logits`` and,
    when requested, the final ``hidden_states``; callers that also need ``loss``,
    ``past_key_values``, etc. read ``.logits`` and build their own output.

    - ``lm_head is None`` (e.g. a non-final pipeline-parallel stage that does not
      own the head): ``hidden_states`` is passed through as ``logits`` so the
      next stage receives it.
    - ``logits_to_keep == 0`` (training default): every position is projected.
      The full range is deliberately *not* sliced, because ``slice(0, None)`` on
      a DTensor is unsupported (it raises on the ``aten.alias`` op under tensor
      parallel with sequence parallelism).
    - ``logits_to_keep`` as a positive int or a tensor of indices: only the
      requested positions are projected. Both 2D ``[T, H]`` (THD/packed) and 3D
      ``[B, S, H]`` (BSHD) hidden states are handled.
    - ``is_thd``: THD/packed inputs yield 2D ``[T, V]`` logits; the leading batch
      dim is restored (``unsqueeze(0)`` -> ``[1, T, V]``) so downstream code sees
      a uniform ``[B, S, V]`` layout. The same restoration is applied to the
      ``hidden_states`` field. Only applied while the tensor is still 2D, so an
      ``inputs_embeds`` path that already produced ``[1, T, *]`` is left
      untouched.
    - The sliced input is cast to the head's compute dtype (``head_compute_dtype``):
      a standalone FSDP2 head's mixed-precision ``param_dtype``, else its weight
      dtype. The projection goes through the ``lm_head`` module (DTensor-aware
      under FSDP2) rather than ``F.linear`` so its FSDP hooks and DTensor
      redistribution run, and its output is returned uncast: a
      head configured for FP32 logits (``logits_dtype=float32``, or a head that is
      FP32 by design) writes FP32 logits, any other head its compute dtype.
    - A bare weight tensor is projected in its compute dtype with FP32 logits.
      This path is used by callers that already own or have materialized the weight.
    - ``output_hidden_states``: when set, the (full-sequence, THD-restored)
      ``hidden_states`` are attached to the output so the fused cross-entropy
      path can recompute logits over every position; otherwise the field is
      ``None``.

    Args:
        lm_head: The head module, a projection weight of shape [vocab, hidden],
            or ``None`` on a pipeline stage without a head.
        hidden_states: Final hidden states, shaped ``[T, H]`` or ``[B, S, H]``.
        logits_to_keep: ``0`` to project every position; a positive int to keep
            the last ``N`` positions; or a tensor of position indices.
        is_thd: Whether the inputs were THD/packed; if so, a 2D logits (and
            hidden-states) result is unsqueezed back to a leading batch dim of 1.
        output_hidden_states: Attach the final hidden states to the output.
        bias: Optional bias [vocab] when passing a local weight tensor.

    Returns:
        A ``CausalLMOutputWithPast`` whose ``logits`` are the projected logits
        (or ``hidden_states`` unchanged when ``lm_head`` is ``None``) and whose
        ``hidden_states`` are the final hidden states when ``output_hidden_states``
        is set, else ``None``.
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
        if isinstance(lm_head, DTensor):
            if bias is not None:
                raise NotImplementedError("FP32 logits with tensor parallelism need a bias-free LM head.")
            logits = _vocab_parallel_fp32_logits(sliced, lm_head)
        elif isinstance(lm_head, torch.Tensor):
            logits = _fp32_logits(sliced.to(lm_head.dtype), lm_head)
            if bias is not None:
                logits = logits + bias.float()
        else:
            logits = lm_head(sliced.to(head_compute_dtype(lm_head)))
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


def fp32_linear_forward(head: nn.Linear, input: torch.Tensor) -> torch.Tensor:
    """Project with the head's current parameters so FSDP and tied weights are respected."""
    return compute_lm_head_logits(head.weight, input, bias=head.bias).logits


def enable_fp32_logits(
    head: nn.Module, *, forward: Callable[[nn.Linear, torch.Tensor], torch.Tensor] = fp32_linear_forward
) -> nn.Module:
    """Configure the head to write FP32 logits while retaining its parameters.

    Native heads with a dedicated projection expose the ``fp32_logits`` switch.
    Plain linears run ``forward(head, input)`` through their existing module hooks.
    """
    if hasattr(head, "fp32_logits"):
        # Baichuan normalizes its weight and Inkling2 computes in FP32 itself.
        head.fp32_logits = True
        return head
    if type(head) is not nn.Linear:
        # Backend linears may have extra state or an empty tensor for a disabled bias.
        if type(head) not in (
            getattr(sys.modules.get("transformer_engine.pytorch.module.linear"), "Linear", None),
            getattr(sys.modules.get("quack.linear"), "Linear", None),
        ):
            raise ValueError(f"FP32 logits require a plain linear kernel backend; got {type(head).__name__}.")
        linear = nn.Linear(head.in_features, head.out_features, bias=False, device="meta")
        linear.weight = head.weight
        if isinstance(head.bias, nn.Parameter):
            linear.bias = head.bias
        head = linear.train(head.training)
    head.forward = partial(forward, head)
    head.fp32_logits = True
    return head


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

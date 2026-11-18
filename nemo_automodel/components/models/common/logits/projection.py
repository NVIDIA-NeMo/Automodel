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

"""LM-head configuration and causal model output packaging."""

import sys
from collections.abc import Callable
from functools import partial

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common.logits.logits_processor import LogitsProcessor

_fp32_processor = LogitsProcessor(head_dtype=torch.float32)


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
        if isinstance(lm_head, torch.Tensor):
            logits = _fp32_processor(lm_head, sliced, bias)
        else:
            logits = lm_head(sliced.to(head_compute_dtype(lm_head)))
    if is_thd and logits.dim() == 2:
        logits = logits.unsqueeze(0)

    hidden_out = None
    if output_hidden_states:
        hidden_out = hidden_states.unsqueeze(0) if is_thd and hidden_states.dim() == 2 else hidden_states

    return CausalLMOutputWithPast(logits=logits, hidden_states=hidden_out)


def fp32_linear_forward(head: nn.Linear, input: torch.Tensor) -> torch.Tensor:
    """Project with the head's current parameters so FSDP and tied weights are respected."""
    return _fp32_processor(head.weight, input, head.bias)


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

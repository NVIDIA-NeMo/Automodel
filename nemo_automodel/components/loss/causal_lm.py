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

"""Shared causal-LM loss calculation for the LLM and VLM recipes."""

from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.loss.mtp import MTPLossConfig, calculate_mtp_loss
from nemo_automodel.components.loss.utils import _get_final_hidden_states, _get_lm_head_weight, calculate_loss


def causal_lm_loss(
    loss_fn: nn.Module,
    model: nn.Module,
    output: Any,
    labels: torch.Tensor,
    mtp_config: MTPLossConfig | None,
    *,
    num_label_tokens: int | None,
    grad_reduce_group: dist.ProcessGroup | None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the main causal-LM loss plus an optional MTP loss.

    Args:
        loss_fn: Configured causal-LM loss module.
        model: Model that owns the LM head used by fused loss implementations.
        output: Model output containing logits of shape ``[batch, sequence,
            vocab]`` or final hidden states of shape ``[batch, sequence,
            hidden]``. Optional MTP fields use the same batch and sequence
            axes per prediction depth.
        labels: Target token ids of shape ``[batch, sequence]`` or ``[tokens]``
            for a flattened THD stream.
        mtp_config: MTP loss settings, required only when ``output`` contains
            MTP predictions.
        num_label_tokens: Global supervised-token denominator. ``None`` keeps
            each loss as an unnormalized local sum for Engine normalization.
        grad_reduce_group: Group that contributes independent fused-loss
            shards, or ``None`` for an unsharded LM head.
        cu_seqlens: Optional THD cumulative sequence offsets of shape
            ``[num_sequences + 1]``.

    Returns:
        Scalar causal-LM loss retaining its autograd graph.
    """
    hidden_states = _get_final_hidden_states(output)
    if isinstance(loss_fn, FusedLinearCrossEntropy) and hidden_states is None:
        raise ValueError("FusedLinearCrossEntropy requires the model to output hidden states")

    lm_weight = (
        loss_fn.materialize_lm_weight(
            _get_lm_head_weight(model),
            grad_reduce_group=grad_reduce_group,
        )
        if isinstance(loss_fn, FusedLinearCrossEntropy)
        else None
    )
    loss = calculate_loss(
        loss_fn,
        logits=getattr(output, "logits", output),
        labels=labels,
        model=model,
        hidden_states=hidden_states,
        lm_weight=lm_weight,
        grad_reduce_group=grad_reduce_group,
        num_label_tokens=num_label_tokens,
    )

    mtp_hidden = getattr(output, "mtp_per_depth_h", None)
    mtp_logits = getattr(output, "mtp_per_depth_logits", None)
    if mtp_hidden is None and mtp_logits is None:
        return loss
    if mtp_config is None:
        raise ValueError("MTP model output requires an MTP loss config")

    scaling_factor = (
        mtp_config.scaling_factor if mtp_config.scaling_factor is not None else output.mtp_loss_scaling_factor
    )
    return loss + calculate_mtp_loss(
        loss_fn,
        mtp_per_depth_h=mtp_hidden,
        mtp_per_depth_logits=mtp_logits,
        labels=labels,
        model=model,
        scaling_factor=scaling_factor,
        num_label_tokens=num_label_tokens,
        ignore_index=mtp_config.ignore_index,
        cu_seqlens=cu_seqlens,
        lm_weight=lm_weight,
        grad_reduce_group=grad_reduce_group,
    )

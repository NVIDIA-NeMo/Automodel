# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Shared per-microbatch forward + LM-loss step for the finetune recipes.

The LLM and VLM finetune recipes ran near-identical ``forward -> loss`` code in
their ``_forward_backward_step`` methods. This module hoists the modality-agnostic
core -- the (``FusedLinearCrossEntropy``-aware) model forward, the main LM loss,
and the optional MTP loss -- into one function so the two recipes cannot drift,
and so other callers (e.g. the tinker ``Engine``) can reuse it and run their own
extraction on the returned model output.

Callers still own everything modality- or context-specific: batch preparation
(device move, CP/THD sharding, ``labels`` pop), the train/sync/fp8 context
managers, the pipeline-parallel path, any extra model-output-driven losses
(e.g. speculative-drafter co-training or the CP full-logits grad touch), and the
backward call.
"""

from typing import Any, Optional

import torch

from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.loss.mtp import calculate_mtp_loss
from nemo_automodel.components.loss.utils import _get_lm_head_weight, calculate_loss
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states

__all__ = ["forward_backward_step"]


def forward_backward_step(
    model: torch.nn.Module,
    batch: dict[str, Any],
    labels: torch.Tensor,
    loss_fn: Any,
    *,
    num_label_tokens: Optional[int] = None,
    mtp_cfg: Any = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    grad_reduce_group: Optional[torch.distributed.ProcessGroup] = None,
) -> tuple[Any, torch.Tensor]:
    """Forward the model and compute the (main + optional MTP) LM loss.

    This is the modality-agnostic core shared by the LLM and VLM finetune recipes'
    non-pipeline forward/backward path. The caller is responsible for preparing
    ``batch`` (device placement, CP/THD sharding, ``filter_forward_kwargs``, and
    popping ``labels``), for the surrounding train/sync/fp8 context managers, for
    any extra model-output-driven losses, and for scaling + calling ``.backward()``
    on the returned loss.

    Args:
        model: The (sharded) model part to forward.
        batch: Model forward kwargs, already device-placed / CP-sharded, with
            ``labels`` popped out by the caller.
        labels: Next-token targets for the loss.
        loss_fn: The recipe's loss module. A ``FusedLinearCrossEntropy`` instance
            triggers the hidden-states path (``logits_to_keep=1`` plus a shared
            LM-head weight reused across the main and MTP losses).
        num_label_tokens: Global label-token count for loss normalization.
        mtp_cfg: MTP config exposing ``scaling_factor`` / ``ignore_index``. When the
            model emits ``mtp_per_depth_*`` outputs and this is not ``None``, the
            MTP loss is added.
        cu_seqlens: THD packing boundaries forwarded to the MTP loss to mask
            cross-sequence label rolls (``None`` for unpacked batches).
        grad_reduce_group: Process group whose ranks contribute independent
            fused-loss shards (the flattened DP-CP group during training).
            Forwarded to ``FusedLinearCrossEntropy.materialize_lm_weight`` so
            LM-head gradients are reduced correctly; pass ``None`` when no
            backward follows (e.g. validation).

    Returns:
        ``(out, local_loss)`` -- the raw model output and the summed (main + MTP)
        loss, not yet scaled or backward-ed.
    """
    use_fused = isinstance(loss_fn, FusedLinearCrossEntropy)
    if use_fused:
        # num_logits_to_keep avoids materializing the full logits matrix in memory.
        out = model(logits_to_keep=1, **batch)
        if "hidden_states" not in out:
            raise ValueError(
                "FusedLinearCrossEntropy requires the model to output hidden states. "
                "Set `output_hidden_states=True` in the model config."
            )
    else:
        out = model(**batch)

    # Materialize the LM head once and share it across the main loss and every
    # MTP depth (fused path) to avoid redundant full_tensor() gathers that
    # accumulate on-device and OOM for long sequences. The grad-reduce group
    # only exists on the fused path, so it is threaded as an extra kwarg to
    # keep non-fused loss callables free of it.
    loss_distributed_kwargs: dict[str, Any] = {}
    shared_lm_weight = None
    if use_fused:
        shared_lm_weight = loss_fn.materialize_lm_weight(
            _get_lm_head_weight(model), grad_reduce_group=grad_reduce_group
        )
        loss_distributed_kwargs["grad_reduce_group"] = grad_reduce_group
    local_loss = calculate_loss(
        loss_fn,
        logits=getattr(out, "logits", out),
        labels=labels,
        model=model,
        hidden_states=get_final_hidden_states(out),
        lm_weight=shared_lm_weight,
        num_label_tokens=num_label_tokens,
        **loss_distributed_kwargs,
    )

    # DSV4-style multi-token-prediction: triggered when the model emits per-depth
    # hidden states / logits. Mutually exclusive with the drafter path, which the
    # caller adds on top of the returned loss.
    mtp_per_depth_h = getattr(out, "mtp_per_depth_h", None)
    mtp_per_depth_logits = getattr(out, "mtp_per_depth_logits", None)
    if (mtp_per_depth_h is not None or mtp_per_depth_logits is not None) and mtp_cfg is not None:
        scaling_factor = mtp_cfg.scaling_factor if mtp_cfg.scaling_factor is not None else out.mtp_loss_scaling_factor
        local_loss = local_loss + calculate_mtp_loss(
            loss_fn,
            mtp_per_depth_h=mtp_per_depth_h,
            mtp_per_depth_logits=mtp_per_depth_logits,
            labels=labels,
            model=model,
            scaling_factor=scaling_factor,
            num_label_tokens=num_label_tokens,
            ignore_index=mtp_cfg.ignore_index,
            # mask cross-boundary MTP label rolls in THD packing (no-op when None)
            cu_seqlens=cu_seqlens,
            lm_weight=shared_lm_weight,
            **loss_distributed_kwargs,
        )

    return out, local_loss

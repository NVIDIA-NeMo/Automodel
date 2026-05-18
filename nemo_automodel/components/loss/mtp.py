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

from typing import Optional

import torch
import torch.nn as nn

from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.loss.utils import _get_final_hidden_states, _get_lm_head_module, calculate_loss
from nemo_automodel.components.models.common.mtp import get_mtp_loss_scaling_factor, roll_tensor


def calculate_mtp_loss(
    loss_fn,
    *,
    mtp_per_depth_h: list[torch.Tensor],
    labels: torch.Tensor,
    model: nn.Module,
    scaling_factor: float = 0.1,
    num_label_tokens: Optional[int] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute the DeepSeek-V3 Multi-Token Prediction auxiliary loss.

    Each depth's CE is dispatched through :func:`calculate_loss` with the
    same loss class as the main path, so MTP inherits FusedLinearCrossEntropy
    / MaskedCrossEntropy memory and numerical characteristics.

    Args:
        loss_fn: Configured per-token loss class (same instance the main
            path uses).
        mtp_per_depth_h: Per-depth hidden states from the model's MTP head,
            one ``[B, S, H]`` tensor per depth.
        labels: Original (unshifted) labels.
        model: The wrapped model; used to fetch the shared LM head when the
            loss class needs materialized logits (non-FusedLinearCE path).
        scaling_factor: Coefficient applied to the summed per-depth CE.
        num_label_tokens: Total non-ignore label tokens (forwarded to the
            base loss for sum-reduction normalization).
        ignore_index: Label value masked out of the CE loss for the trailing
            ``k+1`` rolled positions at depth ``k``.

    Returns:
        Scalar MTP loss with autograd graph.
    """
    D = len(mtp_per_depth_h)
    cur_labels = labels
    total = mtp_per_depth_h[0].new_zeros(())
    for k, h_k in enumerate(mtp_per_depth_h):
        cur_labels = roll_tensor(cur_labels, shifts=-1, dim=-1)
        masked = cur_labels.clone()
        n_invalid = min(k + 1, masked.shape[-1])
        masked[..., -n_invalid:] = ignore_index

        if isinstance(loss_fn, FusedLinearCrossEntropy):
            depth_loss = calculate_loss(
                loss_fn,
                hidden_states=h_k,
                labels=masked,
                model=model,
                num_label_tokens=num_label_tokens,
            )
        else:
            lm_head = _get_lm_head_module(model)
            if lm_head is None:
                raise ValueError("lm_head module not found in model")
            depth_loss = calculate_loss(
                loss_fn,
                logits=lm_head(h_k),
                labels=masked,
                model=model,
                num_label_tokens=num_label_tokens,
            )
        total = total + depth_loss

    return total * (scaling_factor / D)


class PipelineCausalLMLoss(nn.Module):
    """Pipeline schedule loss that can add MTP auxiliary CE on the last stage."""

    def __init__(self, loss_fn: nn.Module, model: nn.Module):
        super().__init__()
        self.loss_fn = loss_fn
        self.model = model

    def forward(self, output, labels: torch.Tensor) -> torch.Tensor:
        if isinstance(output, tuple):
            logits = output[0]
            hidden_states = None
            mtp_per_depth_h = list(output[1:]) if len(output) > 1 else None
            scaling_factor = get_mtp_loss_scaling_factor(self.model)
        else:
            logits = getattr(output, "logits", output)
            hidden_states = _get_final_hidden_states(output)
            mtp_per_depth_h = getattr(output, "mtp_per_depth_h", None)
            scaling_factor = getattr(output, "mtp_loss_scaling_factor", get_mtp_loss_scaling_factor(self.model))

        loss = calculate_loss(
            self.loss_fn,
            logits=logits,
            labels=labels,
            model=self.model,
            hidden_states=hidden_states,
        )
        if mtp_per_depth_h is not None and self.model.training:
            loss = loss + calculate_mtp_loss(
                self.loss_fn,
                mtp_per_depth_h=mtp_per_depth_h,
                labels=labels,
                model=self.model,
                scaling_factor=scaling_factor,
            )
        return loss

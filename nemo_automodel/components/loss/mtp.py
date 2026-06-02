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

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.loss.utils import _get_final_hidden_states, _get_lm_head_module, calculate_loss
from nemo_automodel.components.models.common.mtp import get_mtp_loss_scaling_factor, roll_tensor


def calculate_mtp_loss(
    loss_fn,
    *,
    mtp_per_depth_h: list[torch.Tensor] | None = None,
    mtp_per_depth_logits: list[torch.Tensor] | None = None,
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
    if (mtp_per_depth_h is None) == (mtp_per_depth_logits is None):
        raise ValueError("Provide exactly one of mtp_per_depth_h or mtp_per_depth_logits")

    mtp_outputs = mtp_per_depth_logits if mtp_per_depth_logits is not None else mtp_per_depth_h
    D = len(mtp_outputs)
    cur_labels = labels
    total = mtp_outputs[0].new_zeros(())
    for k, mtp_output in enumerate(mtp_outputs):
        cur_labels = roll_tensor(cur_labels, shifts=-1, dim=-1)
        masked = cur_labels.clone()
        n_invalid = min(k + 1, masked.shape[-1])
        masked[..., -n_invalid:] = ignore_index

        if mtp_per_depth_logits is not None:
            if isinstance(loss_fn, FusedLinearCrossEntropy):
                raise ValueError("MTP logits are incompatible with FusedLinearCrossEntropy")
            depth_loss = calculate_loss(
                loss_fn,
                logits=mtp_output,
                labels=masked,
                model=model,
                num_label_tokens=num_label_tokens,
            )
        elif isinstance(loss_fn, FusedLinearCrossEntropy):
            depth_loss = calculate_loss(
                loss_fn,
                hidden_states=mtp_output,
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
                logits=lm_head(mtp_output),
                labels=masked,
                model=model,
                num_label_tokens=num_label_tokens,
            )
        total = total + depth_loss

    return total * (scaling_factor / D)


class PipelineCausalLMLoss(nn.Module):
    """Pipeline schedule loss that can add MTP auxiliary CE on the last stage."""

    def __init__(
        self,
        loss_fn: nn.Module,
        model: nn.Module,
        scaling_factor: float | None = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.model = model
        self.scaling_factor = scaling_factor
        self.ignore_index = ignore_index

    def forward(self, output, labels: torch.Tensor) -> torch.Tensor:
        if isinstance(output, tuple):
            logits = output[0]
            hidden_states = None
            mtp_per_depth_h = None
            mtp_per_depth_logits = None
            if len(output) > 1:
                if getattr(self.model, "mtp_outputs_are_logits", False):
                    mtp_per_depth_logits = list(output[1:])
                else:
                    mtp_per_depth_h = list(output[1:])
            model_scaling_factor = get_mtp_loss_scaling_factor(self.model)
        else:
            logits = getattr(output, "logits", output)
            hidden_states = _get_final_hidden_states(output)
            mtp_per_depth_h = getattr(output, "mtp_per_depth_h", None)
            mtp_per_depth_logits = getattr(output, "mtp_per_depth_logits", None)
            model_scaling_factor = getattr(output, "mtp_loss_scaling_factor", get_mtp_loss_scaling_factor(self.model))

        loss = calculate_loss(
            self.loss_fn,
            logits=logits,
            labels=labels,
            model=self.model,
            hidden_states=hidden_states,
        )
        if (mtp_per_depth_h is not None or mtp_per_depth_logits is not None) and self.model.training:
            scaling_factor = self.scaling_factor if self.scaling_factor is not None else model_scaling_factor
            loss = loss + calculate_mtp_loss(
                self.loss_fn,
                mtp_per_depth_h=mtp_per_depth_h,
                mtp_per_depth_logits=mtp_per_depth_logits,
                labels=labels,
                model=self.model,
                scaling_factor=scaling_factor,
                ignore_index=self.ignore_index,
            )
        return loss


@dataclass
class MTPLossConfig:
    """Typed config for the Multi-Token-Prediction auxiliary loss.

    MTP is gated on the model emitting per-depth outputs; this config only
    carries its hyperparameters. ``scaling_factor=None`` keeps the
    model-provided value (``out.mtp_loss_scaling_factor`` /
    ``get_mtp_loss_scaling_factor``); set it to override.
    """

    scaling_factor: float | None = None
    ignore_index: int = -100

    def build(self, loss_fn: nn.Module, model: nn.Module) -> PipelineCausalLMLoss:
        """Build the pipeline-schedule, MTP-aware loss for ``loss_fn``/``model``."""
        return PipelineCausalLMLoss(loss_fn, model, scaling_factor=self.scaling_factor, ignore_index=self.ignore_index)

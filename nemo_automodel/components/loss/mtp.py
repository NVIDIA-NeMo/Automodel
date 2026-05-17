# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Multi-Token Prediction auxiliary loss.

DeepSeek-V3-style MTP head emits a list of per-depth hidden states. This
function computes a CE per depth (using :func:`calculate_loss` to inherit
the FusedLinearCrossEntropy / MaskedCrossEntropy code path of the main loss)
and sums them, scaled by ``scaling_factor / num_depths``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from nemo_automodel.components.loss.calculate import calculate_loss
from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy


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
    from nemo_automodel.components.models.common.mtp import roll_tensor

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
            depth_loss = calculate_loss(
                loss_fn,
                logits=model.get_output_embeddings()(h_k),
                labels=masked,
                model=model,
                num_label_tokens=num_label_tokens,
            )
        total = total + depth_loss

    return total * (scaling_factor / D)


__all__ = ["calculate_mtp_loss"]

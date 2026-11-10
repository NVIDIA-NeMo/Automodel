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

"""Training objective of the Deciffusion decision recipe (hard/soft cross-entropy, optional gold CE + KL)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def decision_loss(
    logits: torch.Tensor,
    gold: int,
    soft: torch.Tensor | None = None,
    *,
    kl_weight: float = 0.0,
) -> torch.Tensor:
    """Loss of one decision over its option codes.

    ``kl_weight == 0``: cross-entropy against ``soft`` when given, else against the one-hot gold label.
    ``kl_weight > 0``: ``CE(gold) + kl_weight * KL(soft || p)``; rows without soft labels use gold CE alone.

    Args:
        logits: Option logits, shape ``[options]``, in prompt order.
        gold: Index of the gold option in ``[0, options)``.
        soft: Target distribution, shape ``[options]`` (non-negative, sums to one), or ``None``.
        kl_weight: Weight of the KL term; must be non-negative.

    Returns:
        Scalar loss.
    """
    if kl_weight < 0:
        raise ValueError("kl_weight must be non-negative")
    if logits.dim() != 1 or not 0 <= gold < logits.numel():
        raise ValueError("logits must be [options] and gold an index into the options")
    if soft is not None and soft.shape != logits.shape:
        raise ValueError("soft labels must align with the option logits")
    log_probs = logits.float().log_softmax(-1)
    if kl_weight > 0:
        loss = -log_probs[gold]
        if soft is not None:
            soft = soft.to(log_probs)
            loss = loss + kl_weight * (soft * (soft.clamp_min(1e-12).log() - log_probs)).sum()
        return loss
    target = soft.to(log_probs) if soft is not None else F.one_hot(torch.tensor(gold), logits.numel()).to(log_probs)
    return -(target * log_probs).sum()

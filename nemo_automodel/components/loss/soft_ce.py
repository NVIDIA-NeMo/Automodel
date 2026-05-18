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

"""Loss utilities for EAGLE-3 training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_soft_cross_entropy(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    position_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute masked soft-target cross entropy.

    Important implementation notes:

    1. The target alignment should still follow the original EAGLE-3 training
       flow: target logits / input ids are shifted in target preparation, as in
       the reference SpecForge implementation.
    2. We intentionally do *not* preserve the original loss reduction from the
       EAGLE / SpecForge code. The reference implementation averages over
       ``batch * seq_len`` even when only a small subset of positions is valid.
       That reduction is not a sound masked-loss definition because the loss
       scale changes with padding / sparse supervision density. Here we
       normalize by the number of valid supervised positions, which is the
       correct masked objective.

    Args:
        logits: Draft logits of shape ``[batch, seq_len, draft_vocab_size]``.
        target_probs: Target distributions aligned to the draft vocabulary.
        position_mask: Boolean/0-1 mask of shape ``[batch, seq_len, 1]``.

    Returns:
        Scalar loss normalized by the number of valid positions.
    """
    log_probs = F.log_softmax(logits.float(), dim=-1)
    per_token_loss = -(target_probs * log_probs).sum(dim=-1)
    valid_mask = position_mask.squeeze(-1).to(per_token_loss.dtype)
    valid_count = valid_mask.sum().clamp_min(1.0)
    return (per_token_loss * valid_mask).sum() / valid_count

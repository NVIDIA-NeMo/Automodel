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

"""Decay-weighted cross-entropy loss for DFlash draft training (Eq. 4 of the paper)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from nemo_automodel.components.loss.dllm_loss import DLLMLossOutput


class DFlashDecayLoss(nn.Module):
    """Position-decay cross-entropy loss for DFlash draft model training.

    Implements Eq. 4 of the DFlash paper:

    .. math::
        w_k = \\exp\\!\\left(-\\frac{k-1}{\\gamma}\\right), \\quad k = 1, \\dots, T

    where *k* indexes the predicted positions within a block (k=0 is the clean
    anchor and is not predicted; k=1 is the first masked position).

    Loss is normalised by the sum of effective weights
    ``(w_k * block_mask)``.  Pass *num_tokens* (a global all-reduced count) for
    normalisation consistent across DP replicas and gradient-accumulation steps.

    Paper default γ values (Appendix A.1):

    - block size 16 → γ = 7
    - block size 10 → γ = 5
    - block size  8 → γ = 4

    Args:
        loss_gamma: Decay parameter γ.
    """

    def __init__(self, loss_gamma: float = 7.0):
        super().__init__()
        self.loss_gamma = float(loss_gamma)

    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        block_mask: torch.Tensor,
        num_tokens: int | None = None,
    ) -> DLLMLossOutput:
        """Compute the DFlash decay-weighted loss.

        Args:
            logits: Draft model logits for the predicted block positions,
                shape ``[B, T, V]`` where ``T = block_size - 1``.
            target_ids: Ground-truth token IDs, shape ``[B, T]``.
            block_mask: Float/bool valid-position mask, shape ``[B, T]``.
                Zero entries (padding) are excluded from the loss.
            num_tokens: Optional global token count for loss normalisation.

        Returns:
            :class:`~nemo_automodel.components.loss.dllm_loss.DLLMLossOutput`.
        """
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()

        B, T, V = logits.shape
        token_nll = F.cross_entropy(
            logits.reshape(-1, V),
            target_ids.reshape(-1).to(logits.device),
            reduction="none",
        ).reshape(B, T)

        # w_k = exp(-(k-1)/gamma) for k=1..T  →  exp(-arange(T)/gamma)
        w = torch.exp(-torch.arange(T, device=logits.device, dtype=token_nll.dtype) / self.loss_gamma)
        weights = w.unsqueeze(0) * block_mask.to(token_nll.dtype)  # [B, T]

        loss = (token_nll * weights).sum()
        if num_tokens is not None:
            loss = loss / max(float(num_tokens), 1.0)

        return DLLMLossOutput(total_loss=loss, dllm_loss=loss.detach().clone())

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

"""Loss functions for diffusion LLM (dLLM) training.

All loss classes return :class:`DLLMLossOutput` so the recipe can handle them
uniformly without branching on model type.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor


def _compute_per_token_nll(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token negative log-likelihood, shape ``[B, L]``."""
    if isinstance(logits, DTensor):
        logits = logits.full_tensor()

    V = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, V),
        target_ids.reshape(-1).to(logits.device),
        reduction="none",
    ).reshape(target_ids.shape)


class DLLMLossOutput(NamedTuple):
    """Unified return type for all dLLM loss functions.

    Attributes:
        total_loss: Loss used for backward (may include AR component).
        dllm_loss: Pure diffusion loss for logging/metrics.
    """

    total_loss: torch.Tensor
    dllm_loss: torch.Tensor


class MDLMCrossEntropyLoss(nn.Module):
    """Cross-entropy loss for MDLM training.

    Matches the reference dllm framework (``dllm/core/trainers/mdlm.py``):

    .. math::
        \\text{loss} = \\frac{\\sum_{i \\in \\text{masked}} \\text{CE}_i \\cdot w(t)}{\\sum \\text{maskable}}

    where :math:`w(t) = 1/t` for the ``scheduler`` weight type (linear schedule).
    """

    def __init__(self, fp32_upcast: bool = True):
        super().__init__()
        self.fp32_upcast = fp32_upcast

    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        noise_mask: torch.Tensor,
        p_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        num_diffusion_tokens: Optional[int] = None,
    ) -> DLLMLossOutput:
        """Compute the MDLM cross-entropy loss.

        Args:
            logits: Model output logits, shape ``[B, L, V]``.
            target_ids: Clean (uncorrupted) token IDs, shape ``[B, L]``.
            noise_mask: Boolean mask of corrupted positions, shape ``[B, L]``.
            p_mask: Per-position masking probability, shape ``[B, L]``.
            loss_mask: Supervised positions mask, shape ``[B, L]``.
            num_diffusion_tokens: If provided, used for global normalization
                (total supervised tokens across all grad-acc microbatches).

        Returns:
            :class:`DLLMLossOutput` where ``total_loss == dllm_loss``.
        """
        token_nll = _compute_per_token_nll(logits, target_ids)  # [B, L]
        del logits

        # Effective mask: corrupted AND supervised positions
        mask = noise_mask & loss_mask.bool()  # [B, L]

        # Weight by 1/p_mask (= scheduler weight 1/t for linear schedule)
        p_mask_safe = p_mask.clamp(min=1e-8)
        weighted_nll = token_nll * mask.float() * (1.0 / p_mask_safe)

        loss = weighted_nll.sum()

        # Normalize by total supervised tokens
        if num_diffusion_tokens is not None:
            loss = loss / max(num_diffusion_tokens, 1)

        return DLLMLossOutput(total_loss=loss, dllm_loss=loss.detach().clone())


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
        num_tokens: Optional[int] = None,
        block_size: Optional[int] = None,
    ) -> DLLMLossOutput:
        """Compute the DFlash decay-weighted loss.

        Args:
            logits: Draft model logits for the predicted block positions,
                shape ``[B, T, V]`` where ``T = N * (block_size - 1)``.
            target_ids: Ground-truth token IDs, shape ``[B, T]``.
            block_mask: Float/bool valid-position mask, shape ``[B, T]``.
                Zero entries (padding) are excluded from the loss.
            num_tokens: Optional global token count for loss normalisation.
            block_size: When provided, the decay weights reset at each block
                boundary so that every block's first predicted position has
                weight 1.  Required for multi-block training (N > 1).

        Returns:
            :class:`DLLMLossOutput`.
        """
        token_nll = _compute_per_token_nll(logits, target_ids)  # [B, T]
        B, T = token_nll.shape

        if block_size is not None:
            # Multi-block: replicate per-block decay so weights reset at each boundary.
            T_per = block_size - 1
            n_blocks = T // T_per if T_per > 0 else 1
            w_single = torch.exp(-torch.arange(T_per, device=logits.device, dtype=token_nll.dtype) / self.loss_gamma)
            w = w_single.repeat(n_blocks)
        else:
            # Single-block or legacy: monotone decay over the full T.
            w = torch.exp(-torch.arange(T, device=logits.device, dtype=token_nll.dtype) / self.loss_gamma)

        weights = w.unsqueeze(0) * block_mask.to(token_nll.dtype)  # [B, T]

        loss = (token_nll * weights).sum()
        if num_tokens is not None:
            loss = loss / max(float(num_tokens), 1.0)

        return DLLMLossOutput(total_loss=loss, dllm_loss=loss.detach().clone())

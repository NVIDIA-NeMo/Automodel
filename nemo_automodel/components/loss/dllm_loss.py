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
        loss_mask_ar: Optional[torch.Tensor] = None,
        num_diffusion_tokens: Optional[int] = None,
        num_ar_tokens: Optional[int] = None,
        causal_logits: Optional[torch.Tensor] = None,
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


class HybridDiffusionLLMLoss(nn.Module):
    """Combined diffusion + optional AR loss for hybrid diffusion LLM models.

    Used by Nemotron-Labs-Diffusion. The diffusion component computes
    MDLM-style loss at noise-masked positions, weighted by ``1/p_mask``. An
    optional autoregressive (AR) component adds standard cross-entropy at AR
    positions (the causal branch of model output).

    Total loss = alpha * diffusion_loss + ar_loss.
    """

    def __init__(self, alpha: float = 1.0, fp32_upcast: bool = True):
        """Initialize the hybrid loss.

        Args:
            alpha: Weight for the diffusion loss component.
            fp32_upcast: If True, upcast logits to float32 for numerical stability.
        """
        super().__init__()
        self.alpha = alpha
        self.fp32_upcast = fp32_upcast

    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        noise_mask: torch.Tensor,
        p_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        loss_mask_ar: Optional[torch.Tensor] = None,
        num_diffusion_tokens: Optional[int] = None,
        num_ar_tokens: Optional[int] = None,
        causal_logits: Optional[torch.Tensor] = None,
    ) -> DLLMLossOutput:
        """Compute the hybrid diffusion + AR loss.

        Args:
            logits: Model output logits, shape ``[B, L, V]`` or
                ``[B, L+L_ar, V]`` if the model produces both diffusion and AR
                logits in a single concatenated tensor (legacy path).
            target_ids: Clean token IDs, shape ``[B, L]``.
            noise_mask: Boolean mask of corrupted positions, shape ``[B, L]``.
            p_mask: Per-position masking probability, shape ``[B, L]``.
            loss_mask: Diffusion loss mask (supervised positions), shape ``[B, L]``.
            loss_mask_ar: AR loss mask, shape ``[B, L]``. If None, no AR loss.
            num_diffusion_tokens: Total diffusion label tokens for normalization.
            num_ar_tokens: Total AR label tokens for normalization.
            causal_logits: Optional separate AR logits, shape ``[B, L, V]``.
                When provided, avoids the concat/split of the legacy layout.

        Returns:
            :class:`DLLMLossOutput` with combined ``total_loss`` and the pure
            (alpha-weighted) diffusion loss exposed as ``dllm_loss``.
        """
        input_ids_len = target_ids.shape[1]

        # Legacy path: split concatenated logits when causal_logits not passed
        # separately. Must happen before _compute_per_token_nll consumes the
        # DTensor. For DTensor input we all-gather first (unavoidable for the
        # legacy concat layout).
        if causal_logits is None:
            if isinstance(logits, DTensor):
                logits_full = logits.full_tensor()
                if logits_full.shape[1] > input_ids_len:
                    causal_logits = logits_full[:, input_ids_len:]
                    logits = logits_full[:, :input_ids_len]
                else:
                    logits = logits_full
                del logits_full
            elif logits.shape[1] > input_ids_len:
                causal_logits = logits[:, input_ids_len:]
                logits = logits[:, :input_ids_len]

        # --- Diffusion loss ---
        token_nll = _compute_per_token_nll(logits, target_ids)  # [B, L]
        del logits

        mask = noise_mask & loss_mask.bool()
        p_mask_safe = p_mask.clamp(min=1e-8)

        inv_p = torch.nan_to_num(1.0 / p_mask_safe, posinf=1.0, neginf=1.0)
        masked_weighted = token_nll * inv_p
        dllm_loss = (masked_weighted * mask.float()).sum()
        del token_nll
        if num_diffusion_tokens is not None:
            dllm_loss = dllm_loss / max(num_diffusion_tokens, 1)

        total_loss = self.alpha * dllm_loss

        # --- Optional AR loss ---
        if causal_logits is not None and loss_mask_ar is not None:
            ar_targets = target_ids[:, 1:]
            ar_logits = causal_logits[:, :-1]
            ar_nll = _compute_per_token_nll(ar_logits, ar_targets)
            del causal_logits, ar_logits

            ar_mask = loss_mask_ar[:, 1:].float()
            ar_loss = (ar_nll * ar_mask).sum()
            if num_ar_tokens is not None:
                ar_loss = ar_loss / max(num_ar_tokens, 1)

            total_loss = total_loss + ar_loss

        return DLLMLossOutput(total_loss=total_loss, dllm_loss=(self.alpha * dllm_loss).detach())


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
        del logits
        _, T = token_nll.shape

        if block_size is not None:
            # Multi-block: replicate per-block decay so weights reset at each boundary.
            T_per = block_size - 1
            n_blocks = T // T_per if T_per > 0 else 1
            w_single = torch.exp(-torch.arange(T_per, device=token_nll.device, dtype=token_nll.dtype) / self.loss_gamma)
            w = w_single.repeat(n_blocks)
        else:
            # Single-block or legacy: monotone decay over the full T.
            w = torch.exp(-torch.arange(T, device=token_nll.device, dtype=token_nll.dtype) / self.loss_gamma)

        weights = w.unsqueeze(0) * block_mask.to(token_nll.dtype)  # [B, T]

        loss = (token_nll * weights).sum()
        if num_tokens is not None:
            loss = loss / max(float(num_tokens), 1.0)

        return DLLMLossOutput(total_loss=loss, dllm_loss=loss.detach().clone())

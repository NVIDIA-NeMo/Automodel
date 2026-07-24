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

from typing import NamedTuple, Optional, Tuple

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


def encoder_ar_loss(
    encoder_logits: torch.Tensor,
    input_ids: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    num_tokens: Optional[int] = None,
) -> torch.Tensor:
    """Autoregressive next-token CE on the encoder's causal logits.

    The co-trained encoder loss for ``diffusion_gemma`` SFT: a standard causal
    LM cross-entropy over the clean full sequence, scored where both the current
    and next position are valid (non-pad).

    Args:
        encoder_logits: Encoder logits over the clean sequence, ``[B, S, V]``.
        input_ids: Clean token IDs, ``[B, S]``.
        valid_mask: Boolean non-pad mask ``[B, S]``. If ``None``, all positions count.
        num_tokens: Optional global denominator (summed across grad-acc microbatches);
            defaults to the local valid next-token count.

    Returns:
        Scalar AR loss (mean CE over valid next-token positions).
    """
    logits = encoder_logits[:, :-1, :]
    targets = input_ids[:, 1:]
    nll = _compute_per_token_nll(logits, targets)  # [B, S-1]
    if valid_mask is not None:
        mask = (valid_mask[:, :-1] & valid_mask[:, 1:]).to(nll.dtype)
    else:
        mask = torch.ones_like(nll)
    loss = (nll * mask).sum()
    denom = num_tokens if num_tokens is not None else int(mask.sum().item())
    return loss / max(denom, 1)


class DLLMLossOutput(NamedTuple):
    """Unified return type for all dLLM loss functions.

    Attributes:
        total_loss: Loss used for backward (may include AR component).
        dllm_loss: Pure diffusion loss for logging/metrics.
        draft_correct_per_pos: Per-rank raw count of argmax-correct predictions
            at each block offset k=1..block_size-1, shape ``[block_size-1]``.
            ``None`` when not computed (e.g. block_size unknown).
        draft_count_per_pos: Per-rank raw count of valid predicted positions
            at each block offset, shape ``[block_size-1]``. SUM-allreducing
            ``draft_correct_per_pos`` and ``draft_count_per_pos`` across
            DP/CP replicas and dividing post-reduction yields per-position
            global accuracy; summing across positions before dividing gives
            the overall draft top-1 accuracy.
    """

    total_loss: torch.Tensor
    dllm_loss: torch.Tensor
    draft_correct_per_pos: Optional[torch.Tensor] = None
    draft_count_per_pos: Optional[torch.Tensor] = None


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


class BlockDiffusionCrossEntropyLoss(nn.Module):
    """Flat cross-entropy loss for block-diffusion (``diffusion_gemma``) training.

    The ``diffusion_gemma`` checkpoint uses uniform random-token (D3PM-uniform)
    corruption, not absorbing ``[MASK]``. Its loss is plain mean cross-entropy
    over **all supervised canvas positions** (corrupted AND uncorrupted): the loss
    support is the full selected canvas (``target_mask = canvas_mask``), which is
    NOT noise-gated. ``noise_mask`` is accepted (for diagnostics) but does NOT
    gate the loss support:

    .. math::
        \\text{loss} = \\frac{\\sum_{i \\in \\text{supervised (canvas)}} \\text{CE}_i}{N}

    where ``N`` is the supervised canvas-token count. There is **no** ``1/p`` (``1/t``)
    reweighting (that is the absorbing-kernel ELBO weight, which does not apply
    to the uniform kernel) and **no** autoregressive term. Flatness is a
    property of this class, not of a caller passing ``p_mask = 1``.

    The signature matches :class:`MDLMCrossEntropyLoss` /
    :class:`HybridDiffusionLLMLoss` so the recipe can call it uniformly; the
    ``p_mask`` / ``causal_logits`` / ``loss_mask_ar`` / ``num_ar_tokens``
    arguments are accepted but ignored.
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
        """Compute the flat block-diffusion cross-entropy loss.

        Args:
            logits: Model output logits over the canvas, shape ``[B, L, V]``.
            target_ids: Clean (uncorrupted) canvas token IDs, shape ``[B, L]``.
            noise_mask: Boolean mask of corrupted positions, shape ``[B, L]``.
            p_mask: Ignored (flat loss has no per-token weight).
            loss_mask: Supervised positions mask, shape ``[B, L]``.
            num_diffusion_tokens: If provided, the global corrupted-token count
                used as the normalization denominator (summed across grad-acc
                microbatches). If ``None``, normalizes by the local corrupted
                count in this microbatch.

        Returns:
            :class:`DLLMLossOutput` where ``total_loss == dllm_loss`` (no AR).
        """
        token_nll = _compute_per_token_nll(logits, target_ids)  # [B, L]
        del logits

        # ALL supervised canvas positions (corrupted AND uncorrupted) — matches
        # Google's decoder target_mask = canvas_mask (NOT noise-gated). noise_mask
        # is intentionally unused here; the loss support is the full canvas.
        del noise_mask
        mask = loss_mask.bool().to(token_nll.dtype)
        loss = (token_nll * mask).sum()

        denom = num_diffusion_tokens if num_diffusion_tokens is not None else int(mask.sum().item())
        loss = loss / max(denom, 1)

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
        use_fused_linear_ce: When True, compute the per-token NLL with the
            chunked linear-CE path (:meth:`forward_fused`) — projects the
            LM head and runs cross-entropy in position chunks, each wrapped in
            :func:`torch.utils.checkpoint` so the full ``[B, T, vocab]`` logits
            tensor is never materialised (peak is one chunk). Keeps large
            ``num_blocks_per_sample`` (e.g. paper-default 512) within memory on
            full-vocab targets.

            We deliberately do NOT use ``liger_kernel``'s
            ``LigerFusedLinearCrossEntropyLoss`` here: its custom autograd
            Function computes ``grad_input`` eagerly in forward and only
            integrates with FSDP via the model-patching redirection
            (``apply_liger_kernel_to_*``). Used standalone under FSDP2 the
            gradient does not reach the sharded model params (grad_norm 0).
            The chunked path is plain autograd, so FSDP2 handles it correctly.
        chunk_size: Number of predicted positions per chunk in the chunked
            linear-CE path. Smaller = lower peak memory, more recompute.
        normalize: Loss denominator. ``"tokens"`` (default) divides the
            decay-weighted sum by ``num_tokens``, a global all-reduced count
            that keeps the loss consistent across DP replicas and grad-accum.
            ``"mean"`` divides by the effective weight sum
            ``(w_k * block_mask).sum()`` for a per-call decay-weighted mean.
        loss_gamma: Decay parameter γ. ``None`` disables decay (all predicted
            positions weighted equally).
    """

    def __init__(
        self,
        loss_gamma: Optional[float] = 7.0,
        use_fused_linear_ce: bool = False,
        chunk_size: int = 1024,
        normalize: str = "tokens",
    ):
        super().__init__()
        if normalize not in ("tokens", "mean"):
            raise ValueError(f"normalize must be 'tokens' or 'mean', got {normalize!r}")
        self.loss_gamma = None if loss_gamma is None else float(loss_gamma)
        self.use_fused_linear_ce = bool(use_fused_linear_ce)
        self.chunk_size = int(chunk_size)
        self.normalize = normalize

    def _decay_weights(self, T: int, block_size: Optional[int], device, dtype) -> torch.Tensor:
        """Eq. 4 weights for ``T`` predicted positions, resetting per block.

        Returns all-ones (uniform) when ``loss_gamma is None`` (decay disabled).
        """
        if self.loss_gamma is None:
            return torch.ones(T, device=device, dtype=dtype)
        if block_size is not None:
            T_per = block_size - 1
            n_blocks = T // T_per if T_per > 0 else 1
            w_single = torch.exp(-torch.arange(T_per, device=device, dtype=dtype) / self.loss_gamma)
            return w_single.repeat(n_blocks)
        return torch.exp(-torch.arange(T, device=device, dtype=dtype) / self.loss_gamma)

    def _reduce(
        self,
        token_nll: torch.Tensor,
        block_mask: torch.Tensor,
        num_tokens: Optional[int],
        block_size: Optional[int],
        draft_correct_per_pos: Optional[torch.Tensor] = None,
        draft_count_per_pos: Optional[torch.Tensor] = None,
    ) -> DLLMLossOutput:
        """Apply decay weights + block mask, sum, and normalise."""
        _, T = token_nll.shape
        w = self._decay_weights(T, block_size, token_nll.device, token_nll.dtype)
        weights = w.unsqueeze(0) * block_mask.to(token_nll.dtype)  # [B, T]
        loss = (token_nll * weights).sum()
        if self.normalize == "mean":
            loss = loss / (weights.sum() + 1e-6)
        elif num_tokens is not None:
            loss = loss / max(float(num_tokens), 1.0)
        return DLLMLossOutput(
            total_loss=loss,
            dllm_loss=loss.detach().clone(),
            draft_correct_per_pos=draft_correct_per_pos,
            draft_count_per_pos=draft_count_per_pos,
        )

    @staticmethod
    def _draft_acc_per_pos(
        correct: torch.Tensor,
        block_mask: torch.Tensor,
        block_size: Optional[int],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Per-rank (correct, count) sums per block offset k=1..block_size-1.

        ``correct`` is a ``[B, T]`` bool/float tensor of argmax matches and
        ``block_mask`` excludes padding (T = N * (block_size - 1) when
        ``block_size`` is provided). Reshape to ``[B, N, block_size-1]`` and
        sum over ``(B, N)`` to get per-offset counts of shape
        ``[block_size-1]``. Returns ``(None, None)`` when ``block_size`` is
        unknown (single-block / legacy path).
        """
        if block_size is None or block_size <= 1:
            return None, None
        T_per = block_size - 1
        B, T = correct.shape
        if T % T_per != 0:
            return None, None
        N = T // T_per
        c = correct.to(block_mask.dtype).view(B, N, T_per)
        m = block_mask.view(B, N, T_per)
        correct_per_pos = (c * m).sum(dim=(0, 1))  # [block_size-1]
        count_per_pos = m.sum(dim=(0, 1))  # [block_size-1]
        return correct_per_pos, count_per_pos

    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        block_mask: torch.Tensor,
        num_tokens: Optional[int] = None,
        block_size: Optional[int] = None,
    ) -> DLLMLossOutput:
        """Compute the DFlash decay-weighted loss from pre-computed logits.

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
        correct = logits.argmax(dim=-1) == target_ids  # [B, T]
        del logits
        c_per_pos, n_per_pos = self._draft_acc_per_pos(correct, block_mask, block_size)
        return self._reduce(
            token_nll,
            block_mask,
            num_tokens,
            block_size,
            draft_correct_per_pos=c_per_pos,
            draft_count_per_pos=n_per_pos,
        )

    @staticmethod
    def _chunk_nll(
        hidden_chunk: torch.Tensor,
        lm_head_weight: torch.Tensor,
        lm_head_bias: Optional[torch.Tensor],
        target_chunk: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project one position chunk; return its per-token NLL and argmax-matches.

        Wrapped in :func:`torch.utils.checkpoint` by the caller, so the
        ``[chunk, vocab]`` logits are recomputed in backward rather than held.
        The argmax is non-differentiable, so it adds no backward cost.
        """
        logits = F.linear(hidden_chunk, lm_head_weight, lm_head_bias)  # [chunk, V]
        nll = F.cross_entropy(logits.float(), target_chunk, reduction="none")  # [chunk]
        correct = logits.argmax(dim=-1) == target_chunk  # [chunk]
        return nll, correct

    def forward_fused(
        self,
        hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        target_ids: torch.Tensor,
        block_mask: torch.Tensor,
        num_tokens: Optional[int] = None,
        block_size: Optional[int] = None,
        lm_head_bias: Optional[torch.Tensor] = None,
    ) -> DLLMLossOutput:
        """Chunked linear-CE: never materialises the full logits tensor.

        Projects the LM head + cross-entropy in chunks of ``chunk_size``
        predicted positions, each wrapped in :func:`torch.utils.checkpoint` so
        the ``[chunk, vocab]`` logits are recomputed in backward instead of
        held — peak logit memory is one chunk, not ``[B*T, vocab]``. Pure
        autograd, so the gradient flows correctly through FSDP2 (unlike a
        standalone liger fused-CE Function).

        Args:
            hidden: Draft hidden states for the predicted positions,
                shape ``[B, T, D]`` (``D`` = model dim, NOT vocab).
            lm_head_weight: LM-head projection weight, shape ``[V, D]``.
            target_ids: Ground-truth token IDs, shape ``[B, T]``.
            block_mask: Valid-position mask, shape ``[B, T]``.
            num_tokens / block_size: as in :meth:`forward`.
            lm_head_bias: Optional LM-head bias, shape ``[V]``.

        Returns:
            :class:`DLLMLossOutput`.
        """
        B, T, D = hidden.shape
        flat_hidden = hidden.reshape(-1, D)  # [B*T, D]
        flat_target = target_ids.reshape(-1)  # [B*T]

        nll_parts = []
        correct_parts = []
        for start in range(0, flat_hidden.size(0), self.chunk_size):
            end = start + self.chunk_size
            nll_chunk, correct_chunk = torch.utils.checkpoint.checkpoint(
                self._chunk_nll,
                flat_hidden[start:end],
                lm_head_weight,
                lm_head_bias,
                flat_target[start:end],
                use_reentrant=False,
            )
            nll_parts.append(nll_chunk)
            correct_parts.append(correct_chunk)
        token_nll = torch.cat(nll_parts).reshape(B, T)
        correct = torch.cat(correct_parts).reshape(B, T)
        c_per_pos, n_per_pos = self._draft_acc_per_pos(correct, block_mask, block_size)
        return self._reduce(
            token_nll,
            block_mask,
            num_tokens,
            block_size,
            draft_correct_per_pos=c_per_pos,
            draft_count_per_pos=n_per_pos,
        )


class IDLMLoss(nn.Module):
    """Introspective DLM all-masked loss (Yu et al., 2026; arXiv:2604.11035).

    Operates on the concatenated ``[x_t (L) | x_0 (L)]`` forward output produced
    under the block-diffusion attention mask, where ``x_t`` is the noisy (masked)
    copy and ``x_0`` the clean copy. With a next-token "logit shift" (the hidden
    state at position ``i`` predicts token ``i+1``) the objective combines two
    cross-entropy terms, both supervised on the response (answer) tokens:

    .. math::
        L = \\text{CE}_\\text{noisy} + \\alpha \\cdot \\text{CE}_\\text{clean}

    - ``CE_noisy`` — decode CE on the ``x_t`` half (distribution ``q``): each
      masked token is conditioned on the clean ground-truth prefix.
    - ``CE_clean`` — verify CE on the ``x_0`` half (distribution ``p``): the
      clean copy of the response under strict causal attention.

    With ``auto_balance=True`` the fixed weight is replaced by the detached
    ratio ``CE_noisy / CE_clean`` each step so the two terms stay comparable in
    magnitude (paper Eq. 2, used for the later stride expansions). Otherwise the
    fixed ``clean_loss_weight`` is used (the paper's ``0.2`` for early training).

    Args:
        clean_loss_weight: Fixed ``alpha`` for the clean-copy CE.
        auto_balance: Replace ``alpha`` with ``(CE_noisy / CE_clean).detach()``.
    """

    def __init__(self, clean_loss_weight: float = 0.2, auto_balance: bool = False):
        super().__init__()
        self.clean_loss_weight = float(clean_loss_weight)
        self.auto_balance = bool(auto_balance)

    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        answer_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        seq_len: int,
        num_diffusion_tokens: Optional[int] = None,
    ) -> DLLMLossOutput:
        """Compute the I-DLM block-diffusion loss.

        Args:
            logits: Concatenated forward logits, shape ``[B, 2L, V]`` ordered
                ``[x_t | x_0]``.
            target_ids: Clean token IDs for one copy, shape ``[B, L]``.
            answer_mask: Bool mask of supervised (response) positions, ``[B, L]``.
            valid_mask: Bool/long padding-validity mask, shape ``[B, L]``.
            seq_len: Length ``L`` of one copy.
            num_diffusion_tokens: Global, DP-all-reduced supervised-token count
                used as the loss denominator (summed across grad-accum
                microbatches and data-parallel ranks). Pass this so the loss is a
                proper global token-mean — required for the recipe's
                ``(loss * dp_group_size).backward()`` scaling to give
                DP/grad-accum-invariant gradients. Falls back to the local
                supervised count when ``None`` (single-process use / unit tests).

        Returns:
            :class:`DLLMLossOutput` with the combined ``total_loss`` and
            ``dllm_loss`` set to the decode term ``CE_noisy``.
        """
        noisy_logits = logits[:, :seq_len, :]
        clean_logits = logits[:, seq_len : 2 * seq_len, :]

        # Logit shift: logits[:, i] predicts target[:, i+1]. Both copies supervise
        # the response tokens. _compute_per_token_nll materialises a vocab-sharded
        # DTensor via full_tensor() if needed.
        #
        # Support note (deliberate, differs from the reference by one position):
        # gating on ``answer_mask[:, 1:]`` supervises every position whose NEXT
        # token is a response token, so the loss covers exactly the response
        # tokens — including the FIRST one, predicted from the last prompt
        # position. The official repo instead keeps only masked positions
        # (``logits_to_keep``), which never includes that clean prompt position,
        # so it skips the first response token and scores one position past the
        # answer. Supervising it matters here: at inference the first generated
        # token comes from exactly that hidden state, so leaving it untrained
        # would rely on the base AR behaviour surviving finetuning.
        shift_target = target_ids[:, 1:]
        supervise = answer_mask[:, 1:].bool() & valid_mask[:, 1:].bool()
        weight = supervise.to(torch.float32)
        # Global token count keeps the denominator constant across ranks and
        # microbatches; the ratio in auto_balance is denominator-independent.
        if num_diffusion_tokens is not None:
            denom = max(int(num_diffusion_tokens), 1)
        else:
            denom = supervise.sum().clamp_min(1)

        ce_noisy = (_compute_per_token_nll(noisy_logits[:, :-1, :], shift_target) * weight).sum() / denom
        ce_clean = (_compute_per_token_nll(clean_logits[:, :-1, :], shift_target) * weight).sum() / denom

        if self.auto_balance:
            alpha = (ce_noisy / ce_clean.clamp_min(1e-6)).detach()
        else:
            alpha = self.clean_loss_weight
        loss = ce_noisy + alpha * ce_clean
        return DLLMLossOutput(total_loss=loss, dllm_loss=ce_noisy.detach().clone())

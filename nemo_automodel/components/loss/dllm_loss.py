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

from dataclasses import dataclass
from typing import NamedTuple, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from nemo_automodel.components.loss.chunked_ce import _validate_chunk_len

# Probability floor used throughout the SCDD schedule/ELBO. Quantities that are
# exactly zero at the schedule boundaries (rho -> 1 gives a zero uniform base)
# are clamped to this before a log, which keeps every term finite; the resulting
# bias is far below fp32 resolution and the affected terms carry a zero
# coefficient anyway.
_SCDD_TINY = 1e-30


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
    valid_mask: torch.Tensor | None = None,
    num_tokens: int | None = None,
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
    draft_correct_per_pos: torch.Tensor | None = None
    draft_count_per_pos: torch.Tensor | None = None


@dataclass(frozen=True)
class DFlashLossDetails:
    """DFlash loss internals reused by the online trainer wrappers.

    Attributes:
        token_nll: Per-token NLL, shape ``[batch, blocks, k]``.
        pred_ids: Greedy token IDs, shape ``[batch, blocks, k]``.
        output: Public four-field loss result returned by :class:`DFlashDecayLoss`.
        denominator: Scalar tensor used to normalize ``output.total_loss``.
    """

    token_nll: torch.Tensor
    pred_ids: torch.Tensor
    output: DLLMLossOutput
    denominator: torch.Tensor


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
        loss_mask_ar: torch.Tensor | None = None,
        num_diffusion_tokens: int | None = None,
        num_ar_tokens: int | None = None,
        causal_logits: torch.Tensor | None = None,
        noisy_input_ids: torch.Tensor | None = None,
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
            noisy_input_ids: Ignored (the absorbing kernel needs only
                ``noise_mask``), shape ``[B, L]`` when supplied.

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


@dataclass(frozen=True)
class SCDDSchedule:
    """Marginal of the SCDD forward process at a diffusion time.

    SCDD (Self-Correcting Discrete Diffusion, openreview.net/forum?id=zQKlzKB6I9)
    generalises the
    absorbing masked-diffusion forward process by mixing in uniform transitions,
    so the denoiser sees corrupted-but-plausible tokens during training and
    learns to *correct* them rather than only to fill ``[MASK]``. The marginal
    of a clean token ``x`` at time ``t`` is

    .. math::
        q(z_t \\mid x) = \\gamma_t\\bigl(\\rho_t x + (1-\\rho_t) u\\bigr)
                        + (1-\\gamma_t)\\,m

    where ``u`` is uniform over the non-``[MASK]`` vocabulary and ``m`` is the
    absorbing ``[MASK]`` state.

    Attributes:
        clean_mass: ``gamma_t * rho_t`` — probability the token is *retained*,
            shape ``[batch]``.
        uniform_mass: ``gamma_t * (1 - rho_t)`` — probability the token was
            redrawn from the uniform distribution, shape ``[batch]``.
        absorbed_mass: ``1 - gamma_t`` — probability the token is ``[MASK]``,
            shape ``[batch]``.
        gamma: Probability the token is not ``[MASK]``, shape ``[batch]``.
        rho: Probability the token is retained given that it is not ``[MASK]``,
            shape ``[batch]``.
    """

    clean_mass: torch.Tensor
    uniform_mass: torch.Tensor
    absorbed_mass: torch.Tensor
    gamma: torch.Tensor
    rho: torch.Tensor


def scdd_schedule(
    t: torch.Tensor,
    *,
    max_ratio: float,
    gamma_shape: float,
    t_peak: float,
) -> SCDDSchedule:
    """Evaluate the SCDD forward-process marginal at diffusion time *t*.

    The uniform-noise mass follows a Beta-shaped bump ``c(t) = B t^a (1-t)^b``
    with ``a = gamma_shape * t_peak`` and ``b = gamma_shape * (1 - t_peak)``,
    normalised so that its ratio against the retained mass peaks at *max_ratio*
    at ``t = t_peak``. The retained mass decays linearly, giving the closed form

    ``clean = (1-t)/(1+c)``, ``uniform = c/(1+c)``, ``absorbed = t/(1+c)``.

    Both ``rho`` and ``gamma`` are monotonically decreasing in *t*, which is what
    makes ``[MASK]`` an absorbing state of the induced Markov chain (no
    remasking during sampling).

    Args:
        t: Diffusion times in ``[0, 1]``, shape ``[batch]``. Values outside the
            unit interval are clamped (fractional powers of a negative base are
            undefined).
        max_ratio: Peak uniform-to-retained mass ratio, in ``[0, 1)``. ``0``
            degenerates the process to pure absorbing masked diffusion (MDLM).
        gamma_shape: Total shape mass of the bump; larger values concentrate the
            uniform noise around *t_peak*.
        t_peak: Time in ``(0, 1)`` at which the uniform-noise ratio peaks.

    Returns:
        The :class:`SCDDSchedule` at *t*; every field has shape ``[batch]``.
    """
    if not 0.0 <= max_ratio < 1.0:
        raise ValueError(f"scdd_schedule requires 0 <= max_ratio < 1 (got {max_ratio})")
    if not 0.0 < t_peak < 1.0:
        raise ValueError(f"scdd_schedule requires 0 < t_peak < 1 (got {t_peak})")

    t = t.clamp(0.0, 1.0)
    a = gamma_shape * t_peak
    b = gamma_shape * (1.0 - t_peak)
    peak = (t_peak**a) * ((1.0 - t_peak) ** b)
    scale = (max_ratio / (1.0 - max_ratio)) / peak

    c = scale * torch.pow(t, a) * torch.pow(1.0 - t, b)
    clean_mass = (1.0 - t) / (1.0 + c)
    uniform_mass = c / (1.0 + c)
    absorbed_mass = 1.0 - clean_mass - uniform_mass
    gamma = clean_mass + uniform_mass
    rho = clean_mass / gamma.clamp(min=_SCDD_TINY)

    return SCDDSchedule(
        clean_mass=clean_mass,
        uniform_mass=uniform_mass,
        absorbed_mass=absorbed_mass,
        gamma=gamma,
        rho=rho,
    )


class SCDDLoss(nn.Module):
    """Discrete-time NELBO for SCDD (openreview.net/forum?id=zQKlzKB6I9).

    The forward process mixes an absorbing ``[MASK]`` channel with uniform
    transitions (see :func:`scdd_schedule`), so a position at time ``t`` is
    either ``[MASK]`` or a possibly-wrong non-``[MASK]`` token. The two cases
    contribute different terms to the ELBO:

    * ``z_t = [MASK]`` — the familiar denoising term, the reverse-KL mass that
      the model must place on the clean token when it un-absorbs.
    * ``z_t != [MASK]`` — the **correction** term, the reverse KL of the true
      posterior against the model posterior at an already-visible token. This is
      what trains the model to overwrite its own earlier mistakes, and it is
      scored at every non-``[MASK]`` supervised position, including uncorrupted
      ones (where it vanishes only in the degenerate ``max_ratio = 0`` limit).

    Both terms are scaled by ``num_timesteps`` so the loss is the discrete-time
    NELBO per token rather than a per-step increment.

    Setting ``max_ratio = 0`` removes the uniform channel entirely and the loss
    reduces exactly to the MDLM objective ``-log p(x_0) / t`` at masked
    positions with zero correction term — the invariant the unit tests pin.

    The model output is re-parameterised as a distribution over non-``[MASK]``
    tokens (the ``[MASK]`` logit is driven to ``-inf`` before the log-softmax),
    matching the SCDD backbone parameterisation: the denoiser never predicts the
    absorbing state.

    Unlike the absorbing losses, the ELBO needs the model's probability of
    *every* non-``[MASK]`` token, so it cannot be reduced by a fused
    cross-entropy kernel. The vocabulary-sized work is instead done in position
    chunks wrapped in :func:`torch.utils.checkpoint` (the same treatment
    :meth:`DFlashDecayLoss.forward_fused` gives its LM-head projection), so the
    two ``[positions, vocab]`` fp32 intermediates are recomputed in backward and
    peak activation is one chunk rather than the whole batch.
    """

    def __init__(
        self,
        mask_token_id: int,
        num_timesteps: int = 1000,
        max_ratio: float = 0.1,
        gamma_shape: float = 1.0,
        t_peak: float = 0.5,
        chunk_size: int | None = 1024,
    ):
        """Initialise the SCDD loss.

        Args:
            mask_token_id: Token ID of the absorbing ``[MASK]`` state.
            num_timesteps: Number of discrete diffusion steps ``T``; the loss is
                the ``T``-step NELBO and the reverse step is ``1/T``. At least 2,
                so the grid holds a usable point below the fully absorbed ``t = 1``.
            max_ratio: Peak uniform-to-retained mass ratio of the forward
                process (``0`` degenerates to MDLM).
            gamma_shape: Shape mass of the uniform-noise bump.
            t_peak: Time at which the uniform-noise ratio peaks.
            chunk_size: Number of positions whose vocabulary-sized terms are
                computed at once, each chunk wrapped in
                :func:`torch.utils.checkpoint`. Smaller means lower peak memory
                and more recompute. ``None`` computes every position in one
                shot with no checkpointing — numerically identical, but it holds
                two fp32 ``[batch * sequence, vocab]`` tensors at once.
        """
        super().__init__()
        if num_timesteps < 2:
            raise ValueError(f"SCDDLoss requires num_timesteps >= 2 (got {num_timesteps})")
        self.mask_token_id = int(mask_token_id)
        self.num_timesteps = int(num_timesteps)
        self.max_ratio = float(max_ratio)
        self.gamma_shape = float(gamma_shape)
        self.t_peak = float(t_peak)
        # Same positive-int contract the chunked cross-entropy kernel uses.
        self.chunk_size = None if chunk_size is None else _validate_chunk_len(chunk_size)

    @staticmethod
    def _vocab_terms(
        logits_chunk: torch.Tensor,
        x_0_chunk: torch.Tensor,
        z_t_chunk: torch.Tensor,
        log_base_s_chunk: torch.Tensor,
        log_rho_s_chunk: torch.Tensor,
        mask_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce one position chunk over the vocabulary axis.

        This is the only part of the ELBO whose working set scales with the
        vocabulary, so it is the part the caller wraps in
        :func:`torch.utils.checkpoint`: the two ``[chunk, vocab]`` fp32
        intermediates are then recomputed in backward instead of held. Every
        position is independent, so chunking is exact.

        Args:
            logits_chunk: Model logits, shape ``[chunk, vocab]``.
            x_0_chunk: Clean token IDs, shape ``[chunk]``.
            z_t_chunk: Corrupted token IDs seen by the model, shape ``[chunk]``.
            log_base_s_chunk: ``log`` of the uniform base mass at ``s``, shape
                ``[chunk]``.
            log_rho_s_chunk: ``log`` of the retained-mass ratio at ``s``, shape
                ``[chunk]``.
            mask_token_id: Token ID of the absorbing ``[MASK]`` state.

        Returns:
            Tuple of ``(sum_log, log_at_x0, log_at_zt, log_p_zt)``, each of
            shape ``[chunk]``:

            * ``sum_log`` — the posterior numerator summed over the non-``[MASK]``
              domain.
            * ``log_at_x0`` / ``log_at_zt`` — that numerator at the clean and at
              the corrupted token.
            * ``log_p_zt`` — the denoiser's log-probability of the corrupted token.
        """
        # Driving the [MASK] logit to -inf removes the absorbing state from the
        # denoiser's domain. The fill is done in the logits' own dtype so only
        # the float() cast below pays vocabulary-sized fp32.
        mask_col = torch.tensor([mask_token_id], device=logits_chunk.device)
        logits_chunk = logits_chunk.index_fill(-1, mask_col, float("-inf")).float()  # [chunk, vocab]
        log_denom = torch.logsumexp(logits_chunk, dim=-1)  # [chunk]

        # log p_theta(v) = logits(v) - logsumexp(logits), so the log-softmax
        # never has to be materialised: its per-position normaliser folds into a
        # scalar shift, and it is only needed pointwise at z_t.
        shift = log_rho_s_chunk - log_denom  # [chunk]
        log_term = torch.logaddexp(
            log_base_s_chunk[:, None].expand_as(logits_chunk),
            shift[:, None] + logits_chunk,
        )  # [chunk, vocab] — log( base_s + rho_s * p_theta(v) )

        # At the [MASK] column the logit is -inf, so the term collapses to
        # log(base_s): excluding that column from the vocabulary sum is a scalar
        # subtraction, not a gather.
        sum_log = log_term.sum(dim=-1) - log_base_s_chunk
        log_at_x0 = log_term.gather(-1, x_0_chunk[:, None]).squeeze(-1)
        log_at_zt = log_term.gather(-1, z_t_chunk[:, None]).squeeze(-1)
        log_p_zt = logits_chunk.gather(-1, z_t_chunk[:, None]).squeeze(-1) - log_denom
        return sum_log, log_at_x0, log_at_zt, log_p_zt

    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        noise_mask: torch.Tensor,
        p_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        loss_mask_ar: torch.Tensor | None = None,
        num_diffusion_tokens: int | None = None,
        num_ar_tokens: int | None = None,
        causal_logits: torch.Tensor | None = None,
        noisy_input_ids: torch.Tensor | None = None,
    ) -> DLLMLossOutput:
        """Compute the SCDD discrete-time NELBO.

        Args:
            logits: Model output logits, shape ``[batch, sequence, vocab]``.
            target_ids: Clean token IDs ``x_0``, shape ``[batch, sequence]``.
            noise_mask: Boolean mask of corrupted positions, shape
                ``[batch, sequence]``. Ignored — the SCDD ELBO is supported on
                every supervised position, corrupted or not.
            p_mask: Per-position diffusion time ``t``, shape
                ``[batch, sequence]``, constant along the sequence axis (the
                SCDD forward process draws one ``t`` per sequence). This is the
                contract with
                :meth:`~nemo_automodel.recipes.dllm.strategy.SCDDStrategy.apply_corruption`,
                which samples ``t`` on the ``1/T`` grid and broadcasts it here;
                unlike the absorbing kernels this slot carries the time itself,
                because the ELBO weights need the full schedule at ``t`` and at
                the previous grid point.
            loss_mask: Supervised positions mask, shape ``[batch, sequence]``.
            loss_mask_ar: Ignored (SCDD has no autoregressive term).
            num_diffusion_tokens: If provided, the global supervised-token count
                used as the normalisation denominator (summed across grad-acc
                microbatches). If ``None``, normalises by the local supervised
                count.
            num_ar_tokens: Ignored (SCDD has no autoregressive term).
            causal_logits: Ignored (SCDD has no autoregressive term).
            noisy_input_ids: Corrupted token IDs ``z_t`` the model was fed, shape
                ``[batch, sequence]``. Required: the correction term is a
                function of the visible token, which cannot be recovered from
                ``noise_mask`` alone.

        Returns:
            :class:`DLLMLossOutput` where ``total_loss == dllm_loss``.
        """
        del noise_mask, loss_mask_ar, num_ar_tokens, causal_logits

        if noisy_input_ids is None:
            raise ValueError("SCDDLoss requires noisy_input_ids (the corrupted tokens z_t seen by the model).")

        if isinstance(logits, DTensor):
            logits = logits.full_tensor()
        z_t = noisy_input_ids.to(logits.device)
        x_0 = target_ids.to(logits.device)

        vocab = logits.size(-1)
        # Domain of the denoiser: every token except the absorbing state.
        num_states = vocab - 1

        # --- schedule at t and at the previous grid point s = t - 1/T ---
        step = 1.0 / self.num_timesteps
        t = p_mask[:, 0].to(torch.float32).clamp(0.0, 1.0)  # [batch]
        s = (t - step).clamp(min=0.0)
        sched_t = scdd_schedule(t, max_ratio=self.max_ratio, gamma_shape=self.gamma_shape, t_peak=self.t_peak)
        sched_s = scdd_schedule(s, max_ratio=self.max_ratio, gamma_shape=self.gamma_shape, t_peak=self.t_peak)

        rho_t, rho_s = sched_t.rho, sched_s.rho
        rho_s_safe = rho_s.clamp(min=_SCDD_TINY)
        # Backward transition of the retained/uniform split between s and t.
        clean_transition = rho_t / rho_s_safe
        uniform_transition = (rho_s - rho_t) / rho_s_safe
        # Fraction of the absorbed mass released over one reverse step.
        unmask_coeff = (sched_s.gamma - sched_t.gamma) / (1.0 - sched_t.gamma).clamp(min=_SCDD_TINY)
        base_s = (1.0 - rho_s) / num_states
        base_t = (1.0 - rho_t) / num_states

        # --- vocabulary-sized work, one position chunk at a time ---
        batch, seq_len = x_0.shape
        # Broadcast the per-sequence schedule onto positions so a chunk can span
        # the batch boundary.
        log_base_s = base_s.clamp(min=_SCDD_TINY).log().repeat_interleave(seq_len)  # [batch * sequence]
        log_rho_s = rho_s.clamp(min=_SCDD_TINY).log().repeat_interleave(seq_len)  # [batch * sequence]
        flat_logits = logits.reshape(-1, vocab)
        flat_x0 = x_0.reshape(-1)
        flat_zt = z_t.reshape(-1)
        del logits

        num_positions = flat_logits.size(0)
        chunk = num_positions if self.chunk_size is None else self.chunk_size
        parts = []
        for start in range(0, num_positions, chunk):
            end = start + chunk
            args = (
                flat_logits[start:end],
                flat_x0[start:end],
                flat_zt[start:end],
                log_base_s[start:end],
                log_rho_s[start:end],
                self.mask_token_id,
            )
            if self.chunk_size is None:
                parts.append(self._vocab_terms(*args))
            else:
                parts.append(torch.utils.checkpoint.checkpoint(self._vocab_terms, *args, use_reentrant=False))
        sum_log, log_at_x0, log_at_zt, log_p_zt = (torch.cat(term).reshape(batch, seq_len) for term in zip(*parts))

        # --- z_t == [MASK]: standard denoising term ---
        absorbed_loss = -unmask_coeff[:, None] * (base_s[:, None] * sum_log + rho_s[:, None] * log_at_x0)

        # --- z_t != [MASK]: correction term ---
        log_denom = torch.logaddexp(
            base_t.clamp(min=_SCDD_TINY).log()[:, None].expand_as(log_p_zt),
            rho_t.clamp(min=_SCDD_TINY).log()[:, None] + log_p_zt,
        )  # [batch, sequence]

        # Expectation of log(q/p) over z_s ~ q(. | z_t, x_0), expanded into the
        # four (uniform|retained) x (x_0|z_t) coefficient blocks.
        retained = (z_t == x_0).to(log_denom.dtype)  # [batch, sequence]
        coeff_uniform = (uniform_transition / num_states)[:, None]
        coeff_clean = clean_transition[:, None]
        total = (
            (base_s[:, None] * coeff_uniform) * (sum_log - num_states * log_denom)
            + (rho_s[:, None] * coeff_uniform) * (log_at_x0 - log_denom)
            + (base_s[:, None] * coeff_clean) * (log_at_zt - log_denom)
            + (rho_s[:, None] * coeff_clean * retained) * (log_at_x0 - log_denom)
        )
        correction_loss = -total / (base_t[:, None] + rho_t[:, None] * retained).clamp(min=_SCDD_TINY)

        per_token = torch.where(z_t == self.mask_token_id, absorbed_loss, correction_loss) * self.num_timesteps

        mask = loss_mask.bool().to(per_token.dtype)
        loss = (per_token * mask).sum()
        denom = num_diffusion_tokens if num_diffusion_tokens is not None else int(mask.sum().item())
        loss = loss / max(denom, 1)

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
        loss_mask_ar: torch.Tensor | None = None,
        num_diffusion_tokens: int | None = None,
        num_ar_tokens: int | None = None,
        causal_logits: torch.Tensor | None = None,
        noisy_input_ids: torch.Tensor | None = None,
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
            noisy_input_ids: Ignored (the flat loss scores the clean targets),
                shape ``[B, L]`` when supplied.

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
        loss_mask_ar: torch.Tensor | None = None,
        num_diffusion_tokens: int | None = None,
        num_ar_tokens: int | None = None,
        causal_logits: torch.Tensor | None = None,
        noisy_input_ids: torch.Tensor | None = None,
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
            noisy_input_ids: Ignored (the model applies masking internally),
                shape ``[B, L]`` when supplied.

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


_DFLASH_LOSS_TYPES = {
    "dflash",
    "dpace",
    "dpace-cumulative-confidence-only",
    "dpace-continuation-value-only",
}
_DPACE_LOSS_TYPES = _DFLASH_LOSS_TYPES - {"dflash"}


class DFlashDecayLoss(nn.Module):
    """Position-decay cross-entropy loss for DFlash draft model training.

    Implements Eq. 4 of the DFlash paper:

    .. math::
        w_k = \\exp\\!\\left(-\\frac{k-1}{\\gamma}\\right), \\quad k = 1, \\dots, T

    where *k* indexes the predicted positions within a block (k=0 is the clean
    anchor and is not predicted; k=1 is the first masked position).

    The normalization policy is selected by ``normalize``. Pass ``num_tokens``
    as a global all-reduced denominator for consistency across DP replicas and
    gradient-accumulation steps.

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
        normalize: Loss denominator. ``"tokens"`` (default) divides the weighted
            sum by ``num_tokens``, a global all-reduced count that keeps the loss
            consistent across DP replicas and grad-accum. ``"mean"`` divides by
            the effective weight sum ``(w_k * block_mask).sum()`` for dflash, and
            by ``batch * blocks`` for D-PACE -- whose weights carry the objective's
            signal, so a weight-sum denominator would cancel it (and, being
            per-rank, would bias the DP gradient average). The block count counts
            sampled anchors rather than tokens, so it keeps both ``loss_type``
            values on a comparable scale without changing the D-PACE optimum.
        loss_type: Loss variant. ``"dflash"`` keeps the original decay-weighted
            cross entropy. The ``"dpace*"`` variants use detached Dynamic
            Position-Aware Cross-Entropy weights (D-PACE, arXiv:2605.18810).
        dpace_alpha: Smoothing alpha for D-PACE confidence products.
    """

    def __init__(
        self,
        loss_gamma: float | None = 7.0,
        use_fused_linear_ce: bool = False,
        chunk_size: int = 1024,
        normalize: str = "tokens",
        loss_type: str = "dflash",
        dpace_alpha: float = 0.5,
    ):
        super().__init__()
        if normalize not in ("tokens", "mean"):
            raise ValueError(f"normalize must be 'tokens' or 'mean', got {normalize!r}")
        if loss_type not in _DFLASH_LOSS_TYPES:
            raise ValueError(f"loss_type must be one of {sorted(_DFLASH_LOSS_TYPES)}, got {loss_type!r}")
        if not 0.0 <= dpace_alpha <= 1.0:
            raise ValueError(f"dpace_alpha must be in [0, 1], got {dpace_alpha}")
        self.loss_gamma = None if loss_gamma is None else float(loss_gamma)
        self.use_fused_linear_ce = bool(use_fused_linear_ce)
        self.chunk_size = int(chunk_size)
        self.normalize = normalize
        self.loss_type = loss_type
        self.dpace_alpha = float(dpace_alpha)

    def _decay_weights(self, k: int, device, dtype) -> torch.Tensor:
        """Eq. 4 decay weights for the ``k`` predicted positions of one block.

        Args:
            k: Predicted positions per block (``block_size - 1``).
            device: Device for the returned tensor.
            dtype: Dtype for the returned tensor.

        Returns:
            Tensor of shape ``[k]``; all-ones (uniform) when ``loss_gamma is None``
            (decay disabled).
        """
        if self.loss_gamma is None:
            return torch.ones(k, device=device, dtype=dtype)
        return torch.exp(-torch.arange(k, device=device, dtype=dtype) / self.loss_gamma)

    def _dpace_weight(
        self,
        token_nll: torch.Tensor,
        binary_mask: torch.Tensor,
        binary_mask_bool: torch.Tensor,
    ) -> torch.Tensor:
        """Compute detached D-PACE position weights.

        Args:
            token_nll: Per-token NLL of the draft on the target token, shape
                ``[batch, blocks, k]``; the draft confidence is ``exp(-token_nll)``.
                Taken as the NLL rather than the confidence so the smoothed
                log-confidence stays finite even where ``exp(-token_nll)``
                underflows.
            binary_mask: Float valid-position mask, same shape.
            binary_mask_bool: Boolean valid-position mask, same shape.

        Returns:
            Detached weights of shape ``[batch, blocks, k]``, in ``token_nll``'s
            dtype. The ``cumprod`` / reverse-``cumsum`` run over the last (``k``)
            axis, so the confidence product resets at every block boundary by
            construction. Accumulated in float32: the per-block products span many
            orders of magnitude at small ``dpace_alpha``, which bf16's 8-bit
            mantissa cannot carry.
        """
        nll = token_nll.float()
        smooth = (1.0 - self.dpace_alpha) * torch.exp(-nll) + self.dpace_alpha
        smooth = torch.where(binary_mask_bool, smooth, torch.ones_like(smooth))
        prefix = torch.cumprod(smooth, dim=-1)

        if self.loss_type == "dpace-cumulative-confidence-only":
            return prefix.to(token_nll.dtype)

        if self.loss_type == "dpace":
            suffix = torch.flip(
                torch.cumsum(torch.flip(prefix * binary_mask.to(prefix.dtype), dims=[-1]), dim=-1),
                dims=[-1],
            )
            return suffix.to(token_nll.dtype)

        if self.loss_type == "dpace-continuation-value-only":
            # The same suffix/prefix ratio, evaluated in log space. Computing it as a
            # division underflows for small dpace_alpha: alpha=0 leaves a bare product
            # of block_size-1 confidences, so a merely lukewarm draft (0.3 per position
            # over 15 positions) already reaches 1e-8 and keeps falling, and once both
            # operands flush to zero the ratio is lost. log(smooth) is built from the
            # NLL via logaddexp, so it stays finite even where exp(-nll) is zero.
            log_alpha = nll.new_tensor(self.dpace_alpha).log()
            log_one_minus_alpha = torch.log1p(nll.new_tensor(-self.dpace_alpha))
            log_smooth = torch.logaddexp(log_one_minus_alpha - nll, log_alpha.expand_as(nll))
            log_smooth = torch.where(binary_mask_bool, log_smooth, torch.zeros_like(log_smooth))
            log_prefix = torch.cumsum(log_smooth, dim=-1)
            log_suffix = torch.flip(
                torch.logcumsumexp(
                    torch.flip(log_prefix.masked_fill(~binary_mask_bool, float("-inf")), dims=[-1]),
                    dim=-1,
                ),
                dims=[-1],
            )
            return torch.exp(log_suffix - log_prefix).to(token_nll.dtype)
        raise ValueError(f"unknown D-PACE loss_type {self.loss_type!r}")

    def position_weights(self, token_nll: torch.Tensor, block_mask: torch.Tensor) -> torch.Tensor:
        """Per-position loss weights for this instance's ``loss_type``, unreduced.

        Exposed (beyond :meth:`forward` / :meth:`forward_fused`) so a second
        training objective over the same fixed-anchor block layout -- DFlash 2's
        candidate-selector CE -- can weight its own per-position loss by the exact
        same schedule instead of re-deriving one, keeping "position importance"
        identical between the two objectives by construction.

        Args:
            token_nll: Per-token NLL, shape ``[batch, blocks, k]`` (``k =
                block_size - 1`` predicted positions per block). Only its shape,
                device, and dtype are used for ``loss_type="dflash"``; the D-PACE
                variants read its values as the draft's per-position confidence.
            block_mask: Valid-position mask, same shape ``[batch, blocks, k]``;
                zero entries (padding) are excluded.

        Returns:
            Tensor of shape ``[batch, blocks, k]``: ``block_mask`` scaled by the
            decay weights (``"dflash"``) or the detached D-PACE confidence product
            (``"dpace*"`` -- the ``cumprod`` runs over the last (``k``) axis, so it
            resets per block by construction).
        """
        _, _, k = token_nll.shape
        block_mask = block_mask.to(token_nll.dtype)
        if self.loss_type == "dflash":
            w = self._decay_weights(k, token_nll.device, token_nll.dtype)
            return w.view(1, 1, k) * block_mask  # [batch, blocks, k]
        if self.loss_type in _DPACE_LOSS_TYPES:
            with torch.no_grad():
                dpace_weights = self._dpace_weight(token_nll.detach(), block_mask, block_mask > 0)
            return block_mask * dpace_weights
        raise ValueError(f"unknown loss_type {self.loss_type!r}")

    def _mean_denominator(
        self, weights: torch.Tensor, bsz: int, n_blocks: int, total_blocks: int | None = None
    ) -> torch.Tensor:
        """Denominator for ``normalize="mean"``: weight sum for ``"dflash"``, else block count.

        Args:
            weights: Position weights from :meth:`position_weights`, shape
                ``[batch, blocks, k]``.
            bsz: Batch size (``weights.shape[0]``).
            n_blocks: Sampled-block count (``weights.shape[1]``); used for
                D-PACE only when ``total_blocks`` is not given.
            total_blocks: The trainer's *configured* block-sampling budget (e.g.
                ``num_anchors``), used in place of ``n_blocks`` for D-PACE so the
                denominator is identical on every DP rank. ``n_blocks`` is
                ``min(num_anchors, valid_anchors_in_batch)`` and therefore
                differs per micro-batch and per rank; ``num_anchors`` is a config
                constant.

        Returns:
            Scalar tensor.

        D-PACE is a weighted *sum*: the weight magnitude is the signal (the
        expected accepted-prefix length), so a weight-sum denominator would cancel
        exactly the variation the objective encodes, and a data-dependent
        denominator would desync the DP gradient average (exact only when every
        rank divides by the same constant).

        The published objective divides by the batch size alone. We also divide by
        the block count: it counts sampled anchors, not unmasked tokens, so unlike
        a weight sum it cancels nothing the objective encodes, and it is
        ``num_anchors`` whenever the batch supplies that many valid anchors --
        i.e. a constant, leaving the gradient direction and the optimum unchanged.
        What it buys is scale: without it, flipping ``loss_type`` in the YAML
        moves the effective learning rate by orders of magnitude and the run
        diverges instead of erroring. The reported value is correspondingly
        ``n_blocks`` times smaller than the published one.
        """
        if self.loss_type in _DPACE_LOSS_TYPES:
            blocks = n_blocks if total_blocks is None else total_blocks
            return weights.new_tensor(max(float(bsz * blocks), 1.0))
        return weights.sum() + 1e-6

    def weighted_mean(
        self,
        values: torch.Tensor,
        token_nll: torch.Tensor,
        block_mask: torch.Tensor,
        value_mask: torch.Tensor | None = None,
        total_blocks: int | None = None,
    ) -> torch.Tensor:
        """Weight ``values`` by :meth:`position_weights` and reduce with ``normalize="mean"`` semantics.

        Lets a second training objective over the same fixed-anchor block layout
        (DFlash 2's candidate-selector CE) share this loss's exact position
        weighting and denominator instead of re-deriving both, so switching
        ``loss_type`` (``"dflash"`` <-> a D-PACE variant) rescales every objective
        built on this block layout identically, not only the one :meth:`forward`
        computes directly.

        Args:
            values: Per-position loss values to weight and reduce, shape
                ``[batch, blocks, k]``.
            token_nll: Per-token NLL the position weights are derived from --
                typically the backbone's own per-position NLL over the same
                block, shape ``[batch, blocks, k]``. Ignored for
                ``loss_type="dflash"``.
            block_mask: The mask :meth:`position_weights` builds the schedule
                from, shape ``[batch, blocks, k]`` -- keep this the *base*
                objective's own valid-position mask, not narrowed to some
                condition specific to ``values``. The D-PACE weight is a
                sequential cumprod/cumsum across the block, so narrowing this
                mask changes every other position's weight too, not only the
                excluded one's; narrow with ``value_mask`` instead.
            value_mask: Which positions ``values`` is actually summed over, and
                (for ``loss_type="dflash"``) the weight-sum denominator too --
                matching a single-term reduction's own semantics. Defaults to
                ``block_mask``. Pass a narrower mask (e.g. DFlash 2's
                ``pred_mask * has_target``, valid only where a real supervision
                target exists) to exclude positions from the reduction without
                perturbing the schedule ``block_mask`` builds.
            total_blocks: See :meth:`_mean_denominator`.

        Returns:
            Scalar tensor: the weighted mean of ``values``.
        """
        if value_mask is None:
            value_mask = block_mask
        bsz, n_blocks, _ = block_mask.shape
        weights = self.position_weights(token_nll, block_mask) * value_mask.to(token_nll.dtype)
        denom = self._mean_denominator(weights, bsz, n_blocks, total_blocks=total_blocks)
        return (values * weights).sum() / denom

    def _reduce(
        self,
        token_nll: torch.Tensor,
        block_mask: torch.Tensor,
        num_tokens: int | None,
        draft_correct_per_pos: torch.Tensor | None,
        draft_count_per_pos: torch.Tensor | None,
        total_blocks: int | None = None,
    ) -> tuple[DLLMLossOutput, torch.Tensor]:
        """Build per-position weights, then sum and normalise.

        Args:
            token_nll: Per-token NLL, shape ``[batch, blocks, k]`` (``k =
                block_size - 1`` predicted positions per block).
            block_mask: Valid-position mask, same shape ``[batch, blocks, k]``;
                zero entries (padding) are excluded from the loss.
            num_tokens: Optional global, DP-all-reduced token count used as the
                denominator when ``normalize != "mean"``.
            draft_correct_per_pos: Per-offset argmax-correct counts, shape ``[k]``.
            draft_count_per_pos: Per-offset valid-position counts, shape ``[k]``.
            total_blocks: See :meth:`_mean_denominator`.

        Returns:
            Tuple containing the public :class:`DLLMLossOutput` and the exact
            scalar denominator used for its total loss.
        """
        bsz, n_blocks, _ = token_nll.shape
        weights = self.position_weights(token_nll, block_mask)
        weighted_sum = (token_nll * weights).sum()
        if self.normalize == "mean":
            denom = self._mean_denominator(weights, bsz, n_blocks, total_blocks=total_blocks)
        elif num_tokens is not None:
            denom = weighted_sum.new_tensor(max(float(num_tokens), 1.0))
        else:
            denom = weighted_sum.new_tensor(1.0)
        loss = weighted_sum / denom
        return (
            DLLMLossOutput(
                total_loss=loss,
                dllm_loss=loss.detach().clone(),
                draft_correct_per_pos=draft_correct_per_pos,
                draft_count_per_pos=draft_count_per_pos,
            ),
            denom.detach(),
        )

    @staticmethod
    def _draft_acc_per_pos(
        correct: torch.Tensor,
        block_mask: torch.Tensor,
        report_per_pos: bool = True,
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        """Per-rank (correct, count) sums per block offset.

        Args:
            correct: Bool/float argmax-match tensor, shape ``[batch, blocks, k]``.
            block_mask: Valid-position mask, shape ``[batch, blocks, k]``.
            report_per_pos: Whether the caller supplied an explicit block layout.

        Returns:
            ``(correct_per_pos, count_per_pos)``, each shape ``[k]`` and summed
            over ``(batch, blocks)``. Returns ``(None, None)`` for the legacy
            flattened input without a ``block_size``.
        """
        if not report_per_pos:
            return None, None
        c = correct.to(block_mask.dtype)
        m = block_mask.to(c.dtype)
        correct_per_pos = (c * m).sum(dim=(0, 1))  # [k]
        count_per_pos = m.sum(dim=(0, 1))  # [k]
        return correct_per_pos, count_per_pos

    def _reshape_block_inputs(
        self,
        values: torch.Tensor,
        target_ids: torch.Tensor,
        block_mask: torch.Tensor,
        block_size: int | None,
        *,
        value_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Restore the block axis for the legacy flattened input contract."""
        if values.ndim == 4:
            if block_size is not None and values.shape[2] != block_size - 1:
                raise ValueError(
                    f"block-shaped {value_name} have k={values.shape[2]}, expected block_size - 1={block_size - 1}"
                )
            return values, target_ids, block_mask, True
        if values.ndim != 3:
            raise ValueError(f"{value_name} must have rank 3 or 4, got shape {tuple(values.shape)}")

        batch, tokens, width = values.shape
        if block_size is None:
            if self.loss_type in _DPACE_LOSS_TYPES:
                raise ValueError("block_size is required for flattened D-PACE inputs")
            blocks, positions = 1, tokens
            report_per_pos = False
        else:
            positions = block_size - 1
            if positions <= 0 or tokens % positions != 0:
                raise ValueError(f"flattened token count ({tokens}) must be divisible by block_size - 1 ({positions})")
            blocks = tokens // positions
            report_per_pos = True
        return (
            values.reshape(batch, blocks, positions, width),
            target_ids.reshape(batch, blocks, positions),
            block_mask.reshape(batch, blocks, positions),
            report_per_pos,
        )

    def forward_with_token_nll(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        block_mask: torch.Tensor,
        num_tokens: int | None = None,
        block_size: int | None = None,
        *,
        total_blocks: int | None = None,
    ) -> DFlashLossDetails:
        """Like :meth:`forward`, but also returns the per-token NLL it computed.

        Lets a second training objective over the same fixed-anchor block layout
        (DFlash 2's D-PACE selector weighting) reuse this call's own backbone
        confidence instead of recomputing a second full-vocabulary
        cross-entropy pass over the same ``logits``.

        Args:
            logits: Draft logits shaped ``[batch, blocks, k, vocab]`` or legacy
                flattened ``[batch, tokens, vocab]``.
            target_ids: Token IDs shaped ``[batch, blocks, k]`` or ``[batch, tokens]``.
            block_mask: Valid-position mask with the same leading shape as
                ``target_ids``; zero entries are excluded.
            num_tokens: Optional global token count for loss normalisation.
            block_size: Legacy flattened-input block size. Required for D-PACE
                so its confidence product resets at each block boundary.
            total_blocks: See :meth:`_mean_denominator`.

        Returns:
            :class:`DFlashLossDetails` with block-shaped NLLs and predictions.
        """
        logits, target_ids, block_mask, report_per_pos = self._reshape_block_inputs(
            logits, target_ids, block_mask, block_size, value_name="logits"
        )

        token_nll = _compute_per_token_nll(logits, target_ids)  # [batch, blocks, k]
        pred_ids = logits.argmax(dim=-1)  # [batch, blocks, k]
        del logits
        c_per_pos, n_per_pos = self._draft_acc_per_pos(pred_ids == target_ids, block_mask, report_per_pos)
        output, denominator = self._reduce(
            token_nll,
            block_mask,
            num_tokens,
            draft_correct_per_pos=c_per_pos,
            draft_count_per_pos=n_per_pos,
            total_blocks=total_blocks,
        )
        return DFlashLossDetails(token_nll, pred_ids, output, denominator)

    def forward(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        block_mask: torch.Tensor,
        num_tokens: int | None = None,
        block_size: int | None = None,
        *,
        total_blocks: int | None = None,
    ) -> DLLMLossOutput:
        """Compute the DFlash decay-weighted loss from pre-computed logits.

        Args:
            logits: Draft logits shaped ``[batch, blocks, k, vocab]`` or legacy
                flattened ``[batch, tokens, vocab]``.
            target_ids: Token IDs shaped ``[batch, blocks, k]`` or ``[batch, tokens]``.
            block_mask: Valid-position mask with the same leading shape as
                ``target_ids``; zero entries are excluded.
            num_tokens: Optional global token count for loss normalisation.
            block_size: Legacy flattened-input block size.
            total_blocks: See :meth:`_mean_denominator`.

        Returns:
            :class:`DLLMLossOutput`.
        """
        return self.forward_with_token_nll(
            logits,
            target_ids,
            block_mask,
            num_tokens,
            block_size,
            total_blocks=total_blocks,
        ).output

    @staticmethod
    def _chunk_nll(
        hidden_chunk: torch.Tensor,
        lm_head_weight: torch.Tensor,
        lm_head_bias: torch.Tensor | None,
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
        num_tokens: int | None = None,
        block_size: int | None = None,
        lm_head_bias: torch.Tensor | None = None,
        *,
        total_blocks: int | None = None,
    ) -> DLLMLossOutput:
        """Chunked linear-CE: never materialises the full logits tensor.

        Projects the LM head + cross-entropy in chunks of ``chunk_size``
        predicted positions, each wrapped in :func:`torch.utils.checkpoint` so
        the ``[chunk, vocab]`` logits are recomputed in backward instead of
        held — peak logit memory is one chunk, not ``[batch*blocks*k, vocab]``.
        Pure autograd, so the gradient flows correctly through FSDP2 (unlike a
        standalone liger fused-CE Function).

        Args:
            hidden: Draft hidden states shaped ``[batch, blocks, k, hidden]`` or
                legacy flattened ``[batch, tokens, hidden]``.
            lm_head_weight: LM-head projection weight, shape ``[vocab, hidden]``.
            target_ids: Token IDs shaped ``[batch, blocks, k]`` or ``[batch, tokens]``.
            block_mask: Valid-position mask with the same leading shape as
                ``target_ids``.
            num_tokens: as in :meth:`forward`.
            block_size: Legacy flattened-input block size.
            lm_head_bias: Optional LM-head bias, shape ``[vocab]``.
            total_blocks: See :meth:`_mean_denominator`.

        Returns:
            :class:`DLLMLossOutput`.
        """
        hidden, target_ids, block_mask, report_per_pos = self._reshape_block_inputs(
            hidden, target_ids, block_mask, block_size, value_name="hidden states"
        )
        B, n, k, D = hidden.shape
        flat_hidden = hidden.reshape(-1, D)  # [batch*blocks*k, D]
        flat_target = target_ids.reshape(-1)  # [batch*blocks*k]

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
        token_nll = torch.cat(nll_parts).reshape(B, n, k)
        correct = torch.cat(correct_parts).reshape(B, n, k)
        c_per_pos, n_per_pos = self._draft_acc_per_pos(correct, block_mask, report_per_pos)
        output, _ = self._reduce(
            token_nll,
            block_mask,
            num_tokens,
            draft_correct_per_pos=c_per_pos,
            draft_count_per_pos=n_per_pos,
            total_blocks=total_blocks,
        )
        return output


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
        num_diffusion_tokens: int | None = None,
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

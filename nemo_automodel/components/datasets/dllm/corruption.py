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

"""Data corruption utilities for diffusion LLM (dLLM) training.

Provides masking strategies for dLLM SFT:
- ``corrupt_uniform``: uniform per-sequence corruption
- ``corrupt_blockwise``: per-block weighted corruption with exponential position bias
"""

from __future__ import annotations

import math

import torch


def gumbel_topk(log_w: torch.Tensor, k: int) -> torch.Tensor:
    """Return a bool mask of length ``len(log_w)`` with exactly *k* ``True`` entries.

    Uses the Gumbel-max trick for stochastic top-k selection.
    """
    g = -torch.log(-torch.log(torch.rand_like(log_w) + 1e-9) + 1e-9)
    topk = torch.topk(log_w + g, k).indices
    mask = torch.zeros_like(log_w, dtype=torch.bool)
    mask[topk] = True
    return mask


def _batched_gumbel_topk(log_w: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Batched variable-*k* Gumbel top-k selection.

    Vectorised replacement for calling :func:`gumbel_topk` in a Python loop.
    Each row ``i`` of *log_w* gets exactly ``k[i]`` ``True`` entries selected
    via the Gumbel-max trick.

    Algebraically identical to per-row ``gumbel_topk``: both select the *k*
    indices with the highest ``log_w + Gumbel`` score.  ``torch.sort`` gives
    a full ranking; ``positions < k`` picks the top-k in sorted order;
    ``scatter_`` maps them back to original positions.

    Args:
        log_w: Log-weights, shape ``[N, D]``.  Positions that should never
            be selected must be set to ``-inf``.
        k: Number of positions to select per row, shape ``[N]``.
            Rows with ``k=0`` produce an all-False mask.

    Returns:
        Boolean mask of shape ``[N, D]``.
    """
    g = -torch.log(-torch.log(torch.rand_like(log_w) + 1e-9) + 1e-9)
    _, sorted_indices = torch.sort(log_w + g, dim=1, descending=True)
    # positions[i, j] = j  →  "is this the j-th largest in row i?"
    positions = torch.arange(log_w.shape[1], device=log_w.device).expand_as(log_w)
    selected = positions < k.unsqueeze(1)  # top-k mask in sorted order
    mask = torch.zeros_like(log_w, dtype=torch.bool)
    mask.scatter_(1, sorted_indices, selected)
    return mask


def corrupt_uniform(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    mask_token_id: int,
    eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sequence uniform corruption for MDLM training.

    For each sequence, sample ``t ~ U[0, 1]`` and derive a masking probability
    ``p = (1 - eps) * t + eps``.  Each token at a supervised position (where
    ``loss_mask == 1``) is independently replaced with ``mask_token_id`` with
    probability *p*.

    Args:
        input_ids: Token IDs, shape ``[B, L]``.
        loss_mask: Binary mask indicating supervised positions, shape ``[B, L]``.
        mask_token_id: The token ID used for masking.
        eps: Minimum corruption ratio.

    Returns:
        Tuple of ``(noisy_input_ids, noise_mask, p_mask)`` each of shape ``[B, L]``.
        - ``noisy_input_ids``: input_ids with masked positions replaced.
        - ``noise_mask``: bool mask of which positions were corrupted.
        - ``p_mask``: per-position masking probability (float32).
    """
    B, L = input_ids.shape
    device = input_ids.device

    # t ~ U[0, 1] per sequence
    t = torch.rand((B,), device=device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].expand(B, L)  # (B, L)

    # Sample noise mask: each position independently masked with probability p
    noise_mask = torch.rand((B, L), device=device) < p_mask
    noise_mask = noise_mask & loss_mask.bool()

    noisy_input_ids = torch.where(noise_mask, mask_token_id, input_ids)

    return noisy_input_ids, noise_mask, p_mask.float()


def corrupt_blockwise(
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
    mask_token_id: int,
    block_size: int | None = None,
    eps: float = 1e-3,
    half_life_ratio: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two-stage corruption with optional per-block sampling.

    This function combines three independent concerns that could be separated
    if a future model needs a different mix (e.g., blocks without position
    bias, or sequence-level with bias):

    1. **Sampling scope** — per-sequence vs per-block ``m`` sampling
       (controlled by ``block_size``).
    2. **Selection method** — Gumbel-max top-k for exact-``k`` masking.
    3. **Position bias** — exponential weighting via ``half_life_ratio``.

    Stage 1: Sample ``m ~ U(eps, 1)`` per sequence (or per block), compute
    ``k = round(m * length)`` positions to mask.

    Stage 2: Sample exactly *k* positions using exponentially weighted
    probabilities ``w_i(m) = exp[lambda * (1-m) * i]`` which bias toward
    later positions when ``m`` is small (few masks → mask later tokens) and
    become uniform when ``m`` is large (many masks).

    If ``block_size`` is given, stages 1 and 2 operate independently within
    each contiguous block of that length.

    All operations are fully vectorised (no Python loops over batch or
    blocks) via :func:`_batched_gumbel_topk`.

    Args:
        input_ids: Token IDs, shape ``[B, L]``.
        loss_mask: Binary mask indicating supervised positions, shape ``[B, L]``.
        mask_token_id: The token ID used for masking.
        block_size: If not None, operate block-wise with per-block *m* sampling.
        eps: Minimum corruption ratio.
        half_life_ratio: Controls steepness of positional bias when ``m → 0``.

    Returns:
        Tuple of ``(noisy_input_ids, noise_mask, p_mask)`` each of shape ``[B, L]``.
    """
    B, L = input_ids.shape
    device = input_ids.device
    dtype = torch.float32

    if block_size is None:
        # --- Vectorised per-sequence path ---
        lam_base = math.log(2.0) / (half_life_ratio * L)

        m = eps + (1.0 - eps) * torch.rand(B, device=device)  # [B]
        k = torch.round(m * L).long().clamp(1, L)  # [B]

        p_mask = m.unsqueeze(1).expand(B, L)  # [B, L]
        slope = 1.0 - m  # [B]

        pos = torch.arange(L, device=device, dtype=dtype)  # [L]
        log_w = lam_base * slope.unsqueeze(1) * pos.unsqueeze(0)  # [B, L]

        masked_indices = _batched_gumbel_topk(log_w, k)
    else:
        # --- Vectorised per-block path ---
        num_blocks = math.ceil(L / block_size)
        N = B * num_blocks  # total number of blocks
        padded_L = num_blocks * block_size

        lam_base = math.log(2.0) / (half_life_ratio * block_size)

        # Effective length of each block (last block per sequence may be shorter)
        block_lens = torch.full((N,), block_size, dtype=torch.long, device=device)
        if L % block_size != 0:
            last_len = L % block_size
            last_indices = torch.arange(num_blocks - 1, N, num_blocks, device=device)
            block_lens[last_indices] = last_len

        m = eps + (1.0 - eps) * torch.rand(N, device=device)  # [N]
        k = torch.round(m * block_lens.float()).long()  # [N]
        k = torch.min(k.clamp(min=0), block_lens)  # [N]

        slope = 1.0 - m  # [N]
        pos = torch.arange(block_size, device=device, dtype=dtype)  # [block_size]
        log_w = lam_base * slope.unsqueeze(1) * pos.unsqueeze(0)  # [N, block_size]

        # Invalidate padding positions in partial last blocks so they
        # are never selected by _batched_gumbel_topk.
        valid = pos.unsqueeze(0).expand(N, block_size) < block_lens.unsqueeze(1)
        log_w[~valid] = -float("inf")

        block_masked = _batched_gumbel_topk(log_w, k)

        # Reshape [N, block_size] → [B, padded_L] and trim to [B, L]
        masked_indices = block_masked.view(B, padded_L)[:, :L]

        # Expand per-block m to full sequence length
        p_mask = m.view(B, num_blocks, 1).expand(B, num_blocks, block_size)
        p_mask = p_mask.reshape(B, padded_L)[:, :L]

    # Only mask at supervised positions
    if loss_mask is not None:
        masked_indices[loss_mask == 0] = False

    noisy_input_ids = torch.where(masked_indices, mask_token_id, input_ids)

    return noisy_input_ids, masked_indices, p_mask.float()

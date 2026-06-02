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

"""Conditional-On-Distribution (COD) sampling for P-EAGLE parallel drafting.

P-EAGLE (https://github.com/vllm-project/speculators/pull/480) trains all ``K``
draft depths in a *single* parallel forward rather than EAGLE-3's sequential
test-time-training (TTT) unroll. To keep the flattened multi-depth sequence
affordable, COD subsamples deeper depths with geometric decay: depth 0 keeps all
``n`` positions, depth 1 keeps ``n * r``, depth 2 ``n * r**2``, ..., so the
attention cost drops from ``O((nK)**2)`` to ``O((n * sum r**i)**2)``.

This is a verbatim port of speculators' ``generate_cod_sample_indices`` so the
on-disk draft trains against the exact distribution vLLM's parallel-drafting
runtime samples at inference.
"""

from __future__ import annotations

import torch


def generate_cod_sample_indices(
    seq_length: int,
    loss_mask: torch.Tensor,
    num_depths: int = 8,
    down_sample_ratio: float = 0.7,
    down_sample_ratio_min: float = 0.2,
    filter_position_zero: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate COD sampling indices for one sequence.

    Args:
        seq_length: Length of the (padded) sequence.
        loss_mask: Binary mask of valid training positions, shape ``[1, seq_len]``
            or ``[seq_len]``. Padding / unsupervised positions must be 0.
        num_depths: Number of parallel prediction depths ``K``.
        down_sample_ratio: Geometric decay ratio ``r in (0, 1]``.
        down_sample_ratio_min: Minimum retention ratio floor.
        filter_position_zero: Drop position 0 from the deeper-depth candidate pool
            (it has no preceding token to predict).

    Returns:
        Tuple of:
            anchor_pos: Start position in the original sequence each sampled
                element's chain began from, shape ``[total_sampled]``.
            depth: Which COD round each element belongs to, shape
                ``[total_sampled]``. The reference (target) position of an
                element is ``anchor_pos + depth``.
    """
    loss_mask = loss_mask.squeeze(0)
    device = loss_mask.device
    all_valid_indices = torch.where(loss_mask == 1)[0]

    sample_indices = [torch.arange(seq_length, device=device)]
    n_per_depth = [seq_length]
    prev_indices = all_valid_indices

    for d in range(1, num_depths):
        valid_length = max(0, all_valid_indices.shape[0] - d)
        ratio = max(down_sample_ratio**d, down_sample_ratio_min)
        sample_size = int(valid_length * ratio)

        if sample_size <= 0:
            break

        # Subsample from the candidate pool, or keep all if the pool is small.
        if prev_indices.shape[0] >= sample_size:
            random_selection = torch.randperm(prev_indices.shape[0], device=device)[:sample_size]
            sampled_idx = prev_indices[random_selection]
            sampled_idx = torch.sort(sampled_idx)[0]  # restore causal order
        else:
            sampled_idx = prev_indices

        # Build the next depth's candidate pool: shift by +1 (next-token targets).
        next_candidates = (sampled_idx + 1) % seq_length
        if filter_position_zero:
            next_candidates = next_candidates[next_candidates != 0]
        mask = torch.isin(next_candidates, all_valid_indices)
        prev_indices = next_candidates[mask]

        # ``anchor_pos`` stores the chain start (``sampled_idx - d``); the
        # reference position is recovered as ``anchor_pos + depth``.
        sample_indices.append(sampled_idx - d)
        n_per_depth.append(sampled_idx.shape[0])

    anchor_pos = torch.cat(sample_indices)
    depth = torch.cat(
        [torch.full((n,), i, device=device, dtype=torch.long) for i, n in enumerate(n_per_depth)]
    )

    return anchor_pos, depth

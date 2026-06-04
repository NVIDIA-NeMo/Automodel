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

"""COD sampling for P-EAGLE parallel-group training.

P-EAGLE (Parallel-Drafting EAGLE, arXiv:2602.01469) trains the draft to predict
``num_depths`` tokens per position in a single forward pass. Naively this makes
attention memory scale with ``seq_len * num_depths``; Conditional-On-Distribution
(COD) sampling keeps it bounded by retaining geometrically fewer positions at
deeper prediction offsets: depth 0 keeps all ``n`` valid positions, depth ``d``
keeps ``max(r**d, r_min) * n`` of them.
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
    """Build the packed (anchor_pos, depth) index pair for one sequence.

    Args:
        seq_length: Length of the (single) sequence.
        loss_mask: ``[1, seq_len]`` (or ``[seq_len]``) 0/1 mask of supervised positions.
        num_depths: Number of parallel prediction groups ``K``.
        down_sample_ratio: Geometric decay ``r`` in ``(0, 1]``.
        down_sample_ratio_min: Floor on the retention ratio.
        filter_position_zero: Drop position 0 from a depth's candidate pool (it has
            no preceding token to predict from).

    Returns:
        ``(anchor_pos, depth)`` 1-D long tensors of equal length ``total_sampled``.
        ``orig_positions = anchor_pos + depth`` recovers the actual sequence index
        each packed element supervises.
    """
    loss_mask = loss_mask.squeeze(0)
    device = loss_mask.device
    all_valid_indices = torch.where(loss_mask == 1)[0]

    # Depth 0 always covers every position (the NTP head with real context).
    sample_indices = [torch.arange(seq_length, device=device)]
    n_per_depth = [seq_length]
    prev_indices = all_valid_indices

    for d in range(1, num_depths):
        valid_length = max(0, all_valid_indices.shape[0] - d)
        ratio = max(down_sample_ratio**d, down_sample_ratio_min)
        sample_size = int(valid_length * ratio)
        if sample_size <= 0:
            break

        if prev_indices.shape[0] >= sample_size:
            random_selection = torch.randperm(prev_indices.shape[0], device=device)[:sample_size]
            sampled_idx = prev_indices[random_selection]
            sampled_idx = torch.sort(sampled_idx)[0]  # restore causal order
        else:
            sampled_idx = prev_indices

        # Candidate pool for the next depth: the next-token positions, kept only
        # where they are themselves supervised.
        next_candidates = (sampled_idx + 1) % seq_length
        if filter_position_zero:
            next_candidates = next_candidates[next_candidates != 0]
        mask = torch.isin(next_candidates, all_valid_indices)
        prev_indices = next_candidates[mask]

        # Store the chain anchor (subtract d so anchor_pos + depth == real index).
        sample_indices.append(sampled_idx - d)
        n_per_depth.append(sampled_idx.shape[0])

    anchor_pos = torch.cat(sample_indices)
    depth = torch.cat([torch.full((n,), i, device=device, dtype=torch.long) for i, n in enumerate(n_per_depth)])
    return anchor_pos, depth

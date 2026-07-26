# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#
# Adapted from the Untied Ulysses reference implementation
# (https://github.com/togethercomputer/Untied-Ulysses), Apache-2.0.

"""Ulysses all-to-all reshard primitive.

Trades the sequence axis for the head axis (and back) across the Ulysses
process group. The reference implementation read its process group from a
mutable global; here the group is always an explicit argument.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


@torch.no_grad()
def all_to_all_4d(
    input: torch.Tensor,
    group: dist.ProcessGroup | None,
    scatter_idx: int = 2,
    gather_idx: int = 1,
) -> torch.Tensor:
    """Reshard a 4D ``[batch, seq, heads, head_dim]`` tensor across the Ulysses group.

    Args:
        input: 4D tensor to reshard.
        group: Ulysses process group. ``None`` or a group of size 1 is a no-op
            reshape (single-rank fallback).
        scatter_idx: Axis to scatter, either 2 (heads) or 1 (sequence).
        gather_idx: Axis to gather, either 1 (sequence) or 2 (heads).

    Returns:
        The resharded tensor. With ``scatter_idx=2, gather_idx=1`` the shape goes
        from ``[b, s/P, h, d]`` to ``[b, s, h/P, d]``; with ``scatter_idx=1,
        gather_idx=2`` it goes the other way.

    Raises:
        ValueError: If the input is not 4D or the index pair is unsupported.
    """
    if input.dim() != 4:
        raise ValueError(f"all_to_all_4d expects a 4D tensor, got {input.dim()}D")

    world_size = 1 if group is None else dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # [b, s/P, h, d] -> [b, s, h/P, d]
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * world_size
        shard_hc = hc // world_size

        # [b, s/P, h, d] -> [b, s/P, P, h/P, d] -> [P, s/P, b, h/P, d]
        input_t = input.reshape(bs, shard_seqlen, world_size, shard_hc, hs).transpose(0, 2).contiguous()

        if world_size > 1:
            output = torch.empty_like(input_t)
            dist.all_to_all_single(output, input_t, group=group)
        else:
            output = input_t

        return output.reshape(seqlen, bs, shard_hc, hs).transpose(0, 1)

    if scatter_idx == 1 and gather_idx == 2:
        # [b, s, h/P, d] -> [b, s/P, h, d]
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * world_size
        shard_seqlen = seqlen // world_size

        # [b, s, h/P, d] -> [b, P, s/P, h/P, d] -> [P, h/P, s/P, b, d]
        input_t = (
            input.reshape(bs, world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(world_size, shard_hc, shard_seqlen, bs, hs)
        )

        if world_size > 1:
            output = torch.empty_like(input_t)
            dist.all_to_all_single(output, input_t, group=group)
        else:
            output = input_t

        output = output.reshape(hc, shard_seqlen, bs, hs)
        return output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

    raise ValueError(f"unsupported (scatter_idx, gather_idx) pair: ({scatter_idx}, {gather_idx})")

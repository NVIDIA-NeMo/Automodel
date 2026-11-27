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

"""Target features and key lists shared by every DFly2 draft.

A DFly2 draft layer owns no context projection. It is paired with one sparse-attention
layer of the target and reads that layer's own keys (KV sharing) at the positions that
layer attends from the row before the anchor (index sharing). The target therefore hands
the draft, per paired layer, the key tensor it attended over and the key list of every
row; the draft keeps the row before each anchor and appends its own block.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Key lists are padded to a multiple of this width, matching the sparse attention kernels.
KEY_LIST_ALIGNMENT = 64


@dataclass(frozen=True)
class DFly2TargetFeatures:
    """Frozen target state one DFly2 training step reads.

    Attributes:
        context_keys: One tensor per draft layer, of shape [batch, context_keys, key_dim]:
            the keys and values the paired target layer attended over, in that layer's
            own row order (for example a sliding-window segment followed by a shared
            global segment).
        context_indices: One integer tensor per draft layer, of shape [batch, sequence,
            slots]. Row ``r`` lists the ``context_keys`` rows the paired target layer
            attended from token ``r``, with ``-1`` marking unused slots. Every listed key
            lies at or before ``r`` in ``r``'s document.
        last_hidden_states: Tensor of shape [batch, sequence, hidden], the target's final
            normalized hidden state (the input of its LM head).
    """

    context_keys: tuple[torch.Tensor, ...]
    context_indices: tuple[torch.Tensor, ...]
    last_hidden_states: torch.Tensor

    def __post_init__(self) -> None:
        if len(self.context_keys) != len(self.context_indices):
            raise ValueError("DFly2 target features need one key tensor per key-list tensor")
        batch, sequence = self.last_hidden_states.shape[:2]
        for keys, indices in zip(self.context_keys, self.context_indices):
            if keys.shape[0] != batch or indices.shape[:2] != (batch, sequence):
                raise ValueError("DFly2 context keys and key lists must match last_hidden_states [batch, sequence]")


def build_block_key_indices(
    context_indices: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    *,
    num_context_keys: int,
    block_size: int,
    causal: bool,
) -> torch.Tensor:
    """Build every draft query's key list: the target's picks for its block, then its block.

    Each block reads the key list of the target row just before its anchor, the last token
    the target has verified when the block is drafted. Every query of a block shares that
    list and then sees its own block's draft keys, which follow the ``num_context_keys``
    context rows in the draft attention's key tensor. A block never sees another block, and
    a dropped block (``block_keep_mask`` false) or an anchor at the sequence start sees no
    key, so its output is zero.

    Args:
        context_indices: Integer tensor of shape [batch, sequence, slots] with ``-1``
            unused, as in :class:`DFly2TargetFeatures`.
        anchor_positions: Integer tensor of shape [batch, num_anchors].
        block_keep_mask: Boolean tensor of shape [batch, num_anchors].
        num_context_keys: Number of context rows preceding the draft keys.
        block_size: Draft slots per block, the anchor slot included.
        causal: Whether a draft query sees only the earlier slots of its block.

    Returns:
        Int32 tensor of shape [batch, num_anchors * block_size, padded_slots], the context
        slots followed by ``block_size`` draft slots, ``-1`` padded to a multiple of
        :data:`KEY_LIST_ALIGNMENT`.
    """
    batch, num_anchors = anchor_positions.shape
    device = anchor_positions.device
    rows = (anchor_positions - 1).clamp(min=0)
    context = torch.gather(
        context_indices, 1, rows.unsqueeze(-1).expand(-1, -1, context_indices.shape[-1]).to(torch.long)
    )
    has_context = block_keep_mask & (anchor_positions > 0)
    context = torch.where(has_context.unsqueeze(-1), context, -1).repeat_interleave(block_size, dim=1)

    query = torch.arange(num_anchors * block_size, device=device)
    slot = torch.arange(block_size, device=device)
    draft = num_context_keys + (query // block_size).unsqueeze(-1) * block_size + slot
    visible = block_keep_mask.repeat_interleave(block_size, dim=1).unsqueeze(-1)
    if causal:
        visible = visible & (slot <= (query % block_size).unsqueeze(-1))
    draft = torch.where(visible, draft.expand(batch, -1, -1), -1)

    indices = torch.cat((context.to(torch.int32), draft.to(torch.int32)), dim=-1)
    return F.pad(indices, (0, -indices.shape[-1] % KEY_LIST_ALIGNMENT), value=-1)


def context_key_indices(key_indices: torch.Tensor, num_context_keys: int) -> torch.Tensor:
    """Keep only the context slots of draft key lists, masking the block's own keys.

    Args:
        key_indices: Integer tensor of shape [batch, queries, slots] from
            :func:`build_block_key_indices`.
        num_context_keys: Number of context rows preceding the draft keys.

    Returns:
        Tensor with the shape and dtype of ``key_indices`` where every draft-key slot is ``-1``.
    """
    return torch.where(key_indices < num_context_keys, key_indices, -1)


__all__ = [
    "DFly2TargetFeatures",
    "KEY_LIST_ALIGNMENT",
    "build_block_key_indices",
    "context_key_indices",
]

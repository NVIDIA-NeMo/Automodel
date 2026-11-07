# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Gated-LoRA multi-token-prediction attention mask (Samragh et al., 2025; arXiv:2507.11851).

Built over the depth-major concatenated ``[clean (S) | depth 1 (S) | ... | depth D (S)]`` input, where ``D``
is the number of MTP depths and the mask token of anchor ``t`` and depth ``d`` (1-based) sits at offset
``d * S + t``. The mask tokens of one anchor, one per depth, form that anchor's block:

  - A clean query attends the clean keys at or before it (plain causal attention), never a mask token, so the
    clean copy is exactly the base model.
  - A mask query of anchor ``t`` and depth ``d`` attends the clean keys at or before ``t`` and the mask tokens
    of its own block at depths ``<= d``.
"""

from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from nemo_automodel.components.attention.dflash_mask import _get_compiled_create_block_mask


def _gated_mtp_allowed(q_idx: torch.Tensor, kv_idx: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Return whether query ``q_idx`` may attend key ``kv_idx`` in the gated-MTP layout.

    Args:
        q_idx: Integer Tensor of query positions in the concatenated sequence, broadcastable against ``kv_idx``.
        kv_idx: Integer Tensor of key positions in the concatenated sequence, broadcastable against ``q_idx``.
        seq_len: Length ``S`` of the clean copy.

    Returns:
        Bool Tensor with the broadcast shape of ``q_idx`` and ``kv_idx``.
    """
    clean_q = q_idx < seq_len
    clean_kv = kv_idx < seq_len
    # Anchor and depth of a mask position (depth 0 is the clean copy, where the anchor is the position itself).
    anchor_q = q_idx % seq_len
    anchor_kv = kv_idx % seq_len
    depth_q = q_idx // seq_len
    depth_kv = kv_idx // seq_len
    clean_to_clean = clean_q & clean_kv & (kv_idx <= q_idx)
    mask_to_clean = ~clean_q & clean_kv & (kv_idx <= anchor_q)
    mask_to_block = ~clean_q & ~clean_kv & (anchor_kv == anchor_q) & (depth_kv <= depth_q)
    return clean_to_clean | mask_to_clean | mask_to_block


def _key_valid(valid_mask: torch.Tensor, num_depths: int) -> torch.Tensor:
    """Extend the clean padding mask to the concatenated keys.

    Mask tokens are always valid keys so that no query row is fully masked; the mask tokens of a padded anchor
    only feed positions whose labels are ignored.

    Args:
        valid_mask: Padding-validity mask over the clean copy, Tensor of shape [batch, sequence] (non-zero = real
            token).
        num_depths: Number of MTP depths ``D``.

    Returns:
        Bool Tensor of shape [batch, sequence * (1 + D)].
    """
    batch, seq_len = valid_mask.shape
    mask_keys = torch.ones(batch, seq_len * num_depths, dtype=torch.bool, device=valid_mask.device)
    return torch.cat([valid_mask.bool(), mask_keys], dim=1)


def create_gated_mtp_sdpa_mask(
    num_depths: int,
    valid_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the dense additive gated-MTP mask for SDPA or eager attention.

    Args:
        num_depths: Number of MTP depths ``D``.
        valid_mask: Padding-validity mask over the clean copy, Tensor of shape [batch, sequence] (non-zero = real
            token); padded clean keys are blocked.
        dtype: dtype of the additive mask (the dtype attention runs in).

    Returns:
        Tensor of shape [batch, 1, sequence * (1 + D), sequence * (1 + D)] on ``valid_mask``'s device: ``0`` at
        attended positions and ``-inf`` elsewhere.
    """
    batch, seq_len = valid_mask.shape
    total = seq_len * (1 + num_depths)
    idx = torch.arange(total, device=valid_mask.device)
    allowed = _gated_mtp_allowed(idx.view(-1, 1), idx.view(1, -1), seq_len)
    allowed = allowed.view(1, 1, total, total) & _key_valid(valid_mask, num_depths).view(batch, 1, 1, total)
    zero = torch.zeros((), device=valid_mask.device, dtype=dtype)
    neg_inf = torch.full((), float("-inf"), device=valid_mask.device, dtype=dtype)
    return torch.where(allowed, zero, neg_inf)


def create_gated_mtp_block_mask(
    num_depths: int,
    valid_mask: torch.Tensor,
    *,
    use_compile: bool = True,
) -> BlockMask:
    """Build the sparse FlexAttention :class:`BlockMask` for gated-MTP training.

    Same semantics as :func:`create_gated_mtp_sdpa_mask` without materializing the dense mask. Consumed by the
    ``flex_attention`` backend through the ``attention_mask`` kwarg.

    Args:
        num_depths: Number of MTP depths ``D``.
        valid_mask: Padding-validity mask over the clean copy, Tensor of shape [batch, sequence] (non-zero = real
            token).
        use_compile: Reuse a cached ``torch.compile``'d ``create_block_mask``.

    Returns:
        :class:`~torch.nn.attention.flex_attention.BlockMask` over ``sequence * (1 + D)`` queries and keys.
    """
    batch, seq_len = valid_mask.shape
    total = seq_len * (1 + num_depths)
    key_valid = _key_valid(valid_mask, num_depths)

    def gated_mtp_mask_mod(b, h, q_idx, kv_idx):
        return _gated_mtp_allowed(q_idx, kv_idx, seq_len) & key_valid[b, kv_idx]

    builder = _get_compiled_create_block_mask() if use_compile else create_block_mask
    return builder(gated_mtp_mask_mod, B=batch, H=None, Q_LEN=total, KV_LEN=total, device=valid_mask.device)

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

"""I-DLM block-diffusion attention mask (Yu et al., 2026; arXiv:2604.11035).

Built over the concatenated ``[x_t (L) | x_0 (L)]`` input, where ``x_t`` is the
noisy (masked) copy and ``x_0`` the clean copy. It combines three components
(paper Appendix E):

  - ``M_BD``  — causal self-attention within each noisy ``x_t`` block.
  - ``M_OBC`` — each ``x_t`` token cross-attends clean ``x_0`` tokens in strictly
    earlier blocks (the clean ground-truth prefix the decode is conditioned on).
  - ``M_BC``  — strict token-causal attention within the clean ``x_0`` copy.

With ``block_size == 1`` each ``x_t`` token attends only to itself plus the clean
tokens strictly before it, and ``x_0`` is plain autoregressive.
"""

from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from nemo_automodel.components.attention.dflash_mask import _get_compiled_create_block_mask


def create_idlm_sdpa_mask(
    seq_len: int,
    block_size: int,
    valid_mask: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the dense additive ``[x_t | x_0]`` block-diffusion mask for SDPA.

    The ``x_0`` region is always strict token-causal, matching the paper (§3.1)
    and every released I-DLM checkpoint.

    Args:
        seq_len:  Length ``L`` of one copy (the concatenated length is ``2L``).
        block_size: Diffusion block size (``block_length`` in the paper).
        valid_mask: Padding-validity mask over one copy, shape ``[B, L]``
            (1 = real token); padded keys are blocked in both copies.
        device:   torch device.
        dtype:    dtype for the additive mask (typically the model dtype).

    Returns:
        ``[B, 1, 2L, 2L]`` float tensor: ``0`` at attended positions, ``-inf``
        elsewhere.
    """
    two_l = 2 * seq_len
    q_idx = torch.arange(two_l, device=device).view(1, 1, -1, 1)
    kv_idx = torch.arange(two_l, device=device).view(1, 1, 1, -1)

    x0_q = q_idx >= seq_len
    x0_kv = kv_idx >= seq_len
    block_q = torch.where(x0_q, (q_idx - seq_len) // block_size, q_idx // block_size)
    block_kv = torch.where(x0_kv, (kv_idx - seq_len) // block_size, kv_idx // block_size)

    m_bd = (block_q == block_kv) & (~x0_q) & (~x0_kv) & (q_idx >= kv_idx)
    m_obc = (block_q > block_kv) & x0_kv & (~x0_q)
    m_bc = (q_idx >= kv_idx) & x0_q & x0_kv

    key_valid = torch.cat([valid_mask.bool(), valid_mask.bool()], dim=1).view(valid_mask.size(0), 1, 1, two_l)
    bool_mask = (m_bd | m_obc | m_bc) & key_valid

    neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    return torch.where(bool_mask, zero, neg_inf)


def create_idlm_block_mask(
    seq_len: int,
    block_size: int,
    valid_mask: torch.Tensor,
    *,
    device: torch.device,
    use_compile: bool = True,
) -> "BlockMask":
    """Build a sparse FlexAttention :class:`BlockMask` for I-DLM training.

    Same semantics as :func:`create_idlm_sdpa_mask` but avoids materialising the
    dense ``2L x 2L`` mask — preferred at scale. Consumed by transformers'
    ``flex_attention`` backend when ``_attn_implementation="flex_attention"``;
    pass it via the ``attention_mask`` kwarg.

    Args:
        seq_len:  Length ``L`` of one copy (concatenated length is ``2L``).
        block_size: Diffusion block size.
        valid_mask: Padding-validity mask over one copy, shape ``[B, L]``.
        device:   torch device.
        use_compile: Reuse a cached ``torch.compile``'d ``create_block_mask``.

    Returns:
        :class:`torch.nn.attention.flex_attention.BlockMask`.
    """
    two_l = 2 * seq_len
    B = valid_mask.size(0)
    key_valid = torch.cat([valid_mask.bool(), valid_mask.bool()], dim=1)  # [B, 2L]

    def idlm_mask_mod(b, h, q_idx, kv_idx):
        x0_q = q_idx >= seq_len
        x0_kv = kv_idx >= seq_len
        block_q = torch.where(x0_q, (q_idx - seq_len) // block_size, q_idx // block_size)
        block_kv = torch.where(x0_kv, (kv_idx - seq_len) // block_size, kv_idx // block_size)
        m_bd = (block_q == block_kv) & (~x0_q) & (~x0_kv) & (q_idx >= kv_idx)
        m_obc = (block_q > block_kv) & x0_kv & (~x0_q)
        m_bc = (q_idx >= kv_idx) & x0_q & x0_kv
        return (m_bd | m_obc | m_bc) & key_valid[b, kv_idx]

    builder = _get_compiled_create_block_mask() if use_compile else create_block_mask
    return builder(idlm_mask_mod, B=B, H=None, Q_LEN=two_l, KV_LEN=two_l, device=device)

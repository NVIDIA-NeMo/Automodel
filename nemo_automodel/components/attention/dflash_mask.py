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

"""DFlash sparse-attention masks (SDPA + FlexAttention).

Ports the mask builders from SpecForge (``specforge/core/dflash.py``) so that
multi-anchor DFlash training (up to ~512 anchors per sequence — paper
Appendix A.1) is tractable in memory.

KV layout:  ``[ context (S tokens) | block_0 | block_1 | ... | block_{N-1} ]``
Q  layout:  ``[ block_0 | block_1 | ... | block_{N-1} ]``

Each query in block *b* attends to:

  1. context positions strictly less than ``anchor[b]`` (causal-style prefix)
  2. its own block's noise positions (bidirectional in-block)
  3. nothing else — other blocks are invisible

The context is never queried *from* (the target LM is frozen, we only need
its hidden states), so omitting it from Q halves the attention compute vs.
including context positions in Q.
"""

from __future__ import annotations

import torch

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    _FLEX_AVAILABLE = True
except ImportError:  # pragma: no cover - older PyTorch
    BlockMask = None  # type: ignore[assignment]
    create_block_mask = None  # type: ignore[assignment]
    _FLEX_AVAILABLE = False


def flex_attention_available() -> bool:
    """Whether ``torch.nn.attention.flex_attention`` is importable."""
    return _FLEX_AVAILABLE


def create_dflash_sdpa_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a dense additive attention mask for the SDPA backend.

    Args:
        anchor_positions: ``[B, N]`` anchor positions per sample (long).
        block_keep_mask:  ``[B, N]`` per-sample valid-anchor mask (bool).
        ctx_len:          context length ``S``.
        block_size:       block size.
        device:           torch device.
        dtype:            dtype for the additive mask (typically the model dtype).

    Returns:
        ``[B, 1, N*block_size, S + N*block_size]`` float tensor: ``0`` at
        attended positions, ``-inf`` elsewhere.
    """
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = ctx_len + N * block_size

    q_idx = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)
    kv_idx = torch.arange(KV_LEN, device=device).view(1, 1, 1, -1)

    q_block = q_idx // block_size
    anchor_exp = anchor_positions.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)

    is_ctx = kv_idx < ctx_len
    ctx_visible = is_ctx & (kv_idx < anchor_exp)

    is_noise = kv_idx >= ctx_len
    kv_block = (kv_idx - ctx_len) // block_size
    noise_visible = is_noise & (q_block == kv_block)

    keep = block_keep_mask.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)
    bool_mask = (ctx_visible | noise_visible) & keep

    neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    return torch.where(bool_mask, zero, neg_inf)


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
    device: torch.device,
) -> "BlockMask":
    """Build a sparse FlexAttention :class:`BlockMask` for DFlash training.

    See module docstring for the mask semantics. The returned ``BlockMask`` is
    consumed directly by transformers' ``flex_attention`` backend when
    ``_attn_implementation="flex_attention"`` is set on the draft model — pass
    it via the ``attention_mask`` kwarg.

    Args:
        anchor_positions: ``[B, N]`` anchor positions (long).
        block_keep_mask:  ``[B, N]`` valid-anchor mask (bool).
        ctx_len:          context length.
        block_size:       block size.
        device:           torch device.

    Returns:
        :class:`torch.nn.attention.flex_attention.BlockMask`.
    """
    if not _FLEX_AVAILABLE:
        raise RuntimeError(
            "FlexAttention requires PyTorch >= 2.5 with torch.nn.attention.flex_attention; "
            "fall back to attention_backend='sdpa' for older PyTorch."
        )

    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = ctx_len + N * block_size

    # Capture the input tensors via closure. mask_mod must use only tensor ops
    # (no Python control flow) so torch.compile can trace it.
    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block = q_idx // block_size
        safe_q_block = q_block.clamp(max=N - 1)
        anchor = anchor_positions[b, safe_q_block]

        is_ctx = kv_idx < ctx_len
        ctx_visible = is_ctx & (kv_idx < anchor)

        is_noise = kv_idx >= ctx_len
        kv_block = (kv_idx - ctx_len) // block_size
        noise_visible = is_noise & (q_block == kv_block)

        keep = block_keep_mask[b, safe_q_block]
        in_bounds = q_block < N
        return (ctx_visible | noise_visible) & keep & in_bounds

    return create_block_mask(
        dflash_mask_mod,
        B=B,
        H=None,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=device,
    )

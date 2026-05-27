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

Builds the DFlash block-diagonal attention masks (paper §4.2) so that
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


# Singleton-cached torch.compile of ``create_block_mask``. The mask_mod closure
# changes per step (new anchor positions), but the create_block_mask machinery
# itself is identical — compiling once and reusing avoids per-step Python
# overhead of evaluating mask_mod across the BlockMask grid.
#
# Mirrors the graceful-fallback pattern used by
# nemo_automodel.components.utils.compile_utils.compile_model: if torch.compile
# raises during construction or first call, return the uncompiled
# create_block_mask so training is never blocked by a compile-side bug.
_compiled_create_block_mask = None
_compile_attempted = False
_compile_failed = False


def _get_compiled_create_block_mask():
    """Lazy-initialise a compiled ``create_block_mask`` and cache it.

    Returns the uncompiled fallback if construction or invocation of the
    compiled version raises (matches the fallback semantics of
    ``compile_utils.compile_model``).
    """
    global _compiled_create_block_mask, _compile_attempted, _compile_failed
    if not _FLEX_AVAILABLE:
        raise RuntimeError("FlexAttention not available; cannot compile create_block_mask.")
    if _compile_failed:
        return create_block_mask
    if _compiled_create_block_mask is not None:
        return _compiled_create_block_mask
    if _compile_attempted:
        # Construction succeeded earlier; just return cached.
        return _compiled_create_block_mask or create_block_mask

    _compile_attempted = True
    import torch

    try:
        _compiled_create_block_mask = torch.compile(create_block_mask, dynamic=False)
    except Exception as exc:  # noqa: BLE001 — match compile_model's broad fallback.
        import logging

        logging.getLogger(__name__).warning(
            "torch.compile(create_block_mask) construction failed; falling back to uncompiled "
            "create_block_mask. Reason: %s",
            exc,
        )
        _compile_failed = True
        return create_block_mask
    return _compiled_create_block_mask


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
    use_compile: bool = True,
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
        use_compile:      Cache and reuse a torch.compile'd ``create_block_mask``
            across calls (default True). Set to False when running on PyTorch
            builds that hit Inductor errors during compile.

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

    # ``BLOCK_SIZE`` is left at the default (128). It MUST be a multiple of the
    # underlying flex_attention kernel's BLOCK_M / BLOCK_N (128 on H100); setting
    # it equal to our draft block_size (16) would in theory give finer-grained
    # sparsity but triggers Inductor's "Q and KV block size must be divisible
    # by BLOCK_M and BLOCK_N" lowering error at runtime.
    builder = _get_compiled_create_block_mask() if use_compile else create_block_mask
    try:
        return builder(
            dflash_mask_mod,
            B=B,
            H=None,
            Q_LEN=Q_LEN,
            KV_LEN=KV_LEN,
            device=device,
        )
    except Exception as exc:  # noqa: BLE001 — same graceful-fallback as compile_model.
        if builder is create_block_mask:
            raise
        global _compile_failed
        _compile_failed = True
        import logging

        logging.getLogger(__name__).warning(
            "Compiled create_block_mask invocation failed; falling back to uncompiled for this and "
            "subsequent calls. Reason: %s",
            exc,
        )
        return create_block_mask(
            dflash_mask_mod,
            B=B,
            H=None,
            Q_LEN=Q_LEN,
            KV_LEN=KV_LEN,
            device=device,
        )

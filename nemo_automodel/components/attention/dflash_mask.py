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
  2. its own block's noise positions (bidirectional in-block, or causal in-block
     when ``causal=True`` — see below)
  3. nothing else — other blocks are invisible

The context is never queried *from* (the target LM is frozen, we only need
its hidden states), so omitting it from Q halves the attention compute vs.
including context positions in Q.

``causal=False`` (default) is the DFlash mask: in-block attention is
bidirectional. ``causal=True`` is the JetSpec mask: a query at within-block
offset ``i`` attends only to noise positions at offset ``j <= i``, so each
branch follows the target model's autoregressive order (paper §2.2). Offset 0
(the anchor) still attends to itself, so no query row is ever fully masked.
"""

from __future__ import annotations

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

# Singleton-cached torch.compile of ``create_block_mask``. The mask_mod closure
# changes per step (new anchor positions), but the create_block_mask machinery
# itself is identical — compiling once and reusing avoids per-step Python
# overhead of evaluating mask_mod across the BlockMask grid.
# ``dynamic=True``: the mask shape depends on the per-batch anchor count
# (``min(num_anchors, valid anchors in the batch)``), so a static compile
# re-specializes (a full recompile, a multi-second stall mid-training) every
# time a batch carries a new anchor count. One dynamic compile covers all.
_compiled_create_block_mask = None


def _get_compiled_create_block_mask():
    """Lazy-initialise a compiled ``create_block_mask`` and cache it."""
    global _compiled_create_block_mask
    if _compiled_create_block_mask is None:
        _compiled_create_block_mask = torch.compile(create_block_mask, dynamic=True)
    return _compiled_create_block_mask


def create_dflash_sdpa_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
    causal: bool = False,
    ctx_doc_id: torch.Tensor | None = None,
    anchor_doc_id: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a dense additive attention mask for the SDPA backend.

    Args:
        anchor_positions: ``[B, N]`` anchor positions per sample (long).
        block_keep_mask:  ``[B, N]`` per-sample valid-anchor mask (bool).
        ctx_len:          context length ``S``.
        block_size:       block size.
        device:           torch device.
        dtype:            dtype for the additive mask (typically the model dtype).
        causal:           When True, make in-block attention causal (JetSpec);
            otherwise it is bidirectional (DFlash).
        ctx_doc_id:       ``[B, S]`` long per-context-token document id (sequence
            packing), or ``None`` to disable the per-document constraint.
        anchor_doc_id:    ``[B, N]`` long document id of each anchor. Required when
            ``ctx_doc_id`` is given; a block then attends only to context tokens in
            its anchor's document.

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
    if ctx_doc_id is not None:
        # Packing: a block attends only to context tokens in its anchor's
        # document. Build a per-kv doc id whose noise slots are a sentinel (-1)
        # that never matches an anchor's (>=0) doc id, then compare per block.
        kv_doc = torch.full((B, KV_LEN), -1, dtype=torch.long, device=device)
        kv_doc[:, :ctx_len] = ctx_doc_id
        anchor_doc_exp = anchor_doc_id.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)
        ctx_visible = ctx_visible & (kv_doc.view(B, 1, 1, KV_LEN) == anchor_doc_exp)

    is_noise = kv_idx >= ctx_len
    kv_block = (kv_idx - ctx_len) // block_size
    noise_visible = is_noise & (q_block == kv_block)
    if causal:
        # Block-causal in-block attention (JetSpec): offset i sees only offsets
        # j <= i. Offset 0 (anchor) still sees itself, so no row is fully masked.
        q_in_block = q_idx - q_block * block_size
        kv_in_block = (kv_idx - ctx_len) - kv_block * block_size
        noise_visible = noise_visible & (kv_in_block <= q_in_block)

    keep = block_keep_mask.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)
    # A padding block (``keep`` False) keeps only its in-block (noise) attention,
    # so its query rows are never fully masked. A fully-masked row makes the dense
    # softmax (the sdpa/eager backends) return NaN, which then contaminates the
    # rest of the sample through the next layer's ``0 * NaN`` value aggregation.
    # The block's output is discarded downstream by the loss block mask anyway.
    bool_mask = (ctx_visible & keep) | noise_visible

    neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)
    zero = torch.tensor(0.0, device=device, dtype=dtype)
    return torch.where(bool_mask, zero, neg_inf)


def build_dflash_mask_mod(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
    num_blocks: int,
    causal: bool = False,
    ctx_doc_id: torch.Tensor | None = None,
    anchor_doc_id: torch.Tensor | None = None,
):
    """Return the FlexAttention ``mask_mod`` closure encoding the DFlash mask.

    Factored out of :func:`create_dflash_block_mask` so the doc-aware gating can be
    materialised with ``torch.nn.attention.flex_attention.create_mask`` and tested on
    CPU (FlexAttention's kernel is CUDA-only). The closure uses only tensor ops (no
    Python control flow) so ``torch.compile`` can trace it; see the module docstring
    for the mask semantics and the ``ctx_doc_id`` / ``anchor_doc_id`` packing args.
    """
    N = num_blocks
    doc_aware = ctx_doc_id is not None

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block = q_idx // block_size
        safe_q_block = q_block.clamp(max=N - 1)
        anchor = anchor_positions[b, safe_q_block]

        is_ctx = kv_idx < ctx_len
        ctx_visible = is_ctx & (kv_idx < anchor)
        if doc_aware:
            # Packing: restrict the context prefix to the anchor's document.
            # Clamp the kv index into range for the ctx lookup; the result only
            # gates ``ctx_visible`` (already False on the noise half).
            kv_doc = ctx_doc_id[b, kv_idx.clamp(max=ctx_len - 1)]
            ctx_visible = ctx_visible & (kv_doc == anchor_doc_id[b, safe_q_block])

        is_noise = kv_idx >= ctx_len
        kv_block = (kv_idx - ctx_len) // block_size
        noise_visible = is_noise & (q_block == kv_block)
        if causal:
            # Block-causal in-block attention (JetSpec): offset i sees only
            # offsets j <= i. Offset 0 (anchor) still sees itself.
            q_in_block = q_idx - q_block * block_size
            kv_in_block = (kv_idx - ctx_len) - kv_block * block_size
            noise_visible = noise_visible & (kv_in_block <= q_in_block)

        keep = block_keep_mask[b, safe_q_block]
        in_bounds = q_block < N
        # Padding blocks keep in-block attention so no query row is fully masked
        # (see create_dflash_sdpa_mask; kept consistent across backends, and the
        # output is dropped by the loss block mask anyway).
        return ((ctx_visible & keep) | noise_visible) & in_bounds

    return dflash_mask_mod


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    ctx_len: int,
    block_size: int,
    device: torch.device,
    use_compile: bool = True,
    causal: bool = False,
    ctx_doc_id: torch.Tensor | None = None,
    anchor_doc_id: torch.Tensor | None = None,
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
        causal:           When True, make in-block attention causal (JetSpec);
            otherwise it is bidirectional (DFlash).
        ctx_doc_id:       ``[B, S]`` long per-context-token document id (sequence
            packing), or ``None`` to disable the per-document constraint.
        anchor_doc_id:    ``[B, N]`` long document id of each anchor. Required when
            ``ctx_doc_id`` is given; a block then attends only to context tokens in
            its anchor's document.

    Returns:
        :class:`torch.nn.attention.flex_attention.BlockMask`.
    """
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = ctx_len + N * block_size

    dflash_mask_mod = build_dflash_mask_mod(
        anchor_positions, block_keep_mask, ctx_len, block_size, N, causal, ctx_doc_id, anchor_doc_id
    )

    # ``BLOCK_SIZE`` is left at the default (128). It MUST be a multiple of the
    # underlying flex_attention kernel's BLOCK_M / BLOCK_N (128 on H100); setting
    # it equal to our draft block_size (16) would in theory give finer-grained
    # sparsity but triggers Inductor's "Q and KV block size must be divisible
    # by BLOCK_M and BLOCK_N" lowering error at runtime.
    builder = _get_compiled_create_block_mask() if use_compile else create_block_mask
    return builder(
        dflash_mask_mod,
        B=B,
        H=None,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=device,
    )

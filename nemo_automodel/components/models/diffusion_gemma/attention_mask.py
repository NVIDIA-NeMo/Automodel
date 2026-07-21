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

"""Block-causal training attention mask for ``diffusion_gemma`` block diffusion.

This module builds the *training* decoder attention mask for the
``diffusion_gemma`` block-diffusion model. It is the highest correctness-risk
piece of the SFT path; read the leakage invariant below before changing it.

Layout
------
At inference the model runs the shared transformer twice: once *causally* over
the clean prefix to populate a per-layer KV cache (the "encoder" KV), and once
*bidirectionally* over a single noised ``canvas`` block (the "decoder"). Each
decoder layer concatenates ``[encoder_KV ; canvas_KV]`` along the key axis
(``modeling_diffusion_gemma.py:579-582``), so the decoder query attends over a
key axis of length ``enc_len + canvas_len``.

For training we run the whole sequence at once and supervise all response
blocks jointly (joint block-causal block diffusion). The encoder holds the
**clean** full sequence (prompt + full response); the canvas holds the
**noised** full response. The training mask therefore has shape
``[B, 1, canvas_len, enc_len + canvas_len]`` and splits column-wise into:

* **Left columns** ``[0, enc_len)`` — the clean encoder KV. A canvas query in
  block ``i`` may attend a clean encoder column only if that column belongs to
  a response block **strictly before** block ``i`` (offset-block-causal,
  ``M_OBC``). Prompt columns (encoder positions ``< prefix_len``) are always
  visible.
* **Right columns** ``[enc_len, enc_len + canvas_len)`` — the noised canvas KV.
  Block-diagonal (``M_BD``): canvas block ``i`` attends bidirectionally within
  block ``i`` only, never to another canvas block.

This mirrors the 3-part BD3LM mask (``M_BD`` / ``M_OBC`` / ``M_BC``) in
``dllm-zhz/dllm/core/trainers/bd3lm.py`` adapted to the encoder-KV/canvas
layout. There is no ``M_BC`` term: in BD3LM ``M_BC`` is block-causal attention
*within the clean (x_0) half*, but here the clean tokens live only in the
encoder columns (there is no clean-canvas query half), so it does not apply.

Leakage invariant (THE correctness property)
---------------------------------------------
``M_OBC`` uses a **strict** ``block_q > block_kv`` comparison. A canvas query in
block ``i`` MUST be masked against the clean encoder column at response-relative
position ``i * block_size`` (the first token of its own block) and every later
clean position. Using ``>=`` instead of ``>`` is silent **total leakage**: the
canvas would see the clean answer for the very tokens it is being trained to
denoise, the loss would collapse, and the model would learn nothing useful.
The unit test ``tests/.../test_diffusion_gemma_mask.py`` asserts exactly this
boundary and is the gate for the rest of the SFT work.
"""

from __future__ import annotations

import torch


def _block_ids(num_positions: int, block_size: int, device: torch.device) -> torch.Tensor:
    """Response-relative block index for each of ``num_positions`` positions."""
    return torch.arange(num_positions, device=device) // block_size


def build_block_diffusion_training_mask(
    prefix_lengths: torch.Tensor | int,
    response_length: int,
    enc_len: int,
    block_size: int,
    *,
    sliding_window: int | None = None,
    batch_size: int | None = None,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the block-causal training mask and its sliding-window variant.

    The canvas (decoder query axis) has length ``response_length``. The key axis
    is ``[encoder_KV (enc_len) ; canvas_KV (response_length)]``, so the returned
    masks have shape ``[B, 1, response_length, enc_len + response_length]``.

    Args:
        prefix_lengths: Per-example prompt length(s) in the encoder, i.e. the
            number of leading clean encoder positions that are prompt (always
            attendable). An ``int`` is broadcast to all examples; a 1-D tensor
            of shape ``[B]`` gives per-example prefixes. The response occupies
            encoder positions ``[prefix_length, prefix_length + response_length)``.
        response_length: Canvas length (number of noised response positions).
        enc_len: Total encoder key length (``prefix + response`` and any tail
            padding columns). Must satisfy ``enc_len >= max(prefix) + response_length``.
        block_size: Diffusion block size (``canvas_length``; 256 for the ckpt).
        sliding_window: If given, the sliding variant additionally restricts
            attention to key positions within ``sliding_window`` absolute
            positions of the query (see module docstring for position-id
            convention). If ``None`` the sliding variant equals the full mask.
        batch_size: Batch dimension. Required when ``prefix_lengths`` is an int;
            inferred from the tensor otherwise.
        device: Device for the returned tensors.
        dtype: If ``None`` (default) return boolean *keep* masks (``True`` =
            attend). If a floating dtype, return an **additive** mask
            (``0`` where attended, ``-inf`` where masked) ready to add to
            attention scores.

    Returns:
        Tuple ``(mask_full, mask_sliding)``:

        * ``mask_full`` — ``M_OBC`` (left cols) ∪ ``M_BD`` (right cols), used by
          the full-attention layers ``{5, 11, 17, 23, 29}``.
        * ``mask_sliding`` — ``mask_full`` ∩ sliding-window, used by the 25
          sliding-attention layers.

        Each has shape ``[B, 1, response_length, enc_len + response_length]``.
    """
    device = torch.device(device)

    if isinstance(prefix_lengths, int):
        if batch_size is None:
            raise ValueError("batch_size must be provided when prefix_lengths is an int")
        prefix = torch.full((batch_size,), prefix_lengths, dtype=torch.long, device=device)
    else:
        prefix = prefix_lengths.to(device=device, dtype=torch.long)
        if prefix.ndim != 1:
            raise ValueError(f"prefix_lengths tensor must be 1-D [B], got shape {tuple(prefix.shape)}")
        if batch_size is None:
            batch_size = prefix.shape[0]
        elif batch_size != prefix.shape[0]:
            raise ValueError(f"batch_size {batch_size} != prefix_lengths length {prefix.shape[0]}")

    if (prefix < 0).any():
        raise ValueError("prefix_lengths must be non-negative")
    if (prefix + response_length > enc_len).any():
        raise ValueError(
            f"enc_len ({enc_len}) too small: need prefix + response_length <= enc_len "
            f"(max prefix {int(prefix.max())}, response_length {response_length})"
        )

    canvas_len = response_length  # key axis is [enc_len ; canvas_len]

    # Canvas query block index (response-relative): [canvas_len]
    q_block = _block_ids(canvas_len, block_size, device)  # [Lq]

    # --- Left columns: clean encoder KV -> M_OBC (offset-block-causal) ---
    enc_pos = torch.arange(enc_len, device=device)  # [enc_len], absolute encoder position
    # Response-relative offset of each encoder column, per example: [B, enc_len].
    # Prompt columns (offset < 0) get a sentinel block id of -1 so that any
    # canvas block (block_q >= 0) strictly exceeds them => prompt always visible.
    enc_rel = enc_pos[None, :] - prefix[:, None]  # [B, enc_len]
    enc_block = torch.where(
        enc_rel >= 0,
        enc_rel // block_size,
        torch.full_like(enc_rel, -1),
    )  # [B, enc_len]
    # Columns that are tail padding beyond the response are never attendable.
    enc_is_valid = enc_rel < response_length  # [B, enc_len]

    # M_OBC: strict block_q > block_kv. THE leakage invariant lives here.
    m_obc = (q_block[None, :, None] > enc_block[:, None, :]) & enc_is_valid[:, None, :]  # [B, Lq, enc_len]

    # --- Right columns: noised canvas KV -> M_BD (block-diagonal) ---
    kv_block = _block_ids(canvas_len, block_size, device)  # [Lkv_canvas]
    m_bd = q_block[:, None] == kv_block[None, :]  # [Lq, canvas_len]
    m_bd = m_bd[None].expand(batch_size, -1, -1)  # [B, Lq, canvas_len]

    keep = torch.cat([m_obc, m_bd], dim=2)  # [B, Lq, key_len]
    keep = keep.unsqueeze(1)  # [B, 1, Lq, key_len]

    # --- Sliding-window variant ---
    if sliding_window is None:
        keep_sliding = keep.clone()
    else:
        # BLOCK-ANCHORED encoder sliding window, matching the reference inference geometry
        # (``create_diffusion_decoder_attention_mask``). When canvas block b is denoised,
        # the encoder KV cache holds exactly prompt + response blocks 0..b-1, i.e. it ENDS
        # at the absolute boundary ``valid_cache_b = prefix + b*block_size``. A sliding
        # layer keeps the last ``sliding_window`` cache columns ending at that boundary:
        # ``[valid_cache_b - sliding_window + 1, valid_cache_b)``. This window is CONSTANT
        # for every canvas query in block b (anchored to the block boundary, NOT to the
        # query's own position) — it is *not* a per-query band.
        #
        # (An earlier version used a per-query symmetric band ``|q_abs - k_abs| < sw``.
        # That shifts the window off the cache end for later-in-block queries, starving
        # them of the previous-block context the reference feeds them at inference — a
        # train/inference parity break on every sliding layer. Note also that Google's
        # SFT trains all canvases in parallel, not one at a time, so the old "one canvas
        # at a time" rationale was incorrect.)
        #
        # The encoder UPPER bound (k < valid_cache_b) is already imposed by M_OBC
        # (block-causal: enc_block < q_block); the sliding window adds only the LOWER
        # bound. ``enc_abs >= 0`` auto-clamps it, so an early block whose cache is shorter
        # than the window (valid_cache_b < sw) gets no lower bound — matching the reference's
        # ``sliding_start_idx = 0`` branch. Canvas (M_BD) columns get NO sliding band: the
        # current block is always fully attended within itself (the reference pads the canvas
        # region all-True) and M_BD already confines canvas keys to the block.
        block_start = q_block * block_size  # [Lq] response-relative start of each query's block
        valid_cache = prefix[:, None] + block_start[None, :]  # [B, Lq] abs cache end for the block
        enc_abs = torch.arange(enc_len, device=device)  # [enc_len]
        enc_within = enc_abs[None, None, :] >= (valid_cache[:, :, None] - sliding_window + 1)  # [B, Lq, enc_len]
        canvas_within = torch.ones((batch_size, canvas_len, canvas_len), dtype=torch.bool, device=device)
        within = torch.cat([enc_within, canvas_within], dim=2)  # [B, Lq, key_len]
        keep_sliding = keep & within[:, None]

    if dtype is None:
        return keep, keep_sliding

    return _to_additive(keep, dtype), _to_additive(keep_sliding, dtype)


def _to_additive(keep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert a boolean keep-mask to an additive mask (0 / -inf)."""
    additive = torch.zeros(keep.shape, dtype=dtype, device=keep.device)
    additive.masked_fill_(~keep, torch.finfo(dtype).min)
    return additive

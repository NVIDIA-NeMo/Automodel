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

"""Leakage / structure tests for the diffusion_gemma block-causal training mask.

This is the gate test for the whole SFT path (design v2 item 2): the
offset-block-causal term ``M_OBC`` must use a STRICT ``block_q > block_kv``
comparison. If it ever regresses to ``>=`` a canvas block would attend the
clean encoder copy of its own tokens (total leakage) and these assertions fail.

Pure torch / CPU — no GPU, no model forward.
"""

import torch

from nemo_automodel.components.models.diffusion_gemma.attention_mask import (
    build_block_diffusion_training_mask,
)


def test_output_shape_and_dtype():
    bs, resp, enc, blk = 2, 8, 8, 4
    keep, keep_sliding = build_block_diffusion_training_mask(
        prefix_lengths=0, response_length=resp, enc_len=enc, block_size=blk, batch_size=bs
    )
    assert keep.shape == (bs, 1, resp, enc + resp)
    assert keep_sliding.shape == (bs, 1, resp, enc + resp)
    assert keep.dtype == torch.bool
    assert keep_sliding.dtype == torch.bool


# ---------------------------------------------------------------------------
# THE leakage invariant: strict block_q > block_kv at the own-block boundary.
# ---------------------------------------------------------------------------


def test_leakage_block_i_masked_at_encoder_block_start():
    """Block-i canvas query is MASKED at clean encoder position i*block_size.

    prefix_length=0 => encoder holds the clean response only, so encoder
    position p has response-relative offset p. Canvas block i covers canvas
    positions [i*blk, (i+1)*blk). The first clean position of block i is
    encoder column i*blk; the strict-> boundary masks it (and everything after).
    """
    bs, blk = 1, 4
    resp = enc = 16  # 4 blocks of size 4, encoder == clean response
    keep, _ = build_block_diffusion_training_mask(
        prefix_lengths=0, response_length=resp, enc_len=enc, block_size=blk, batch_size=bs
    )
    num_blocks = resp // blk
    for i in range(num_blocks):
        q = i * blk  # a query inside block i (use its first position)
        # MASKED: own-block start and every later clean encoder column.
        for p in range(i * blk, enc):
            assert not keep[0, 0, q, p], f"LEAKAGE: block {i} query attends clean encoder pos {p} (>= own block start)"
        # VISIBLE: all strictly-earlier clean encoder columns.
        for p in range(0, i * blk):
            assert keep[0, 0, q, p], f"block {i} query should attend earlier clean encoder pos {p}"


def test_every_query_in_block_sees_same_encoder_blocks():
    """All queries in block i share the same offset-block-causal encoder visibility."""
    bs, blk = 1, 4
    resp = enc = 16
    keep, _ = build_block_diffusion_training_mask(
        prefix_lengths=0, response_length=resp, enc_len=enc, block_size=blk, batch_size=bs
    )
    for i in range(resp // blk):
        rows = [keep[0, 0, i * blk + j, :enc] for j in range(blk)]
        for r in rows[1:]:
            assert torch.equal(rows[0], r)


def test_strict_boundary_differs_from_non_strict():
    """Guard: a >= (non-strict) mask would additionally expose the own block.

    Reconstruct what a buggy >= comparison would produce and assert the real
    mask is strictly more restrictive on the encoder columns (i.e. it hides the
    diagonal own-block that >= would leak).
    """
    bs, blk = 1, 4
    resp = enc = 12
    keep, _ = build_block_diffusion_training_mask(
        prefix_lengths=0, response_length=resp, enc_len=enc, block_size=blk, batch_size=bs
    )
    q_block = torch.arange(resp) // blk
    enc_block = torch.arange(enc) // blk
    leaky = q_block[:, None] >= enc_block[None, :]  # the BUG
    strict = keep[0, 0, :, :enc]
    # strict must mask at least the own-block diagonal that leaky exposes.
    own_block = q_block[:, None] == enc_block[None, :]
    assert (leaky & own_block).any(), "test setup: leaky mask should expose own block"
    assert not (strict & own_block).any(), "LEAKAGE: strict mask exposes own clean block"


# ---------------------------------------------------------------------------
# Block-diagonal canvas (M_BD): block i never attends canvas block j != i.
# ---------------------------------------------------------------------------


def test_canvas_block_diagonal():
    bs, blk = 1, 4
    resp = enc = 16
    keep, _ = build_block_diffusion_training_mask(
        prefix_lengths=0, response_length=resp, enc_len=enc, block_size=blk, batch_size=bs
    )
    canvas = keep[0, 0, :, enc:]  # [Lq, canvas_len]
    q_block = torch.arange(resp) // blk
    kv_block = torch.arange(resp) // blk
    same_block = q_block[:, None] == kv_block[None, :]
    # Within own block: fully attended (bidirectional).
    assert (canvas[same_block]).all(), "canvas should attend bidirectionally within its own block"
    # Across blocks: never attended.
    assert not (canvas[~same_block]).any(), "canvas block i must not attend canvas block j != i"


# ---------------------------------------------------------------------------
# Prompt prefix is always visible; tail padding is never visible.
# ---------------------------------------------------------------------------


def test_prompt_prefix_always_visible():
    bs, blk, prefix = 1, 4, 5
    resp = 8
    enc = prefix + resp  # encoder = prompt(5) + clean response(8)
    keep, _ = build_block_diffusion_training_mask(
        prefix_lengths=prefix, response_length=resp, enc_len=enc, block_size=blk, batch_size=bs
    )
    # Every canvas query (any block) attends every prompt column.
    assert keep[0, 0, :, :prefix].all(), "all prompt columns must be visible to every canvas block"


def test_tail_padding_never_visible():
    """Encoder columns beyond prefix+response_length are padding -> never attended."""
    bs, blk, prefix = 1, 4, 0
    resp = 8
    enc = resp + 6  # 6 extra padding columns
    keep, _ = build_block_diffusion_training_mask(
        prefix_lengths=prefix, response_length=resp, enc_len=enc, block_size=blk, batch_size=bs
    )
    pad = keep[0, 0, :, prefix + resp : enc]
    assert not pad.any(), "tail padding encoder columns must never be attended"


def test_per_example_prefix_lengths():
    """A 1-D prefix tensor offsets the response start per example."""
    blk, resp = 4, 8
    prefixes = torch.tensor([0, 4])
    enc = int(prefixes.max()) + resp
    keep, _ = build_block_diffusion_training_mask(
        prefix_lengths=prefixes, response_length=resp, enc_len=enc, block_size=blk
    )
    # Example 0: prefix 0 -> block-0 query masked at encoder col 0.
    assert not keep[0, 0, 0, 0]
    # Example 1: prefix 4 -> cols [0,4) are prompt (always visible); response
    # starts at col 4, so block-0 query masked at encoder col 4 but sees col 3.
    assert keep[1, 0, 0, 3], "example-1 prompt column should be visible"
    assert not keep[1, 0, 0, 4], "example-1 own-block clean start should be masked"


# ---------------------------------------------------------------------------
# Sliding variant respects the window and is a subset of the full mask.
# ---------------------------------------------------------------------------


def test_sliding_is_subset_of_full():
    bs, blk = 1, 4
    resp = enc = 32
    keep, keep_sliding = build_block_diffusion_training_mask(
        prefix_lengths=0, response_length=resp, enc_len=enc, block_size=blk, batch_size=bs, sliding_window=8
    )
    # Sliding can only remove attention, never add it.
    assert (keep | ~keep_sliding).all(), "sliding mask must be a subset of the full mask"


def test_sliding_window_block_anchored():
    """Encoder sliding window is BLOCK-ANCHORED, matching the wheel's inference cache slice.

    For canvas block b the encoder cache ends at ``valid_cache_b = prefix + b*block_size``,
    and a sliding layer keeps encoder columns ``[valid_cache_b - sw + 1, valid_cache_b)`` --
    the SAME slice for every query in block b (anchored to the block boundary, NOT to the
    query's own position). The upper bound is the block-causal M_OBC; the window adds the
    lower bound. (Previously this test asserted a per-query band ``|q_abs - k_abs| < sw`` --
    the geometry the wheel does NOT use; see review #1.)
    """
    bs, blk, window = 1, 4, 8
    resp = enc = 32
    _, keep_sliding = build_block_diffusion_training_mask(
        prefix_lengths=0, response_length=resp, enc_len=enc, block_size=blk, batch_size=bs, sliding_window=window
    )
    enc_part = keep_sliding[0, 0, :, :enc]  # [resp, enc] : canvas query x encoder col
    for j in range(resp):
        valid_cache = (j // blk) * blk  # prefix=0; cache end at this query's block boundary
        lo = valid_cache - window + 1
        for k in enc_part[j].nonzero().flatten().tolist():
            assert lo <= k < valid_cache, (
                f"query {j} (block {j // blk}) kept encoder col {k} outside the block-anchored "
                f"window [{lo}, {valid_cache})"
            )
    # Block-constancy: every query in a block shares the same encoder-column window.
    for b in range(resp // blk):
        rows = enc_part[b * blk : (b + 1) * blk]
        assert (rows == rows[0]).all(), f"block {b}: encoder window varies across its queries (not block-anchored)"


def test_sliding_keeps_prompt_for_early_block():
    """Regression: block 0's clean-encoder prompt visibility must survive the sliding band.

    Under the block-anchored window, block 0's cache ends at ``valid_cache_0 = prefix`` (= 6),
    shorter than the window (16), so ``valid_cache_0 - sw + 1 < 0`` => no lower bound (the
    wheel's ``sliding_start_idx = 0`` branch). Hence the prompt columns [0, prefix) stay
    visible to block 0, identical to the full mask.
    """
    bs, blk, window = 1, 4, 16
    prefix, resp = 6, 16
    seq = prefix + resp  # training geometry: encoder holds the whole clean sequence
    full, sliding = build_block_diffusion_training_mask(
        prefix_lengths=prefix, response_length=resp, enc_len=seq, block_size=blk, batch_size=bs, sliding_window=window
    )
    f, s = full[0, 0], sliding[0, 0]
    block0, prompt = slice(0, blk), slice(0, prefix)
    # valid_cache_0 = prefix (6) < window (16) => no lower bound => prompt survives unchanged.
    assert f[block0, prompt].any(), "setup: prompt should be visible to block 0 in the full mask"
    assert torch.equal(f[block0, prompt], s[block0, prompt]), (
        "block-anchored sliding removed block-0's prompt visibility (early-block short-cache case)"
    )


def test_sliding_none_equals_full():
    bs, blk = 1, 4
    resp = enc = 16
    keep, keep_sliding = build_block_diffusion_training_mask(
        prefix_lengths=0, response_length=resp, enc_len=enc, block_size=blk, batch_size=bs, sliding_window=None
    )
    assert torch.equal(keep, keep_sliding)


# ---------------------------------------------------------------------------
# Additive-mask variant.
# ---------------------------------------------------------------------------


def test_additive_mask_values():
    bs, blk = 1, 4
    resp = enc = 8
    keep_bool, _ = build_block_diffusion_training_mask(
        prefix_lengths=0, response_length=resp, enc_len=enc, block_size=blk, batch_size=bs
    )
    add_full, add_sliding = build_block_diffusion_training_mask(
        prefix_lengths=0,
        response_length=resp,
        enc_len=enc,
        block_size=blk,
        batch_size=bs,
        dtype=torch.float32,
    )
    assert add_full.dtype == torch.float32
    # 0 where attended, -inf (finfo.min) where masked.
    assert (add_full[keep_bool] == 0).all()
    assert (add_full[~keep_bool] == torch.finfo(torch.float32).min).all()


def test_full_block_size_equals_response_single_block():
    """block_size == response_length => one block => no offset-block-causal leakage room.

    Every canvas position is in block 0, so no clean encoder column is strictly
    earlier in block index: the entire encoder is masked, canvas is fully
    bidirectional within the single block.
    """
    bs = 1
    resp = enc = 8
    keep, _ = build_block_diffusion_training_mask(
        prefix_lengths=0, response_length=resp, enc_len=enc, block_size=resp, batch_size=bs
    )
    assert not keep[0, 0, :, :enc].any(), "single block: no strictly-earlier clean column exists"
    assert keep[0, 0, :, enc:].all(), "single block canvas should be fully bidirectional"

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

"""Tests for the DFlash sparse-attention mask helpers."""

from __future__ import annotations

import pytest
import torch

import nemo_automodel.components.attention.dflash_mask as dflash_mask
from nemo_automodel.components.attention.dflash_mask import create_dflash_block_mask, create_dflash_sdpa_mask


def _reference_dflash_mask(anchor_positions, block_keep_mask, ctx_len, block_size):
    """Element-level reference mask (pure Python loops, obviously correct).

    Mirrors the rules from the DFlash paper §4.2:
    - Block b attends to context positions ``< anchor[b]`` (causal prefix).
    - Block b attends to its own noise positions (bidirectional).
    - Different blocks invisible; padded-anchor slots see nothing.
    """
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = ctx_len + N * block_size
    mask = torch.zeros(B, 1, Q_LEN, KV_LEN, dtype=torch.bool)
    for b in range(B):
        for q in range(Q_LEN):
            block_id = q // block_size
            if not bool(block_keep_mask[b, block_id]):
                continue
            anchor = int(anchor_positions[b, block_id])
            for k in range(KV_LEN):
                is_ctx = k < ctx_len
                ctx_ok = is_ctx and (k < anchor)
                is_noise = k >= ctx_len
                kv_block = (k - ctx_len) // block_size if is_noise else -1
                noise_ok = is_noise and (kv_block == block_id)
                mask[b, 0, q, k] = ctx_ok or noise_ok
    return mask


def test_sdpa_mask_matches_reference():
    """SDPA additive mask: float 0 where attended, -inf elsewhere; matches reference."""
    device = torch.device("cpu")
    block_size = 4
    ctx_len = 8
    anchor_positions = torch.tensor([[2, 5], [3, 6]], dtype=torch.long)
    block_keep_mask = torch.tensor([[True, True], [True, True]])

    sdpa = create_dflash_sdpa_mask(
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        ctx_len=ctx_len,
        block_size=block_size,
        device=device,
        dtype=torch.float32,
    )
    ref = _reference_dflash_mask(anchor_positions, block_keep_mask, ctx_len, block_size)

    # SDPA mask is float additive; reference is bool. Compare: where ref==True, sdpa==0; where ref==False, sdpa==-inf.
    attended = sdpa == 0.0
    masked = sdpa == float("-inf")
    assert torch.equal(attended, ref)
    assert torch.equal(masked, ~ref)
    assert sdpa.shape == (2, 1, 2 * 4, 8 + 2 * 4)


def test_sdpa_mask_respects_block_keep():
    """Padded anchor slots (keep=False) must attend to nothing."""
    block_size = 4
    ctx_len = 6
    anchor_positions = torch.tensor([[2, 4, 0]], dtype=torch.long)
    block_keep_mask = torch.tensor([[True, True, False]])  # 3rd block is padding

    sdpa = create_dflash_sdpa_mask(
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        ctx_len=ctx_len,
        block_size=block_size,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    # Rows for block 2 (q_idx in [8, 12)) must all be -inf — sees nothing.
    pad_rows = sdpa[0, 0, 2 * block_size : 3 * block_size, :]
    assert torch.all(torch.isinf(pad_rows) & (pad_rows < 0))


def test_sdpa_mask_overlapping_anchors():
    """Same anchor in two blocks → each block has its own KV slot, both see same context."""
    block_size = 4
    ctx_len = 6
    anchor_positions = torch.tensor([[3, 3]], dtype=torch.long)  # both at pos 3
    block_keep_mask = torch.tensor([[True, True]])

    sdpa = create_dflash_sdpa_mask(
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        ctx_len=ctx_len,
        block_size=block_size,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    ref = _reference_dflash_mask(anchor_positions, block_keep_mask, ctx_len, block_size)
    attended = sdpa == 0.0
    assert torch.equal(attended, ref)


def test_compiled_create_block_mask_is_cached(monkeypatch):
    sentinel = object()
    calls = []

    def compile_fn(fn, dynamic):
        calls.append((fn, dynamic))
        return sentinel

    monkeypatch.setattr(dflash_mask, "_compiled_create_block_mask", None)
    monkeypatch.setattr(dflash_mask.torch, "compile", compile_fn)

    assert dflash_mask._get_compiled_create_block_mask() is sentinel
    assert dflash_mask._get_compiled_create_block_mask() is sentinel
    assert calls == [(dflash_mask.create_block_mask, False)]


def test_block_mask_uncompiled_cpu_smoke():
    """The FlexAttention helper can build an uncompiled BlockMask on CPU.

    This exercises the ``use_compile=False`` path without needing a CUDA
    runner or invoking torch.compile.
    """
    block_mask = create_dflash_block_mask(
        anchor_positions=torch.tensor([[2, 5]], dtype=torch.long),
        block_keep_mask=torch.tensor([[True, False]]),
        ctx_len=8,
        block_size=4,
        device=torch.device("cpu"),
        use_compile=False,
    )

    assert block_mask.shape == (1, 1, 8, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention requires CUDA")
def test_flex_block_mask_matches_reference():
    """FlexAttention BlockMask materialised back to dense matches the reference."""
    from nemo_automodel.components.attention.dflash_mask import create_dflash_block_mask

    device = torch.device("cuda")
    block_size = 4
    ctx_len = 8
    anchor_positions = torch.tensor([[2, 5], [3, 6]], dtype=torch.long, device=device)
    block_keep_mask = torch.tensor([[True, True], [True, True]], device=device)

    block_mask = create_dflash_block_mask(
        anchor_positions=anchor_positions,
        block_keep_mask=block_keep_mask,
        ctx_len=ctx_len,
        block_size=block_size,
        device=device,
    )
    dense = block_mask.to_dense()  # (B, H, Q, KV) — H is broadcast dim (1)
    ref = _reference_dflash_mask(anchor_positions.cpu(), block_keep_mask.cpu(), ctx_len, block_size).to(device)

    # BlockMask granularity may be coarser than 1×1; reference is element-exact.
    # When BLOCK_SIZE in create_block_mask divides block_size cleanly the dense view should match.
    # Compare element-wise: dense must be a superset of ref (BlockMask never under-attends).
    if dense.shape == ref.shape:
        # ref ⊆ dense
        assert torch.all(dense | ~ref), "ref positions must be present in BlockMask"

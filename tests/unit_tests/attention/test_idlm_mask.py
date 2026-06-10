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

"""Tests for the I-DLM block-diffusion attention mask helpers."""

from __future__ import annotations

import pytest
import torch

from nemo_automodel.components.attention.idlm_mask import create_idlm_block_mask, create_idlm_sdpa_mask


def _reference_idlm_mask(seq_len, block_size, valid_mask, use_regular_causal=True):
    """Element-level reference for block_diff_mask over the [x_t | x_0] concat.

    Mirrors the official ``block_diff_mask`` (paper Appendix E):
    - ``x_t`` token attends itself's block causally (M_BD) and clean ``x_0``
      tokens in strictly earlier blocks (M_OBC).
    - ``x_0`` token is strict token-causal (M_BC).
    Padded keys (in either copy) are never attended.
    """
    B = valid_mask.shape[0]
    two_l = 2 * seq_len
    mask = torch.zeros(B, 1, two_l, two_l, dtype=torch.bool)
    for b in range(B):
        for q in range(two_l):
            x0_q = q >= seq_len
            bq = (q - seq_len) // block_size if x0_q else q // block_size
            for k in range(two_l):
                x0_kv = k >= seq_len
                bkv = (k - seq_len) // block_size if x0_kv else k // block_size
                m_bd = (bq == bkv) and (not x0_q) and (not x0_kv) and (q >= k)
                m_obc = (bq > bkv) and x0_kv and (not x0_q)
                if use_regular_causal:
                    m_bc = (q >= k) and x0_q and x0_kv
                else:
                    m_bc = (bq >= bkv) and x0_q and x0_kv
                key_ok = bool(valid_mask[b, k % seq_len])
                mask[b, 0, q, k] = (m_bd or m_obc or m_bc) and key_ok
    return mask


def _check_matches(seq_len, block_size, valid_mask):
    sdpa = create_idlm_sdpa_mask(seq_len, block_size, valid_mask, device=torch.device("cpu"), dtype=torch.float32)
    ref = _reference_idlm_mask(seq_len, block_size, valid_mask)
    assert sdpa.shape == (valid_mask.shape[0], 1, 2 * seq_len, 2 * seq_len)
    assert torch.equal(sdpa == 0.0, ref)
    assert torch.equal(torch.isinf(sdpa) & (sdpa < 0), ~ref)


def test_sdpa_mask_block_size_1_matches_reference():
    _check_matches(4, 1, torch.ones(2, 4, dtype=torch.long))


def test_sdpa_mask_block_size_2_matches_reference():
    _check_matches(6, 2, torch.ones(2, 6, dtype=torch.long))


def test_sdpa_mask_blocks_padded_keys():
    valid_mask = torch.ones(1, 4, dtype=torch.long)
    valid_mask[0, 3] = 0  # pad the last token
    sdpa = create_idlm_sdpa_mask(4, 1, valid_mask, device=torch.device("cpu"), dtype=torch.float32)
    # Padded token 3 is a key in both copies (columns 3 and 7) — never attended.
    assert torch.all(torch.isinf(sdpa[0, 0, :, 3]) & (sdpa[0, 0, :, 3] < 0))
    assert torch.all(torch.isinf(sdpa[0, 0, :, 7]) & (sdpa[0, 0, :, 7] < 0))


def test_block_mask_uncompiled_cpu_smoke():
    """The FlexAttention helper builds an uncompiled BlockMask on CPU."""
    block_mask = create_idlm_block_mask(
        4, 1, torch.ones(1, 4, dtype=torch.long), device=torch.device("cpu"), use_compile=False
    )
    assert block_mask.shape == (1, 1, 8, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention requires CUDA")
@pytest.mark.parametrize("block_size", [1, 2])
def test_flex_block_mask_matches_sdpa_outputs(block_size):
    """flex_attention with the BlockMask and SDPA with the dense mask agree on the same q/k/v."""
    from torch.nn.attention.flex_attention import flex_attention

    device = torch.device("cuda")
    torch.manual_seed(0)
    B, H, L, D = 2, 2, 8, 16
    q, k, v = (torch.randn(B, H, 2 * L, D, device=device, dtype=torch.float32) for _ in range(3))
    valid_mask = torch.ones(B, L, dtype=torch.long, device=device)

    dense = create_idlm_sdpa_mask(L, block_size, valid_mask, device=device, dtype=torch.float32)
    out_sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=dense)

    block_mask = create_idlm_block_mask(L, block_size, valid_mask, device=device, use_compile=False)
    out_flex = flex_attention(q, k, v, block_mask=block_mask)

    assert torch.allclose(out_sdpa, out_flex, atol=1e-5), f"max diff {(out_sdpa - out_flex).abs().max().item():.3e}"

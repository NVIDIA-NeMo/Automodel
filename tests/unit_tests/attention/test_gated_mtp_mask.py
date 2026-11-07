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

"""Tests for the gated-LoRA MTP attention mask helpers."""

from __future__ import annotations

import pytest
import torch

from nemo_automodel.components.attention.gated_mtp_mask import create_gated_mtp_block_mask, create_gated_mtp_sdpa_mask


def _reference_gated_mtp_mask(num_depths, valid_mask):
    """Element-level reference built from an explicit list of the layout's tokens.

    Returns:
        Bool Tensor of shape [batch, 1, sequence * (1 + depths), sequence * (1 + depths)].
    """
    batch, seq_len = valid_mask.shape
    # (anchor, depth) of every token in the depth-major [clean | depth 1 | ... | depth D] order; depth 0 = clean.
    tokens = [(t, d) for d in range(num_depths + 1) for t in range(seq_len)]
    total = len(tokens)
    mask = torch.zeros(batch, 1, total, total, dtype=torch.bool)
    for b in range(batch):
        for q, (anchor_q, depth_q) in enumerate(tokens):
            for k, (anchor_k, depth_k) in enumerate(tokens):
                if depth_k == 0:
                    # clean keys: at or before the query's clean position (its anchor), and not padding
                    allowed = anchor_k <= anchor_q and bool(valid_mask[b, anchor_k])
                else:
                    # mask keys: only from mask queries of the same anchor at the same or a deeper depth
                    allowed = depth_q > 0 and anchor_k == anchor_q and depth_k <= depth_q
                mask[b, 0, q, k] = allowed
    return mask


@pytest.mark.parametrize("num_depths", [1, 3])
def test_sdpa_mask_matches_reference(num_depths):
    valid_mask = torch.ones(2, 5, dtype=torch.long)
    valid_mask[1, 4] = 0
    sdpa = create_gated_mtp_sdpa_mask(num_depths, valid_mask, dtype=torch.float32)
    reference = _reference_gated_mtp_mask(num_depths, valid_mask)
    assert sdpa.shape == reference.shape
    assert torch.equal(sdpa == 0.0, reference)
    assert torch.equal(torch.isneginf(sdpa), ~reference)


def test_sdpa_mask_has_no_fully_masked_rows():
    # Padded anchors still attend their own mask token, so SDPA never softmaxes an all -inf row.
    valid_mask = torch.tensor([[1, 1, 0, 0]])
    sdpa = create_gated_mtp_sdpa_mask(2, valid_mask, dtype=torch.float32)
    assert (sdpa == 0.0).any(dim=-1)[0, 0, valid_mask.shape[1] :].all()


def test_block_mask_matches_sdpa_mask_on_cpu():
    valid_mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
    block_mask = create_gated_mtp_block_mask(2, valid_mask, use_compile=False)
    sdpa = create_gated_mtp_sdpa_mask(2, valid_mask, dtype=torch.float32)
    total = valid_mask.shape[1] * 3
    assert block_mask.shape == (2, 1, total, total)
    # Evaluate the mask_mod on every (query, key) pair; the BlockMask is built from the same mask_mod.
    q_idx = torch.arange(total).view(-1, 1)
    kv_idx = torch.arange(total).view(1, -1)
    for b in range(2):
        dense = block_mask.mask_mod(torch.tensor(b), torch.tensor(0), q_idx, kv_idx)
        assert torch.equal(dense, sdpa[b, 0] == 0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention requires CUDA")
def test_flex_block_mask_matches_sdpa_outputs():
    """flex_attention with the BlockMask and SDPA with the dense mask agree on the same q/k/v."""
    from torch.nn.attention.flex_attention import flex_attention

    device = torch.device("cuda")
    torch.manual_seed(0)
    num_depths, seq_len = 3, 32
    batch, heads, head_dim = 2, 2, 16
    total = seq_len * (1 + num_depths)
    q, k, v = (torch.randn(batch, heads, total, head_dim, device=device) for _ in range(3))
    valid_mask = torch.ones(batch, seq_len, dtype=torch.long, device=device)
    valid_mask[1, -5:] = 0

    dense = create_gated_mtp_sdpa_mask(num_depths, valid_mask, dtype=torch.float32)
    out_sdpa = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=dense)
    block_mask = create_gated_mtp_block_mask(num_depths, valid_mask, use_compile=False)
    out_flex = flex_attention(q, k, v, block_mask=block_mask)

    # Different kernels agree only up to kernel precision; a mask bug shows O(0.1-1) differences.
    assert torch.allclose(out_sdpa, out_flex, atol=2e-3, rtol=1e-2), (out_sdpa - out_flex).abs().max()

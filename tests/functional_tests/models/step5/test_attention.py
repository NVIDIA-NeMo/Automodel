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

"""CUDA forward/backward coverage for Step-5's fused sparse attention."""

import pytest
import torch
from torch.nn import functional as F

from nemo_automodel.components.models.step5.attention import fused_attention
from nemo_automodel.components.models.step5.fa4 import fa4_attention
from nemo_automodel.components.models.step5.sparse import RegionSelection


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("backend", ["flex", "fa4"])
@pytest.mark.parametrize("batch,query_start", [(1, 0), (2, 517)])
def test_fused_attention_matches_independent_mask_and_gradients(
    sparse: bool, backend: str, batch: int, query_start: int
) -> None:
    torch.manual_seed(51)
    length, dim = 1153, 192
    q = torch.randn(batch, length - query_start, 4, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch, length, 2, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    positions = torch.cat((torch.arange(137), torch.arange(length - 137))).cuda()[None]
    positions = positions.expand(batch, -1).contiguous()
    docs = (positions == 0).cumsum(-1)
    padding = torch.zeros_like(positions, dtype=torch.bool)
    padding[:, -3:] = True
    selection = None
    distance = positions[:, :, None] - positions[:, None, :]
    mask = (distance >= 0) & (docs[:, :, None] == docs[:, None, :]) & ~padding[:, None, :]
    mask = mask[:, None].expand(-1, 4, -1, -1)
    if sparse:
        regions = (torch.arange(length, device="cuda")[None] // 8).expand(batch, -1).contiguous()
        # Independent selection: current region plus one group-specific old region.
        ids = torch.stack((torch.zeros_like(regions), regions), -1)[:, None].expand(-1, 2, -1, -1).clone().int()
        ids[:, 1, :, 0] = (regions - 2).clamp(min=0)
        selection = RegionSelection(
            ids[:, :, query_start:].contiguous(), regions, docs[:, query_start:], docs, int(regions.max()) + 1
        )
        chosen = (regions[:, None, None, :, None] == ids[:, :, :, None, :]).any(-1)
        mask = mask & chosen.repeat_interleave(2, dim=1)
    else:
        mask = mask & (distance[:, None] < 65)
    attention = fa4_attention if backend == "fa4" else fused_attention
    out = attention(
        q,
        k,
        v,
        positions[:, query_start:],
        positions,
        padding,
        docs,
        query_start=query_start,
        window=None if sparse else 65,
        selection=selection,
    )
    reference = F.scaled_dot_product_attention(
        q.float().transpose(1, 2),
        k.float().transpose(1, 2).repeat_interleave(2, dim=1),
        v.float().transpose(1, 2).repeat_interleave(2, dim=1),
        attn_mask=mask[:, :, query_start:],
    ).transpose(1, 2)
    torch.testing.assert_close(out.float(), reference, atol=0.015, rtol=0.015)
    upstream = torch.randn_like(out)
    actual_grad = torch.autograd.grad(out, (q, k, v), upstream)
    expected_grad = torch.autograd.grad(reference, (q, k, v), upstream.float())
    for actual, expected in zip(actual_grad, expected_grad):
        relative_rms = (
            actual.float() - expected.float()
        ).square().mean().sqrt() / expected.float().square().mean().sqrt()
        assert relative_rms < 0.012, relative_rms.item()
        assert torch.isfinite(actual).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_indexer_scores_and_region_bitmap() -> None:
    """Check actual kernels against independent FP32 math and set membership."""
    from nemo_automodel.components.models.step5.kernels import indexer_score_tile, pack_region_bitmap

    torch.manual_seed(515)
    query = torch.randn(2, 173, 4, 4, 256, device="cuda").to(torch.float8_e4m3fn).to(torch.bfloat16)
    summary = torch.randn(2, 333, 256, device="cuda").to(torch.float8_e4m3fn).to(torch.bfloat16)
    weights = torch.randn(2, 173, 4, 4, device="cuda")
    actual = indexer_score_tile(query, weights, summary, 49, 117)
    dots = torch.einsum("bqghd,brd->bgqhr", query[:, 49:117].float(), summary.float())
    expected = (dots.relu() * weights[:, 49:117].permute(0, 2, 1, 3).unsqueeze(-1)).sum(-2)
    relative = (actual - expected).square().sum().sqrt() / expected.square().sum().sqrt()
    assert relative < 1e-5

    ids = torch.tensor([0, 31, 32, 32, 96, 97], device="cuda", dtype=torch.int32)
    ids = ids[None, None, None].expand(2, 3, 5, -1)
    bitmap = pack_region_bitmap(ids, 97)
    regions = torch.arange(97, device="cuda")
    unpacked = ((bitmap[..., regions // 32] >> (regions % 32)) & 1).bool()
    oracle = (ids[..., None] == regions).any(-2)
    torch.testing.assert_close(unpacked, oracle, rtol=0, atol=0)

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

"""Mixed B2 CSR/FA4 with independent packed document and selected-key math."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from nemo_automodel.components.models.step5.indexed_attention import selected_attention
from nemo_automodel.components.models.step5.sparse import RegionSelection, region_metadata

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("long_row,partner", [(0, "prefix"), (1, "prefix"), (1, "allpad"), (0, "partialpad")])
def test_mixed_rows_preserve_independent_forward_and_gradients(long_row: int, partner: str) -> None:
    """Exercise real 32K dispatch, shifted CP queries, doc resets and padded partners."""
    torch.manual_seed(6713)
    keys, start, queries, groups, heads, dim = 40960, 38912, 64, 1, 16, 192
    cpu_positions = torch.arange(keys)[None].expand(2, -1).clone()
    cpu_padding = torch.zeros_like(cpu_positions, dtype=torch.bool)
    short = 1 - long_row
    cpu_positions[short] %= 1024
    if partner != "prefix":
        valid = 2048 if partner == "allpad" else start + 37
        cpu_padding[short, valid:] = True
        cpu_positions[short, valid:] = 0
    cpu_docs = ((cpu_positions == 0) & ~cpu_padding).cumsum(1)
    positions, padding = cpu_positions.cuda(), cpu_padding.cuda()
    metadata = region_metadata(
        positions,
        padding,
        query_start=start,
        queries=queries,
        region_size=8,
        topk=512,
        indexed_csa=True,
    )
    assert metadata.indexed_layout is not None
    assert metadata.indexed_layout.row_eligible == tuple(row == long_row for row in range(2))
    assert metadata.causal_row_layouts[short] is not None
    ids = metadata.dense_region_ids[:, None].clone()
    # The long row is a single real 40K document. Pick 512 complete historical
    # regions and its current region, independently of the production selector.
    current = torch.arange(start, start + queries, device="cuda") // 8
    ids[long_row, 0, :, :512] = torch.arange(512, device="cuda")
    ids[long_row, 0, :, 512] = current
    selection = RegionSelection(
        ids,
        metadata.token_regions,
        metadata.query_documents,
        metadata.documents,
        metadata.region_count,
        indexed_layout=metadata.indexed_layout,
        causal_row_layouts=metadata.causal_row_layouts,
    )
    query = torch.randn(2, queries, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    joint = torch.randn(keys, 2, 2, groups, dim, device="cuda", dtype=torch.bfloat16)
    key, value = joint.transpose(0, 1).unbind(2)
    key, value = key.detach().requires_grad_(), value.detach().requires_grad_()
    output = selected_attention(
        query,
        key,
        value,
        positions[:, start : start + queries],
        positions,
        padding,
        selection=selection,
        region_size=8,
        query_start=start,
    )
    allowed = (
        (cpu_positions[:, start : start + queries, None] >= cpu_positions[:, None])
        & (cpu_docs[:, start : start + queries, None] == cpu_docs[:, None])
        & ~cpu_padding[:, None]
    )
    physical = torch.arange(keys)
    chosen = (physical[None] < 4096) | (physical[None] // 8 == torch.arange(start, start + queries)[:, None] // 8)
    allowed[long_row] &= chosen
    rq, rk, rv = [t.detach().float().requires_grad_() for t in (query, key, value)]
    reference_rows = []
    for row in range(2):
        if row == short and partner == "allpad":
            singleton = int(torch.nonzero(allowed[row, 0]).item())
            assert torch.all(allowed[row].sum(-1) == 1)
            reference_rows.append(
                rv[row, singleton].repeat_interleave(heads, dim=0)[None].expand(queries, -1, -1)
                + (rq[row].sum() + rk[row].sum()) * 0
            )
        else:
            reference_rows.append(
                F.scaled_dot_product_attention(
                    rq[row].transpose(0, 1)[None],
                    rk[row].transpose(0, 1)[None].repeat_interleave(heads, dim=1),
                    rv[row].transpose(0, 1)[None].repeat_interleave(heads, dim=1),
                    attn_mask=allowed[row].cuda()[None, None],
                )[0].transpose(0, 1)
            )
    reference = torch.stack(reference_rows)
    torch.testing.assert_close(output.float(), reference, atol=0.015, rtol=0.015)
    upstream = torch.randn_like(output)
    actual = torch.autograd.grad(output, (query, key, value), upstream)
    expected = torch.autograd.grad(reference, (rq, rk, rv), upstream.float())
    for a, e in zip(actual, expected):
        assert torch.isfinite(a).all()
        assert (a.float() - e).norm() / e.norm() < 0.012
    if partner == "allpad":
        assert torch.count_nonzero(actual[0][short]) == 0
        assert torch.count_nonzero(actual[1][short]) == 0
        assert actual[2][short].abs().sum() > 0
    for grad in actual[1:]:
        assert torch.count_nonzero(grad[short][padding[short]]) == 0
    # A swapped-row region layout must not accidentally satisfy the oracle.
    wrong = replace(selection, region_ids=selection.region_ids.flip(0))
    bad = selected_attention(
        query,
        key,
        value,
        positions[:, start : start + queries],
        positions,
        padding,
        selection=wrong,
        region_size=8,
        query_start=start,
    )
    with pytest.raises(AssertionError):
        torch.testing.assert_close(bad.float(), reference, atol=0.015, rtol=0.015)

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

"""CUDA mixed sparse/prefix B2 dispatch below the indexed-attention threshold."""

from collections.abc import Callable
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from nemo_automodel.components.models.step5 import fa4
from nemo_automodel.components.models.step5.sparse import RegionSelection, region_metadata

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize(
    "long_row,partner,start",
    [
        (0, "prefix", 4090),
        (1, "prefix", 6120),
        (0, "prefix", 6120),
        (0, "partialpad", 6120),
        (1, "partialpad", 6120),
        (0, "allpad", 6120),
        (1, "allpad", 6120),
    ],
)
def test_below_threshold_mixed_rows_keep_views_and_independent_math(
    monkeypatch: pytest.MonkeyPatch, long_row: int, partner: str, start: int
) -> None:
    """Exercise real FA4 kernels, document resets, sparse positions, and random VJPs."""
    torch.manual_seed(9413)
    keys, queries, groups, heads, dim = 8192, 96, 4, 64, 192
    short = 1 - long_row
    cpu_positions = torch.arange(keys)[None].expand(2, -1).clone()
    cpu_padding = torch.zeros_like(cpu_positions, dtype=torch.bool)
    cpu_positions[short] %= 1024
    valid = {"prefix": keys, "partialpad": start + 61, "allpad": 2048}[partner]
    cpu_positions[short, valid:] = 0
    cpu_padding[short, valid:] = True
    cpu_docs = ((cpu_positions == 0) & ~cpu_padding).cumsum(1)
    positions, padding = cpu_positions.cuda(), cpu_padding.cuda()
    metadata = region_metadata(
        positions, padding, query_start=start, queries=queries, region_size=8, topk=512, indexed_csa=True
    )
    assert metadata.indexed_layout is None
    assert metadata.causal_row_layouts[long_row] is None
    assert metadata.causal_row_layouts[short] is not None
    ids = metadata.dense_region_ids[:, None].expand(-1, groups, -1, -1).clone()
    current = torch.arange(start, start + queries, device="cuda") // 8
    ids[long_row, :, :, :512] = torch.arange(512, device="cuda")
    ids[long_row, :, :, 512] = current
    original_ids = ids.clone()
    selection = RegionSelection(
        ids,
        metadata.token_regions,
        metadata.query_documents,
        metadata.documents,
        metadata.region_count,
        causal_row_layouts=metadata.causal_row_layouts,
    )
    query = torch.randn(2, queries, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    # Match the native CP gather -> transpose -> K/V unbind storage layout.
    joint = torch.randn(keys, 2, 2, groups, dim, device="cuda", dtype=torch.bfloat16)
    key, value = joint.transpose(0, 1).unbind(2)
    key, value = key.detach().requires_grad_(), value.detach().requires_grad_()
    assert not key.is_contiguous() and not value.is_contiguous()
    custom_calls = []
    load = fa4._load_fa4

    def observe_load(window: int | None, bitmap: bool, slots: int, providers: int) -> tuple[Callable, type, Callable]:
        attention, block_lists, mask = load(window, bitmap, slots, providers)

        def observe(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, ...]:
            """Observe real kernel inputs without changing their execution.

            Args:
                q: Provider-folded tensor [groups, queries, heads_per_group, dim].
                k: Strided tensor [groups, global_keys, 1, dim].
                v: Strided tensor with the same shape as k.
                kwargs: Unchanged FA4 keyword arguments, including metadata.

            Returns:
                Native output [groups, queries, heads_per_group, dim] and LSE.
            """
            assert q.untyped_storage().data_ptr() == query.untyped_storage().data_ptr()
            assert k.untyped_storage().data_ptr() == joint.untyped_storage().data_ptr()
            assert v.untyped_storage().data_ptr() == joint.untyped_storage().data_ptr()
            assert q.shape[0] == groups
            custom_calls.append(q.shape)
            return attention(q, k, v, **kwargs)

        return observe, block_lists, mask

    monkeypatch.setattr(fa4, "_load_fa4", observe_load)
    output = fa4.fa4_attention(
        query,
        key,
        value,
        positions[:, start : start + queries],
        positions,
        padding,
        metadata.documents,
        query_start=start,
        window=None,
        selection=selection,
    )
    assert len(custom_calls) == 1
    allowed = (
        (cpu_positions[:, start : start + queries, None] >= cpu_positions[:, None])
        & (cpu_docs[:, start : start + queries, None] == cpu_docs[:, None])
        & ~cpu_padding[:, None]
    )
    physical = torch.arange(keys)
    chosen = (physical[None] < 4096) | (physical[None] // 8 == torch.arange(start, start + queries)[:, None] // 8)
    # The tested sparse bucket really excludes legal causal keys.
    dropped = (allowed[long_row] & ~chosen).sum(-1)
    assert (dropped > 0).any()
    allowed[long_row] &= chosen
    rq, rk, rv = [t.detach().float().requires_grad_() for t in (query, key, value)]
    reference_rows = []
    for row in range(2):
        if row == short and partner == "allpad":
            singleton = int(torch.nonzero(allowed[row, 0]).item())
            assert torch.all(allowed[row].sum(-1) == 1)
            reference_rows.append(
                rv[row, singleton].repeat_interleave(heads // groups, dim=0)[None].expand(queries, -1, -1)
                + (rq[row].sum() + rk[row].sum()) * 0
            )
        else:
            with sdpa_kernel(SDPBackend.MATH):
                reference_rows.append(
                    F.scaled_dot_product_attention(
                        rq[row].transpose(0, 1)[None],
                        rk[row].transpose(0, 1)[None].repeat_interleave(heads // groups, dim=1),
                        rv[row].transpose(0, 1)[None].repeat_interleave(heads // groups, dim=1),
                        attn_mask=allowed[row].cuda()[None, None],
                    )[0].transpose(0, 1)
                )
    reference = torch.stack(reference_rows)
    for row in range(2):
        torch.testing.assert_close(output[row].float(), reference[row], atol=0.015, rtol=0.015)
    for bucket in (dropped == 0, dropped > 0):
        if bucket.any():
            torch.testing.assert_close(
                output[long_row, bucket].float(), reference[long_row, bucket], atol=0.015, rtol=0.015
            )
    upstream = torch.randn_like(output)
    actual = torch.autograd.grad(output, (query, key, value), upstream)
    expected = torch.autograd.grad(reference, (rq, rk, rv), upstream.float())
    for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
        for row in range(2):
            assert torch.isfinite(a[row]).all()
            if partner == "allpad" and row == short and index < 2:
                assert torch.count_nonzero(a[row]) == 0
                assert torch.count_nonzero(e[row]) == 0
            else:
                assert (a[row].float() - e[row]).norm() / e[row].norm() < 0.012
    if partner == "allpad":
        assert actual[2][short].abs().sum() > 0
    for grad in actual[1:]:
        assert torch.count_nonzero(grad[short][padding[short]]) == 0
    assert torch.equal(selection.region_ids, original_ids)
    # Keeping the old B2 fallback must fail the no-copy contract before the
    # kernel launch; this is an allocation-path negative control, not a math pass.
    with pytest.raises(AssertionError):
        fa4.fa4_attention(
            query,
            key,
            value,
            positions[:, start : start + queries],
            positions,
            padding,
            metadata.documents,
            query_start=start,
            window=None,
            selection=replace(selection, causal_row_layouts=()),
        )

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("region_count,expected_bitmap", [(16386, True), (16539, False), (33025, False)])
def test_fa4_membership_crossover_preserves_native_192_outputs_and_gradients(
    monkeypatch: pytest.MonkeyPatch, region_count: int, expected_bitmap: bool
) -> None:
    """Check exact-route attention across the packed-region bitmap crossover."""
    from nemo_automodel.components.models.step5 import fa4

    torch.manual_seed(419)
    keys, queries, heads, dim = region_count * 8, 4, 16, 192
    start = keys - queries
    q = torch.zeros(1, queries, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.zeros(1, keys, 1, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    positions = torch.arange(keys, device="cuda")[None]
    documents = torch.ones_like(positions)
    routes = (
        torch.cat(
            (
                torch.arange(512, device="cuda", dtype=torch.int32),
                torch.tensor([region_count - 1], device="cuda", dtype=torch.int32),
            )
        )[None, None, None]
        .expand(1, 1, queries, -1)
        .contiguous()
    )
    selection = RegionSelection(routes, positions // 8, documents[:, start:], documents, region_count)
    calls = []
    original = fa4._load_fa4

    def load(window, bitmap, slots, provider_groups):
        calls.append(bitmap)
        return original(window, bitmap, slots, provider_groups)

    monkeypatch.setattr(fa4, "_load_fa4", load)
    output = fa4.fa4_attention(
        q,
        k,
        v,
        positions[:, start:],
        positions,
        torch.zeros_like(positions, dtype=torch.bool),
        documents,
        query_start=start,
        window=None,
        selection=selection,
    )
    assert calls == [expected_bitmap]
    # Every QK score is zero. The independent answer is the mean of precisely
    # the selected historical tokens plus the causal prefix of the last region.
    key_ids = torch.cat((torch.arange(4096, device="cuda"), torch.arange(keys - 8, keys, device="cuda")))
    ref_q = q.detach().float().requires_grad_()
    ref_k = k.detach().float().requires_grad_()
    ref_v = v.detach().float().requires_grad_()
    selected_k = ref_k[0, key_ids, 0]
    selected_v = ref_v[0, key_ids, 0]
    scores = torch.einsum("qhd,kd->hqk", ref_q[0], selected_k) * dim**-0.5
    allowed = key_ids[None] <= positions[0, start:, None]
    probabilities = scores.masked_fill(~allowed[None], -torch.inf).softmax(-1)
    expected = torch.einsum("hqk,kd->qhd", probabilities, selected_v)[None]
    gradient = torch.randn_like(output)
    output.backward(gradient)
    expected.backward(gradient.float())
    torch.testing.assert_close(output.float(), expected, atol=3e-4, rtol=5e-3)
    for actual, reference in ((q.grad, ref_q.grad), (k.grad, ref_k.grad), (v.grad, ref_v.grad)):
        torch.testing.assert_close(actual.float(), reference, atol=5e-5, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("lengths,query_start", [((1031, 17, 3048), 0), ((2051, 4096, 2045), 4000)])
def test_fa4_fully_selected_packed_prefix_skips_membership(
    monkeypatch: pytest.MonkeyPatch, lengths: tuple[int, ...], query_start: int
) -> None:
    """Check document-causal forward/backward on irregular 4K/8K packed shards."""
    from nemo_automodel.components.models.step5 import fa4
    from nemo_automodel.components.models.step5.sparse import region_metadata

    torch.manual_seed(516)
    length, heads, groups, dim = sum(lengths), 4, 2, 192
    positions = torch.cat([torch.arange(n, device="cuda") for n in lengths])[None]
    documents = (positions == 0).cumsum(-1)
    padding = torch.zeros_like(positions, dtype=torch.bool)
    padding[:, -3:] = True
    local = length - query_start
    q = torch.randn(1, local, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, length, groups, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    meta = region_metadata(positions, padding, query_start=query_start, queries=local, region_size=8, topk=512)
    assert not meta.score_tiles
    selection = RegionSelection(
        meta.dense_region_ids[:, None].expand(-1, groups, -1, -1),
        meta.token_regions,
        meta.query_documents,
        meta.documents,
        meta.region_count,
        all_regions_selected=True,
    )

    def forbidden_bitmap(*args):
        raise AssertionError("Fully selected document prefixes must not build membership bitmaps")

    monkeypatch.setattr(fa4, "pack_region_bitmap", forbidden_bitmap)
    output = fa4.fa4_attention(
        q,
        k,
        v,
        positions[:, query_start:],
        positions,
        padding,
        documents,
        query_start=query_start,
        window=None,
        selection=selection,
    )
    allowed = (
        (positions[:, query_start:, None] >= positions[:, None, :])
        & (documents[:, query_start:, None] == documents[:, None, :])
        & ~padding[:, None, :]
    )
    expected = F.scaled_dot_product_attention(
        q.float().transpose(1, 2),
        k.float().transpose(1, 2).repeat_interleave(heads // groups, dim=1),
        v.float().transpose(1, 2).repeat_interleave(heads // groups, dim=1),
        attn_mask=allowed[:, None],
    ).transpose(1, 2)
    torch.testing.assert_close(output.float(), expected, atol=0.015, rtol=0.015)
    gradient = torch.randn_like(output)
    actual_grad = torch.autograd.grad(output, (q, k, v), gradient)
    expected_grad = torch.autograd.grad(expected, (q, k, v), gradient.float())
    for actual, reference in zip(actual_grad, expected_grad):
        relative_rms = (
            actual.float() - reference.float()
        ).square().sum().sqrt() / reference.float().square().sum().sqrt()
        assert relative_rms < 0.012


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "lengths,query_start,queries",
    [((1031, 17, 3048), 137, 3800), ((2051, 4096, 2045), 4000, 4096), ((8192,), 2000, 2000), ((1, 2, 1, 3), 1, 5)],
)
def test_fa4_native_varlen_prefix_matches_packed_document_mask(
    monkeypatch: pytest.MonkeyPatch, lengths: tuple[int, ...], query_start: int, queries: int
) -> None:
    """Exercise document-prefix K, truncated final K, and a shard inside one doc."""
    from nemo_automodel.components.models.step5 import fa4
    from nemo_automodel.components.models.step5.sparse import region_metadata

    torch.manual_seed(716)
    keys, heads, groups, dim = sum(lengths), 4, 2, 192
    positions = torch.cat([torch.arange(n, device="cuda") for n in lengths])[None]
    documents = (positions == 0).cumsum(-1)
    padding = torch.zeros_like(positions, dtype=torch.bool)
    if keys > 32:
        padding[:, -3:] = True
    meta = region_metadata(positions, padding, query_start=query_start, queries=queries, region_size=8, topk=512)
    assert meta.causal_layout is not None
    selection = RegionSelection(
        meta.dense_region_ids[:, None].expand(-1, groups, -1, -1),
        meta.token_regions,
        meta.query_documents,
        meta.documents,
        meta.region_count,
        all_regions_selected=True,
        causal_layout=meta.causal_layout,
    )
    q = torch.randn(1, queries, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    # CP transport stacks K/V before gathering, then unbinds them. Preserve
    # that non-contiguous sequence stride when exercising native varlen.
    kv = torch.randn(1, keys, 2, groups, dim, device="cuda", dtype=torch.bfloat16)
    k, v = (part.detach().requires_grad_() for part in kv.unbind(2))
    calls = []
    native = fa4._load_fa4_varlen()

    def invoke(*args, **kwargs):
        calls.append((args[0].shape, args[1].shape, kwargs["causal"]))
        return native(*args, **kwargs)

    def forbidden_blocks(*args, **kwargs):
        raise AssertionError("Native document-causal prefixes must not build sparse block lists")

    monkeypatch.setattr(fa4, "_load_fa4_varlen", lambda: invoke)
    monkeypatch.setattr(fa4, "_selected_block_lists", forbidden_blocks)
    output = fa4.fa4_attention(
        q,
        k,
        v,
        positions[:, query_start : query_start + queries],
        positions,
        padding,
        documents,
        query_start=query_start,
        window=None,
        selection=selection,
    )
    assert len(calls) == 1 and calls[0][2]
    allowed = (
        (positions[:, query_start : query_start + queries, None] >= positions[:, None, :])
        & (documents[:, query_start : query_start + queries, None] == documents[:, None, :])
        & ~padding[:, None, :]
    )
    expected = F.scaled_dot_product_attention(
        q.float().transpose(1, 2),
        k.float().transpose(1, 2).repeat_interleave(heads // groups, dim=1),
        v.float().transpose(1, 2).repeat_interleave(heads // groups, dim=1),
        attn_mask=allowed[:, None],
    ).transpose(1, 2)
    torch.testing.assert_close(output.float(), expected, atol=0.015, rtol=0.015)
    gradient = torch.randn_like(output)
    actual_grad = torch.autograd.grad(output, (q, k, v), gradient)
    expected_grad = torch.autograd.grad(expected, (q, k, v), gradient.float())
    for actual, reference in zip(actual_grad, expected_grad):
        relative_rms = (
            actual.float() - reference.float()
        ).square().sum().sqrt() / reference.float().square().sum().sqrt()
        assert relative_rms < 0.012

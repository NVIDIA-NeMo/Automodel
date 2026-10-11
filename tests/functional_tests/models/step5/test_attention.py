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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("lengths", [(1031, 3065), (2051, 4096, 2045), (8192,)])
def test_fa4_native_varlen_contiguous_cp_outputs_and_qkv_gradients(lengths: tuple[int, ...]) -> None:
    """Compare identical routes on full Q versus two contiguous Q shards.

    Static routes choose the first 512 historical regions and the current
    region. The single-document 8K case therefore genuinely drops keys; its
    full-Q call uses sparse FA4 while only the first CP shard uses varlen.
    """
    from nemo_automodel.components.models.step5.sparse import region_metadata

    torch.manual_seed(718)
    length, heads, groups, dim = sum(lengths), 4, 2, 192
    positions = torch.cat([torch.arange(n, device="cuda") for n in lengths])[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    full_meta = region_metadata(positions, padding, query_start=0, queries=length, region_size=8, topk=512)
    current = full_meta.token_regions
    first = current - positions // 8
    historical = first.unsqueeze(-1) + torch.arange(512, device="cuda")
    historical = torch.where(historical < current.unsqueeze(-1), historical, full_meta.region_count)
    ids = torch.cat((historical, current.unsqueeze(-1)), dim=-1).sort(dim=-1).values.int()
    ids = ids[:, None].expand(-1, groups, -1, -1).contiguous()
    q = torch.randn(1, length, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(1, length, 2, groups, dim, device="cuda", dtype=torch.bfloat16)
    k, v = (part.detach().requires_grad_() for part in kv.unbind(2))
    dispatch: list[bool] = []

    def call(start: int, stop: int) -> torch.Tensor:
        meta = region_metadata(positions, padding, query_start=start, queries=stop - start, region_size=8, topk=512)
        dispatch.append(meta.causal_layout is not None)
        selection = RegionSelection(
            ids[:, :, start:stop].contiguous(),
            meta.token_regions,
            meta.query_documents,
            meta.documents,
            meta.region_count,
            all_regions_selected=not meta.score_tiles,
            causal_layout=meta.causal_layout,
        )
        return fa4_attention(
            q[:, start:stop],
            k,
            v,
            positions[:, start:stop],
            positions,
            padding,
            meta.documents,
            query_start=start,
            window=None,
            selection=selection,
        )

    cp1 = call(0, length)
    cp2 = torch.cat((call(0, length // 2), call(length // 2, length)), dim=1)
    assert dispatch == ([False, True, False] if lengths == (8192,) else [True, True, True])
    torch.testing.assert_close(cp1, cp2, atol=0.015, rtol=0.015)
    upstream = torch.randn_like(cp1)
    grad1 = torch.autograd.grad(cp1, (q, k, v), upstream)
    grad2 = torch.autograd.grad(cp2, (q, k, v), upstream)
    metrics = {}
    for name, expected, actual in zip(("output", "dq", "dk", "dv"), (cp1, *grad1), (cp2, *grad2)):
        relative_rms = ((actual.float() - expected.float()).square().sum() / expected.float().square().sum()).sqrt()
        metrics[name] = {
            "relative_rms": float(relative_rms),
            "max_abs": float((actual.float() - expected.float()).abs().max()),
        }
        assert relative_rms < 0.012
    print({"lengths": lengths, "native_varlen_dispatch": dispatch, "cp1_cp2": metrics})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "lengths,padded,query_start", [((1031, 3058), 7, 0), ((2051, 4096, 2038), 7, 4096), ((3000,), 1096, 0)]
)
def test_fa4_native_varlen_trailing_padding_preserves_outputs_and_value_gradients(
    monkeypatch: pytest.MonkeyPatch, lengths: tuple[int, ...], padded: int, query_start: int
) -> None:
    """Padding queries still read the final document's first V, including GQA."""
    from dataclasses import replace

    from nemo_automodel.components.models.step5 import fa4
    from nemo_automodel.components.models.step5.sparse import region_metadata

    torch.manual_seed(720)
    keys, heads, groups, dim = sum(lengths) + padded, 4, 2, 192
    positions = torch.cat(
        [torch.arange(n, device="cuda") for n in lengths] + [torch.zeros(padded, device="cuda", dtype=torch.long)]
    )[None]
    padding = torch.arange(keys, device="cuda")[None] >= sum(lengths)
    documents = ((positions == 0) & ~padding).cumsum(-1)
    queries = keys - query_start
    valid_queries = queries - padded
    meta = region_metadata(positions, padding, query_start=query_start, queries=queries, region_size=8, topk=512)
    assert meta.causal_layout is not None
    assert meta.causal_layout.valid_queries == valid_queries
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
    kv = torch.randn(1, keys, 2, groups, dim, device="cuda", dtype=torch.bfloat16)
    k, v = (part.detach().requires_grad_() for part in kv.unbind(2))
    native = fa4._load_fa4_varlen()
    native_queries: list[int] = []

    def invoke(*args, **kwargs):
        native_queries.append(args[0].shape[0])
        return native(*args, **kwargs)

    monkeypatch.setattr(fa4, "_load_fa4_varlen", lambda: invoke)

    def call(routes: RegionSelection) -> torch.Tensor:
        return fa4.fa4_attention(
            q,
            k,
            v,
            positions[:, query_start:],
            positions,
            padding,
            documents,
            query_start=query_start,
            window=None,
            selection=routes,
        )

    output = call(selection)
    original = call(replace(selection, causal_layout=None))
    assert native_queries == [valid_queries]
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
    for actual in (output, original):
        torch.testing.assert_close(actual.float(), expected, atol=0.015, rtol=0.015)
    torch.testing.assert_close(output[:, valid_queries:], original[:, valid_queries:], atol=0, rtol=0)
    upstream = torch.randn_like(output)
    actual_grad = torch.autograd.grad(output, (q, k, v), upstream, retain_graph=True)
    old_grad = torch.autograd.grad(original, (q, k, v), upstream)
    reference_grad = torch.autograd.grad(expected, (q, k, v), upstream.float())
    metrics: dict[str, list[float]] = {}
    for name, actuals in (("trim", actual_grad), ("original", old_grad)):
        metrics[name] = []
        for actual, reference in zip(actuals, reference_grad):
            relative = ((actual.float() - reference.float()).square().sum() / reference.float().square().sum()).sqrt()
            assert relative < 0.012
            metrics[name].append(float(relative))

    # A label mask is not an attention mask. Exercise nonzero gradients from
    # padded queries explicitly, rather than relying on SFT's ignored labels.
    pad_upstream = upstream.clone()
    pad_upstream[:, :valid_queries] = 0
    dq, dk, dv = torch.autograd.grad(output, (q, k, v), pad_upstream)
    assert torch.count_nonzero(dq) == torch.count_nonzero(dk) == 0
    expected_dv = torch.zeros_like(v)
    expected_dv[:, sum(lengths[:-1])] = (
        upstream[:, valid_queries:].reshape(1, padded, groups, heads // groups, dim).float().sum((1, 3))
    ).to(v.dtype)
    torch.testing.assert_close(dv, expected_dv, rtol=0, atol=0)
    assert torch.count_nonzero(dv) > 0
    print({"lengths": lengths, "padding": padded, "query_start": query_start, "gradient_relative_rms": metrics})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fa4_zero_valid_query_shard_keeps_original_padding_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    """An entirely padded CP rank conservatively stays on the existing backend."""
    from nemo_automodel.components.models.step5 import fa4
    from nemo_automodel.components.models.step5.sparse import region_metadata

    torch.manual_seed(721)
    keys, start, valid, heads, groups, dim = 8192, 4096, 3000, 4, 2, 192
    positions = torch.zeros(1, keys, device="cuda", dtype=torch.long)
    positions[:, :valid] = torch.arange(valid, device="cuda")
    padding = torch.arange(keys, device="cuda")[None] >= valid
    meta = region_metadata(positions, padding, query_start=start, queries=keys - start, region_size=8, topk=512)
    assert meta.causal_layout is None and not meta.score_tiles
    selection = RegionSelection(
        meta.dense_region_ids[:, None].expand(-1, groups, -1, -1),
        meta.token_regions,
        meta.query_documents,
        meta.documents,
        meta.region_count,
        all_regions_selected=True,
    )

    def forbidden_varlen():
        raise AssertionError("An empty valid query prefix must retain the original kernel")

    monkeypatch.setattr(fa4, "_load_fa4_varlen", forbidden_varlen)
    q = torch.randn(1, keys - start, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, keys, groups, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    output = fa4.fa4_attention(
        q,
        k,
        v,
        positions[:, start:],
        positions,
        padding,
        meta.documents,
        query_start=start,
        window=None,
        selection=selection,
    )
    expected = v[:, :1].repeat_interleave(heads // groups, dim=2).expand_as(output)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    gradient = torch.ones_like(output)
    dq, dk, dv = torch.autograd.grad(output, (q, k, v), gradient)
    assert torch.count_nonzero(dq) == torch.count_nonzero(dk) == 0
    expected_dv = torch.zeros_like(v)
    expected_dv[:, 0] = (keys - start) * (heads // groups)
    torch.testing.assert_close(dv, expected_dv, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("spans", [(256, 128, 96, 64, 1), (1, 64, 96, 128, 256)])
@pytest.mark.parametrize("queries,heads,groups", [(4096, 4, 2), (4096, 64, 4), (8192, 64, 4)])
def test_varlen_shard_bounds_reuse_kernels_for_short_documents(
    spans: tuple[int, ...], queries: int, heads: int, groups: int
) -> None:
    """Document lengths cannot introduce late single-block kernel variants."""
    from flash_attn.cute import interface

    from nemo_automodel.components.models.step5.sparse import region_metadata

    torch.manual_seed(812)
    dim, documents_count = 192, 16
    caches = (interface._flash_attn_fwd.compile_cache, interface._flash_attn_bwd.compile_cache)
    for case, span in enumerate(spans):
        valid = documents_count * span
        positions = torch.cat(
            (
                torch.arange(span, device="cuda").repeat(documents_count),
                torch.zeros(queries - valid, device="cuda", dtype=torch.long),
            )
        )[None]
        padding = torch.arange(queries, device="cuda")[None] >= valid
        meta = region_metadata(positions, padding, query_start=0, queries=queries, region_size=8, topk=512)
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
        # The production stack/unbind layout has a noncontiguous token stride.
        kv = torch.randn(1, queries, 2, groups, dim, device="cuda", dtype=torch.bfloat16)
        k, v = (item.detach().requires_grad_() for item in kv.unbind(2))
        before = [set(cache.cache) for cache in caches]
        actual = fa4_attention(
            q,
            k,
            v,
            positions,
            positions,
            padding,
            meta.documents,
            query_start=0,
            window=None,
            selection=selection,
        )
        upstream = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, (q, k, v), upstream)
        torch.cuda.synchronize()
        if case:
            assert [set(cache.cache) for cache in caches] == before, "Short documents compiled another kernel variant"
        # Independent per-document FP32 math, including the existing padding
        # convention: each padded query attends the final document's first V.
        reference_q, reference_k, reference_v = (tensor.detach().float().requires_grad_() for tensor in (q, k, v))
        document_outputs = []
        for start in range(0, valid, span):
            stop = start + span
            document_outputs.append(
                F.scaled_dot_product_attention(
                    reference_q[:, start:stop].transpose(1, 2),
                    reference_k[:, start:stop].repeat_interleave(heads // groups, dim=2).transpose(1, 2),
                    reference_v[:, start:stop].repeat_interleave(heads // groups, dim=2).transpose(1, 2),
                    is_causal=True,
                ).transpose(1, 2)
            )
        if valid < queries:
            document_outputs.append(
                reference_v[:, valid - span : valid - span + 1]
                .repeat_interleave(heads // groups, dim=2)
                .expand(-1, queries - valid, -1, -1)
            )
        expected = torch.cat(document_outputs, dim=1)
        torch.testing.assert_close(actual.float(), expected, atol=0.015, rtol=0.015)
        expected_grads = torch.autograd.grad(expected, (reference_q, reference_k, reference_v), upstream.float())
        for index, (observed, oracle) in enumerate(zip(actual_grads, expected_grads)):
            if span == 1 and index < 2:
                # A singleton softmax is constant: both Q/K derivatives are
                # mathematically zero. FP32 SDPA also has tiny cancellation
                # residuals, so a relative-to-zero comparison is undefined.
                torch.testing.assert_close(observed.float(), torch.zeros_like(oracle), atol=1e-5, rtol=0)
                torch.testing.assert_close(oracle, torch.zeros_like(oracle), atol=1e-5, rtol=0)
            else:
                relative = (observed.float() - oracle).norm() / oracle.norm()
                assert relative < 0.012, (case, span, index, relative.item())
            assert torch.isfinite(observed).all()
        assert torch.count_nonzero(actual_grads[0][:, valid:]) == 0
        assert torch.count_nonzero(actual_grads[1][:, valid:]) == 0

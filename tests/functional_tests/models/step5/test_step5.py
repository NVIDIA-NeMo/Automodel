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

import copy
import json
import os
from collections.abc import Callable, Iterator
from dataclasses import replace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch._dynamo.utils import counters
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile
from torch.utils.checkpoint import checkpoint

from nemo_automodel._transformers.infrastructure import _apply_trainability_policy
from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
from nemo_automodel.components.models.step3p5.layers import Step3p5RMSNorm
from nemo_automodel.components.models.step5 import fa4
from nemo_automodel.components.models.step5.configuration import Step5Config
from nemo_automodel.components.models.step5.fa4 import _region_tiles, fa4_attention
from nemo_automodel.components.models.step5.layers import Step5RMSNorm, fused_attention
from nemo_automodel.components.models.step5.rope import Step5RotaryEmbedding
from nemo_automodel.components.models.step5.sparse import RegionSelection, Step5SparseIndexer, region_metadata
from nemo_automodel.components.optim.optimizer import _trainable_params_or_groups
from tests.unit_tests.models.step5.test_attention import _indexer, _oracle
from tests.unit_tests.models.step5.test_model import tiny_config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_requires_torchrun = pytest.mark.skipif(
    not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) not in (2, 4),
    reason="Requires a two-rank CUDA torchrun launch; four-rank HSDP is an additional local gate",
)


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
    from nemo_automodel.components.models.step5.sparse import indexer_score_tile, pack_region_bitmap

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

    def load(window, bitmap, slots, provider_groups, **kwargs):
        calls.append(bitmap)
        return original(window, bitmap, slots, provider_groups, **kwargs)

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


@pytest.mark.parametrize(
    "lengths,keys,start,queries",
    [
        (((1031, 17, 3040), (17, 1029, 3019)), 4096, 137, 3959),
        (((2051, 4096, 1949), (512, 4096, 1536)), 8192, 4000, 4096),
        (((2051, 1848), (97,)), 4096, 2048, 2048),
        (((63,), (53, 84)), 4096, 2048, 2048),
    ],
)
def test_b2_native_varlen_matches_independent_packed_forward_and_gradients(
    monkeypatch: pytest.MonkeyPatch,
    lengths: tuple[tuple[int, ...], tuple[int, ...]],
    keys: int,
    start: int,
    queries: int,
) -> None:
    """Cover distinct documents, shifted shards, trailing pads and wholly padded rows."""
    torch.manual_seed(8117)
    positions = torch.stack(
        [
            torch.cat([torch.arange(n) for n in row] + [torch.zeros(keys - sum(row), dtype=torch.long)])
            for row in lengths
        ]
    ).cuda()
    padding = torch.stack([torch.arange(keys) >= sum(row) for row in lengths]).cuda()
    documents = ((positions == 0) & ~padding).cumsum(1)
    metadata = region_metadata(positions, padding, query_start=start, queries=queries, region_size=8, topk=512)
    assert not metadata.score_tiles
    assert metadata.causal_batch_layouts is not None
    groups, heads, dim = 2, 4, 192
    selection = RegionSelection(
        metadata.dense_region_ids[:, None].expand(-1, groups, -1, -1).contiguous(),
        metadata.token_regions,
        metadata.query_documents,
        metadata.documents,
        metadata.region_count,
        all_regions_selected=True,
        causal_batch_layouts=metadata.causal_batch_layouts,
    )
    query = torch.randn(2, queries, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    joint = torch.randn(keys, 2, 2, groups, dim, device="cuda", dtype=torch.bfloat16)
    key, value = joint.transpose(0, 1).unbind(2)
    key, value = key.detach().requires_grad_(), value.detach().requires_grad_()
    assert not key.is_contiguous() and not value.is_contiguous()
    calls = []
    original = fa4._load_fa4_varlen()

    def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, ...]:
        """Observe native varlen inputs without changing execution.

        Args:
            q: Tensor [valid_queries, query_heads, head_dim].
            k: Tensor [prefix_keys, kv_heads, head_dim].
            v: Tensor with the same shape as k.
            kwargs: Unchanged FA4 keyword options.

        Returns:
            The native output [valid_queries, query_heads, head_dim] and optional LSE.
        """
        assert k.untyped_storage().data_ptr() == joint.untyped_storage().data_ptr()
        assert v.untyped_storage().data_ptr() == joint.untyped_storage().data_ptr()
        calls.append((q.shape[0], k.shape[0]))
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(fa4, "_load_fa4_varlen", lambda: attention)
    output = fa4.fa4_attention(
        query,
        key,
        value,
        positions[:, start : start + queries],
        positions,
        padding,
        documents,
        query_start=start,
        window=None,
        selection=selection,
    )
    assert calls == [
        (layout.valid_queries, layout.key_stop - layout.key_start)
        for layout in metadata.causal_batch_layouts
        if layout.valid_queries
    ]
    ref_q, ref_k, ref_v = [tensor.detach().float().requires_grad_() for tensor in (query, key, value)]
    allowed = (
        (positions[:, start : start + queries, None] >= positions[:, None, :])
        & (documents[:, start : start + queries, None] == documents[:, None, :])
        & ~padding[:, None, :]
    )
    sdpa_reference = F.scaled_dot_product_attention(
        ref_q.transpose(1, 2),
        ref_k.transpose(1, 2).repeat_interleave(heads // groups, dim=1),
        ref_v.transpose(1, 2).repeat_interleave(heads // groups, dim=1),
        attn_mask=allowed[:, None],
    ).transpose(1, 2)
    reference_rows = []
    for row, valid_lengths in enumerate(lengths):
        if start >= sum(valid_lengths):
            # Independently prove a singleton using CPU document math. A
            # softmax implementation can leave tiny nonzero dQ/dK residuals,
            # whereas a one-key probability is identically one.
            cpu_positions = torch.cat(
                [torch.arange(n) for n in valid_lengths] + [torch.zeros(keys - sum(valid_lengths), dtype=torch.long)]
            )
            cpu_padding = torch.arange(keys) >= sum(valid_lengths)
            cpu_documents = ((cpu_positions == 0) & ~cpu_padding).cumsum(0)
            cpu_allowed = (
                (cpu_positions[start : start + queries, None] >= cpu_positions[None])
                & (cpu_documents[start : start + queries, None] == cpu_documents[None])
                & ~cpu_padding[None]
            )
            assert torch.all(cpu_allowed.sum(-1) == 1)
            singleton = torch.nonzero(cpu_allowed[0]).item()
            assert torch.all(cpu_allowed[:, singleton])
            one_value = ref_v[row, singleton].repeat_interleave(heads // groups, dim=0)
            reference_rows.append(one_value[None].expand(queries, -1, -1) + (ref_q[row].sum() + ref_k[row].sum()) * 0)
        else:
            reference_rows.append(sdpa_reference[row])
    reference = torch.stack(reference_rows)
    torch.testing.assert_close(output.float(), reference, atol=0.015, rtol=0.015)
    upstream = torch.randn_like(output)
    actual = torch.autograd.grad(output, (query, key, value), upstream)
    expected = torch.autograd.grad(reference, (ref_q, ref_k, ref_v), upstream.float())
    for actual_gradient, expected_gradient in zip(actual, expected):
        assert torch.isfinite(actual_gradient).all()
        norm = expected_gradient.square().sum().sqrt()
        if norm == 0:
            assert torch.count_nonzero(actual_gradient) == 0
        else:
            relative = (actual_gradient.float() - expected_gradient).square().sum().sqrt() / norm
            assert relative < 0.012, relative.item()
    for row, valid_lengths in enumerate(lengths):
        for gradient in actual[1:]:
            assert torch.count_nonzero(gradient[row, sum(valid_lengths) :]) == 0
        if start >= sum(valid_lengths):
            assert torch.count_nonzero(actual[0][row]) == 0
            assert torch.count_nonzero(actual[1][row]) == 0
            assert actual[2][row].abs().sum() > 0
    if all(start >= sum(row) for row in lengths) and len(lengths[1]) > 1:
        # Deliberately reading the previous document must fail this oracle.
        wrong_layout = replace(metadata.causal_batch_layouts[1], padding_key=0)
        wrong_selection = replace(selection, causal_batch_layouts=(metadata.causal_batch_layouts[0], wrong_layout))
        wrong_output = fa4.fa4_attention(
            query,
            key,
            value,
            positions[:, start : start + queries],
            positions,
            padding,
            documents,
            query_start=start,
            window=None,
            selection=wrong_selection,
        )
        with pytest.raises(AssertionError):
            torch.testing.assert_close(wrong_output.float(), reference, atol=0.015, rtol=0.015)


@pytest.mark.parametrize("defect", ["sparse_row", "invalid_row"])
def test_b2_unproven_row_dispatches_only_the_existing_fallback(monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    """Intercept dispatch only; numerical CUDA coverage is the independent test above."""
    positions = torch.arange(32, device="cuda")[None].expand(2, -1).clone()
    topk = 512
    if defect == "sparse_row":
        positions[0] = positions[0] % 4
        topk = 1
    else:
        positions[1, 10] = 9
    padding = torch.zeros_like(positions, dtype=torch.bool)
    metadata = region_metadata(positions, padding, query_start=0, queries=32, region_size=8, topk=topk)
    assert metadata.causal_batch_layouts is None
    selection = RegionSelection(
        metadata.dense_region_ids[:, None].expand(-1, 2, -1, -1).contiguous(),
        metadata.token_regions,
        metadata.query_documents,
        metadata.documents,
        metadata.region_count,
        all_regions_selected=not metadata.score_tiles,
    )

    class FallbackObserved(Exception):
        pass

    def forbidden_varlen() -> None:
        raise AssertionError("An unproven B2 row must not enter native varlen")

    def existing_fallback(window: int | None, bitmap: bool, slots: int, provider_groups: int, **kwargs: int) -> None:
        raise FallbackObserved

    monkeypatch.setattr(fa4, "_load_fa4_varlen", forbidden_varlen)
    monkeypatch.setattr(fa4, "_load_fa4", existing_fallback)
    query = torch.zeros(2, 32, 4, 192, device="cuda", dtype=torch.bfloat16)
    key = torch.zeros(2, 32, 2, 192, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(FallbackObserved):
        fa4.fa4_attention(
            query,
            key,
            key,
            positions,
            positions,
            padding,
            metadata.documents,
            query_start=0,
            window=None,
            selection=selection,
        )


def _reference_vjp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    upstream: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute an independent FP32 mathematical VJP with bounded query chunks.

    Args:
        query: BF16 tensor [1, queries, providers * 16, 192].
        key: BF16 tensor [1, keys, providers, 192], possibly a strided KV-unbind view.
        value: BF16 tensor with the key layout.
        upstream: BF16 output gradient with the query layout.

    Returns:
        FP32 output/dQ with query layout and dK/dV with key layout. No dense
        full-sequence query-key mask or score allocation is constructed.
    """
    q = query.detach().float().requires_grad_()
    k = key.detach().float().requires_grad_()
    v = value.detach().float().requires_grad_()
    output, dq = torch.empty_like(q), torch.empty_like(q)
    dk, dv = torch.zeros_like(k), torch.zeros_like(v)
    keys = torch.arange(key.shape[1], device=query.device)[None]
    with sdpa_kernel(SDPBackend.MATH):
        for start in range(0, query.shape[1], 256):
            stop = min(start + 256, query.shape[1])
            positions = torch.arange(start, stop, device=query.device)[:, None]
            # Each provider chooses a different legal 512-region interval,
            # plus the current region. At 8K this catches head/group mixups.
            providers = key.shape[2]
            group = torch.arange(providers, device=query.device)[:, None, None]
            current = positions // 8
            lower = (current - 512).clamp_min(0)[None] * group // max(providers - 1, 1)
            selected = ((keys // 8 >= lower) & (keys // 8 < lower + 512)) | (keys // 8 == current)
            allowed = ((keys <= positions)[None] & selected).repeat_interleave(16, dim=0)[None]
            chunk = q[:, start:stop]
            result = F.scaled_dot_product_attention(
                chunk.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attn_mask=allowed,
                enable_gqa=True,
            ).transpose(1, 2)
            grads = torch.autograd.grad(result, (chunk, k, v), upstream[:, start:stop].float())
            output[:, start:stop] = result.detach()
            dq[:, start:stop] = grads[0]
            dk += grads[1]
            dv += grads[2]
    return output, dq, dk, dv


def _relative_rms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Compare floating tensors without retaining their graph.

    Args:
        actual: Floating tensor of arbitrary shape [...].
        expected: Nonzero reference with the same shape.

    Returns:
        Scalar relative root-mean-square error.
    """
    difference = actual.float() - expected.float()
    return float(difference.square().sum().sqrt() / expected.float().square().sum().sqrt())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("length", [4096, 8192])
@pytest.mark.parametrize("direction", ["forward", "backward", "both", "canonical"])
def test_forward_full_blocks_match_fixed_routes_and_independent_vjp(
    monkeypatch: pytest.MonkeyPatch,
    length: int,
    direction: str,
) -> None:
    """Exercise real D192/GQA16 FA4 with noncontiguous interleaved KV.

    The 4K component intentionally omits causal_layout to test custom FA4.
    Native all-selected 4K dispatch remains covered by the existing CP runner.
    """
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Forward full-block option is restricted to SM90")
    torch.manual_seed(583)
    providers = 4 if length == 8192 else 1
    query = torch.randn(1, length, providers * 16, 192, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(1, length, 2, providers, 192, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key, value = kv.unbind(2)
    assert not key.is_contiguous() and not value.is_contiguous()
    positions = torch.arange(length, device="cuda")[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    metadata = region_metadata(
        positions,
        padding,
        query_start=0,
        queries=length,
        region_size=8,
        topk=512,
        csa_forward_full_blocks=True,
        csa_backward_full_blocks=True,
    )
    current = positions // 8
    lower = (current - 512).clamp_min(0)[:, None] * torch.arange(providers, device="cuda")[None, :, None]
    lower = lower // max(providers - 1, 1)
    historical = lower[..., None] + torch.arange(512, device="cuda")
    historical = torch.where(historical < current[:, None, :, None], historical, metadata.region_count)
    ids = torch.cat((historical, current[:, None, :, None].expand(-1, providers, -1, -1)), dim=-1)
    ids = ids.sort(dim=-1).values.int()
    selection = RegionSelection(
        ids,
        metadata.token_regions,
        metadata.query_documents,
        metadata.documents,
        metadata.region_count,
    )
    upstream = torch.randn_like(query)
    full_block_counts = []
    original = fa4._split_forward_full_blocks if direction == "forward" else fa4._split_backward_full_blocks

    def record_split(
        counts: torch.Tensor,
        indices: torch.Tensor,
        *,
        prefix_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | fa4._BackwardBlockPartition:
        """Observe forward or backward metadata while preserving the real kernel call.

        Args:
            counts: Int32 counts [provider_batches, heads, query_tiles];
                either [groups, 1, query_tiles] or [1, query_heads, query_tiles].
            indices: Int32 indices [provider_batches, heads, query_tiles, key_tiles];
                backward swaps the query_tiles and key_tiles axes for both inputs.
            prefix_length: Proven initial all-selected query count.

        Returns:
            Partial counts, partial indices and full counts in their original
            semantic layouts, unchanged from the production helper. Backward
            also returns an independent full-index tensor matching indices.
        """
        result = original(counts, indices, prefix_length=prefix_length)
        full_block_counts.append(result[2].detach().cpu())
        return result

    monkeypatch.setattr(
        fa4, "_split_forward_full_blocks" if direction == "forward" else "_split_backward_full_blocks", record_split
    )
    runs = []
    memory = []
    for enabled in (False, True) if direction == "forward" else (False, False, True):
        chosen = replace(
            selection,
            forward_full_block_prefix=metadata.forward_full_block_prefix
            if direction in ("both", "canonical") or (direction == "forward" and enabled)
            else 0,
            canonical_region_size=metadata.canonical_region_size if direction == "canonical" and enabled else 0,
            backward_full_block_prefix=metadata.backward_full_block_prefix
            if direction == "canonical" or (direction != "forward" and enabled)
            else 0,
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = torch.cuda.memory_allocated()
        output = fa4.fa4_attention(
            query,
            key,
            value,
            positions,
            positions,
            padding,
            metadata.documents,
            query_start=0,
            window=None,
            selection=chosen,
        )
        forward_peak = torch.cuda.max_memory_allocated()
        gradients = torch.autograd.grad(output, (query, kv), upstream)
        torch.cuda.synchronize()
        memory.append(
            {
                "allocated_before_bytes": allocated_before,
                "forward_peak_allocated_bytes": forward_peak,
                "forward_backward_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
        )
        runs.append((output.detach(), gradients[0], gradients[1][:, :, 0], gradients[1][:, :, 1]))
    assert len(full_block_counts) == (3 if direction == "canonical" else 1) and full_block_counts[0].sum() > 0
    assert full_block_counts[0].shape[:2] == ((1, providers * 16) if providers > 1 else (1, 1))
    aa_metrics = {}
    if direction != "forward":
        for name, first, repeat in zip(("dQ", "dK", "dV"), runs[0][1:], runs[1][1:], strict=True):
            aa_metrics[name] = {
                "relative_rms": _relative_rms(repeat, first),
                "maxabs": float((repeat.float() - first.float()).abs().max()),
            }
            assert aa_metrics[name]["relative_rms"] < 0.012
    # Backward-only splitting leaves forward lists unchanged. Forward splitting
    # also retains identical K traversal; this is a measured bitwise hypothesis.
    # Identical block traversal is a bitwise-forward hypothesis under test,
    # not an excuse to relax a mismatch.
    assert torch.equal(runs[0][0].view(torch.uint8), runs[-1][0].view(torch.uint8))
    torch.cuda.reset_peak_memory_stats()
    expected = _reference_vjp(query, key, value, upstream)
    oracle_peak = torch.cuda.max_memory_allocated()
    pair_metrics = {}
    for name, baseline, candidate in zip(("dQ", "dK", "dV"), runs[0][1:], runs[-1][1:], strict=True):
        relative = _relative_rms(candidate, baseline)
        maximum = float((candidate.float() - baseline.float()).abs().max())
        assert relative < 0.012, (name, relative, maximum)
        pair_metrics[name] = {"relative_rms": relative, "maxabs": maximum}
    metrics = []
    for result in runs:
        torch.testing.assert_close(result[0].float(), expected[0], atol=0.015, rtol=0.015)
        errors = [_relative_rms(a, e) for a, e in zip(result[1:], expected[1:], strict=True)]
        assert all(error < 0.012 for error in errors), errors
        assert all(torch.isfinite(tensor).all() for tensor in result)
        metrics.append(errors)
    bucket_errors = {}
    pair_buckets = {}
    for label, start, stop in (("prefix", 0, min(length, 4104)), ("sparse", 4104, length)):
        if start < stop:
            torch.testing.assert_close(
                runs[-1][0][:, start:stop].float(), expected[0][:, start:stop], atol=0.015, rtol=0.015
            )
            bucket_errors[label] = _relative_rms(runs[-1][1][:, start:stop], expected[1][:, start:stop])
            assert bucket_errors[label] < 0.012
            pair_buckets[label] = {
                "relative_rms": _relative_rms(runs[-1][1][:, start:stop], runs[0][1][:, start:stop]),
                "maxabs": float((runs[-1][1][:, start:stop].float() - runs[0][1][:, start:stop].float()).abs().max()),
            }
            assert pair_buckets[label]["relative_rms"] < 0.012
    print(
        json.dumps(
            {
                "length": length,
                "direction": direction,
                "off_off_gradient_control": aa_metrics,
                "head_dim": 192,
                "gqa": 16,
                "providers": providers,
                "q_stride": query.stride(),
                "kv_stride": kv.stride(),
                "key_stride": key.stride(),
                "value_stride": value.stride(),
                "full_blocks": int(full_block_counts[0].sum()),
                "forward_bitwise": True,
                "gradient_relative_rms_each_vs_fp32": metrics,
                "candidate_vs_baseline_gradients": pair_metrics,
                "candidate_vs_baseline_dq_buckets": pair_buckets,
                "memory_baseline_candidate": memory,
                "oracle_peak_allocated_bytes": oracle_peak,
                "query_gradient_buckets": bucket_errors,
                "truly_sparse_queries": max(0, length - 4104),
                "scope": "Correctness only: fixed legal routes, synthetic nonzero upstream, and synchronized metadata observation. No performance or checkpoint/native model parity claim.",
            }
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("region_count,bitmap", [(16386, True), (16539, False)])
@pytest.mark.parametrize("queries", [4, 128])
def test_long_history_preserves_each_provider_and_query_offset(
    monkeypatch: pytest.MonkeyPatch, region_count: int, bitmap: bool, queries: int
) -> None:
    """Use independent group-specific gathered-key math, including the fallback."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Native D192 head layout is restricted to SM90")
    torch.manual_seed(584)
    groups, heads_per_group, dim = 2, 16, 192
    keys = region_count * 8
    start = keys - queries
    q = torch.randn(1, queries, groups * heads_per_group, dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, keys, 2, groups, dim, device="cuda", dtype=torch.bfloat16)
    q.requires_grad_()
    k, v = (part.detach().requires_grad_() for part in kv.unbind(2))
    positions = torch.arange(keys, device="cuda")[None]
    documents = torch.ones_like(positions)
    # Disjoint historical intervals make accidental group sharing observable.
    historical = torch.arange(512, device="cuda")[None, None, None]
    historical = historical + torch.arange(groups, device="cuda")[None, :, None, None] * 512
    historical = historical.expand(1, groups, queries, -1)
    current = (positions[:, start:] // 8)[:, None, :, None].expand(1, groups, -1, -1)
    selection = RegionSelection(
        torch.cat((historical, current), dim=-1).int(),
        positions // 8,
        documents[:, start:],
        documents,
        region_count,
    )
    dispatch = []
    original = fa4._load_fa4

    def load(
        window: int | None,
        use_bitmap: bool,
        slots: int,
        provider_groups: int,
        *,
        heads_per_group: int = 0,
        **kwargs: int,
    ) -> tuple[Callable, type, Callable]:
        dispatch.append((use_bitmap, heads_per_group))
        return original(window, use_bitmap, slots, provider_groups, heads_per_group=heads_per_group, **kwargs)

    monkeypatch.setattr(fa4, "_load_fa4", load)
    actual = fa4.fa4_attention(
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
    # Four queries cannot amortize the per-head metadata over an output copy;
    # 128 queries can. This asserts actual dispatch, followed by real math.
    assert dispatch == [(bitmap, heads_per_group if queries == 128 else 0)]
    rq, rk, rv = (tensor.detach().float().requires_grad_() for tensor in (q, k, v))
    expected_parts = []
    with sdpa_kernel(SDPBackend.MATH):
        for group in range(groups):
            old = torch.arange(group * 4096, (group + 1) * 4096, device="cuda")
            tail = torch.arange(start // 8 * 8, keys, device="cuda")
            chosen = torch.cat((old, tail))
            # Gather at most 4224 keys. No full-history dense mask/score is used.
            allowed = (chosen[None] <= positions[0, start:, None]) & (
                (chosen[None] < (group + 1) * 4096) | (chosen[None] // 8 == positions[0, start:, None] // 8)
            )
            output = F.scaled_dot_product_attention(
                rq[:, :, group * heads_per_group : (group + 1) * heads_per_group].transpose(1, 2),
                rk[:, chosen, group : group + 1].transpose(1, 2),
                rv[:, chosen, group : group + 1].transpose(1, 2),
                attn_mask=allowed,
                enable_gqa=True,
            ).transpose(1, 2)
            expected_parts.append(output)
    expected = torch.cat(expected_parts, dim=2)
    torch.testing.assert_close(actual.float(), expected, atol=0.015, rtol=0.015)
    upstream = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, (q, k, v), upstream)
    expected_grad = torch.autograd.grad(expected, (rq, rk, rv), upstream.float())
    for name, observed, reference in zip(("dQ", "dK", "dV"), actual_grad, expected_grad, strict=True):
        assert torch.isfinite(observed).all()
        relative = ((observed.float() - reference).square().sum() / reference.square().sum()).sqrt()
        assert relative < 0.012, (name, float(relative))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("lengths,start", [((1031, 3065), 0), ((8192,), 4096), ((2051, 4096, 2045), 4096)])
def test_native_swa_matches_original_and_independent_sdpa(
    lengths: tuple[int, ...], start: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(1127)
    keys, dim, heads, groups, window = sum(lengths), 192, 4, 2, 512
    queries = keys - start
    positions = torch.cat([torch.arange(length) for length in lengths]).cuda()[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    metadata = region_metadata(
        positions, padding, query_start=start, queries=queries, region_size=8, topk=512, window=window
    )
    assert metadata.window_layout is not None
    query = torch.randn(1, queries, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(1, keys, groups, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    first_key = max(0, start - window + 1)
    left_padding = max(first_key % 112 if start else 0, metadata.window_layout.leading_padding)
    halo = F.pad(torch.stack((key, value), dim=2)[:, first_key:], (0, 0, 0, 0, 0, 0, left_padding, 0))
    halo_key, halo_value = halo.unbind(dim=2)
    assert halo_key.stride(1) == 2 * groups * dim
    key_positions = F.pad(positions[:, first_key:], (left_padding, 0))
    key_padding = F.pad(padding[:, first_key:], (left_padding, 0), value=True)
    documents = ((key_positions == 0) & ~key_padding).long().cumsum(1)
    arguments = dict(query_start=start - first_key + left_padding, window=window, selection=None)
    original = fa4.fa4_attention(
        query, halo_key, halo_value, positions[:, start:], key_positions, key_padding, documents, **arguments
    )

    def forbidden_mask(*_args, **_kwargs):
        raise AssertionError("Native SWA must not build block lists or load the custom mask")

    monkeypatch.setattr(fa4, "_load_fa4", forbidden_mask)
    monkeypatch.setattr(fa4, "_block_lists", forbidden_mask)
    actual = fa4.fa4_attention(
        query,
        halo_key,
        halo_value,
        positions[:, start:],
        key_positions,
        key_padding,
        documents,
        window_layout=metadata.window_layout,
        **arguments,
    )
    # Independent global-token predicate; no production layout or halo indexing.
    global_documents = (positions == 0).long().cumsum(-1)
    distances = positions[:, start:, None] - positions[:, None]
    mask = ((global_documents[:, start:, None] == global_documents[:, None]) & (distances >= 0) & (distances < window))[
        :, None
    ]
    expected = F.scaled_dot_product_attention(
        query.float().transpose(1, 2),
        key.float().transpose(1, 2).repeat_interleave(heads // groups, dim=1),
        value.float().transpose(1, 2).repeat_interleave(heads // groups, dim=1),
        attn_mask=mask,
    ).transpose(1, 2)
    torch.testing.assert_close(actual.float(), expected, atol=0.015, rtol=0.015)
    torch.testing.assert_close(original.float(), expected, atol=0.015, rtol=0.015)
    upstream = torch.randn_like(actual)
    reference_grad = torch.autograd.grad(expected, (query, key, value), upstream.float())
    metrics: dict[str, list[float]] = {}
    for name, output in (("native", actual), ("original", original)):
        gradients = torch.autograd.grad(output, (query, key, value), upstream, retain_graph=True)
        metrics[name] = []
        for gradient, reference in zip(gradients, reference_grad):
            relative = ((gradient.float() - reference.float()).square().sum() / reference.float().square().sum()).sqrt()
            assert relative < 0.012 and torch.isfinite(gradient).all()
            metrics[name].append(float(relative))
    print({"lengths": lengths, "start": start, "left_padding": left_padding, "gradient_relative_rms": metrics})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("lengths,pad,start", [((4093,), 3, 0), ((2051, 4096, 2038), 7, 4096), ((4089, 4), 3, 0)])
def test_native_swa_padding_preserves_original_outputs_and_gradients(
    lengths: tuple[int, ...], pad: int, start: int
) -> None:
    torch.manual_seed(1872)
    valid = sum(lengths)
    total = valid + pad
    positions = torch.cat(tuple(torch.arange(n) for n in lengths) + (torch.zeros(pad, dtype=torch.long),)).cuda()[None]
    padding = torch.arange(total, device="cuda")[None] >= valid
    metadata = region_metadata(
        positions, padding, query_start=start, queries=total - start, region_size=8, topk=512, window=512
    )
    layout = metadata.window_layout
    assert layout is not None and layout.valid_queries == valid - start
    query = torch.randn(1, total - start, 4, 192, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn(1, total, 2, 192, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    value = torch.randn_like(key, requires_grad=True)
    first = max(0, start - 511)
    old_padding = first % 112
    new_padding = max(old_padding, layout.leading_padding)
    outputs = []
    for left, native in ((old_padding, False), (new_padding, True)):
        packed = F.pad(torch.stack((key, value), dim=2)[:, first:], (0, 0, 0, 0, 0, 0, left, 0))
        kh, vh = packed.unbind(2)
        kp = F.pad(positions[:, first:], (left, 0))
        padmask = F.pad(padding[:, first:], (left, 0), value=True)
        docs = ((kp == 0) & ~padmask).long().cumsum(-1)
        outputs.append(
            fa4.fa4_attention(
                query,
                kh,
                vh,
                positions[:, start:],
                kp,
                padmask,
                docs,
                query_start=start - first + left,
                window=512,
                selection=None,
                window_layout=layout if native else None,
            )
        )
    original, actual = outputs
    torch.testing.assert_close(actual[:, -pad:], original[:, -pad:], rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    pad_upstream = upstream.clone()
    pad_upstream[:, :-pad] = 0
    # The padding branch must preserve dV as well as dQ/dK, including the short
    # final document where padded queries can read its first V.
    original_pad_grad = torch.autograd.grad(original, (query, key, value), pad_upstream, retain_graph=True)
    actual_pad_grad = torch.autograd.grad(actual, (query, key, value), pad_upstream, retain_graph=True)
    for got, want in zip(actual_pad_grad, original_pad_grad):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
    # Independently evaluate only valid query semantics. Padded rows have the
    # original kernel's physical block pruning, explicitly checked above.
    docs = ((positions == 0) & ~padding).long().cumsum(-1)
    distance = positions[:, start:valid, None] - positions[:, None]
    mask = (docs[:, start:valid, None] == docs[:, None]) & (distance >= 0) & (distance < 512) & ~padding[:, None]
    expected = F.scaled_dot_product_attention(
        query[:, : valid - start].float().transpose(1, 2),
        key.float().transpose(1, 2).repeat_interleave(2, 1),
        value.float().transpose(1, 2).repeat_interleave(2, 1),
        attn_mask=mask[:, None],
    ).transpose(1, 2)
    torch.testing.assert_close(actual[:, : valid - start].float(), expected, atol=0.015, rtol=0.015)
    reference_grad = torch.autograd.grad(expected, (query, key, value), upstream[:, : valid - start].float())
    valid_upstream = upstream.clone()
    valid_upstream[:, -pad:] = 0
    gradients = torch.autograd.grad(actual, (query, key, value), valid_upstream)
    metrics = []
    for got, want in zip(gradients, reference_grad):
        relative = ((got.float() - want.float()).square().sum() / want.float().square().sum()).sqrt()
        assert relative < 0.012 and torch.isfinite(got).all()
        metrics.append(float(relative))
    print(
        {
            "lengths": lengths,
            "padding": pad,
            "start": start,
            "valid_grad_relative": metrics,
            "padding_output_nonzero": bool(actual[:, -pad:].any()),
            "padding_dv_nonzero": bool(actual_pad_grad[2].any()),
        }
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("spans", [(256, 128, 96, 64, 1), (1, 64, 96, 128, 256)])
@pytest.mark.parametrize("queries,heads,groups", [(4096, 4, 2), (4096, 64, 4), (8192, 64, 4)])
def test_native_swa_bounds_reuse_kernels_for_many_short_documents(
    spans: tuple[int, ...], queries: int, heads: int, groups: int
) -> None:
    """Document counts/lengths cannot select late single-block variants."""
    from flash_attn.cute import interface

    torch.manual_seed(2017)
    caches = (interface._flash_attn_fwd.compile_cache, interface._flash_attn_bwd.compile_cache)
    seen_schedulers: set[bool] = set()
    for case, span in enumerate(spans):
        full, remainder = divmod(queries, span)
        if full > 320:
            # Exercise binary batch lookup without tens of GiB of per-document
            # backward padding scratch for thousands of singleton documents.
            full, remainder = 320, 0
        valid = full * span + remainder
        positions = torch.cat(
            (
                torch.arange(span, device="cuda").repeat(full),
                torch.arange(remainder, device="cuda"),
                torch.zeros(queries - valid, device="cuda", dtype=torch.long),
            )
        )[None]
        padding = torch.arange(queries, device="cuda")[None] >= valid
        meta = region_metadata(positions, padding, query_start=0, queries=queries, region_size=8, topk=512, window=512)
        assert meta.window_layout is not None
        q = torch.randn(1, queries, heads, 192, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        kv = torch.randn(1, queries, 2, groups, 192, device="cuda", dtype=torch.bfloat16)
        k, v = (item.detach().requires_grad_() for item in kv.unbind(2))
        before = [set(cache.cache) for cache in caches]
        actual = fa4.fa4_attention(
            q,
            k,
            v,
            positions,
            positions,
            padding,
            meta.documents,
            query_start=0,
            window=512,
            selection=None,
            window_layout=meta.window_layout,
        )
        upstream = torch.randn_like(actual)
        upstream[:, valid:] = 0
        got = torch.autograd.grad(actual, (q, k, v), upstream)
        torch.cuda.synchronize()
        # FA4 intentionally switches to binary batch lookup above 256
        # documents. Reuse is required within each source-defined scheduler
        # class; maxlen must not add further single-block variants.
        scheduler = full + bool(remainder) > interface.BIN_BATCH_SEARCH_THRESH
        if scheduler in seen_schedulers:
            assert [set(cache.cache) for cache in caches] == before, (
                "Short SWA documents compiled another kernel variant"
            )
        seen_schedulers.add(scheduler)
        rq, rk, rv = (x.detach().float().requires_grad_() for x in (q, k, v))
        # Evaluate equal-length documents as a batch, independently of the
        # production cu_seqlens. No quadratic whole-packed attention mask.
        prefix = full * span
        eq = rq[0, :prefix].reshape(full, span, heads, 192).transpose(1, 2)
        ek = rk[0, :prefix].reshape(full, span, groups, 192).transpose(1, 2).repeat_interleave(heads // groups, 1)
        ev = rv[0, :prefix].reshape(full, span, groups, 192).transpose(1, 2).repeat_interleave(heads // groups, 1)
        expected = (
            F.scaled_dot_product_attention(eq, ek, ev, is_causal=True).transpose(1, 2).reshape(1, prefix, heads, 192)
        )
        if remainder:
            tail = F.scaled_dot_product_attention(
                rq[:, prefix:valid].transpose(1, 2),
                rk[:, prefix:valid].transpose(1, 2).repeat_interleave(heads // groups, 1),
                rv[:, prefix:valid].transpose(1, 2).repeat_interleave(heads // groups, 1),
                is_causal=True,
            ).transpose(1, 2)
            expected = torch.cat((expected, tail), dim=1)
        torch.testing.assert_close(actual[:, :valid].float(), expected, atol=0.015, rtol=0.015)
        want = torch.autograd.grad(expected, (rq, rk, rv), upstream[:, :valid].float())
        for index, (observed, reference) in enumerate(zip(got, want)):
            if span == 1 and index < 2:
                torch.testing.assert_close(observed.float(), torch.zeros_like(reference), atol=1e-5, rtol=0)
                torch.testing.assert_close(reference, torch.zeros_like(reference), atol=1e-5, rtol=0)
            else:
                relative = (observed.float() - reference).norm() / reference.norm()
                assert relative < 0.012, (span, index, float(relative))
            assert torch.isfinite(observed).all()
        print(
            {
                "queries": queries,
                "heads": heads,
                "groups": groups,
                "span": span,
                "documents": full + bool(remainder),
                "cache_sizes": [len(cache.cache) for cache in caches],
            }
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("queries", [1024, 1001])
def test_region_tiles_reuse_graph_across_packed_region_counts(queries: int) -> None:
    """Check real 128K shapes against independent sorted-set membership."""
    torch._dynamo.reset()
    counters.clear()
    generator = torch.Generator().manual_seed(719)
    first_graphs = None
    for region_count in (16386, 16539, 16481):
        # One selected set per 128-query tile; preserve the production CP4K strides.
        sets = torch.randint(region_count, (1, 4, 32, 513), generator=generator, dtype=torch.int32)
        sets[..., -1] = region_count  # Unused-slot sentinel must not select a real key.
        ids = sets.repeat_interleave(128, dim=2).cuda()[:, :, :queries]
        token_regions = (torch.arange(131072, dtype=torch.int64) * region_count // 131072)[None]
        expected = []
        for width in (112, 96):
            starts = torch.arange(0, 131072, width)
            first = token_regions[0, starts]
            last = token_regions[0, (starts + width - 1).clamp(max=131071)]
            grid = torch.empty(1, 4, 8, len(starts), dtype=torch.bool)
            for group in range(4):
                for row in range(8):
                    selected = sets[0, group, row].unique(sorted=True).long()
                    selected = selected[selected < region_count]
                    left = torch.searchsorted(selected, first)
                    right = torch.searchsorted(selected, last, right=True)
                    grid[0, group, row] = right > left
            expected.append(grid)
        actual = _region_tiles(ids, token_regions.cuda(), region_count, 112, 96)
        torch.cuda.synchronize()
        for result, oracle in zip(actual, expected):
            torch.testing.assert_close(result.cpu(), oracle, rtol=0, atol=0)
        graphs = counters["stats"]["unique_graphs"]
        print(f"regions={region_count}, compiled_graphs={graphs}", flush=True)
        if first_graphs is None:
            first_graphs = graphs
            assert first_graphs > 0
        else:
            assert graphs == first_graphs, (region_count, first_graphs, graphs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_region_bitmap_reuses_kernel_across_region_counts() -> None:
    """Changing packed region counts must not specialize the bitmap packer."""
    from nemo_automodel.components.models.step5.sparse import _pack_regions, pack_region_bitmap

    generator = torch.Generator().manual_seed(829)
    first_kernels = None
    for regions in (16386, 16411, 16390, 2049, 4097):
        ids = torch.randint(regions, (1, 2, 17, 9), generator=generator, dtype=torch.int32)
        ids[..., 0] = 31  # Exercise the sign bit of each int32 word.
        ids[..., -2] = regions - 1
        ids[..., -1] = regions  # Unused-slot sentinel.
        words = (regions + 31) // 32
        expected = torch.zeros((1, 2, 17, words), dtype=torch.int32)
        for group in range(2):
            for query in range(17):
                packed = [0] * words
                for region in ids[0, group, query].tolist():
                    if region < regions:
                        packed[region // 32] |= 1 << (region % 32)
                expected[0, group, query] = torch.tensor(
                    [value if value < 2**31 else value - 2**32 for value in packed], dtype=torch.int32
                )
        actual = pack_region_bitmap(ids.cuda(), regions)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        kernels = len(_pack_regions.device_caches[torch.cuda.current_device()][0])
        print(f"bitmap_regions={regions}, words={words}, compiled_kernels={kernels}", flush=True)
        if first_kernels is None:
            first_kernels = kernels
            assert first_kernels > 0
        else:
            assert kernels == first_kernels, (regions, first_kernels, kernels)


@pytest.fixture(autouse=True)
def ieee_fp32_projections() -> Iterator[None]:
    """Keep CPU/CUDA FP32 projection checks independent of global TF32 defaults."""
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize("offset,length", [(4090, 30), (8064, 128)])
@pytest.mark.parametrize("tied", [False, True])
def test_cuda_eight_k_selection_matches_cpu_stable_sort(offset: int, length: int, tied: bool) -> None:
    module = _indexer(topk=512, region_size=8)
    if tied:
        with torch.no_grad():
            module.w_proj.weight.zero_()
    cuda = copy.deepcopy(module).cuda()
    positions = torch.arange(8192)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    hidden = torch.randn(1, 8192, 32)
    keys, logits = module.project_keys(hidden, positions)
    local, local_positions = hidden[:, offset : offset + length], positions[:, offset : offset + length]
    expected = _oracle(module, local, local_positions, keys, logits, positions, padding, offset=offset)
    global_positions, global_padding = positions.cuda(), padding.cuda()
    # Global K/Z projection outputs are held fixed between CPU and CUDA; local
    # Q/weight projections execute on their respective devices in FP32.
    # Native distributed checkpoint loading is covered separately.
    actual = cuda(
        local.cuda(),
        local_positions.cuda(),
        keys.cuda(),
        logits.cuda(),
        global_positions,
        global_padding,
        query_start=offset,
    )
    torch.testing.assert_close(actual.region_ids.cpu(), expected, rtol=0, atol=0)
    active = actual.region_ids[0, 0].cpu()
    assert (active < actual.region_count).sum(-1).max() == 513


def test_cuda_128k_packed_short_documents_use_shared_prefix_routes() -> None:
    module = _indexer(topk=512, region_size=8).cuda()
    positions = torch.arange(4096, device="cuda").repeat(32)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    offset, queries = 17 * 4096, 4096
    meta = region_metadata(positions, padding, query_start=offset, queries=queries, region_size=8, topk=512)
    assert not meta.needs_projection and not meta.score_tiles
    hidden = torch.randn(1, queries, 32, device="cuda")
    actual = module(hidden, positions[:, offset : offset + queries], None, None, positions, padding, metadata=meta)
    assert actual.region_ids.stride(1) == 0
    assert actual.region_ids[0, 0, 0, 0].item() == 17 * 512
    assert actual.region_ids[0, 0, -1, :512].tolist() == list(range(17 * 512, 18 * 512))
    assert actual.region_ids[0, 0, -1, 512].item() == meta.region_count


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    error = actual.float() - expected.float()
    scale = expected.float().square().mean().sqrt().clamp_min(1e-12)
    return {
        "relative_l2": (error.square().mean().sqrt() / scale).item(),
        "max_over_rms": (error.abs().max() / scale).item(),
        "max_abs": error.abs().max().item(),
        "different_fraction": (actual != expected).float().mean().item(),
        "over_1pct_rms_fraction": (error.abs() > 0.01 * scale).float().mean().item(),
    }


def _time_cuda(call: Callable[[], object]) -> float:
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(20):
        call()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) / 20


@pytest.mark.parametrize("sequence", [4096, 8192])
@pytest.mark.parametrize("kind", ["residual", "residual_cast", "query", "key", "final"])
def test_norm_cuda_forward_backward_recompute(sequence: int, kind: str, record_property) -> None:
    torch.manual_seed(983)
    shape = (1, sequence, 64, 192) if kind == "query" else (1, sequence, 4096)
    if kind == "key":
        shape = (1, sequence, 4, 192)
    dtype = torch.float32 if kind in ("residual", "residual_cast") else torch.bfloat16
    output_dtype = torch.bfloat16 if kind == "residual_cast" else dtype
    reference = Step3p5RMSNorm(shape[-1]).cuda()
    candidate = Step5RMSNorm(shape[-1], compile_norm=True).cuda()
    with torch.no_grad():
        reference.weight.copy_(torch.linspace(-0.7, 0.7, shape[-1], device="cuda") + 0.00123)
        candidate.weight.copy_(reference.weight)
    original = torch.randn(shape, device="cuda", dtype=dtype)
    if kind == "key":
        # Actual key projection is a strided slice of interleaved GQA QKV.
        original = torch.randn(1, sequence, 4, 18, 192, device="cuda", dtype=dtype)[..., 16, :]
    upstream = torch.randn_like(original, dtype=output_dtype)
    expected_input = original.detach().requires_grad_()
    actual_input = original.detach().requires_grad_()
    expected = reference(expected_input).to(output_dtype)
    actual = candidate(actual_input, output_dtype=output_dtype)
    expected_grads = torch.autograd.grad(expected, (expected_input, reference.weight), upstream)
    actual_grads = torch.autograd.grad(actual, (actual_input, candidate.weight), upstream)
    rows = {}
    for name, observed, control in zip(
        ("output", "input_gradient", "weight_gradient"), (actual, *actual_grads), (expected, *expected_grads)
    ):
        assert torch.isfinite(observed).all()
        rows[name] = _metrics(observed, control)
        assert rows[name]["relative_l2"] < 0.002, (name, rows[name])
        assert rows[name]["max_over_rms"] < 0.05, (name, rows[name])
    with torch.no_grad():
        inference = candidate(original, output_dtype=output_dtype)
    rows["grad_no_grad"] = _metrics(inference, actual.detach())
    assert torch.equal(inference, actual.detach()), rows["grad_no_grad"]
    replay_input = original.detach().requires_grad_()
    replay = checkpoint(candidate, replay_input, use_reentrant=False, output_dtype=output_dtype)
    replay_grads = torch.autograd.grad(replay, (replay_input, candidate.weight), upstream)
    for observed, control in zip((replay, *replay_grads), (actual, *actual_grads)):
        torch.testing.assert_close(observed, control, rtol=0, atol=0)
    record_property("metrics", json.dumps(rows))

    # Bracket the optimized call with the same eager control; never assert timing.
    def baseline_step() -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(
            reference(expected_input).to(output_dtype), (expected_input, reference.weight), upstream
        )

    def candidate_step() -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(
            candidate(actual_input, output_dtype=output_dtype), (actual_input, candidate.weight), upstream
        )

    record_property(
        "cuda_fwb_ms",
        json.dumps(
            {
                "baseline_before": _time_cuda(baseline_step),
                "compiled": _time_cuda(candidate_step),
                "baseline_after": _time_cuda(baseline_step),
            }
        ),
    )
    peaks = {}
    for name, call in [("baseline", baseline_step), ("compiled", candidate_step)]:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        call()
        torch.cuda.synchronize()
        peaks[name] = torch.cuda.max_memory_allocated() - before
    record_property("incremental_fwb_peak_bytes", json.dumps(peaks))


def _metrics_qnorm_layout(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    """Compare tensors with the same semantic shape.

    Args:
        actual: Observed query [batch, sequence, heads, dim], QKV gradient
            [batch, sequence, groups, heads_per_group + 2, dim], or scale [dim].
        reference: Independent reference with the same shape and dtype.

    Returns:
        Relative L2, maximum error over reference RMS, and absolute error.
    """
    difference = actual.float() - reference.float()
    rms = reference.float().square().mean().sqrt().clamp_min(1e-12)
    return {
        "relative_l2": (difference.square().mean().sqrt() / rms).item(),
        "max_over_rms": (difference.abs().max() / rms).item(),
        "max_abs": difference.abs().max().item(),
        "different_fraction": (actual != reference).float().mean().item(),
    }


@pytest.mark.parametrize("sequence", [4096, 8192])
def test_qnorm_grouped_cuda_math_replay_and_no_input_clone(
    sequence: int, record_property: Callable[[str, str], None]
) -> None:
    torch.manual_seed(1139)
    module = Step5RMSNorm(192, compile_norm=True).cuda()
    with torch.no_grad():
        module.weight.copy_(torch.linspace(-0.7, 0.7, 192, device="cuda") + 0.00123)
    projection = torch.randn(1, sequence, 4, 18, 192, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    query = projection[..., :16, :]
    assert not query.is_contiguous()
    normalized = module(query)
    assert normalized.is_contiguous()
    actual = normalized.flatten(2, 3)
    assert actual.untyped_storage().data_ptr() == normalized.untyped_storage().data_ptr()
    assert module.weight.dtype == torch.float32 and actual.dtype == torch.bfloat16
    upstream = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad(actual, (projection, module.weight), upstream)

    reference_projection = projection.detach().clone().requires_grad_()
    reference_weight = module.weight.detach().clone().requires_grad_()
    expected = (
        F.rms_norm(reference_projection[..., :16, :].float(), (192,), reference_weight + 1, module.variance_epsilon)
        .bfloat16()
        .flatten(2, 3)
    )
    reference_gradients = torch.autograd.grad(expected, (reference_projection, reference_weight), upstream)
    metrics = {}
    for name, observed, reference in zip(
        ("output", "qkv_gradient", "weight_gradient"), (actual, *actual_gradients), (expected, *reference_gradients)
    ):
        assert torch.isfinite(observed).all()
        metrics[name] = _metrics_qnorm_layout(observed, reference)
        # Existing Step5 norm-fusion gates; neither is widened for this layout.
        assert metrics[name]["relative_l2"] < 0.002, (name, metrics[name])
        assert metrics[name]["max_over_rms"] < 0.05, (name, metrics[name])
    assert torch.count_nonzero(actual_gradients[0][..., 16:, :]) == 0

    # The previous compiled two-dimensional path is a layout control, not the
    # mathematical oracle. It deliberately retains the pre-norm Q copy.
    legacy = module(query.reshape(1, sequence, 64, 192))
    legacy_gradients = torch.autograd.grad(legacy, (projection, module.weight), upstream)
    for name, observed, reference in zip(
        ("legacy_output", "legacy_qkv_gradient", "legacy_weight_gradient"),
        (actual, *actual_gradients),
        (legacy, *legacy_gradients),
    ):
        metrics[name] = _metrics_qnorm_layout(observed, reference)
        assert metrics[name]["relative_l2"] < 0.002, (name, metrics[name])
        assert metrics[name]["max_over_rms"] < 0.05, (name, metrics[name])

    replay = checkpoint(module, query, use_reentrant=False).flatten(2, 3)
    replay_gradients = torch.autograd.grad(replay, (projection, module.weight), upstream)
    repeat = module(query).flatten(2, 3)
    repeat_gradients = torch.autograd.grad(repeat, (projection, module.weight), upstream)
    with torch.no_grad():
        inference = module(query).flatten(2, 3)
    for observed, reference in zip(
        (replay, *replay_gradients, repeat, *repeat_gradients, inference),
        (actual, *actual_gradients, actual, *actual_gradients, actual),
    ):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)

    traces = {}
    peaks = {}
    for label in ("legacy", "grouped"):
        with torch.no_grad():
            # Compile no-grad specializations before observing actual execution.
            for _ in range(2):
                if label == "legacy":
                    result = module(query.reshape(1, sequence, 64, 192))
                else:
                    result = module(query).flatten(2, 3)
                del result
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as trace:
                if label == "legacy":
                    result = module(query.reshape(1, sequence, 64, 192))
                else:
                    result = module(query).flatten(2, 3)
                torch.cuda.synchronize()
            peaks[label] = torch.cuda.max_memory_allocated() - before
            del result
        events = trace.events()
        clones = [event.input_shapes for event in events if event.name == "aten::clone"]
        compiled = sorted({event.name for event in events if "Compiled" in event.name})
        kernels = sorted({event.name for event in events if event.device_type == torch.autograd.DeviceType.CUDA})
        traces[label] = {"clone_shapes": clones, "compiled_events": compiled, "cuda_kernels": kernels}
        assert compiled and kernels, traces[label]
    assert traces["legacy"]["clone_shapes"], traces
    assert not traces["grouped"]["clone_shapes"], traces
    # This proves materialization elimination in the executed compiled entry,
    # independently of the CPU/FakeTensor storage checks.
    expected_copy_bytes = query.numel() * query.element_size()
    assert peaks["legacy"] - peaks["grouped"] >= expected_copy_bytes, peaks
    record_property("numerical_metrics", json.dumps(metrics))
    record_property("compiled_execution", json.dumps(traces))
    record_property("incremental_forward_peak_bytes", json.dumps(peaks))


@pytest.mark.parametrize("sequence", [4096, 8192])
@pytest.mark.parametrize(
    "span,heads,dim", [(32, 16, 256), (32, 1, 256), (64, 64, 192), (64, 4, 192), (192, 64, 192), (192, 4, 192)]
)
def test_rope_cuda_long_positions_packed_backward_recompute(
    sequence: int, span: int, heads: int, dim: int, record_property
) -> None:
    torch.manual_seed(985)
    config = Step5Config()
    config.rope_theta = 10000.0 if span == 192 else 10000000.0
    rope = Step5RotaryEmbedding(config, 0, span=span).cuda()
    positions = torch.arange(sequence, device="cuda").unsqueeze(0) + 131072 - sequence
    positions[:, sequence // 2 :] = torch.arange(sequence - sequence // 2, device="cuda")
    hidden = torch.randn(1, sequence, heads, dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    half = span // 2
    # Original inference arithmetic and explicit BF16 table, kept outside candidate helpers.
    inverse = 1.0 / config.rope_theta ** (torch.arange(0, span, 2, device="cuda", dtype=torch.float32) / span)
    phases = positions.float().unsqueeze(-1) * inverse
    cos = phases.cos().bfloat16().float().unsqueeze(-2)
    sin = phases.sin().bfloat16().float().unsqueeze(-2)
    left, right = hidden[..., :half].float(), hidden[..., half:span].float()
    expected = torch.cat(
        ((left * cos - right * sin).bfloat16(), (left * sin + right * cos).bfloat16(), hidden[..., span:]), -1
    )
    actual = rope(hidden, positions)
    upstream = torch.randn_like(hidden)
    expected_grad = torch.autograd.grad(expected, hidden, upstream)[0]
    actual_grad = torch.autograd.grad(actual, hidden, upstream)[0]
    rows = {}
    for name, observed, reference in [("output", actual, expected), ("input_gradient", actual_grad, expected_grad)]:
        delta = observed.float() - reference.float()
        rms = reference.float().square().mean().sqrt().clamp_min(1e-12)
        rows[name] = {
            "relative_l2": (delta.square().mean().sqrt() / rms).item(),
            "max_over_rms": (delta.abs().max() / rms).item(),
            "max_abs": delta.abs().max().item(),
        }
        assert torch.isfinite(observed).all()
        assert rows[name]["relative_l2"] < 0.002, rows[name]
        assert rows[name]["max_over_rms"] < 0.05, rows[name]
    with torch.no_grad():
        no_grad = rope(hidden, positions)
    assert torch.equal(actual.detach(), no_grad)
    replay = checkpoint(rope, hidden, positions, use_reentrant=False)
    replay_grad = torch.autograd.grad(replay, hidden, upstream)[0]
    torch.testing.assert_close(replay, actual, rtol=0, atol=0)
    torch.testing.assert_close(replay_grad, actual_grad, rtol=0, atol=0)
    assert torch.equal(actual[..., span:], hidden[..., span:])
    record_property("metrics", json.dumps(rows))
    # Application-only bracket: table generation is unchanged and excluded on both sides.
    from nemo_automodel.components.models.step5.rope import _apply_rotary_compiled

    def baseline_step() -> torch.Tensor:
        a, b = hidden[..., :half].float(), hidden[..., half:span].float()
        output = torch.cat(((a * cos - b * sin).bfloat16(), (a * sin + b * cos).bfloat16(), hidden[..., span:]), -1)
        return torch.autograd.grad(output, hidden, upstream)[0]

    def candidate_step() -> torch.Tensor:
        output = _apply_rotary_compiled(hidden, cos, sin, span)
        return torch.autograd.grad(output, hidden, upstream)[0]

    record_property(
        "cuda_apply_fwb_ms",
        json.dumps(
            {
                "baseline_before": _time_cuda(baseline_step),
                "compiled": _time_cuda(candidate_step),
                "baseline_after": _time_cuda(baseline_step),
            }
        ),
    )
    peaks = {}
    for name, call in [("baseline", baseline_step), ("compiled", candidate_step)]:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        call()
        torch.cuda.synchronize()
        peaks[name] = torch.cuda.max_memory_allocated() - before
    record_property("incremental_fwb_peak_bytes", json.dumps(peaks))


@pytest.mark.parametrize("heads", [1, 16])
@pytest.mark.parametrize("sequence", [0, 1])
def test_rope_cuda_short_indexer_projection(heads: int, sequence: int) -> None:
    config = Step5Config()
    rope = Step5RotaryEmbedding(config, 3, span=32).cuda()
    hidden = torch.randn(1, sequence, heads, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    positions = torch.arange(sequence, device="cuda").unsqueeze(0) + 131071
    actual = rope(hidden, positions)
    assert actual.shape == hidden.shape
    assert actual.dtype == hidden.dtype
    if sequence == 0:
        assert actual is hidden
    upstream = torch.randn_like(hidden)
    actual_grad = torch.autograd.grad(actual, hidden, upstream)[0]
    with torch.no_grad():
        no_grad = rope(hidden, positions)
    torch.testing.assert_close(no_grad, actual, rtol=0, atol=0)
    replay = checkpoint(rope, hidden, positions, use_reentrant=False)
    replay_grad = torch.autograd.grad(replay, hidden, upstream)[0]
    torch.testing.assert_close(replay, actual, rtol=0, atol=0)
    torch.testing.assert_close(replay_grad, actual_grad, rtol=0, atol=0)


@pytest.mark.parametrize("span", [64, 192])
@pytest.mark.parametrize("sequence", [0, 1, 4096, 8192])
def test_paired_rope_cuda_output_qkv_gradient_and_recompute(sequence: int, span: int, record_property) -> None:
    torch.manual_seed(988)
    config = Step5Config()
    config.rope_theta = 10000000.0 if span == 64 else 10000.0
    rope = Step5RotaryEmbedding(config, 3 if span == 64 else 0, span=span).cuda()
    source = torch.randn(1, sequence, 4, 18, 192, device="cuda", dtype=torch.bfloat16)
    expected_input = source.clone().requires_grad_()
    actual_input = source.clone().requires_grad_()
    positions = torch.arange(sequence, device="cuda").unsqueeze(0) + 131072 - sequence
    positions[:, sequence // 2 :] = torch.arange(sequence - sequence // 2, device="cuda")

    def qk(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Original interleaved QKV source: key is a strided view.
        return value[..., :16, :].reshape(1, sequence, 64, 192), value[..., 16, :]

    expected_q, expected_k = qk(expected_input)
    expected = rope(expected_q, positions), rope(expected_k, positions)
    actual = rope.forward_pair(*qk(actual_input), positions)
    upstream = tuple(torch.randn_like(value) for value in expected)
    expected_grad = torch.autograd.grad(expected, expected_input, upstream)[0]
    actual_grad = torch.autograd.grad(actual, actual_input, upstream)[0]
    for observed, reference in zip((*actual, actual_grad), (*expected, expected_grad)):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    with torch.no_grad():
        no_grad = rope.forward_pair(*qk(source), positions)
    for observed, reference in zip(no_grad, actual):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    replay_input = source.clone().requires_grad_()
    replay = checkpoint(rope.forward_pair, *qk(replay_input), positions, use_reentrant=False)
    replay_grad = torch.autograd.grad(replay, replay_input, upstream)[0]
    for observed, reference in zip((*replay, replay_grad), (*actual, actual_grad)):
        torch.testing.assert_close(observed, reference, rtol=0, atol=0)
    if sequence < 4096:
        return

    def baseline_step() -> torch.Tensor:
        query, key = qk(expected_input)
        result = rope(query, positions), rope(key, positions)
        return torch.autograd.grad(result, expected_input, upstream)[0]

    def paired_step() -> torch.Tensor:
        result = rope.forward_pair(*qk(actual_input), positions)
        return torch.autograd.grad(result, actual_input, upstream)[0]

    record_property(
        "cuda_pair_fwb_ms",
        json.dumps(
            {
                "baseline_before": _time_cuda(baseline_step),
                "paired": _time_cuda(paired_step),
                "baseline_after": _time_cuda(baseline_step),
            }
        ),
    )
    peaks = {}
    for name, call in [("baseline", baseline_step), ("paired", paired_step)]:
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        call()
        torch.cuda.synchronize()
        peaks[name] = torch.cuda.max_memory_allocated() - before
    record_property("incremental_fwb_peak_bytes", json.dumps(peaks))


class Fixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.indexer = Step5SparseIndexer(tiny_config(), 3, dtype=torch.float32)
        self.output = nn.Linear(32, 32, bias=False, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, empty: bool) -> torch.Tensor:
        """Exercise the frozen selector and return differentiable projections.

        Args:
            hidden: FP32 activations shaped ``[batch, sequence, 32]``.
            empty: Whether the selector receives an empty sequence while the
                trainable projection still receives all input tokens.

        Returns:
            FP32 projected activations shaped ``[batch, sequence, 32]``.
        """
        positions = torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0)
        padding = torch.zeros_like(positions, dtype=torch.bool)
        with torch.no_grad():
            projected = hidden[:, :0] if empty else hidden
            projected_positions = positions[:, :0] if empty else positions
            keys, logits = self.indexer.project_keys(projected, projected_positions)
            self.indexer(projected, projected_positions, keys, logits, positions, padding)
        return self.output(hidden)


@pytest.fixture(scope="module")
def world():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    yield dist.get_world_size()
    dist.destroy_process_group()


def _input(rank: int, step: int) -> torch.Tensor:
    return torch.arange(4 * 32, device="cuda").reshape(1, 4, 32).float() / 137 + rank / 7 + step / 11


@_requires_torchrun
@pytest.mark.parametrize("hsdp", [False, True])
@pytest.mark.parametrize("ac", [False, True])
def test_indexer_stays_immutable_across_updates_and_distributed_layouts(world, hsdp, ac):
    torch.manual_seed(1031)
    reference = Fixture().cuda()
    model = copy.deepcopy(reference)
    initial = {name: value.detach().clone() for name, value in model.indexer.named_parameters()}
    shape = (2, world // 2) if hsdp else (world,)
    mesh = init_device_mesh("cuda", shape, mesh_dim_names=("replicate", "shard") if hsdp else ("shard",))
    # A real FP32 group contains both frozen selector and trainable output
    # parameters, which exercises the mixed-requires-grad storage contract.
    fully_shard_by_dtype(
        model,
        mesh,
        MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32),
        None,
        reshard_after_forward=True,
    )
    _apply_trainability_policy(model, peft_enabled=False, freeze_config=None, strict=False)
    optimizer = torch.optim.AdamW(_trainable_params_or_groups(model, []), lr=0.01, weight_decay=0.1)
    reference_optimizer = torch.optim.AdamW(_trainable_params_or_groups(reference, []), lr=0.01, weight_decay=0.1)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
        expected = reference.output(torch.cat([_input(rank, step) for rank in range(world)]))
        expected.square().mean().backward()
        hidden = _input(dist.get_rank(), step).requires_grad_()
        empty = bool(dist.get_rank() % 2)
        output = checkpoint(model, hidden, empty, use_reentrant=False) if ac else model(hidden, empty)
        output.square().mean().backward()
        assert output.dtype == torch.float32
        gradient = model.output.weight.grad.full_tensor()
        torch.testing.assert_close(gradient, reference.output.weight.grad, atol=2e-7, rtol=2e-5)
        torch.testing.assert_close(gradient.norm(), reference.output.weight.grad.norm(), atol=2e-7, rtol=2e-5)
        for name, parameter in model.indexer.named_parameters():
            assert not parameter.requires_grad and parameter.grad is None, name
            assert parameter not in optimizer.state, name
            assert isinstance(parameter, DTensor) and parameter.dtype == torch.float32, name
            torch.testing.assert_close(parameter.full_tensor(), initial[name], atol=0, rtol=0)
        optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(model.output.weight.full_tensor(), reference.output.weight, atol=2e-7, rtol=2e-5)
        for name, parameter in model.indexer.named_parameters():
            torch.testing.assert_close(parameter.full_tensor(), initial[name], atol=0, rtol=0)
            assert parameter not in optimizer.state, name

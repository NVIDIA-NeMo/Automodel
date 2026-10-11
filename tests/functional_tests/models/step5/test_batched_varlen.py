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

"""CUDA B2 causal-prefix attention, retaining each packed row's boundaries."""

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from nemo_automodel.components.models.step5 import fa4
from nemo_automodel.components.models.step5.sparse import RegionSelection, region_metadata

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


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
                [torch.arange(n) for n in valid_lengths]
                + [torch.zeros(keys - sum(valid_lengths), dtype=torch.long)]
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
            reference_rows.append(
                one_value[None].expand(queries, -1, -1) + (ref_q[row].sum() + ref_k[row].sum()) * 0
            )
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

    def existing_fallback(window: int | None, bitmap: bool, slots: int, provider_groups: int) -> None:
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

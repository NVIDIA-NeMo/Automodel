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

"""H100 D192 native SWA window kernels, halo views and document boundaries."""

import pytest
import torch
from torch.nn import functional as F

from nemo_automodel.components.models.step5 import fa4
from nemo_automodel.components.models.step5.sparse import region_metadata


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

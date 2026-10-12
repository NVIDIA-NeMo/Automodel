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

"""FA4 head ownership across long-history membership and payload boundaries."""

from collections.abc import Callable

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from nemo_automodel.components.models.step5 import fa4
from nemo_automodel.components.models.step5.sparse import RegionSelection


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
    ) -> tuple[Callable, type, Callable]:
        dispatch.append((use_bitmap, heads_per_group))
        return original(window, use_bitmap, slots, provider_groups, heads_per_group=heads_per_group)

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

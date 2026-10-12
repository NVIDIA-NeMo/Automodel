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

"""Execute the production mask's scalar contract with an independent token oracle."""

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.models.step5 import fa4
from nemo_automodel.components.models.step5.sparse import region_metadata


@pytest.fixture
def scalar_cute(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Interpret CuTe scalar operations on CPU; this does not validate CUDA compilation."""

    def identity(value: object) -> object:
        return value

    def to_bool(value: object, dtype: type) -> bool:
        return bool(value)

    modules = {
        "flash_attn.cute.interface": SimpleNamespace(flash_attn_func=identity),
        "flash_attn.cute.block_sparsity": SimpleNamespace(BlockSparseTensorsTorch=object),
        "cutlass": SimpleNamespace(
            const_expr=identity, Int32=int, Boolean=bool, min=min, max=max, range_constexpr=range
        ),
        "cutlass.cute": SimpleNamespace(jit=identity),
        "flash_attn.cute.utils": SimpleNamespace(scalar_to_ssa=to_bool),
    }

    def safe_import(name: str) -> tuple[bool, object]:
        return True, modules[name]

    fa4._load_fa4.cache_clear()
    monkeypatch.setattr(fa4, "safe_import", safe_import)
    yield
    fa4._load_fa4.cache_clear()


@pytest.mark.parametrize("region_size", [3, 8])
@pytest.mark.parametrize("bitmap", [False, True])
@pytest.mark.parametrize("head_layout", [False, True])
def test_actual_mask_matches_independent_membership_and_bounds(
    scalar_cute: None, region_size: int, bitmap: bool, head_layout: bool
) -> None:
    length, groups, slots = region_size * 7, 2, 3
    positions = torch.arange(length)[None]
    documents = torch.ones_like(positions)
    padding = torch.zeros_like(positions, dtype=torch.bool)
    token_regions = positions // region_size
    ids = torch.full((1, groups, length, slots), 7, dtype=torch.int32)
    routes: list[list[set[int]]] = [[], []]
    for group in range(groups):
        for query in range(length):
            current = query // region_size
            history = list(range(current))
            selected = history[:2] if group == 0 else history[-2:]
            selected = sorted(set(selected + [current]))
            routes[group].append(set(selected))
            ids[0, group, query, : len(selected)] = torch.tensor(selected)
    membership = ids
    if bitmap:
        membership = torch.zeros((1, groups, length, 1), dtype=torch.int32)
        for group in range(groups):
            for query in range(length):
                membership[0, group, query, 0] = sum(1 << v for v in routes[group][query])
    aux = [positions, positions, documents, documents, padding, membership, token_regions]
    seq = SimpleNamespace(seqlen_q=length, seqlen_k=length)
    masks = [
        fa4._load_fa4(
            None, bitmap, slots, groups, heads_per_group=16 if head_layout else 0, canonical_region_size=width
        )[2]
        for width in (0, region_size)
    ]
    for group in range(groups):
        batch, head = ([0], [group * 16 + 15]) if head_layout else ([group], [15])
        for query in range(length + 2):
            for key in range(length + 2):
                expected = (
                    query < length
                    and key < length
                    and key <= query
                    and key // region_size in routes[group][min(query, length - 1)]
                )
                assert all(mask(batch, head, [query], [key], seq, aux) == expected for mask in masks)


@pytest.mark.parametrize("defect", ["region_size", "document", "padding"])
def test_scalar_oracle_rejects_wrong_specialization(scalar_cute: None, defect: str) -> None:
    length = 24
    positions = torch.arange(length)[None]
    docs = torch.ones_like(positions)
    padding = torch.zeros_like(positions, dtype=torch.bool)
    ids = torch.tensor([0, 2, 3], dtype=torch.int32)[None, None, None].expand(1, 1, length, 3)
    if defect == "document":
        docs[:, 8:] += 1
    if defect == "padding":
        padding[:, 0] = True
    aux = [positions, positions, docs, docs, padding, ids, positions // 8]
    seq = SimpleNamespace(seqlen_q=length, seqlen_k=length)
    mask = fa4._load_fa4(None, False, 3, 1, canonical_region_size=3 if defect == "region_size" else 8)[2]

    def assert_correct() -> None:
        for query in range(length):
            for key in range(length):
                expected = (
                    key <= query and not padding[0, key] and docs[0, query] == docs[0, key] and key // 8 in (0, 2, 3)
                )
                assert mask([0], [0], [query], [key], seq, aux) == bool(expected)

    with pytest.raises(AssertionError):
        assert_correct()


@pytest.mark.parametrize("region_size", [3, 8])
def test_proof_uses_actual_region_width_and_remains_opt_in(region_size: int) -> None:
    positions = torch.arange(79)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    kwargs = dict(query_start=0, queries=79, region_size=region_size, topk=2)
    assert region_metadata(positions, padding, **kwargs).canonical_region_size == 0
    assert (
        region_metadata(positions, padding, csa_forward_full_blocks=True, **kwargs).canonical_region_size == region_size
    )
    assert (
        region_metadata(positions, padding, csa_backward_full_blocks=True, **kwargs).canonical_region_size
        == region_size
    )

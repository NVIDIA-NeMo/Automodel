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

"""Narrow temporary IDs without changing learned selection or sentinel ordering."""

from collections.abc import Sequence
from dataclasses import replace

import pytest
import torch

from nemo_automodel.components.models.step5.sparse import region_metadata
from tests.unit_tests.models.step5.test_sparse_metadata import _indexer, _oracle


def _capture_integer_work(monkeypatch: pytest.MonkeyPatch) -> tuple[list[torch.dtype], list[torch.dtype]]:
    """Observe real value-topk and concatenation dtypes without replacing their math."""
    topk_dtypes: list[torch.dtype] = []
    concat_dtypes: list[torch.dtype] = []
    original_topk = torch.Tensor.topk
    original_cat = torch.cat

    def topk(
        tensor: torch.Tensor, k: int, dim: int = -1, largest: bool = True, sorted: bool = True
    ) -> torch.return_types.topk:
        """Run the original top-k and record integer inputs.

        Args:
            tensor: Values of shape [..., regions] with arbitrary leading dimensions.
            k: Number of retained values.
            dim: Selected tensor axis.
            largest: Select greatest values when true.
            sorted: Sort the returned values when true.

        Returns:
            Original values and indices with the selected axis shortened to k.
        """
        if not tensor.is_floating_point():
            topk_dtypes.append(tensor.dtype)
        return original_topk(tensor, k, dim=dim, largest=largest, sorted=sorted)

    def cat(tensors: Sequence[torch.Tensor], dim: int = 0) -> torch.Tensor:
        """Run the original concatenation and record integer result dtypes.

        Args:
            tensors: Tensors with equal shape except along the concatenated axis.
            dim: Concatenated axis.

        Returns:
            Original concatenation, allocating independent output storage.
        """
        result = original_cat(tensors, dim=dim)
        if all(not tensor.is_floating_point() for tensor in tensors):
            concat_dtypes.append(result.dtype)
        return result

    monkeypatch.setattr(torch.Tensor, "topk", topk)
    monkeypatch.setattr(torch, "cat", cat)
    return topk_dtypes, concat_dtypes


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("offset,length", [(0, 35), (3, 13), (17, 18)])
def test_indexer_temporary_ids_preserve_independent_selection(
    monkeypatch: pytest.MonkeyPatch, tied: bool, offset: int, length: int
) -> None:
    module = _indexer()
    if tied:
        with torch.no_grad():
            module.w_proj.weight.zero_()
    positions = torch.cat([torch.arange(n) for n in (3, 11, 1, 20)])[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    hidden = torch.randn(1, 35, 32)
    keys, logits = module.project_keys(hidden, positions)
    local, local_positions = hidden[:, offset : offset + length], positions[:, offset : offset + length]
    meta = region_metadata(positions, padding, query_start=offset, queries=length, region_size=2, topk=2)
    expected = _oracle(module, local, local_positions, keys, logits, positions, padding, offset=offset)
    original_current = meta.current.clone()
    topk_dtypes, concat_dtypes = _capture_integer_work(monkeypatch)
    actual = module(local, local_positions, keys, logits, positions, padding, query_start=offset, metadata=meta)
    torch.testing.assert_close(actual.region_ids, expected, rtol=0, atol=0)
    assert topk_dtypes and set(topk_dtypes) == {torch.int32}
    assert concat_dtypes and set(concat_dtypes) == {torch.int32}
    assert actual.region_ids.dtype == torch.int32 and actual.region_ids.shape[-1] == 3
    assert meta.current.dtype == torch.int64
    torch.testing.assert_close(meta.current, original_current, rtol=0, atol=0)
    assert bool(meta.sparse_queries.any())


@pytest.mark.parametrize("sentinel", [2**31 - 2, 2**31 - 1, 2**31, 2**31 + 1])
def test_indexer_temporary_sentinel_width_boundary(monkeypatch: pytest.MonkeyPatch, sentinel: int) -> None:
    """Exercise real forward's sentinel branch with small referenced region storage.

    Only the unused ID sentinel is inflated; all score tiles still reference the
    original small region tables. This is a temporary-ID component boundary
    test, not evidence for billion-token region_metadata or model execution.
    The wide template on the fallback case preserves the pre-existing caller's
    storage rather than claiming the normal int32 route can hold this sentinel.
    """
    module = _indexer()
    with torch.no_grad():
        module.w_proj.weight.zero_()
    positions = torch.arange(12)[None]
    padding = torch.zeros_like(positions, dtype=torch.bool)
    hidden = torch.randn(1, 12, 32)
    keys, logits = module.project_keys(hidden, positions)
    meta = region_metadata(positions, padding, query_start=0, queries=12, region_size=2, topk=2)
    assert meta.region_sources is not None
    expected = _oracle(module, hidden, positions, keys, logits, positions, padding, offset=0).long()
    expected.masked_fill_(expected == meta.region_count, sentinel)
    dtype = torch.int32 if sentinel <= 2**31 - 1 else torch.int64
    template = meta.dense_region_ids.long()
    template.masked_fill_(template == meta.region_count, sentinel)
    inflated = replace(meta, region_count=sentinel, dense_region_ids=template.to(dtype))
    topk_dtypes, concat_dtypes = _capture_integer_work(monkeypatch)
    actual = module(hidden, positions, keys, logits, positions, padding, metadata=inflated)
    torch.testing.assert_close(actual.region_ids.long(), expected, rtol=0, atol=0)
    assert topk_dtypes and set(topk_dtypes) == {dtype}
    assert concat_dtypes and set(concat_dtypes) == {dtype}
    assert (actual.region_ids[0, 0, 0] == sentinel).sum() == 2
    assert actual.region_ids[0, 0, -1].tolist() == [0, 1, 5]

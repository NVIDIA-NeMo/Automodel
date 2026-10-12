# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent packed-boundary oracle and cache lifetime checks."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from torch._ops import OpOverload
from torch.utils._python_dispatch import TorchDispatchMode

from nemo_automodel.components.models.glm5_next import cp


class _NoTensorOperations(TorchDispatchMode):
    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> None:
        raise AssertionError(f"Cached document ranges performed {func}")


def _expected_ranges(row: list[int]) -> tuple[tuple[int, int], ...]:
    ranges = []
    start = 0
    while start < len(row):
        end = start + 1
        while end < len(row) and row[end] == row[start]:
            end += 1
        if row[start] > 0:
            ranges.append((start, end))
        start = end
    return tuple(ranges)


@pytest.mark.parametrize(
    "rows",
    [
        [[1, 1, 2, 2, 0, 0], [0, 3, 3, 0, 3, 3]],
        [[0, 0, 0, 0, 0, 0], [7, 7, 7, 7, 7, 7]],
        [[-1, -1, 5, 5, 0, 6], [1, 2, 3, 4, 5, 6]],
        [[], []],
        [[1] * 4096, [1] * 4096],
        [[1] * 2049 + [0] * 17 + [2] * 2030, [0] * 512 + [4] * 3584],
    ],
)
@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))],
)
def test_document_ranges_match_independent_row_oracle_and_cache_without_tensor_ops(
    rows: list[list[int]], device: str
) -> None:
    ids = torch.tensor(rows, dtype=torch.int32, device=device)
    original = ids.clone()
    context = cp.Glm5NextPackedContext(ids, original_seq_len=ids.shape[1])
    expected = tuple(_expected_ranges(row) for row in rows)
    with patch.object(cp, "segment_cu_seqlens", wraps=cp.segment_cu_seqlens) as segmented:
        actual = tuple(context.row_document_ranges(row) for row in range(len(rows)))
        assert segmented.call_count == len(rows)
    assert actual == expected
    torch.testing.assert_close(ids, original, rtol=0, atol=0)
    # FSDP rebuilds dataclass kwargs. Global document ranges remain global when
    # a context describes a different local CP interval; this is metadata only.
    rebuilt = replace(context, cp_size=2, seq_start=ids.shape[1] // 2)
    with _NoTensorOperations():
        for _ in range(3):
            for row in range(len(rows)):
                assert context.row_document_ranges(row) == expected[row]
                assert rebuilt.row_document_ranges(row) == expected[row]
    fresh = cp.Glm5NextPackedContext(ids.clone())
    assert fresh._document_ranges is not context._document_ranges
    assert not fresh._document_ranges


def test_merging_documents_or_rows_fails_independent_boundary_oracle() -> None:
    expected = _expected_ranges([1, 1, 2, 2])
    assert expected == ((0, 2), (2, 4))
    assert _expected_ranges([1, 1, 1, 1]) != expected
    separate_rows = (((0, 4096),), ((0, 4096),))
    flattened = (_expected_ranges([1] * 8192),)
    assert flattened != separate_rows

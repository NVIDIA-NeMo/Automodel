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


@pytest.mark.parametrize("sequence", [0, 1, 4096])
def test_unmasked_boundaries_are_known_without_reading_device_values(sequence: int) -> None:
    from nemo_automodel.components.models.glm5_next.model import _packed_context_from_inputs

    ids = torch.empty((2, sequence), dtype=torch.long, device="meta")
    context = _packed_context_from_inputs(ids, None, None, None, None)
    expected_boundaries = [0, sequence] if sequence else [0]
    expected_ranges = ((0, sequence),) if sequence else ()
    with patch.object(cp, "segment_cu_seqlens", side_effect=AssertionError("Unexpected dynamic boundary scan")):
        for row in range(2):
            assert context.row_document_ranges(row) == expected_ranges
            device_boundaries, cpu_boundaries = context.row_cu_seqlens(row)
            assert device_boundaries.device.type == "meta"
            assert cpu_boundaries.device.type == "cpu"
            assert cpu_boundaries.tolist() == expected_boundaries


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))],
)
@pytest.mark.parametrize("sequence", [0, 1, 4096])
def test_unmasked_boundary_cache_matches_explicit_single_document(device: str, sequence: int) -> None:
    from nemo_automodel.components.models.glm5_next.model import _packed_context_from_inputs

    ids = torch.zeros((2, sequence), dtype=torch.long, device=device)
    fast = _packed_context_from_inputs(ids, None, None, None, None)
    explicit = _packed_context_from_inputs(ids, torch.ones_like(ids), None, None, None)
    torch.testing.assert_close(fast.doc_ids, explicit.doc_ids, rtol=0, atol=0)
    for row in range(2):
        assert fast.row_document_ranges(row) == explicit.row_document_ranges(row)
        for actual, expected in zip(fast.row_cu_seqlens(row), explicit.row_cu_seqlens(row), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("metadata_kind", ["packed_ids", "attention_mask", "cu_seqlens", "padding_mask"])
def test_unmasked_fast_path_preserves_explicit_metadata(metadata_kind: str) -> None:
    from nemo_automodel.components.models.glm5_next.model import _packed_context_from_inputs

    ids = torch.zeros((1, 8), dtype=torch.long)
    documents = torch.tensor([[1, 1, 2, 2, 2, 0, 0, 0]], dtype=torch.int32)
    mask = documents if metadata_kind == "attention_mask" else None
    packed = documents if metadata_kind == "packed_ids" else None
    cu = torch.tensor([0, 2, 5]) if metadata_kind == "cu_seqlens" else None
    padding = documents == 0 if metadata_kind == "padding_mask" else None
    context = _packed_context_from_inputs(ids, mask, padding, cu, packed)
    expected = ((0, 5),) if metadata_kind == "padding_mask" else ((0, 2), (2, 5))
    assert context.row_document_ranges(0) == expected

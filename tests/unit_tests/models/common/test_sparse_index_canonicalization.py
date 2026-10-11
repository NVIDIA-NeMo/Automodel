# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent sparse-index ordering and invalid-entry contracts."""

import pytest
import torch

from nemo_automodel.components.models.common.cudnn_sparse_attention import _compact_and_sort_indices


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("key_count", [0, 1, 7, 17])
def test_sparse_index_canonicalization(dtype: torch.dtype, strided: bool, key_count: int) -> None:
    """Retain duplicate valid keys and normalize every invalid entry to -1."""
    indices = torch.tensor(
        [[-1, -8, -2, -1, -3, -5], [0, 16, 0, 17, 7, 6], [6, 5, 6, 4, 3, 2]],
        dtype=dtype,
    )
    if strided:
        storage = torch.empty(indices.shape[0], indices.shape[1] * 2, dtype=dtype)
        storage[:, ::2] = indices
        indices = storage[:, ::2]
        assert not indices.is_contiguous()
    expected, expected_lengths = [], []
    for row in indices.tolist():
        valid = sorted(key for key in row if 0 <= key < key_count)
        expected_lengths.append(len(valid))
        expected.append(valid + [-1] * (len(row) - len(valid)))
    actual, lengths = _compact_and_sort_indices(indices, key_count)
    assert actual.tolist() == expected
    assert lengths.tolist() == expected_lengths
    assert actual.dtype == lengths.dtype == torch.int32
    assert actual.is_contiguous() and lengths.is_contiguous()

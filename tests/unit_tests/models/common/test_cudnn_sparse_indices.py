# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import pytest
import torch

from nemo_automodel.components.models.common.cudnn_sparse_attention import _compact_and_sort_indices


def _reference(indices: torch.Tensor, key_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The previous compact-then-sort implementation, kept as the oracle."""
    valid = (indices >= 0) & (indices < key_count)
    topk_length = valid.sum(dim=-1, dtype=torch.int32)
    positions = torch.arange(indices.size(-1), device=indices.device).view(1, -1)
    compact_order = torch.where(valid, positions, torch.full_like(positions, indices.size(-1))).argsort(dim=-1)
    indices = torch.gather(indices, -1, compact_order)
    compact_valid = torch.gather(valid, -1, compact_order)
    indices = indices.masked_fill(~compact_valid, -1)
    prefix = positions < topk_length.unsqueeze(-1)
    sort_key = torch.where(prefix, indices, torch.full_like(indices, key_count))
    sort_order = sort_key.argsort(dim=-1)
    indices = torch.gather(indices, -1, sort_order)
    sorted_valid = torch.gather(prefix.expand_as(indices), -1, sort_order)
    return indices.masked_fill(~sorted_valid, -1).to(torch.int32).contiguous(), topk_length.contiguous()


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_single_sort_matches_the_compact_then_sort_reference(dtype: torch.dtype, seed: int) -> None:
    """Ascending valid prefix, -1 suffix and lengths are identical to the two-pass algorithm."""
    generator = torch.Generator().manual_seed(seed)
    key_count = 97
    # Mix of valid coordinates (with duplicates), negative sentinels and out-of-range values.
    indices = torch.randint(-5, key_count + 8, (37, 24), generator=generator).to(dtype)
    indices[3] = -1  # a fully invalid row
    indices[4] = torch.arange(24, dtype=dtype)  # already sorted
    indices[5] = torch.full((24,), 7, dtype=dtype)  # duplicates only
    got, got_len = _compact_and_sort_indices(indices, key_count)
    want, want_len = _reference(indices, key_count)
    assert got.dtype == torch.int32 and got.is_contiguous()
    torch.testing.assert_close(got, want)
    torch.testing.assert_close(got_len, want_len)
    # The valid prefix is ascending and nothing valid follows a -1.
    for row, length in zip(got, got_len.tolist(), strict=True):
        prefix = row[:length]
        assert bool((prefix[1:] >= prefix[:-1]).all()) and bool((prefix >= 0).all())
        assert bool((row[length:] == -1).all())

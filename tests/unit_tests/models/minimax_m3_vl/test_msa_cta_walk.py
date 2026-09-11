# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Host CTA-walk rules behind the MSA backward task tables.

The device picks the walk from the exact task count, which only it knows, while the host launches
``grid_launch_bound`` CTAs from the capacity. Two rules therefore have to hold for *every* count
the capacity admits, not just the ones a fixture happens to produce: the walk covers each task row
exactly once, and the bound is never smaller than the grid the device chooses. Both are pure
integer arithmetic, so they are exhaustible on the CPU; a sampled test cannot see either failure.
"""

from itertools import chain

import pytest

from nemo_automodel.components.models.minimax_m3_vl.kernels.msa_schedule import (
    ROWS_PER_CTA_LARGE,
    chunk_map,
    grid_launch_bound,
    rows_per_cta,
)

_SM_COUNTS = (1, 8, 148)


def _cta_row_interval(bidx: int, num_rows: int, walk_rows: int, num_full_ctas: int, tail_rows: int) -> tuple[int, int]:
    """Host mirror of the [row_lo, row_hi) interval a CTA reads from the descriptor in msa_backward_sm100."""
    if bidx >= num_full_ctas:
        row_lo = num_full_ctas * walk_rows + (bidx - num_full_ctas) * tail_rows
        walk_rows = tail_rows
    else:
        row_lo = bidx * walk_rows
    return row_lo, min(row_lo + walk_rows, num_rows)


def _walk(num_rows: int, num_sms: int) -> list[tuple[int, int]]:
    """Return the [row_lo, row_hi) interval every launched CTA claims, in block index order."""
    walk_rows = rows_per_cta(num_rows)
    num_full_ctas, tail_rows, grid_ctas = chunk_map(num_rows, walk_rows, num_sms)
    return [_cta_row_interval(b, num_rows, walk_rows, num_full_ctas, tail_rows) for b in range(grid_ctas)]


def _task_counts(num_sms: int) -> chain[int]:
    """Task counts reaching all four chunk_map shapes in both rows/CTA regimes, for this SM count.

    The low band crosses ``ROWS_PER_CTA_SWITCH``; the high band is where a wave of the large walk
    length first fits, which is what puts an exact wave and a real tail into the large regime.
    """
    return chain(range(3000), range(num_sms * ROWS_PER_CTA_LARGE, num_sms * ROWS_PER_CTA_LARGE + 512))


@pytest.mark.parametrize("num_sms", _SM_COUNTS)
def test_the_cta_walk_covers_every_task_row_exactly_once(num_sms: int) -> None:
    for num_rows in _task_counts(num_sms):
        intervals = _walk(num_rows, num_sms)
        bounds = [0, *(row_hi for _, row_hi in intervals)]
        # Contiguous, ascending and ending exactly at num_rows == every row owned by one CTA.
        assert [row_lo for row_lo, _ in intervals] == bounds[:-1], num_rows
        assert bounds[-1] == num_rows, num_rows


def test_chunk_map_returns_the_four_documented_walk_shapes() -> None:
    # (num_full_ctas, tail_rows, grid_ctas) for 8 SMs; rows/CTA is 4 below the switch, 64 above.
    assert chunk_map(0, 4, 8) == (0, 1, 0)  # no rows, no launch
    assert chunk_map(32, 4, 8) == (8, 1, 8)  # one exact wave, no tail
    assert chunk_map(49, 4, 8) == (8, 3, 14)  # one wave plus a tail of 3 rows per CTA
    assert chunk_map(1, 4, 8) == (0, 4, 1)  # tail below 3 rows: no wave, every CTA walks 4
    assert chunk_map(2401, 64, 8) == (32, 45, 40)
    assert chunk_map(2560, 64, 8) == (40, 1, 40)


@pytest.mark.parametrize("num_sms", _SM_COUNTS)
# 2000 is below ROWS_PER_CTA_SWITCH: without it a bound that keeps only the large-regime term
# (capacity // 64 + num_sms) still passes every other capacity here.
@pytest.mark.parametrize("capacity", (0, 1, 2000, 20000, 200000))
def test_the_launch_bound_covers_every_count_the_capacity_admits(capacity: int, num_sms: int) -> None:
    worst = max(chunk_map(n, rows_per_cta(n), num_sms)[2] for n in range(capacity + 1))
    assert worst <= grid_launch_bound(capacity, num_sms)

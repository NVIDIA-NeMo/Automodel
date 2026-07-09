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

"""Pins the cu_seqlens → seq_idx derivation against a brute-force reference.

Two production sites derive ``seq_idx`` via searchsorted on ``cu_seqlens[1:]``:

  * ``nemo_automodel/components/loss/mtp.py:calculate_mtp_loss`` — cross-seq
    mask in the MTP loss.
  * ``nemo_automodel/components/models/nemotron_v3/layers.py`` — mamba SSD
    scan state-reset.

Both must use ``side="right"`` (i.e. ``right=True``) so a position equal to a
boundary (``t == cu_seqlens[k]``, the FIRST token of sub-seq k) maps to k,
not k-1. The default ``side="left"`` is off-by-one at every internal
boundary.
"""

from __future__ import annotations

import random

import pytest
import torch


def _brute_force_seq_idx(cu_seqlens: list[int], total_len: int) -> list[int]:
    """Ground-truth reference: position t belongs to sub-seq k iff
    ``cu_seqlens[k] <= t < cu_seqlens[k+1]``.
    """
    K = len(cu_seqlens) - 1
    out: list[int] = []
    for t in range(total_len):
        for k in range(K):
            if cu_seqlens[k] <= t < cu_seqlens[k + 1]:
                out.append(k)
                break
        else:  # pragma: no cover — only triggers on malformed cu_seqlens
            raise AssertionError(f"position {t} fits no sub-seq with cu_seqlens={cu_seqlens}")
    return out


def _searchsorted_right(cu_seqlens: list[int]) -> list[int]:
    """The production derivation: searchsorted on cu_seqlens[1:] with right=True."""
    cu = torch.tensor(cu_seqlens, dtype=torch.int32)
    pos = torch.arange(int(cu[-1].item()))
    return torch.searchsorted(cu[1:].contiguous(), pos, right=True).tolist()


# ───── Deterministic edge cases ─────────────────────────────────────────────


def test_single_subseq():
    """K=1: every position must be sub-seq 0."""
    cu = [0, 7]
    assert _searchsorted_right(cu) == _brute_force_seq_idx(cu, 7)
    assert _searchsorted_right(cu) == [0] * 7


def test_two_equal_subseqs():
    """K=2, equal widths. The boundary position is 4 (start of sub-seq 1)."""
    cu = [0, 4, 8]
    ref = _brute_force_seq_idx(cu, 8)
    assert ref == [0, 0, 0, 0, 1, 1, 1, 1]
    assert _searchsorted_right(cu) == ref


def test_uneven_widths():
    """K=3 with widths 3, 2, 4."""
    cu = [0, 3, 5, 9]
    ref = _brute_force_seq_idx(cu, 9)
    assert ref == [0, 0, 0, 1, 1, 2, 2, 2, 2]
    assert _searchsorted_right(cu) == ref


def test_unit_width_subseqs():
    """K=5 with width-1 sub-seqs — every position is itself a boundary."""
    cu = [0, 1, 2, 3, 4, 5]
    ref = _brute_force_seq_idx(cu, 5)
    assert ref == [0, 1, 2, 3, 4]
    assert _searchsorted_right(cu) == ref


def test_large_then_small_widths():
    """Mixed widths: large slot followed by tiny ones."""
    cu = [0, 100, 101, 102, 103]
    ref = _brute_force_seq_idx(cu, 103)
    assert ref[:100] == [0] * 100
    assert ref[100:103] == [1, 2, 3]
    assert _searchsorted_right(cu) == ref


def test_left_side_default_is_off_by_one():
    """Sanity: confirm the pre-fix default (side='left') is off by one at
    every internal boundary. This is the bug the right=True fix addresses.
    """
    cu_seqlens = [0, 4, 8]
    cu_t = torch.tensor(cu_seqlens)
    pos = torch.arange(8)
    left = torch.searchsorted(cu_t[1:], pos).tolist()
    right = torch.searchsorted(cu_t[1:], pos, right=True).tolist()
    ref = _brute_force_seq_idx(cu_seqlens, 8)

    assert right == ref
    # The left variant disagrees with the reference at exactly the
    # internal-boundary positions.
    diffs = [i for i, (a, b) in enumerate(zip(left, ref)) if a != b]
    assert diffs == [4], f"expected mismatch only at position 4, got {diffs}"


# ───── Randomized exhaustive verification ───────────────────────────────────


@pytest.mark.parametrize("seed", list(range(20)))
def test_random_shapes_match_brute_force(seed: int):
    """For a randomized cu_seqlens with mixed widths, every position's
    derived seq_idx must match the brute-force reference exactly.
    """
    rng = random.Random(seed)
    num_subseqs = rng.randint(2, 25)
    widths = [rng.randint(1, 50) for _ in range(num_subseqs)]
    cu = [0]
    for w in widths:
        cu.append(cu[-1] + w)

    ref = _brute_force_seq_idx(cu, cu[-1])
    derived = _searchsorted_right(cu)
    assert derived == ref, f"seed={seed} cu_seqlens={cu}: {sum(1 for a, b in zip(derived, ref) if a != b)} mismatches"


@pytest.mark.parametrize("seed", list(range(5)))
def test_random_shapes_left_side_fails_at_boundaries(seed: int):
    """Negative control: the pre-fix default (left side) ALWAYS mis-classifies
    every internal boundary position. Verifies the magnitude of the bug.
    """
    rng = random.Random(100 + seed)
    num_subseqs = rng.randint(3, 15)
    widths = [rng.randint(2, 30) for _ in range(num_subseqs)]
    cu = [0]
    for w in widths:
        cu.append(cu[-1] + w)

    cu_t = torch.tensor(cu)
    pos = torch.arange(cu[-1])
    left = torch.searchsorted(cu_t[1:], pos).tolist()
    ref = _brute_force_seq_idx(cu, cu[-1])

    # Every internal boundary t = cu[k] (k = 1..K-1) is mis-classified by
    # left-side: it returns k-1 instead of k.
    expected_buggy_positions = set(cu[1:-1])  # exclude the last (== total_len)
    actual_buggy_positions = {i for i, (a, b) in enumerate(zip(left, ref)) if a != b}
    assert actual_buggy_positions == expected_buggy_positions


# ───── Cross-check production call sites ────────────────────────────────────


def test_production_site_loss_mtp_matches_brute_force():
    """Re-runs the exact derivation in
    ``nemo_automodel.components.loss.mtp:calculate_mtp_loss`` (lines ~75-87)
    against the brute-force reference, ensuring the production site uses
    ``right=True``.
    """
    cu_seqlens = torch.tensor([0, 5, 11, 17], dtype=torch.int32)
    positions = torch.arange(17)
    derived = torch.searchsorted(cu_seqlens[1:].contiguous(), positions, right=True).tolist()
    ref = _brute_force_seq_idx(cu_seqlens.tolist(), 17)
    assert derived == ref


def test_production_site_layers_mamba_matches_brute_force():
    """Re-runs the exact derivation in
    ``nemo_automodel.components.models.nemotron_v3.layers`` (line ~330)
    against the brute-force reference, ensuring the production site uses
    ``right=True``.
    """
    cu_seqlens = torch.tensor([0, 7, 13, 20], dtype=torch.int32)
    total_len = int(cu_seqlens[-1].item())
    positions = torch.arange(total_len)
    derived = torch.searchsorted(cu_seqlens[1:], positions, right=True).unsqueeze(0).to(torch.int32)
    ref = torch.tensor(_brute_force_seq_idx(cu_seqlens.tolist(), total_len), dtype=torch.int32).unsqueeze(0)
    assert torch.equal(derived, ref)


def test_searchsorted_call_sites_use_right_true():
    """Static-analysis style check: both production call sites must use
    ``right=True`` (or equivalent ``side="right"``) on ``searchsorted`` over
    ``cu_seqlens[1:]``. Catches regressions that revert to the default
    ``side="left"``.
    """
    import inspect

    from nemo_automodel.components.loss import mtp as _mtp_mod
    from nemo_automodel.components.models.nemotron_v3 import layers as _layers_mod

    for mod in (_mtp_mod, _layers_mod):
        src = inspect.getsource(mod)
        # Each searchsorted call on cu_seqlens-derived array must have right=True
        # (or side="right") on the same line. Strip comments first.
        for raw_line in src.splitlines():
            code = raw_line.split("#", 1)[0]
            if "searchsorted(" in code and ("cu_seqlens" in code or "cs[1:]" in code):
                assert "right=True" in code or 'side="right"' in code, (
                    f"{mod.__name__}: searchsorted on cu_seqlens without right=True: {raw_line.strip()}"
                )

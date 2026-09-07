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

"""CPU contracts for the MSA per-call costs that are hoisted out of the hot path."""

import pytest
import torch

from nemo_automodel.components.models.minimax_m3_vl import _msa as msa
from nemo_automodel.components.models.minimax_m3_vl import model as m3_model


def test_align_backward_tensor_scatters_in_place_without_a_copy() -> None:
    compact = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    positions = torch.tensor([5, 0, 3, 1], dtype=torch.int64)

    aligned = msa._align_backward_tensor(compact, positions, 8)

    assert aligned.shape == (8, 2, 3)
    assert aligned.is_contiguous()
    torch.testing.assert_close(aligned[positions], compact, rtol=0, atol=0)
    unwritten = torch.ones(8, dtype=torch.bool).index_fill_(0, positions, False)
    assert torch.count_nonzero(aligned[unwritten]) == 0


def _doc_ids() -> torch.Tensor:
    """Return int64 packed ids[1, 262] with two documents and a padding tail."""
    ids = torch.zeros(1, 262, dtype=torch.int64)
    ids[0, :128] = 1
    ids[0, 128:200] = 2
    return ids


@pytest.fixture
def builds(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count the _MSAPackedLayout.build calls model._memoized_msa_layout makes."""
    counter = [0]
    build = msa._MSAPackedLayout.build

    def counting(doc_ids: torch.Tensor) -> msa._MSAPackedLayout:
        counter[0] += 1
        return build(doc_ids)

    monkeypatch.setattr(m3_model._MSAPackedLayout, "build", staticmethod(counting))
    return counter


def test_layout_is_built_once_per_microbatch_and_released_with_it(builds: list[int]) -> None:
    packed_seq_ids = _doc_ids()
    # Seven virtual pipeline stages of one microbatch resolve their own doc_ids from the same batch
    # tensor and must share one layout.
    layouts = [m3_model._memoized_msa_layout(packed_seq_ids.clone(), packed_seq_ids) for _ in range(7)]
    assert builds[0] == 1
    assert all(layout is layouts[0] for layout in layouts)

    # A second microbatch is a different tensor, so it gets its own layout.
    other_ids = _doc_ids()
    assert m3_model._memoized_msa_layout(other_ids.clone(), other_ids) is not layouts[0]
    assert builds[0] == 2

    # The memo holds the batch weakly: dropping it drops the layout entry.
    assert len(m3_model._MSA_LAYOUT_MEMO) == 2
    del packed_seq_ids, other_ids, layouts
    assert len(m3_model._MSA_LAYOUT_MEMO) == 0


def test_layout_is_rebuilt_when_the_batch_carries_no_packed_ids(builds: list[int]) -> None:
    # A single-document pack takes this path (the collator emits _packed_seq_ids only when some row
    # holds two documents), so the stages of that microbatch must still agree on the geometry.
    doc_ids = _doc_ids()
    first = m3_model._memoized_msa_layout(doc_ids, None)
    second = m3_model._memoized_msa_layout(doc_ids, None)

    assert builds[0] == 2 and first is not second
    assert len(m3_model._MSA_LAYOUT_MEMO) == 0
    assert torch.equal(first._cu_seqlens, second._cu_seqlens) and first._max_seqlen == second._max_seqlen

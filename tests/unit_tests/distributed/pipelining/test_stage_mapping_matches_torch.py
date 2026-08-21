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

"""Guard against drift between our stage assignment and PyTorch's.

``stage_ids_this_rank`` decides which model parts a rank materializes, while
``torch.distributed.pipelining`` independently computes
``PipelineStage.stage_index_to_group_rank`` to decide which peer each stage
sends to. If the two ever disagree, stages send activations to the wrong rank,
which surfaces as a hang or as silently corrupted activations rather than as an
error. These tests pin the two together so a PyTorch upgrade that changes the
mapping fails here instead of in a multi-node run.
"""

import pytest

from nemo_automodel.components.distributed.pipelining.module_plan import stage_ids_this_rank

torch_utils = pytest.importorskip("torch.distributed.pipelining._utils")


def _torch_rank_to_stages(pp_size: int, num_stages: int, style: str) -> dict[int, list[int]]:
    """Return PyTorch's rank-to-stage assignment for the given topology."""
    return torch_utils.generate_rank_to_stage_mapping(pp_size, num_stages, style)


@pytest.mark.parametrize(
    ("pp_size", "num_stages"),
    [(2, 2), (2, 4), (2, 6), (4, 4), (4, 8), (4, 12), (8, 8), (8, 16), (3, 9)],
)
def test_loop_style_matches_torch(pp_size, num_stages):
    """Looped assignment must match PyTorch rank for rank."""
    expected = _torch_rank_to_stages(pp_size, num_stages, "loop")
    for pp_rank in range(pp_size):
        assert list(stage_ids_this_rank(pp_rank, pp_size, num_stages, "loop")) == sorted(expected[pp_rank])


@pytest.mark.parametrize(("pp_size", "num_stages"), [(2, 4), (4, 8), (8, 16)])
def test_v_style_matches_torch(pp_size, num_stages):
    """V-shaped assignment must match PyTorch rank for rank."""
    expected = _torch_rank_to_stages(pp_size, num_stages, "v")
    for pp_rank in range(pp_size):
        assert sorted(stage_ids_this_rank(pp_rank, pp_size, num_stages, "v")) == sorted(expected[pp_rank])


def test_every_stage_is_owned_exactly_once():
    """The per-rank assignments must partition the global stages."""
    for style in ("loop", "v"):
        owned: list[int] = []
        for pp_rank in range(4):
            owned.extend(stage_ids_this_rank(pp_rank, 4, 8, style))
        assert sorted(owned) == list(range(8))


def test_v_style_rejects_topologies_it_does_not_implement():
    """V schedules assume exactly two stages per rank."""
    with pytest.raises(ValueError, match="2 stages per rank"):
        stage_ids_this_rank(0, 4, 12, "v")


def test_unknown_style_is_rejected():
    """An unrecognized assignment style must not fall through silently."""
    with pytest.raises(ValueError, match="Unknown pipeline stage assignment style"):
        stage_ids_this_rank(0, 2, 4, "spiral")

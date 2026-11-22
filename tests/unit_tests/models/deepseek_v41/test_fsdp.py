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
"""Pipeline-stage preparation of the DeepSeek V4.1 parallelizer."""

from types import SimpleNamespace

import pytest

from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.models.deepseek_v41.fsdp import PARALLELIZER
from tests.unit_tests.models.deepseek_v41.test_vision_packing import _model

pytestmark = pytest.mark.cpu_only


def _mesh_context(owner_size: int, ep_size: int = 1) -> SimpleNamespace:
    owner = SimpleNamespace(size=lambda: owner_size, get_group=lambda: object())
    return SimpleNamespace(
        pp_size=2, ep_size=ep_size, strategy_config=FSDP2Config(), device_mesh={"dp_shard_cp": owner}
    )


def test_pp_with_data_parallelism_needs_expert_parallelism_only_with_engram() -> None:
    PARALLELIZER._prepare_model(_model(full_schedule=True, engram=False), _mesh_context(owner_size=2))
    with pytest.raises(ValueError, match="requires expert parallelism"):
        PARALLELIZER._prepare_model(_model(full_schedule=True, engram=True), _mesh_context(owner_size=2))


def test_single_owner_stage_keeps_the_local_engram_table() -> None:
    model = _model(full_schedule=True, engram=True)
    PARALLELIZER._prepare_model(model, _mesh_context(owner_size=1))
    tables = [layer.engram.embed for layer in model.model.layers.values() if layer.engram is not None]
    assert tables and all(table.process_group is None and table.owner_world_size == 1 for table in tables)

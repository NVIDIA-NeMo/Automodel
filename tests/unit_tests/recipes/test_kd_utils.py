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

from types import SimpleNamespace

import pytest
import torch
from torch.distributed.tensor.experimental import _attention

from nemo_automodel.components.distributed.config import DDPConfig, FSDP2Config
from nemo_automodel.recipes import kd_utils


def test_materialize_teacher_logits_unshards_cp_and_removes_padding(monkeypatch):
    cp_mesh = SimpleNamespace(size=lambda: 2)

    class Mesh:
        mesh_dim_names = ("cp",)

        def __getitem__(self, name):
            assert name == "cp"
            return cp_mesh

    local = torch.tensor([[[1.0], [2.0]]])

    def unshard(mesh, tensors, seq_dims):
        assert mesh is cp_mesh
        assert tensors == [local]
        assert seq_dims == [1]
        return [torch.tensor([[[1.0], [2.0], [3.0], [0.0]]])]

    monkeypatch.setattr(_attention, "context_parallel_unshard", unshard)

    result = kd_utils.materialize_teacher_logits(local, device_mesh=Mesh(), sequence_length=3)

    assert torch.equal(result, torch.tensor([[[1.0], [2.0], [3.0]]]))


def test_shared_kd_setup_keeps_existing_world(monkeypatch):
    shared = SimpleNamespace(strategy_config=FSDP2Config())

    def build(cfg, **kwargs):
        return shared

    monkeypatch.setattr(kd_utils, "create_distributed_setup_from_config", build)

    setups = kd_utils.create_kd_distributed_setups({"distributed": {}}, world_size=4)

    assert setups.student is shared
    assert setups.teacher is shared
    assert setups.student_ranks == (0, 1, 2, 3)
    assert setups.teacher_ranks == (0, 1, 2, 3)
    assert not setups.separate


def test_separate_kd_setup_assigns_contiguous_disjoint_ranks(monkeypatch):
    calls = []

    def build(cfg, **kwargs):
        calls.append((cfg, kwargs))
        return SimpleNamespace(strategy_config=FSDP2Config())

    monkeypatch.setattr(kd_utils, "create_distributed_setup_from_config", build)
    cfg = {
        "separate_meshes": True,
        "distributed": {"strategy": "fsdp2", "dp_size": 2},
        "teacher_distributed": {"strategy": "fsdp2", "dp_size": 1, "tp_size": 2},
    }

    setups = kd_utils.create_kd_distributed_setups(cfg, world_size=4)

    assert setups.student_ranks == (0, 1)
    assert setups.teacher_ranks == (2, 3)
    assert setups.separate
    assert calls[0][1] == {"world_size": 2, "ranks": (0, 1)}
    assert calls[1][1] == {"world_size": 2, "ranks": (2, 3)}


@pytest.mark.parametrize(
    ("cfg", "message"),
    [
        (
            {"distributed": {}, "teacher_distributed": {"dp_size": 1}},
            "teacher_distributed requires separate_meshes=true",
        ),
        (
            {"separate_meshes": True, "distributed": {"dp_size": 2}},
            "requires a teacher_distributed section",
        ),
        (
            {
                "separate_meshes": True,
                "distributed": {"tp_size": 2},
                "teacher_distributed": {"dp_size": 2},
            },
            "distributed.dp_size must be set",
        ),
        (
            {
                "separate_meshes": True,
                "distributed": {"dp_size": 2},
                "teacher_distributed": {"dp_size": 1},
            },
            "student=2.*teacher=1 != world_size=4",
        ),
    ],
)
def test_kd_setup_rejects_ambiguous_rank_splits(cfg, message):
    with pytest.raises(ValueError, match=message):
        kd_utils.create_kd_distributed_setups(cfg, world_size=4)


def test_separate_kd_setup_rejects_ddp(monkeypatch):
    monkeypatch.setattr(
        kd_utils,
        "create_distributed_setup_from_config",
        lambda cfg, **kwargs: SimpleNamespace(strategy_config=DDPConfig()),
    )
    cfg = {
        "separate_meshes": True,
        "distributed": {"strategy": "ddp", "dp_size": 2},
        "teacher_distributed": {"strategy": "fsdp2", "dp_size": 2},
    }

    with pytest.raises(ValueError, match="DDP is not supported"):
        kd_utils.create_kd_distributed_setups(cfg, world_size=4)

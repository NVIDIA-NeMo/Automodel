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

from types import SimpleNamespace

import torch

from nemo_automodel.recipes.llm.train_ft import _warmup_mesh_collectives


class _RecordingDist:
    """Test double for torch.distributed that records all_reduce calls."""

    def __init__(self, monkeypatch, initialized: bool = True):
        self.calls: list[object] = []
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: initialized)
        monkeypatch.setattr(torch.distributed, "all_reduce", self._all_reduce)

    def _all_reduce(self, tensor, group=None):
        assert tensor.shape == (1,)
        assert (tensor == 0).all()
        self.calls.append(group)


def _mesh(groups: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        mesh_dim_names=tuple(groups),
        get_group=lambda mesh_dim: groups[mesh_dim],
    )


def test_noop_when_distributed_is_not_initialized(monkeypatch):
    dist = _RecordingDist(monkeypatch, initialized=False)

    _warmup_mesh_collectives(_mesh({"dp": object()}))

    assert dist.calls == []


def test_one_all_reduce_per_unique_group_plus_world(monkeypatch):
    dist = _RecordingDist(monkeypatch)
    dp_group, pp_group, ep_group = object(), object(), object()
    device_mesh = _mesh({"dp": dp_group, "pp": pp_group})
    # The moe mesh shares the dp group with the device mesh; it must not be
    # warmed up twice.
    moe_mesh = _mesh({"ep": ep_group, "dp": dp_group})

    _warmup_mesh_collectives(device_mesh, moe_mesh, None)

    assert dist.calls == [dp_group, pp_group, ep_group, None]


def test_skips_meshes_without_dim_names(monkeypatch):
    dist = _RecordingDist(monkeypatch)

    _warmup_mesh_collectives(SimpleNamespace(mesh_dim_names=None), None)

    # Only the final default-group (world) all-reduce runs.
    assert dist.calls == [None]

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import importlib
import weakref
from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.distributed.fsdp_patches import (
    _PostForwardMeshInfoCache,
    patch_fsdp_post_forward_mesh_cache,
)


class _Mesh:
    def __init__(self):
        self.mesh = torch.arange(8)
        self.device_type = "cuda"

    def __hash__(self):
        return 17

    def __eq__(self, other):
        return isinstance(other, _Mesh)


class _PostInfo:
    def __init__(self, mesh, shard_mesh_dim, replicate_mesh_dim):
        self.mesh = mesh
        self.shard_mesh_dim = shard_mesh_dim
        self.replicate_mesh_dim = replicate_mesh_dim


@pytest.fixture
def cache_runtime(monkeypatch):
    init = importlib.import_module("torch.distributed.fsdp._fully_shard._fsdp_init")
    full = importlib.import_module("torch.distributed.fsdp._fully_shard._fully_shard")
    original = init._get_post_forward_mesh_info
    while isinstance(original, _PostForwardMeshInfoCache):
        original = original._original
    monkeypatch.setattr(init, "_get_post_forward_mesh_info", original)
    monkeypatch.setattr(full, "_get_post_forward_mesh_info", original)
    constructed = []

    def device_mesh(device_type, tensor):
        result = (device_type, tensor)
        constructed.append(result)
        return result

    monkeypatch.setattr(init, "DeviceMesh", device_mesh)
    monkeypatch.setattr(init, "HSDPMeshInfo", _PostInfo)
    patch_fsdp_post_forward_mesh_cache()
    return init, full, original, constructed


def test_partial_mesh_cache_deduplicates_both_aliases_and_installs_once(cache_runtime):
    init, full, _, constructed = cache_runtime
    mesh = _Mesh()
    infos = [SimpleNamespace(mesh=mesh, shard_mesh_size=8) for _ in range(4)]
    cached = init._get_post_forward_mesh_info
    assert full._get_post_forward_mesh_info is cached
    first = cached(4, infos[0])
    for info in infos[1:]:
        assert full._get_post_forward_mesh_info(4, info) is first
    assert len(constructed) == 1
    patch_fsdp_post_forward_mesh_cache()
    assert init._get_post_forward_mesh_info is cached
    assert full._get_post_forward_mesh_info is cached


def test_partial_mesh_cache_does_not_merge_equal_distinct_source_meshes(cache_runtime):
    init, _, _, constructed = cache_runtime
    first_source, second_source = _Mesh(), _Mesh()
    assert first_source == second_source
    first = init._get_post_forward_mesh_info(4, SimpleNamespace(mesh=first_source, shard_mesh_size=8))
    second = init._get_post_forward_mesh_info(4, SimpleNamespace(mesh=second_source, shard_mesh_size=8))
    assert first is not second
    assert len(constructed) == 2


def test_partial_mesh_cache_separates_reshard_sizes(cache_runtime):
    init, _, _, constructed = cache_runtime
    info = SimpleNamespace(mesh=_Mesh(), shard_mesh_size=8)
    first = init._get_post_forward_mesh_info(2, info)
    second = init._get_post_forward_mesh_info(4, info)
    assert first is not second
    assert len(constructed) == 2


def test_partial_mesh_cache_weak_value_owns_source_only_until_last_user(cache_runtime):
    init, _, _, _ = cache_runtime
    source = _Mesh()
    source_ref = weakref.ref(source)
    info = SimpleNamespace(mesh=source, shard_mesh_size=8)
    first = init._get_post_forward_mesh_info(4, info)
    same = init._get_post_forward_mesh_info(4, info)
    value_ref = weakref.ref(first)
    del source, info, first
    gc.collect()
    assert value_ref() is same
    assert source_ref() is not None
    del same
    gc.collect()
    assert value_ref() is None
    assert source_ref() is None
    assert len(init._get_post_forward_mesh_info._cache) == 0


@pytest.mark.parametrize("reshard", [True, False, 1, 8, 0, 3, 9, None, "4"])
def test_partial_mesh_cache_preserves_upstream_controls_and_validation(cache_runtime, reshard):
    init, _, original, constructed = cache_runtime
    info = SimpleNamespace(mesh=_Mesh(), shard_mesh_size=8)
    try:
        expected = original(reshard, info)
    except ValueError as error:
        with pytest.raises(ValueError, match=str(error).replace("(", r"\(").replace(")", r"\)")):
            init._get_post_forward_mesh_info(reshard, info)
    else:
        assert init._get_post_forward_mesh_info(reshard, info) is expected
    assert constructed == []
    assert len(init._get_post_forward_mesh_info._cache) == 0


def test_partial_mesh_cache_original_negative_control_constructs_per_unit(cache_runtime):
    _, _, original, constructed = cache_runtime
    mesh = _Mesh()
    first = original(4, SimpleNamespace(mesh=mesh, shard_mesh_size=8))
    second = original(4, SimpleNamespace(mesh=mesh, shard_mesh_size=8))
    assert first is not second
    assert len(constructed) == 2

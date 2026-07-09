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

import pytest
import torch
import torch.distributed as dist

from nemo_automodel.components.training.prewarm import (
    PrewarmConfig,
    _collect_gdn_autotune_shapes,
    _prewarm_comm_groups,
    _prewarm_cublas_backward,
    _prewarm_fla_gdn_autotune,
)


class _FakeGDN(torch.nn.Module):
    """Minimal stand-in for a gated-delta-net attention module."""

    def __init__(self, num_v_heads: int = 4, head_k_dim: int = 8, head_v_dim: int = 16):
        super().__init__()
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.in_proj_qkv = torch.nn.Linear(4, 4)
        self.chunk_gated_delta_rule = object()  # presence is what the discovery checks


@pytest.fixture
def single_rank_gloo():
    """Initialize a single-rank gloo process group for the duration of a test."""
    if dist.is_initialized():
        pytest.skip("a process group is already initialized in this session")
    dist.init_process_group(backend="gloo", rank=0, world_size=1, store=dist.HashStore())
    try:
        yield
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# PrewarmConfig
# ---------------------------------------------------------------------------


def test_prewarm_config_defaults_all_off():
    cfg = PrewarmConfig()
    assert cfg.cublas_backward is False
    assert cfg.fla_gdn_autotune is False
    assert cfg.comm_groups is False


def test_apply_runs_only_enabled_prewarms(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "nemo_automodel.components.training.prewarm._prewarm_cublas_backward",
        lambda device: calls.append(("cublas", device)),
    )
    monkeypatch.setattr(
        "nemo_automodel.components.training.prewarm._prewarm_fla_gdn_autotune",
        lambda model_parts, device: calls.append(("fla", device)),
    )
    monkeypatch.setattr(
        "nemo_automodel.components.training.prewarm._prewarm_comm_groups",
        lambda model_parts, device, pp_mesh=None: calls.append(("comm", pp_mesh)),
    )
    PrewarmConfig(cublas_backward=True, comm_groups=True).apply(
        model_parts=[torch.nn.Linear(2, 2)],
        device=torch.device("cpu"),
        pp_mesh="pp-mesh",
    )
    assert calls == [("cublas", torch.device("cpu")), ("comm", "pp-mesh")]

    calls.clear()
    PrewarmConfig().apply(model_parts=[], device=None)
    assert calls == []


def test_recipe_config_exposes_typed_prewarm_section():
    from nemo_automodel.recipes._typed_config import RecipeConfig

    cfg = RecipeConfig({"prewarm": {"cublas_backward": True, "comm_groups": True}})
    prewarm = cfg.prewarm
    assert isinstance(prewarm, PrewarmConfig)
    assert prewarm.cublas_backward is True
    assert prewarm.fla_gdn_autotune is False
    assert prewarm.comm_groups is True

    assert RecipeConfig({}).prewarm is None


def test_recipe_config_rejects_unknown_prewarm_keys():
    from nemo_automodel.recipes._typed_config import RecipeConfig

    with pytest.raises(TypeError):
        _ = RecipeConfig({"prewarm": {"cublas_backwards": True}}).prewarm


# ---------------------------------------------------------------------------
# cuBLAS backward prewarm
# ---------------------------------------------------------------------------


def test_cublas_prewarm_skips_without_cuda_device():
    assert _prewarm_cublas_backward(None) is False
    assert _prewarm_cublas_backward(torch.device("cpu")) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_cublas_prewarm_runs_on_gpu():
    assert _prewarm_cublas_backward(torch.device("cuda", torch.cuda.current_device())) is True


# ---------------------------------------------------------------------------
# fla GDN autotune prewarm
# ---------------------------------------------------------------------------


def test_collect_gdn_autotune_shapes_finds_and_dedups():
    class _Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gdn_a = _FakeGDN(4, 8, 16)
            self.gdn_b = _FakeGDN(4, 8, 16)  # duplicate shape, deduped
            self.gdn_c = _FakeGDN(2, 8, 16)
            self.plain = torch.nn.Linear(4, 4)  # no GDN attrs, ignored

    shapes = _collect_gdn_autotune_shapes([_Wrapper()])
    assert set(shapes) == {(4, 8, 16, torch.float32), (2, 8, 16, torch.float32)}
    assert shapes[(4, 8, 16, torch.float32)] == "gdn_a"


def test_collect_gdn_autotune_shapes_requires_gdn_op():
    module = _FakeGDN()
    del module.chunk_gated_delta_rule
    assert _collect_gdn_autotune_shapes([module]) == {}


def test_fla_prewarm_skips_without_cuda_device():
    assert _prewarm_fla_gdn_autotune([_FakeGDN()], torch.device("cpu")) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_fla_prewarm_skips_without_gdn_modules():
    device = torch.device("cuda", torch.cuda.current_device())
    assert _prewarm_fla_gdn_autotune([torch.nn.Linear(2, 2)], device) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_fla_prewarm_populates_autotune_cache_on_gpu():
    pytest.importorskip("fla")
    device = torch.device("cuda", torch.cuda.current_device())
    module = _FakeGDN(num_v_heads=2, head_k_dim=64, head_v_dim=64).to(device, torch.bfloat16)
    assert _prewarm_fla_gdn_autotune([module], device) is True


# ---------------------------------------------------------------------------
# Comm-group prewarm
# ---------------------------------------------------------------------------


def test_comm_groups_prewarm_skips_without_dist_init():
    assert _prewarm_comm_groups([torch.nn.Linear(2, 2)], torch.device("cpu")) == 0


def test_comm_groups_prewarm_warms_shard_groups(single_rank_gloo):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    mesh = init_device_mesh("cpu", (1,))
    module = torch.nn.Linear(4, 4, bias=False)
    module.weight = torch.nn.Parameter(distribute_tensor(module.weight.detach(), mesh, [Shard(0)]))
    assert _prewarm_comm_groups([module], torch.device("cpu")) == 1

    # Replicate-placed parameters define no shard groups.
    replicated = torch.nn.Linear(4, 4, bias=False)
    replicated.weight = torch.nn.Parameter(distribute_tensor(replicated.weight.detach(), mesh, [Replicate()]))
    assert _prewarm_comm_groups([replicated], torch.device("cpu")) == 0

    # The PP group is discovered via pp_mesh even though no param shards on it.
    assert _prewarm_comm_groups([replicated], torch.device("cpu"), pp_mesh=mesh) == 1


def test_comm_groups_prewarm_ignores_regular_tensors(single_rank_gloo):
    assert _prewarm_comm_groups([torch.nn.Linear(4, 4)], torch.device("cpu")) == 0

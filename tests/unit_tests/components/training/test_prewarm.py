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

import logging
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

from nemo_automodel.components.training import prewarm
from nemo_automodel.components.training.prewarm import (
    PrewarmConfig,
    _collect_gdn_autotune_shapes,
    _prewarm_comm_groups,
    _prewarm_cublas_backward,
    _prewarm_fla_gdn_autotune,
    _prewarm_fla_gdn_cp_kernels,
    _prewarm_fla_gdn_end_to_end,
    _triton_kernel_accepts,
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


def test_apply_continues_after_prewarm_failures(monkeypatch, caplog):
    calls = []

    def fail(label):
        def _raise(*args, **kwargs):
            calls.append(label)
            raise RuntimeError(label)

        return _raise

    monkeypatch.setattr(prewarm, "_prewarm_cublas_backward", fail("cublas"))
    monkeypatch.setattr(prewarm, "_prewarm_fla_gdn_autotune", fail("fla"))
    monkeypatch.setattr(prewarm, "_prewarm_comm_groups", fail("comm"))

    with caplog.at_level(logging.ERROR, logger=prewarm.__name__):
        PrewarmConfig(cublas_backward=True, fla_gdn_autotune=True, comm_groups=True).apply(
            model_parts=[torch.nn.Linear(2, 2)],
            device=torch.device("cpu"),
        )

    assert calls == ["cublas", "fla", "comm"]
    assert "cuBLAS backward prewarm failed" in caplog.text
    assert "fla GDN autotune prewarm failed" in caplog.text
    assert "Communication-group prewarm failed" in caplog.text


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
def test_cublas_prewarm_runs_without_advancing_rng_on_gpu():
    device = torch.device("cuda", torch.cuda.current_device())
    rng_before = torch.cuda.get_rng_state(device)
    assert _prewarm_cublas_backward(device) is True
    assert torch.equal(torch.cuda.get_rng_state(device), rng_before)


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
def test_fla_end_to_end_prewarm_preserves_rng_with_stubbed_op(monkeypatch):
    device = torch.device("cuda", torch.cuda.current_device())
    output = torch.ones((1, 2, 2, 4), device=device, requires_grad=True)
    fake_gdn_op = Mock(return_value=(output, None))
    monkeypatch.setattr(prewarm, "safe_import_from", lambda *args: (True, fake_gdn_op))

    rng_before = torch.cuda.get_rng_state(device)
    assert _prewarm_fla_gdn_end_to_end({(2, 4, 4, torch.float32): "gdn"}, device, seq_len=2) is True
    assert torch.equal(torch.cuda.get_rng_state(device), rng_before)
    fake_gdn_op.assert_called_once()
    assert output.grad is not None
    assert torch.equal(output.grad, torch.ones_like(output))


def test_triton_kernel_accepts_unwraps_wrappers_and_validates_args():
    jit_fn = SimpleNamespace(arg_names=["a", "b", "c"])
    autotuner = SimpleNamespace(fn=SimpleNamespace(fn=jit_fn))  # Autotuner(Heuristics(JITFunction))

    assert _triton_kernel_accepts(autotuner, frozenset(("a", "b")), "kernel") is True
    assert _triton_kernel_accepts(autotuner, frozenset(("a", "missing")), "kernel") is False
    # Objects exposing no arg_names anywhere in the fn chain are rejected.
    assert _triton_kernel_accepts(object(), frozenset(("a",)), "kernel") is False


def test_fla_cp_kernel_prewarm_skips_when_fla_unavailable(monkeypatch, caplog):
    monkeypatch.setattr(prewarm, "safe_import", lambda name: (False, None))
    monkeypatch.setattr(prewarm, "safe_import_from", lambda module, name: (False, None))

    with caplog.at_level(logging.INFO, logger=prewarm.__name__):
        _prewarm_fla_gdn_cp_kernels({(2, 8, 16, torch.float32): "gdn"}, torch.device("cpu"), seq_len=16)

    assert "fla CP kernels not importable" in caplog.text


def test_fla_cp_kernel_prewarm_skips_on_kernel_signature_mismatch(monkeypatch, caplog):
    launches = []

    class _DriftedKernel:
        """Triton-like kernel whose parameter list no longer matches the launch contract."""

        arg_names = ["q", "k", "renamed_everything_else"]

        def __getitem__(self, grid):
            return lambda **kwargs: launches.append(kwargs)

    fake_triton = SimpleNamespace(next_power_of_2=lambda n: n, cdiv=lambda a, b: -(-a // b))
    monkeypatch.setattr(prewarm, "safe_import", lambda name: (True, fake_triton))
    monkeypatch.setattr(prewarm, "safe_import_from", lambda module, name: (True, _DriftedKernel()))

    with caplog.at_level(logging.WARNING, logger=prewarm.__name__):
        _prewarm_fla_gdn_cp_kernels({(2, 8, 16, torch.float32): "gdn"}, torch.device("cpu"), seq_len=16)

    assert launches == []  # nothing may be launched on signature drift
    assert "pre_process_bwd_kernel_merged" in caplog.text
    assert "merge_fwd_bwd_kernel" in caplog.text


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


def _run_two_rank_comm_group_prewarm(rank: int, world_size: int, init_file: str) -> None:
    """Exercise sharded and pipeline groups with real two-rank collectives."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,))

        sharded = torch.nn.Linear(4, 4, bias=False)
        sharded.weight = torch.nn.Parameter(distribute_tensor(sharded.weight.detach(), mesh, [Shard(0)]))
        assert _prewarm_comm_groups([sharded], torch.device("cpu")) == 1

        replicated = torch.nn.Linear(4, 4, bias=False)
        replicated.weight = torch.nn.Parameter(distribute_tensor(replicated.weight.detach(), mesh, [Replicate()]))
        assert _prewarm_comm_groups([replicated], torch.device("cpu"), pp_mesh=mesh) == 1

        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_comm_groups_prewarm_warms_groups_on_two_ranks(tmp_path):
    init_file = tmp_path / "prewarm_pg"
    torch.multiprocessing.spawn(
        _run_two_rank_comm_group_prewarm,
        args=(2, str(init_file)),
        nprocs=2,
        join=True,
    )

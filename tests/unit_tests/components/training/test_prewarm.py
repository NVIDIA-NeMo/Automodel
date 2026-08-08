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

from nemo_automodel.components.training import _mamba_ssd_prewarm, prewarm
from nemo_automodel.components.training._mamba_ssd_prewarm import (
    _collect_mamba_ssd_autotune_shapes,
    _prewarm_mamba_ssd_autotune,
    _prewarm_mamba_ssd_end_to_end,
)
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


class _FakeMambaSSD(torch.nn.Module):
    """Minimal stand-in for a mixer backed by the Mamba SSD operators."""

    def __init__(
        self,
        num_heads: int = 4,
        head_dim: int = 8,
        state_size: int = 16,
        n_groups: int = 2,
        chunk_size: int = 32,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.ssm_state_size = state_size
        self.n_groups = n_groups
        self.chunk_size = chunk_size
        self.cp = None
        self.in_proj = torch.nn.Linear(4, 4)


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
    assert cfg.mamba_ssd_autotune is False
    assert cfg.comm_groups is False


def test_apply_runs_only_enabled_prewarms(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "nemo_automodel.components.training.prewarm._prewarm_cublas_backward",
        lambda device: calls.append(("cublas", device)),
    )
    monkeypatch.setattr(
        "nemo_automodel.components.training.prewarm._prewarm_fla_gdn_autotune",
        lambda model_parts, device, batch_size: calls.append(("fla", device, batch_size)),
    )
    monkeypatch.setattr(
        "nemo_automodel.components.training.prewarm._prewarm_mamba_ssd_autotune",
        lambda model_parts, device: calls.append(("mamba", device)),
    )
    monkeypatch.setattr(
        "nemo_automodel.components.training.prewarm._prewarm_comm_groups",
        lambda model_parts, device, pp_mesh=None: calls.append(("comm", pp_mesh)),
    )

    PrewarmConfig(cublas_backward=True, fla_gdn_autotune=True, mamba_ssd_autotune=True, comm_groups=True).apply(
        model_parts=[torch.nn.Linear(2, 2)],
        device=torch.device("cpu"),
        batch_size=4,
        pp_mesh="pp-mesh",
    )
    assert calls == [
        ("cublas", torch.device("cpu")),
        ("fla", torch.device("cpu"), 4),
        ("mamba", torch.device("cpu")),
        ("comm", "pp-mesh"),
    ]

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
    monkeypatch.setattr(prewarm, "_prewarm_mamba_ssd_autotune", fail("mamba"))
    monkeypatch.setattr(prewarm, "_prewarm_comm_groups", fail("comm"))

    with caplog.at_level(logging.ERROR, logger=prewarm.__name__):
        PrewarmConfig(
            cublas_backward=True,
            fla_gdn_autotune=True,
            mamba_ssd_autotune=True,
            comm_groups=True,
        ).apply(
            model_parts=[torch.nn.Linear(2, 2)],
            device=torch.device("cpu"),
        )

    assert calls == ["cublas", "fla", "mamba", "comm"]
    assert "cuBLAS backward prewarm failed" in caplog.text
    assert "fla GDN autotune prewarm failed" in caplog.text
    assert "Mamba SSD autotune prewarm failed" in caplog.text
    assert "Communication-group prewarm failed" in caplog.text


def test_recipe_config_exposes_typed_prewarm_section():
    from nemo_automodel.recipes._typed_config import RecipeConfig

    cfg = RecipeConfig({"prewarm": {"cublas_backward": True, "mamba_ssd_autotune": True, "comm_groups": True}})
    prewarm = cfg.prewarm
    assert isinstance(prewarm, PrewarmConfig)
    assert prewarm.cublas_backward is True
    assert prewarm.fla_gdn_autotune is False
    assert prewarm.mamba_ssd_autotune is True
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


def test_fla_end_to_end_prewarm_preserves_rng_with_stubbed_op(monkeypatch):
    device = torch.device("cpu")
    outputs = []

    def fake_gdn_op(q, k, v, **kwargs):
        output = torch.ones_like(v, requires_grad=True)
        outputs.append(output)
        return output, None

    fake_gdn_op = Mock(side_effect=fake_gdn_op)
    monkeypatch.setattr(prewarm, "safe_import_from", lambda *args: (True, fake_gdn_op))

    rng_before = torch.random.get_rng_state()
    assert (
        _prewarm_fla_gdn_end_to_end(
            {(2, 4, 4, torch.float32): "gdn"},
            device,
            seq_len=2,
            batch_size=3,
        )
        is True
    )
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert fake_gdn_op.call_count == 2
    dense_call, packed_call = fake_gdn_op.call_args_list
    assert dense_call.args[0].shape == (3, 2, 2, 4)
    assert dense_call.kwargs["cu_seqlens"] is None
    assert packed_call.args[0].shape == (1, 2, 2, 4)
    assert torch.equal(packed_call.kwargs["cu_seqlens"], torch.tensor([0, 2], device=device))
    assert all(output.grad is not None for output in outputs)


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
# Mamba SSD autotune prewarm
# ---------------------------------------------------------------------------


def test_collect_mamba_ssd_autotune_shapes_uses_local_cp_geometry_and_dedups():
    class _Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mamba_a = _FakeMambaSSD()
            self.mamba_b = _FakeMambaSSD()  # duplicate shape, deduped
            self.mamba_cp = _FakeMambaSSD(num_heads=8, n_groups=4)
            self.mamba_cp.cp = SimpleNamespace(num_heads_local=4, n_groups_local=2)
            self.plain = torch.nn.Linear(4, 4)

    shapes = _collect_mamba_ssd_autotune_shapes([_Wrapper()])
    assert set(shapes) == {(4, 8, 16, 2, 32, torch.float32)}
    assert shapes[(4, 8, 16, 2, 32, torch.float32)] == "mamba_a"


def test_mamba_ssd_prewarm_skips_without_cuda_device():
    assert _prewarm_mamba_ssd_autotune([_FakeMambaSSD()], torch.device("cpu")) is False


def test_mamba_ssd_end_to_end_matches_keyed_geometry_and_preserves_rng(monkeypatch):
    captured = {}

    def fake_mamba_ssd_op(x, dt, a, b, c, chunk_size, *, D, z, dt_bias, seq_idx, dt_softplus):
        captured.update(
            x=x,
            dt=dt,
            a=a,
            b=b,
            c=c,
            chunk_size=chunk_size,
            d=D,
            z=z,
            dt_bias=dt_bias,
            seq_idx=seq_idx,
            dt_softplus=dt_softplus,
        )
        return x

    monkeypatch.setattr(_mamba_ssd_prewarm, "safe_import_from", lambda *args: (True, fake_mamba_ssd_op))
    rng_before = torch.random.get_rng_state()

    shapes = {(4, 8, 16, 2, 32, torch.float32): "mamba"}
    assert _prewarm_mamba_ssd_end_to_end(shapes, torch.device("cpu")) is True

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert captured["x"].shape == (1, 64, 4, 8)
    assert captured["dt"].shape == (1, 64, 4)
    assert captured["a"].shape == (4,)
    assert captured["b"].shape == (1, 64, 2, 16)
    assert captured["c"].shape == (1, 64, 2, 16)
    assert captured["d"].shape == (4,)
    assert captured["dt_bias"].shape == (4,)
    assert captured["seq_idx"].shape == (1, 64)
    assert captured["seq_idx"].dtype == torch.int32
    assert captured["chunk_size"] == 32
    assert captured["z"] is None
    assert captured["dt_softplus"] is True
    assert captured["x"].grad is not None


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

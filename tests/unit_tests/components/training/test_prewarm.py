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
    _dry_run_warmup,
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


class _TinyLM(torch.nn.Module):
    """Tiny embedding + linear-head language model for dry-run tests."""

    def __init__(self, vocab_size: int = 32, hidden: int = 8):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.lm_head = torch.nn.Linear(hidden, vocab_size)

    def get_input_embeddings(self) -> torch.nn.Embedding:
        return self.embed

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute logits.

        Args:
            input_ids: Token ids of shape ``[B, S]`` (B = batch, S = sequence).

        Returns:
            Logits of shape ``[B, S, V]`` (V = vocab size).
        """
        return self.lm_head(self.embed(input_ids))


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
    assert cfg.dry_run is False


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
    monkeypatch.setattr(
        "nemo_automodel.components.training.prewarm._dry_run_warmup",
        lambda model_parts, device, pp_enabled=False: calls.append(("dry_run", pp_enabled)),
    )

    PrewarmConfig(cublas_backward=True, comm_groups=True).apply(
        model_parts=[torch.nn.Linear(2, 2)],
        device=torch.device("cpu"),
        pp_mesh="pp-mesh",
        pp_enabled=False,
    )
    assert calls == [("cublas", torch.device("cpu")), ("comm", "pp-mesh")]

    calls.clear()
    PrewarmConfig().apply(model_parts=[], device=None)
    assert calls == []


def test_recipe_config_exposes_typed_prewarm_section():
    from nemo_automodel.recipes._typed_config import RecipeConfig

    cfg = RecipeConfig({"prewarm": {"cublas_backward": True, "dry_run": True}})
    prewarm = cfg.prewarm
    assert isinstance(prewarm, PrewarmConfig)
    assert prewarm.cublas_backward is True
    assert prewarm.fla_gdn_autotune is False
    assert prewarm.dry_run is True

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


# ---------------------------------------------------------------------------
# Dry-run warmup (RFC)
# ---------------------------------------------------------------------------


def test_dry_run_warmup_skips_when_pp_enabled():
    assert _dry_run_warmup([_TinyLM()], torch.device("cpu"), pp_enabled=True) is False


def test_dry_run_warmup_is_side_effect_free_on_cpu():
    model = _TinyLM()
    params_before = {name: p.detach().clone() for name, p in model.named_parameters()}
    rng_before = torch.get_rng_state()

    assert _dry_run_warmup([model], torch.device("cpu"), seq_len=8) is True

    for name, p in model.named_parameters():
        assert p.grad is None, f"{name} kept a gradient after the dry run"
        assert torch.equal(p.detach(), params_before[name]), f"{name} changed during the dry run"
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_dry_run_warmup_swallows_forward_failures():
    class _Broken(_TinyLM):
        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            """Raise unconditionally; ``input_ids`` is an unused ``[B, S]`` id tensor."""
            raise RuntimeError("boom")

    assert _dry_run_warmup([_Broken()], torch.device("cpu")) is False


def test_dry_run_warmup_skips_without_vocab_size():
    model = torch.nn.Linear(4, 4)  # no config and no get_input_embeddings
    assert _dry_run_warmup([model], torch.device("cpu")) is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
def test_dry_run_warmup_runs_on_gpu():
    device = torch.device("cuda", torch.cuda.current_device())
    model = _TinyLM().to(device)
    assert _dry_run_warmup([model], device, seq_len=8) is True
    assert all(p.grad is None for p in model.parameters())

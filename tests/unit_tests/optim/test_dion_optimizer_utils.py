# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import math

import pytest
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    def __init__(self, with_lm_head=True, with_bias=False):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 4)
        self.linear = nn.Linear(4, 4, bias=with_bias)
        if with_lm_head:
            self.lm_head = nn.Linear(4, 10, bias=False)


class FakeMesh:
    """Simple stand-in for a named DeviceMesh-like object."""

    def __init__(self, mapping: dict, ndim: int = 1):
        self._mapping = dict(mapping)
        self.ndim = ndim

    def __getitem__(self, key):
        if key not in self._mapping:
            raise KeyError(key)
        return self._mapping[key]


class FakeSubmesh:
    """A 1-D submesh stub returned from a 2-D mesh lookup."""

    def __init__(self, mapping: dict | None = None):
        self.ndim = 1
        self._mapping = mapping or {}

    def __getitem__(self, key):
        if key not in self._mapping:
            raise KeyError(key)
        return self._mapping[key]


# ---------------------------------------------------------------------------
# Tests for is_dion_optimizer()
# ---------------------------------------------------------------------------


class TestIsDionOptimizer:
    def test_returns_true_for_dion_module(self):
        from nemo_automodel.components.optim.dion import is_dion_optimizer

        optimizer_factory = type("SomeOpt", (), {"__module__": "dion.optimizers"})
        assert is_dion_optimizer(optimizer_factory) is True

    def test_returns_true_for_known_names(self):
        from nemo_automodel.components.optim.dion import is_dion_optimizer

        for name in ("Dion", "Dion2", "Muon", "NorMuon"):
            optimizer_factory = type(name, (), {"__module__": "some.module"})
            assert is_dion_optimizer(optimizer_factory) is True, f"Expected True for {name}"

    def test_returns_false_for_non_dion(self):
        from nemo_automodel.components.optim.dion import is_dion_optimizer

        optimizer_factory = type("Adam", (), {"__module__": "torch.optim"})
        assert is_dion_optimizer(optimizer_factory) is False

    def test_returns_false_when_no_target(self):
        from nemo_automodel.components.optim.dion import is_dion_optimizer

        assert is_dion_optimizer(object()) is False


# ---------------------------------------------------------------------------
# Tests for _separate_param_groups()
# ---------------------------------------------------------------------------


class TestSeparateParamGroups:
    def _call(self, **kwargs):
        from nemo_automodel.components.optim.dion import _separate_param_groups

        return _separate_param_groups(**kwargs)

    def test_basic_grouping(self):
        model = TinyModel()
        groups = self._call(model=model, base_lr=1e-3, scalar_opt="adamw", weight_decay=0.01)

        # matrix_params, vector_params, embed_params, lm_head_params
        assert len(groups) == 4
        # matrix group (linear.weight) has no explicit algorithm key
        assert "algorithm" not in groups[0]
        # vector group
        assert groups[1]["algorithm"] == "adamw"
        # embed group
        assert groups[2]["algorithm"] == "adamw"
        assert groups[2]["weight_decay"] == 0.0
        # lm_head group
        assert groups[3]["algorithm"] == "adamw"
        assert groups[3]["weight_decay"] == 0.0

    def test_no_lm_head(self):
        model = TinyModel(with_lm_head=False)
        groups = self._call(model=model, base_lr=1e-3, scalar_opt="adamw", weight_decay=0.01)
        # Only 3 groups: matrix, vector, embed (no lm_head)
        assert len(groups) == 3

    def test_auto_lm_head_lr(self):
        model = TinyModel()
        base_lr = 1e-3
        groups = self._call(model=model, base_lr=base_lr, scalar_opt="adamw", weight_decay=0.0)
        lm_head_group = groups[3]
        # lm_head.weight shape is (10, 4), d_in = 4
        expected_lr = base_lr / math.sqrt(4.0)
        assert lm_head_group["lr"] == pytest.approx(expected_lr)

    def test_explicit_lm_head_lr(self):
        model = TinyModel()
        groups = self._call(model=model, base_lr=1e-3, scalar_opt="adamw", weight_decay=0.0, lm_head_lr=5e-5)
        assert groups[3]["lr"] == pytest.approx(5e-5)

    def test_scalar_lr_and_embed_lr_overrides(self):
        model = TinyModel()
        groups = self._call(
            model=model,
            base_lr=1e-3,
            scalar_opt="lion",
            weight_decay=0.0,
            scalar_lr=2e-4,
            embed_lr=3e-4,
        )
        # vector group uses scalar_lr
        assert groups[1]["lr"] == pytest.approx(2e-4)
        # embed group uses embed_lr
        assert groups[2]["lr"] == pytest.approx(3e-4)

    def test_scalar_lr_defaults_embed_lr(self):
        model = TinyModel()
        groups = self._call(
            model=model,
            base_lr=1e-3,
            scalar_opt="adamw",
            weight_decay=0.0,
            scalar_lr=7e-4,
        )
        # embed_lr defaults to scalar_lr when not provided
        assert groups[2]["lr"] == pytest.approx(7e-4)

    def test_scalar_betas_and_eps(self):
        model = TinyModel()
        groups = self._call(
            model=model,
            base_lr=1e-3,
            scalar_opt="adamw",
            weight_decay=0.0,
            scalar_betas=(0.9, 0.999),
            scalar_eps=1e-8,
        )
        for g in groups[1:]:  # all scalar groups
            assert g["beta1"] == pytest.approx(0.9)
            assert g["beta2"] == pytest.approx(0.999)
            assert g["epsilon"] == pytest.approx(1e-8)

    def test_requires_grad_false_skipped(self):
        model = TinyModel()
        # Freeze linear
        for p in model.linear.parameters():
            p.requires_grad = False
        groups = self._call(model=model, base_lr=1e-3, scalar_opt="adamw", weight_decay=0.0)
        # matrix group should be empty (linear was the only 2D non-embed non-lm_head)
        assert len(groups[0]["params"]) == 0

    def test_bias_goes_to_vector_group(self):
        model = TinyModel(with_bias=True)
        groups = self._call(model=model, base_lr=1e-3, scalar_opt="adamw", weight_decay=0.0)
        # vector group should have the bias param (1D)
        vector_shapes = [p.shape for p in groups[1]["params"]]
        assert any(len(s) == 1 for s in vector_shapes)


# ---------------------------------------------------------------------------
# Tests for _get_dion_mesh()
# ---------------------------------------------------------------------------


class TestGetDionMesh:
    def _call(self, mesh):
        from nemo_automodel.components.optim.dion import _get_dion_mesh

        return _get_dion_mesh(mesh)

    def test_none_returns_none(self):
        assert self._call(None) is None

    def test_1d_mesh_returned_as_is(self):
        mesh = FakeMesh({}, ndim=1)
        assert self._call(mesh) is mesh

    def test_no_ndim_returned_as_is(self):
        mesh = object()  # no ndim attribute
        assert self._call(mesh) is mesh

    def test_multidim_mesh_extracts_submesh(self):
        inner_submesh = FakeSubmesh()
        inner_submesh.ndim = 1
        dp_2d = FakeSubmesh({"dp_shard_cp": inner_submesh})
        mesh = FakeMesh({("dp_replicate", "dp_shard_cp"): dp_2d}, ndim=2)
        result = self._call(mesh)
        assert result is inner_submesh

    def test_multidim_mesh_fallback_on_key_error(self):
        mesh = FakeMesh({}, ndim=2)
        # KeyError when accessing ("dp_replicate", "dp_shard_cp")
        result = self._call(mesh)
        # Falls back to returning mesh itself
        assert result is mesh


# ---------------------------------------------------------------------------
# Tests for build_dion_optimizer()
# ---------------------------------------------------------------------------


class FakeDionConfig:
    """Minimal stand-in for a dion-family OptimizerConfig.

    ``build_dion_optimizer`` reads settings off the config via attribute access,
    so any object exposing the same attributes works.
    """

    def __init__(
        self,
        *,
        lr=1e-3,
        weight_decay=0.0,
        scalar_opt="adamw",
        scalar_betas=None,
        scalar_eps=None,
        scalar_lr=None,
        embed_lr=None,
        lm_head_lr=None,
        no_compile=False,
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.scalar_opt = scalar_opt
        self.scalar_betas = scalar_betas
        self.scalar_eps = scalar_eps
        self.scalar_lr = scalar_lr
        self.embed_lr = embed_lr
        self.lm_head_lr = lm_head_lr
        self.no_compile = no_compile


class TestBuildDionOptimizer:
    """``build_dion_optimizer`` reads its settings off the passed config and returns
    ``(param_groups, mesh_kwargs)``; it does not instantiate the optimizer (the
    typed config assembles its own kwargs, splatting ``mesh_kwargs``, and
    instantiates)."""

    def _build(self, monkeypatch, config, model=None, mesh=None, mesh_kwarg="distributed_mesh"):
        from nemo_automodel.components.optim import dion as optim_dion

        monkeypatch.setattr(optim_dion, "_import_error", None, raising=False)
        if model is None:
            model = TinyModel()
        return optim_dion.build_dion_optimizer(config, model, device_mesh=mesh, mesh_kwarg=mesh_kwarg)

    def test_returns_param_groups_and_mesh_kwargs(self, monkeypatch):
        param_groups, mesh_kwargs = self._build(monkeypatch, FakeDionConfig(lr=1e-3, weight_decay=0.05))
        assert isinstance(param_groups, list)
        assert len(param_groups) >= 2
        assert mesh_kwargs == {}

    def test_resolves_1d_mesh(self, monkeypatch):
        """A 1-D device mesh is returned as-is, keyed under ``mesh_kwarg``."""
        mesh = FakeMesh({"dp_replicate": object(), "dp_shard_cp": object()}, ndim=1)
        _, mesh_kwargs = self._build(monkeypatch, FakeDionConfig(), mesh=mesh)
        assert mesh_kwargs == {"distributed_mesh": mesh}

    def test_mesh_kwarg_arg_controls_key(self, monkeypatch):
        """The ``mesh_kwarg`` argument controls the key the mesh is returned under."""
        mesh = FakeMesh({}, ndim=1)
        _, mesh_kwargs = self._build(monkeypatch, FakeDionConfig(), mesh=mesh, mesh_kwarg="outer_shard_mesh")
        assert mesh_kwargs == {"outer_shard_mesh": mesh}

    def test_mesh_kwarg_none_omits_mesh(self, monkeypatch):
        """``mesh_kwarg=None`` never includes the mesh, even when one is provided."""
        mesh = FakeMesh({}, ndim=1)
        _, mesh_kwargs = self._build(monkeypatch, FakeDionConfig(), mesh=mesh, mesh_kwarg=None)
        assert mesh_kwargs == {}

    def test_resolves_submesh_from_multidim(self, monkeypatch):
        """A multi-dim mesh is reduced to its 1-D dp_shard_cp submesh."""
        inner = FakeSubmesh()
        inner.ndim = 1
        dp_2d = FakeSubmesh({"dp_shard_cp": inner})
        mesh = FakeMesh({("dp_replicate", "dp_shard_cp"): dp_2d}, ndim=2)
        _, mesh_kwargs = self._build(monkeypatch, FakeDionConfig(), mesh=mesh)
        assert mesh_kwargs == {"distributed_mesh": inner}

    def test_none_mesh_returns_empty(self, monkeypatch):
        """device_mesh=None yields no mesh kwargs."""
        _, mesh_kwargs = self._build(monkeypatch, FakeDionConfig(), mesh=None)
        assert mesh_kwargs == {}

    def test_import_error_raises(self, monkeypatch):
        from nemo_automodel.components.optim import dion as optim_dion

        monkeypatch.setattr(optim_dion, "_import_error", ImportError("no dion"), raising=False)

        with pytest.raises(RuntimeError, match="Failed to import Dion"):
            optim_dion.build_dion_optimizer(FakeDionConfig(), TinyModel())

    def test_scalar_config_drives_param_groups(self, monkeypatch):
        """scalar_opt / scalar_betas / scalar_eps and the scalar/embed/lm_head LRs
        are read off the config and applied to the scalar param groups."""
        param_groups, _ = self._build(
            monkeypatch,
            FakeDionConfig(
                lr=1e-3,
                weight_decay=0.01,
                scalar_opt="lion",
                scalar_betas=[0.9, 0.95],
                scalar_eps=1e-8,
                scalar_lr=5e-4,
                embed_lr=3e-4,
                lm_head_lr=1e-5,
            ),
        )
        # groups: matrix, vector, embed, lm_head
        assert param_groups[1]["algorithm"] == "lion"
        assert param_groups[1]["lr"] == pytest.approx(5e-4)
        assert param_groups[2]["lr"] == pytest.approx(3e-4)
        assert param_groups[3]["lr"] == pytest.approx(1e-5)

    def test_param_groups_structure(self, monkeypatch):
        """Verify param groups have the right structure."""
        param_groups, _ = self._build(monkeypatch, FakeDionConfig(), model=TinyModel())
        # 4 groups: matrix, vector, embed, lm_head
        assert len(param_groups) == 4
        for g in param_groups:
            assert "params" in g
            assert isinstance(g["params"], list)


# ---------------------------------------------------------------------------
# Tests for base_recipe.py: synchronize_for_checkpoint() integration
# ---------------------------------------------------------------------------


class TestSynchronizeForCheckpoint:
    """Test the synchronize_for_checkpoint loop added in BaseRecipe.save_checkpoint.

    The logic under test (base_recipe.py lines 286-290):
        optimizers = optimizer if isinstance(optimizer, list) else [optimizer]
        for opt in optimizers:
            if hasattr(opt, "synchronize_for_checkpoint"):
                opt.synchronize_for_checkpoint()
    """

    @staticmethod
    def _run_sync_logic(optimizer):
        """Reproduce the exact synchronize_for_checkpoint pattern from base_recipe.py."""
        optimizers = optimizer if isinstance(optimizer, list) else [optimizer]
        for opt in optimizers:
            if hasattr(opt, "synchronize_for_checkpoint"):
                opt.synchronize_for_checkpoint()

    def test_single_optimizer_with_sync(self):
        class DionLikeOpt:
            def __init__(self):
                self.sync_called = False

            def synchronize_for_checkpoint(self):
                self.sync_called = True

        opt = DionLikeOpt()
        self._run_sync_logic(opt)
        assert opt.sync_called is True

    def test_single_optimizer_without_sync(self):
        """Regular optimizer without synchronize_for_checkpoint — should not error."""

        class RegularOpt:
            pass

        opt = RegularOpt()
        self._run_sync_logic(opt)  # no error

    def test_list_of_optimizers_all_with_sync(self):
        class DionLikeOpt:
            def __init__(self):
                self.sync_called = False

            def synchronize_for_checkpoint(self):
                self.sync_called = True

        opts = [DionLikeOpt(), DionLikeOpt()]
        self._run_sync_logic(opts)
        assert all(o.sync_called for o in opts)

    def test_list_of_optimizers_mixed(self):
        """Mix of Dion (has sync) and regular (no sync) optimizers."""

        class DionLikeOpt:
            def __init__(self):
                self.sync_called = False

            def synchronize_for_checkpoint(self):
                self.sync_called = True

        class RegularOpt:
            pass

        dion_opt = DionLikeOpt()
        regular_opt = RegularOpt()
        self._run_sync_logic([dion_opt, regular_opt])
        assert dion_opt.sync_called is True
        # regular_opt has no sync method — just verify no error

    def test_empty_list(self):
        """Empty optimizer list — should not error."""
        self._run_sync_logic([])


# ---------------------------------------------------------------------------
# Tests for the OptimizerFromFactoryConfig escape hatch via build_optimizer
# ---------------------------------------------------------------------------


class TestBuildOptimizerFactoryConfig:
    """The factory escape hatch calls ``factory(params=..., **kwargs)`` directly,
    once per model part (Dion-family optimizers use the typed MuonConfig instead).
    """

    @staticmethod
    def _make_simple_model():
        return nn.Linear(4, 4)

    @staticmethod
    def _make_parts_model():
        class PartsModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.part1 = nn.Linear(4, 4)
                self.part2 = nn.Linear(4, 4)
                self.parts = [self.part1, self.part2]

            def forward(self, x):
                return self.part2(self.part1(x))

        return PartsModel()

    def test_factory_single_model(self):
        from nemo_automodel.components.optim.optimizer import OptimizerFromFactoryConfig, build_optimizer

        model = self._make_simple_model()
        instantiate_calls = []

        def fake_optimizer_factory(params=None, **kwargs):
            instantiate_calls.append(params)
            return "regular_opt"

        optimizers = build_optimizer(model, OptimizerFromFactoryConfig(factory=fake_optimizer_factory))

        assert len(instantiate_calls) == 1
        assert len(instantiate_calls[0]) > 0  # trainable params passed
        assert optimizers == ["regular_opt"]

    def test_factory_with_parts(self):
        from nemo_automodel.components.optim.optimizer import OptimizerFromFactoryConfig, build_optimizer

        model = self._make_parts_model()
        instantiate_calls = []

        def fake_optimizer_factory(params=None, **kwargs):
            instantiate_calls.append(params)
            return f"opt_{len(instantiate_calls)}"

        optimizers = build_optimizer(model, OptimizerFromFactoryConfig(factory=fake_optimizer_factory))

        assert len(instantiate_calls) == 2
        assert len(optimizers) == 2

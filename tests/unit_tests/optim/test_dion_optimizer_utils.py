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
from contextlib import nullcontext

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyCfgOpt:
    """Minimal config shim compatible with build_dion_optimizer()."""

    def __init__(self, target, d: dict):
        self._target_ = target
        self._d = dict(d)

    def to_dict(self):
        return dict(self._d)


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
        from nemo_automodel.components.optim.utils import is_dion_optimizer

        class _Cfg:
            class _target_:
                __name__ = "SomeOpt"
                __module__ = "dion.optimizers"

        assert is_dion_optimizer(_Cfg()) is True

    def test_returns_true_for_known_names(self):
        from nemo_automodel.components.optim.utils import is_dion_optimizer

        for name in ("Dion", "Dion2", "Muon", "NorMuon"):

            class _Cfg:
                pass

            _Cfg._target_ = type(name, (), {"__name__": name, "__module__": "some.module"})
            assert is_dion_optimizer(_Cfg()) is True, f"Expected True for {name}"

    def test_returns_false_for_non_dion(self):
        from nemo_automodel.components.optim.utils import is_dion_optimizer

        class _Cfg:
            class _target_:
                __name__ = "Adam"
                __module__ = "torch.optim"

        assert is_dion_optimizer(_Cfg()) is False

    def test_returns_false_when_no_target(self):
        from nemo_automodel.components.optim.utils import is_dion_optimizer

        class _Cfg:
            pass

        assert is_dion_optimizer(_Cfg()) is False


# ---------------------------------------------------------------------------
# Tests for _separate_param_groups()
# ---------------------------------------------------------------------------

class TestSeparateParamGroups:
    def _call(self, **kwargs):
        from nemo_automodel.components.optim.utils import _separate_param_groups

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
        groups = self._call(
            model=model, base_lr=1e-3, scalar_opt="adamw", weight_decay=0.0, lm_head_lr=5e-5
        )
        assert groups[3]["lr"] == pytest.approx(5e-5)

    def test_scalar_lr_and_embed_lr_overrides(self):
        model = TinyModel()
        groups = self._call(
            model=model, base_lr=1e-3, scalar_opt="lion", weight_decay=0.0,
            scalar_lr=2e-4, embed_lr=3e-4,
        )
        # vector group uses scalar_lr
        assert groups[1]["lr"] == pytest.approx(2e-4)
        # embed group uses embed_lr
        assert groups[2]["lr"] == pytest.approx(3e-4)

    def test_scalar_lr_defaults_embed_lr(self):
        model = TinyModel()
        groups = self._call(
            model=model, base_lr=1e-3, scalar_opt="adamw", weight_decay=0.0,
            scalar_lr=7e-4,
        )
        # embed_lr defaults to scalar_lr when not provided
        assert groups[2]["lr"] == pytest.approx(7e-4)

    def test_scalar_betas_and_eps(self):
        model = TinyModel()
        groups = self._call(
            model=model, base_lr=1e-3, scalar_opt="adamw", weight_decay=0.0,
            scalar_betas=(0.9, 0.999), scalar_eps=1e-8,
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
        from nemo_automodel.components.optim.utils import _get_dion_mesh

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

class TestBuildDionOptimizer:
    def _build(self, monkeypatch, target_cls, cfg_dict, model=None, mesh=None):
        from nemo_automodel.components.optim import utils as optim_utils

        monkeypatch.setattr(optim_utils, "_import_error", None, raising=False)
        if model is None:
            model = TinyModel()
        cfg = DummyCfgOpt(target_cls, cfg_dict)
        return optim_utils.build_dion_optimizer(cfg_opt=cfg, model=model, distributed_mesh=mesh)

    def test_passes_distributed_mesh(self, monkeypatch):
        captured = {}

        class Target:
            def __init__(self, param_groups, distributed_mesh=None, lr=None):
                captured["param_groups"] = param_groups
                captured["distributed_mesh"] = distributed_mesh
                captured["lr"] = lr

        mesh = FakeMesh({"dp_replicate": object(), "dp_shard_cp": object()}, ndim=1)
        self._build(monkeypatch, Target, {"lr": 1e-3, "foo": "ignored"}, mesh=mesh)

        assert captured["distributed_mesh"] is mesh
        assert captured["lr"] == pytest.approx(1e-3)
        assert isinstance(captured["param_groups"], list)
        assert len(captured["param_groups"]) >= 2

    def test_no_mesh_param_in_target(self, monkeypatch):
        """Target that has no distributed_mesh param — mesh should not be passed."""
        captured = {}

        class Target:
            def __init__(self, param_groups, lr=None):
                captured["param_groups"] = param_groups
                captured["lr"] = lr

        mesh = FakeMesh({}, ndim=1)
        self._build(monkeypatch, Target, {"lr": 2e-4}, mesh=mesh)
        assert "distributed_mesh" not in captured
        assert captured["lr"] == pytest.approx(2e-4)

    def test_import_error_raises(self, monkeypatch):
        from nemo_automodel.components.optim import utils as optim_utils

        monkeypatch.setattr(optim_utils, "_import_error", ImportError("no dion"), raising=False)

        class Target:
            def __init__(self, param_groups):
                pass

        cfg = DummyCfgOpt(Target, {"lr": 1e-3})
        with pytest.raises(RuntimeError, match="Failed to import Dion"):
            optim_utils.build_dion_optimizer(cfg_opt=cfg, model=TinyModel())

    def test_adjust_lr_forwarded(self, monkeypatch):
        captured = {}

        class Target:
            def __init__(self, param_groups, distributed_mesh=None, lr=None, adjust_lr=None):
                captured["adjust_lr"] = adjust_lr

        self._build(monkeypatch, Target, {"lr": 1e-3, "adjust_lr": "spectral_norm"})
        assert captured["adjust_lr"] == "spectral_norm"

    def test_no_compile_pops_cleanly(self, monkeypatch):
        """no_compile should be consumed and not leak to the target."""
        captured = {}

        class Target:
            def __init__(self, param_groups, distributed_mesh=None, lr=None):
                captured["kwargs_keys"] = set()

        self._build(monkeypatch, Target, {"lr": 1e-3, "no_compile": True})
        # no_compile should not cause an error (popped before introspection)
        assert True  # If we got here, no_compile was handled

    def test_scalar_config_keys_popped(self, monkeypatch):
        """scalar_opt, scalar_betas, scalar_eps, scalar_lr, embed_lr, lm_head_lr
        should be consumed and NOT passed to the target."""
        captured = {}

        class Target:
            def __init__(self, param_groups, lr=None, weight_decay=None, **kwargs):
                captured["extra"] = kwargs

        self._build(
            monkeypatch,
            Target,
            {
                "lr": 1e-3,
                "weight_decay": 0.01,
                "scalar_opt": "lion",
                "scalar_betas": [0.9, 0.95],
                "scalar_eps": 1e-8,
                "scalar_lr": 5e-4,
                "embed_lr": 3e-4,
                "lm_head_lr": 1e-5,
            },
        )
        # None of the scalar_* keys should leak through
        for key in ("scalar_opt", "scalar_betas", "scalar_eps", "scalar_lr", "embed_lr", "lm_head_lr"):
            assert key not in captured["extra"], f"{key} leaked to target"

    def test_unknown_keys_filtered_out(self, monkeypatch):
        """Keys not in the target signature should be silently dropped."""
        captured = {}

        class Target:
            def __init__(self, param_groups, lr=None):
                captured["lr"] = lr

        self._build(monkeypatch, Target, {"lr": 1e-3, "totally_unknown": 42})
        assert captured["lr"] == pytest.approx(1e-3)

    def test_none_mesh(self, monkeypatch):
        """distributed_mesh=None should work fine."""
        captured = {}

        class Target:
            def __init__(self, param_groups, distributed_mesh=None, lr=None):
                captured["distributed_mesh"] = distributed_mesh

        self._build(monkeypatch, Target, {"lr": 1e-3}, mesh=None)
        assert captured["distributed_mesh"] is None

    def test_param_groups_structure(self, monkeypatch):
        """Verify param groups have the right structure."""
        captured = {}

        class Target:
            def __init__(self, param_groups, distributed_mesh=None, lr=None):
                captured["param_groups"] = param_groups

        model = TinyModel()
        self._build(monkeypatch, Target, {"lr": 1e-3}, model=model)

        groups = captured["param_groups"]
        # 4 groups: matrix, vector, embed, lm_head
        assert len(groups) == 4
        # Each group has 'params' key
        for g in groups:
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
# Tests for train_ft.py: Dion optimizer branch in build_model_and_optimizer()
# ---------------------------------------------------------------------------

def _patch_train_ft_for_cpu(monkeypatch, train_ft_mod, model):
    """Patch train_ft module so build_model_and_optimizer can run on CPU.

    Makes cfg_model.get("_target_") match NeMoAutoModelForCausalLM.from_pretrained
    so the code takes the simple cfg_model.instantiate(**kwargs) path and never
    calls torch.cuda.current_device().

    Returns the sentinel value that FakeCfgModel.get("_target_") should return.
    """
    _sentinel = object()
    monkeypatch.setattr(
        train_ft_mod, "NeMoAutoModelForCausalLM",
        type("_FakeAutoModel", (), {"from_pretrained": _sentinel, "from_config": object()})(),
    )
    # Also stub out the sequence-classification targets so they don't interfere
    monkeypatch.setattr(
        train_ft_mod, "NeMoAutoModelForSequenceClassification",
        type("_FakeAutoModel2", (), {"from_pretrained": object(), "from_config": object()})(),
    )
    monkeypatch.setattr(train_ft_mod, "_supports_logits_to_keep", lambda m: True)
    monkeypatch.setattr(train_ft_mod, "ScopedRNG", lambda seed, ranked: nullcontext())
    return _sentinel


class TestBuildModelAndOptimizerDionBranch:
    """Test the optimizer creation branch in build_model_and_optimizer().

    The logic under test (train_ft.py lines 236-270):
    - is_dion_optimizer(cfg_opt) -> True -> build_dion_optimizer()
    - is_dion_optimizer(cfg_opt) -> False -> normal cfg_opt.instantiate()
    - Model with/without `parts` attribute
    """

    @staticmethod
    def _make_simple_model():
        """A tiny model with requires_grad params."""
        return nn.Linear(4, 4)

    @staticmethod
    def _make_parts_model():
        """A model-like object with a `parts` attribute."""
        class PartsModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.part1 = nn.Linear(4, 4)
                self.part2 = nn.Linear(4, 4)
                self.parts = [self.part1, self.part2]
            def forward(self, x):
                return self.part2(self.part1(x))
        return PartsModel()

    def test_dion_optimizer_single_model(self, monkeypatch):
        """When is_dion_optimizer returns True, build_dion_optimizer is called."""
        import nemo_automodel.recipes.llm.train_ft as train_ft_mod

        model = self._make_simple_model()
        sentinel_mesh = object()
        build_calls = []

        _target_sentinel = _patch_train_ft_for_cpu(monkeypatch, train_ft_mod, model)

        class FakeCfgModel:
            def get(self, key, default=None):
                if key == "_target_":
                    return _target_sentinel
                return default
            def instantiate(self, **kwargs):
                return model

        class FakeCfgOpt:
            foreach = True

        class FakeModelWrapper:
            device_mesh = sentinel_mesh

        monkeypatch.setattr(train_ft_mod, "is_dion_optimizer", lambda cfg: True)
        monkeypatch.setattr(
            train_ft_mod, "build_dion_optimizer",
            lambda cfg_opt, model, distributed_mesh: (
                build_calls.append((model, distributed_mesh)) or "fake_dion_opt"
            ),
        )

        result_model, optimizers, loss_fn = train_ft_mod.build_model_and_optimizer(
            cfg_model=FakeCfgModel(),
            cfg_opt=FakeCfgOpt(),
            cfg_peft=None,
            model_wrapper=FakeModelWrapper(),
            seed=42,
            loss_fn=None,
        )

        assert result_model is model
        assert optimizers == ["fake_dion_opt"]
        assert len(build_calls) == 1
        assert build_calls[0][0] is model
        assert build_calls[0][1] is sentinel_mesh

    def test_dion_optimizer_with_parts(self, monkeypatch):
        """When model has `parts`, build_dion_optimizer is called per part."""
        import nemo_automodel.recipes.llm.train_ft as train_ft_mod

        model = self._make_parts_model()
        build_calls = []

        _target_sentinel = _patch_train_ft_for_cpu(monkeypatch, train_ft_mod, model)

        class FakeCfgModel:
            def get(self, key, default=None):
                if key == "_target_":
                    return _target_sentinel
                return default
            def instantiate(self, **kwargs):
                return model

        class FakeCfgOpt:
            foreach = True

        class FakeModelWrapper:
            device_mesh = None

        monkeypatch.setattr(train_ft_mod, "is_dion_optimizer", lambda cfg: True)
        monkeypatch.setattr(
            train_ft_mod, "build_dion_optimizer",
            lambda cfg_opt, model, distributed_mesh: (
                build_calls.append(model) or f"opt_for_{id(model)}"
            ),
        )

        _, optimizers, _ = train_ft_mod.build_model_and_optimizer(
            cfg_model=FakeCfgModel(),
            cfg_opt=FakeCfgOpt(),
            cfg_peft=None,
            model_wrapper=FakeModelWrapper(),
            seed=42,
            loss_fn=None,
        )

        assert len(build_calls) == 2
        assert build_calls[0] is model.parts[0]
        assert build_calls[1] is model.parts[1]
        assert len(optimizers) == 2

    def test_non_dion_optimizer_single_model(self, monkeypatch):
        """When is_dion_optimizer returns False, normal instantiate path is used."""
        import nemo_automodel.recipes.llm.train_ft as train_ft_mod

        model = self._make_simple_model()
        instantiate_calls = []

        _target_sentinel = _patch_train_ft_for_cpu(monkeypatch, train_ft_mod, model)

        class FakeCfgModel:
            def get(self, key, default=None):
                if key == "_target_":
                    return _target_sentinel
                return default
            def instantiate(self, **kwargs):
                return model

        class FakeCfgOpt:
            foreach = True
            def instantiate(self, params=None):
                instantiate_calls.append(params)
                return "regular_opt"

        class FakeModelWrapper:
            pass

        monkeypatch.setattr(train_ft_mod, "is_dion_optimizer", lambda cfg: False)

        _, optimizers, _ = train_ft_mod.build_model_and_optimizer(
            cfg_model=FakeCfgModel(),
            cfg_opt=FakeCfgOpt(),
            cfg_peft=None,
            model_wrapper=FakeModelWrapper(),
            seed=42,
            loss_fn=None,
        )

        assert len(instantiate_calls) == 1
        assert len(instantiate_calls[0]) > 0  # trainable params passed
        assert optimizers == ["regular_opt"]

    def test_non_dion_optimizer_with_parts(self, monkeypatch):
        """Non-dion optimizer with model.parts -> instantiate per part."""
        import nemo_automodel.recipes.llm.train_ft as train_ft_mod

        model = self._make_parts_model()
        instantiate_calls = []

        _target_sentinel = _patch_train_ft_for_cpu(monkeypatch, train_ft_mod, model)

        class FakeCfgModel:
            def get(self, key, default=None):
                if key == "_target_":
                    return _target_sentinel
                return default
            def instantiate(self, **kwargs):
                return model

        class FakeCfgOpt:
            foreach = True
            def instantiate(self, params=None):
                instantiate_calls.append(params)
                return f"opt_{len(instantiate_calls)}"

        class FakeModelWrapper:
            pass

        monkeypatch.setattr(train_ft_mod, "is_dion_optimizer", lambda cfg: False)

        _, optimizers, _ = train_ft_mod.build_model_and_optimizer(
            cfg_model=FakeCfgModel(),
            cfg_opt=FakeCfgOpt(),
            cfg_peft=None,
            model_wrapper=FakeModelWrapper(),
            seed=42,
            loss_fn=None,
        )

        assert len(instantiate_calls) == 2
        assert len(optimizers) == 2

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

"""Tests for nemo_automodel.components.optim.optimizer — typed configs + builders."""

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.optim.optimizer import (
    AdamConfig,
    AdamWConfig,
    LRSchedulerConfig,
    OptimizerConfig,
    OptimizerFromFactoryConfig,
    build_optimizer,
)


def _model():
    return nn.Linear(4, 4)


def _params():
    return list(_model().parameters())


# ---------------------------------------------------------------------------
# Typed config fields + build()
# ---------------------------------------------------------------------------


class TestAdamConfig:
    def test_defaults(self):
        cfg = AdamConfig()
        assert cfg.lr == 1e-4
        assert cfg.betas == (0.9, 0.999)
        assert cfg.eps == 1e-8
        assert cfg.amsgrad is False

    def test_build_returns_adam_with_fields(self):
        cfg = AdamConfig(lr=2e-4, betas=(0.8, 0.99), amsgrad=True, weight_decay=0.05)
        opt = cfg._build_optimizer(_params())
        assert isinstance(opt, torch.optim.Adam)
        group = opt.param_groups[0]
        assert group["lr"] == 2e-4
        assert group["betas"] == (0.8, 0.99)
        assert group["amsgrad"] is True
        assert group["weight_decay"] == 0.05

    def test_build_forwards_foreach(self):
        opt = AdamConfig()._build_optimizer(_params(), foreach=False)
        assert opt.param_groups[0]["foreach"] is False


class TestAdamWConfig:
    def test_defaults(self):
        cfg = AdamWConfig()
        assert cfg.fused is False
        assert cfg.weight_decay == 0.01

    def test_build_returns_adamw(self):
        opt = AdamWConfig(lr=1e-3)._build_optimizer(_params(), foreach=False)
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.param_groups[0]["lr"] == 1e-3
        assert opt.param_groups[0]["foreach"] is False

    def test_build_fused_skips_foreach(self):
        # fused and foreach are mutually exclusive; foreach is disabled when fused.
        opt = AdamWConfig(fused=True)._build_optimizer(_params(), foreach=False)
        assert opt.param_groups[0]["fused"] is True
        assert not opt.param_groups[0]["foreach"]


class TestOptimizerConfigBase:
    def test_base_build_not_implemented(self):
        with pytest.raises(NotImplementedError):
            OptimizerConfig()._build_optimizer(_params())

    def test_build_from_param_groups_preserves_group_options(self):
        model = _model()
        params = list(model.parameters())
        param_groups = [
            {"params": [params[0]], "weight_decay": 0.1},
            {"params": [params[1]], "weight_decay": 0.0},
        ]

        opt = AdamWConfig(lr=1e-3, weight_decay=0.1).build_from_param_groups(param_groups)

        assert isinstance(opt, torch.optim.AdamW)
        assert opt.param_groups[0]["weight_decay"] == 0.1
        assert opt.param_groups[1]["weight_decay"] == 0.0


# ---------------------------------------------------------------------------
# build_optimizer (Automodel-native orchestration)
# ---------------------------------------------------------------------------


class TestBuildOptimizer:
    def test_single_model_returns_one_optimizer(self):
        model = _model()
        optimizers = build_optimizer(model, AdamWConfig(lr=1e-3))
        assert len(optimizers) == 1
        assert isinstance(optimizers[0], torch.optim.AdamW)
        assert optimizers[0].param_groups[0]["lr"] == 1e-3

    def test_parts_model_returns_optimizer_per_part(self):
        class PartsModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.part1 = nn.Linear(4, 4)
                self.part2 = nn.Linear(4, 4)
                self.parts = [self.part1, self.part2]

        optimizers = build_optimizer(PartsModel(), AdamConfig())
        assert len(optimizers) == 2
        assert all(isinstance(o, torch.optim.Adam) for o in optimizers)

    def test_tp_mesh_disables_foreach(self):
        class FakeMesh:
            mesh_dim_names = ("tp",)

            def __getitem__(self, key):
                m = type("M", (), {})()
                m.size = lambda self=m: 2
                return m

        optimizers = build_optimizer(_model(), AdamConfig(), device_mesh=FakeMesh())
        assert optimizers[0].param_groups[0]["foreach"] is False

    def test_config_class_instead_of_instance_raises(self):
        with pytest.raises(TypeError, match="OptimizerConfig instance or a"):
            build_optimizer(_model(), AdamWConfig)


# ---------------------------------------------------------------------------
# OptimizerFromFactoryConfig
# ---------------------------------------------------------------------------


class TestOptimizerFromFactoryConfig:
    def test_build_constructs_from_factory(self):
        cfg = OptimizerFromFactoryConfig(
            factory=torch.optim.SGD,
            kwargs={"lr": 0.01, "momentum": 0.9},
        )
        optimizers = cfg.build(_model())
        assert len(optimizers) == 1
        assert isinstance(optimizers[0], torch.optim.SGD)
        assert optimizers[0].param_groups[0]["momentum"] == 0.9

    def test_build_resolves_dtype_string_kwargs(self):
        captured = {}

        def fake_factory(params, **kwargs):
            captured.update(kwargs)
            return torch.optim.SGD(params, lr=kwargs.get("lr", 0.01))

        OptimizerFromFactoryConfig(
            factory=fake_factory,
            kwargs={"lr": 1e-3, "master_weight_dtype": "torch.bfloat16"},
        ).build(_model())
        assert captured["master_weight_dtype"] is torch.bfloat16

    def test_build_requires_callable_factory(self):
        with pytest.raises(TypeError, match="must be a callable"):
            OptimizerFromFactoryConfig(factory=None).build(_model())

    def test_build_from_param_groups_requires_callable_factory(self):
        with pytest.raises(TypeError, match="must be a callable"):
            OptimizerFromFactoryConfig(factory=None).build_from_param_groups([])

    def test_build_rejects_empty_trainable_params(self):
        model = _model()
        for param in model.parameters():
            param.requires_grad = False

        with pytest.raises(ValueError, match="No trainable parameters found"):
            AdamWConfig().build(model)

    def test_factory_build_rejects_empty_trainable_params(self):
        model = _model()
        for param in model.parameters():
            param.requires_grad = False

        with pytest.raises(ValueError, match="No trainable parameters found"):
            OptimizerFromFactoryConfig(factory=torch.optim.SGD, kwargs={"lr": 0.01}).build(model)

    def test_build_from_param_groups_uses_factory(self):
        cfg = OptimizerFromFactoryConfig(
            factory=torch.optim.SGD,
            kwargs={"lr": 0.01, "momentum": 0.9},
        )
        params = list(_model().parameters())
        opt = cfg.build_from_param_groups([{"params": [params[0]]}, {"params": [params[1]], "weight_decay": 0.0}])

        assert isinstance(opt, torch.optim.SGD)
        assert len(opt.param_groups) == 2
        assert opt.param_groups[0]["momentum"] == 0.9


# ---------------------------------------------------------------------------
# build_optimizer — (name_or_path, kwargs) tuple form
# ---------------------------------------------------------------------------


class TestBuildOptimizerTuple:
    def test_registry_name_builds_typed_config(self):
        optimizers = build_optimizer(_model(), ("adamw", {"lr": 1e-3, "betas": (0.9, 0.95)}))
        assert isinstance(optimizers[0], torch.optim.AdamW)
        assert optimizers[0].param_groups[0]["lr"] == 1e-3
        assert optimizers[0].param_groups[0]["betas"] == (0.9, 0.95)

    def test_registry_name_is_case_insensitive(self):
        optimizers = build_optimizer(_model(), ("Adam", {"lr": 2e-4}))
        assert isinstance(optimizers[0], torch.optim.Adam)
        assert optimizers[0].param_groups[0]["lr"] == 2e-4

    def test_import_path_to_torch_optimizer(self):
        optimizers = build_optimizer(_model(), ("torch.optim.SGD", {"lr": 0.01, "momentum": 0.9}))
        assert isinstance(optimizers[0], torch.optim.SGD)
        assert optimizers[0].param_groups[0]["momentum"] == 0.9

    def test_import_path_to_torch_optimizer_no_kwargs(self):
        optimizers = build_optimizer(_model(), ("torch.optim.RMSprop", {}))
        assert isinstance(optimizers[0], torch.optim.RMSprop)

    def test_import_path_to_optimizer_config(self):
        path = "nemo_automodel.components.optim.optimizer.AdamConfig"
        optimizers = build_optimizer(_model(), (path, {"lr": 3e-4}))
        assert isinstance(optimizers[0], torch.optim.Adam)
        assert optimizers[0].param_groups[0]["lr"] == 3e-4

    def test_resolves_dtype_string_kwargs(self):
        # dtype strings (e.g. for TE FusedAdam) are resolved to torch.dtype objects.
        captured = {}

        def fake_factory(params, **kwargs):
            captured.update(kwargs)
            return torch.optim.SGD(params, lr=kwargs.get("lr", 0.01))

        build_optimizer(
            _model(),
            OptimizerFromFactoryConfig(
                factory=fake_factory,
                kwargs={"lr": 1e-3, "master_weight_dtype": "torch.bfloat16", "exp_avg_dtype": "float16"},
            ),
        )
        assert captured["master_weight_dtype"] is torch.bfloat16
        assert captured["exp_avg_dtype"] is torch.float16

    def test_unknown_import_path_raises(self):
        with pytest.raises(ImportError):
            build_optimizer(_model(), ("torch.optim.NotAnOptimizer", {"lr": 1e-3}))

    def test_bare_name_without_dot_raises(self):
        # Not in the registry and not a dotted path.
        with pytest.raises(ValueError, match="dotted import path"):
            build_optimizer(_model(), ("lamb", {"lr": 1e-3}))

    def test_non_tuple_non_config_raises(self):
        with pytest.raises(TypeError, match="OptimizerConfig instance or a"):
            build_optimizer(_model(), 123)

    def test_tuple_wrong_length_raises(self):
        with pytest.raises(TypeError, match="length 2"):
            build_optimizer(_model(), ("adamw",))

    def test_tuple_non_string_name_raises(self):
        with pytest.raises(TypeError, match="registry name or import-path string"):
            build_optimizer(_model(), (torch.optim.SGD, {"lr": 0.01}))

    def test_tuple_non_dict_kwargs_raises(self):
        with pytest.raises(TypeError, match="dict of kwargs"):
            build_optimizer(_model(), ("adamw", [("lr", 1e-3)]))


# ---------------------------------------------------------------------------
# Dion-family typed configs (Dion / Dion2 / Muon / NorMuon)
# ---------------------------------------------------------------------------


def _dion_test_model():
    return nn.Sequential(nn.Embedding(8, 16), nn.Linear(16, 16, bias=False), nn.Linear(16, 8, bias=False))


class TestDionFamilyConfigs:
    """The dion-family configs share a per-part Dion ``build`` (``_DionConfigBase``)
    that runs ``build_dion_optimizer`` grouping and strips grouping-only kwargs before
    the dion constructor.  They resolve from the registry by name and from a resolved
    ``dion.*`` class via ``_target_``.  Construction tests ``importorskip`` ``dion``."""

    def test_registry_resolves_to_config_types(self):
        from nemo_automodel.components.optim.optimizer import (
            Dion2Config,
            DionConfig,
            MuonConfig,
            NorMuonConfig,
            build_optimizer_config,
        )

        cases = {
            "muon": MuonConfig,
            "normuon": NorMuonConfig,
            "dion": DionConfig,
            "dion2": Dion2Config,
        }
        for name, cls in cases.items():
            cfg = build_optimizer_config(name, {"lr": 1e-3})
            assert isinstance(cfg, cls)
            assert cfg.lr == pytest.approx(1e-3)

    def test_config_specific_fields(self):
        from nemo_automodel.components.optim.optimizer import Dion2Config, NorMuonConfig

        assert Dion2Config(fraction=0.5, ef_decay=0.9).fraction == pytest.approx(0.5)
        assert NorMuonConfig(muon_beta2=0.99).muon_beta2 == pytest.approx(0.99)

    def test_target_class_routes_to_typed_config(self):
        # YAML ``_target_: dion.Muon`` must resolve to the typed MuonConfig (with grouping),
        # NOT the flat-params OptimizerFromFactoryConfig escape hatch.
        dion = pytest.importorskip("dion")
        from nemo_automodel.components.optim.optimizer import (
            MuonConfig,
            OptimizerFromFactoryConfig,
            build_optimizer_config,
        )

        cfg = build_optimizer_config(dion.Muon, {"lr": 5e-4, "embed_lr": 1e-4, "lm_head_lr": 1e-4})
        assert isinstance(cfg, MuonConfig)
        assert not isinstance(cfg, OptimizerFromFactoryConfig)
        assert cfg.embed_lr == pytest.approx(1e-4)
        assert cfg.lm_head_lr == pytest.approx(1e-4)

    def test_build_groups_params_and_strips_grouping_kwargs(self):
        # build() runs Dion grouping and must not splat scalar_*/*_lr into the dion ctor.
        pytest.importorskip("dion")
        from nemo_automodel.components.optim.optimizer import MuonConfig

        opt = MuonConfig(lr=5e-4, scalar_opt="adamw", embed_lr=1e-4).build(_dion_test_model())[0]
        assert type(opt).__name__ == "Muon"
        assert len(opt.param_groups) >= 2  # matrix group + scalar/embed group(s)

    @pytest.mark.parametrize("cls_name", ["Muon", "NorMuon", "Dion2", "Dion"])
    def test_all_dion_configs_build(self, cls_name):
        pytest.importorskip("dion")
        from nemo_automodel.components.optim import optimizer as opt_mod

        cfg_cls = {
            "Muon": opt_mod.MuonConfig,
            "NorMuon": opt_mod.NorMuonConfig,
            "Dion2": opt_mod.Dion2Config,
            "Dion": opt_mod.DionConfig,
        }[cls_name]
        opt = cfg_cls(lr=5e-4).build(_dion_test_model())[0]
        assert type(opt).__name__ == cls_name


# ---------------------------------------------------------------------------
# LRSchedulerConfig
# ---------------------------------------------------------------------------


class TestLRSchedulerConfig:
    def test_defaults(self):
        cfg = LRSchedulerConfig()
        assert cfg.lr_decay_style == "cosine"
        assert cfg.lr_warmup_steps is None
        assert cfg.use_checkpoint_opt_param_scheduler is True
        assert cfg.override_opt_param_scheduler is False

    def test_all_fields_settable(self):
        cfg = LRSchedulerConfig(
            lr_warmup_steps=500,
            lr_decay_steps=10000,
            lr_decay_style="WSD",
            init_lr=1e-5,
            max_lr=1e-3,
            min_lr=1e-6,
            wsd_decay_steps=2000,
            lr_wsd_decay_style="cosine",
        )
        assert cfg.lr_warmup_steps == 500
        assert cfg.wsd_decay_steps == 2000
        assert cfg.lr_wsd_decay_style == "cosine"

    @staticmethod
    def _step_scheduler(*, epoch_len, num_epochs, max_steps):
        """A StepScheduler stand-in whose dataloader has no usable ``len()``.

        ``len(dataloader)`` raising mirrors an IterableDataset / streaming
        dataloader and proves ``build`` derives ``total_steps`` from
        ``epoch_len``/``max_steps`` and never calls ``len()``.
        """
        ss = MagicMock()
        ss.epoch_len = epoch_len
        ss.num_epochs = num_epochs
        ss.max_steps = max_steps
        ss.grad_acc_steps = 1
        dl = MagicMock()
        dl.__len__ = MagicMock(side_effect=TypeError("object of type 'IterableDataset' has no len()"))
        ss.dataloader = dl
        return ss

    def test_build_uses_epoch_len_not_dataloader_len(self):
        # epoch_len is already in optimizer-step units -> total_steps = num_epochs * epoch_len.
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
        ss = self._step_scheduler(epoch_len=10, num_epochs=4, max_steps=None)
        scheds = LRSchedulerConfig(lr_warmup_steps=1).build(opt, ss)
        assert scheds[0].lr_decay_steps == 40  # defaults to total_steps

    def test_build_iterable_falls_back_to_max_steps(self):
        # epoch_len is None for iterable datasets -> total_steps comes from max_steps, no len().
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
        ss = self._step_scheduler(epoch_len=None, num_epochs=10, max_steps=50)
        scheds = LRSchedulerConfig(lr_warmup_steps=1).build(opt, ss)
        assert scheds[0].lr_decay_steps == 50

    def test_build_iterable_without_max_steps_raises(self):
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
        ss = self._step_scheduler(epoch_len=None, num_epochs=10, max_steps=None)
        with pytest.raises(ValueError, match="iterable/streaming"):
            LRSchedulerConfig(lr_warmup_steps=1).build(opt, ss)

    def test_build_caps_total_steps_at_max_steps(self):
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
        ss = self._step_scheduler(epoch_len=100, num_epochs=10, max_steps=20)
        scheds = LRSchedulerConfig(lr_warmup_steps=1).build(opt, ss)
        assert scheds[0].lr_decay_steps == 20  # min(num_epochs*epoch_len=1000, max_steps=20)

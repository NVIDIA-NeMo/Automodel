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

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.optim.optimizer import (
    AdamConfig,
    AdamWConfig,
    LRSchedulerConfig,
    OptimizerConfig,
    _resolve_dotted_path,
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
        opt = cfg.build(_params())
        assert isinstance(opt, torch.optim.Adam)
        group = opt.param_groups[0]
        assert group["lr"] == 2e-4
        assert group["betas"] == (0.8, 0.99)
        assert group["amsgrad"] is True
        assert group["weight_decay"] == 0.05

    def test_build_forwards_foreach(self):
        opt = AdamConfig().build(_params(), foreach=False)
        assert opt.param_groups[0]["foreach"] is False


class TestAdamWConfig:
    def test_defaults(self):
        cfg = AdamWConfig()
        assert cfg.fused is False
        assert cfg.weight_decay == 0.01

    def test_build_returns_adamw(self):
        opt = AdamWConfig(lr=1e-3).build(_params(), foreach=False)
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.param_groups[0]["lr"] == 1e-3
        assert opt.param_groups[0]["foreach"] is False

    def test_build_fused_skips_foreach(self):
        # fused and foreach are mutually exclusive; foreach must not be forwarded.
        opt = AdamWConfig(fused=True).build(_params(), foreach=False)
        assert opt.param_groups[0]["fused"] is True
        assert opt.param_groups[0]["foreach"] is None


class TestOptimizerConfigBase:
    def test_base_build_not_implemented(self):
        with pytest.raises(NotImplementedError):
            OptimizerConfig().build(_params())


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

    def test_config_with_kwargs_raises(self):
        with pytest.raises(ValueError, match="must be set on the config"):
            build_optimizer(_model(), AdamWConfig(), lr=1e-3)

    def test_config_class_instead_of_instance_raises(self):
        with pytest.raises(TypeError, match="instance, not the class"):
            build_optimizer(_model(), AdamWConfig)


# ---------------------------------------------------------------------------
# build_optimizer — dotted-path / class form (integration escape hatch)
# ---------------------------------------------------------------------------


class TestBuildOptimizerEscapeHatch:
    def test_dotted_path_string(self):
        optimizers = build_optimizer(_model(), "torch.optim.AdamW", lr=1e-3, betas=(0.9, 0.95))
        assert isinstance(optimizers[0], torch.optim.AdamW)
        assert optimizers[0].param_groups[0]["lr"] == 1e-3

    def test_resolved_class(self):
        optimizers = build_optimizer(_model(), torch.optim.SGD, lr=0.01, momentum=0.9)
        assert isinstance(optimizers[0], torch.optim.SGD)
        assert optimizers[0].param_groups[0]["momentum"] == 0.9

    def test_arbitrary_new_optimizer_no_typed_config(self):
        # Adding support for a new optimizer requires no code change here:
        # any importable optimizer + kwargs works.
        optimizers = build_optimizer(_model(), "torch.optim.RMSprop", lr=0.01, alpha=0.95)
        assert isinstance(optimizers[0], torch.optim.RMSprop)
        assert optimizers[0].param_groups[0]["alpha"] == 0.95


class TestResolveDottedPath:
    def test_resolve_adamw(self):
        assert _resolve_dotted_path("torch.optim.AdamW") is torch.optim.AdamW

    def test_bad_path_no_dot(self):
        with pytest.raises(ValueError, match="Expected a dotted path"):
            _resolve_dotted_path("AdamW")

    def test_bad_class(self):
        with pytest.raises(ImportError, match="Cannot find"):
            _resolve_dotted_path("torch.optim.NonExistentOptimizer")


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

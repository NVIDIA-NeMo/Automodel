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

"""Tests for nemo_automodel.components.optim.config — OptimizerConfig hierarchy + LRSchedulerConfig."""

import pytest

from nemo_automodel.components.optim.config import (
    AdamConfig,
    AdamWConfig,
    FlashAdamWConfig,
    FusedAdamConfig,
    LRSchedulerConfig,
    MuonConfig,
    OptimizerConfig,
    _resolve_optimizer,
)

# ---------------------------------------------------------------------------
# _resolve_optimizer
# ---------------------------------------------------------------------------


class TestResolveOptimizer:
    def test_resolve_torch_adamw(self):
        import torch

        cls = _resolve_optimizer("torch.optim.AdamW")
        assert cls is torch.optim.AdamW

    def test_resolve_torch_sgd(self):
        import torch

        cls = _resolve_optimizer("torch.optim.SGD")
        assert cls is torch.optim.SGD

    def test_resolve_bad_path_no_dot(self):
        with pytest.raises(ValueError, match="Expected a dotted path"):
            _resolve_optimizer("AdamW")

    def test_resolve_bad_module(self):
        with pytest.raises(ImportError):
            _resolve_optimizer("nonexistent.module.Foo")

    def test_resolve_bad_class(self):
        with pytest.raises(ImportError, match="Cannot find"):
            _resolve_optimizer("torch.optim.NonExistentOptimizer")


# ---------------------------------------------------------------------------
# OptimizerConfig base
# ---------------------------------------------------------------------------


class TestOptimizerConfig:
    def test_defaults(self):
        cfg = OptimizerConfig()
        assert cfg.name == "torch.optim.AdamW"
        assert cfg.lr == 1e-4
        assert cfg.weight_decay == 0.01
        assert cfg.extra_kwargs == {}

    def test_to_kwargs(self):
        cfg = OptimizerConfig(lr=1e-3, extra_kwargs={"momentum": 0.9})
        kwargs = cfg.to_kwargs()
        assert kwargs == {"lr": 1e-3, "weight_decay": 0.01, "momentum": 0.9}

    def test_extra_kwargs_override(self):
        cfg = OptimizerConfig(extra_kwargs={"weight_decay": 0.05})
        kwargs = cfg.to_kwargs()
        # extra_kwargs should override the base field via dict merge order
        assert kwargs["weight_decay"] == 0.05


# ---------------------------------------------------------------------------
# Typed subclasses — to_kwargs
# ---------------------------------------------------------------------------


class TestAdamConfig:
    def test_defaults(self):
        cfg = AdamConfig()
        assert cfg.name == "torch.optim.Adam"
        assert cfg.betas == (0.9, 0.999)

    def test_to_kwargs(self):
        cfg = AdamConfig(lr=2e-4, betas=(0.8, 0.99), amsgrad=True)
        kwargs = cfg.to_kwargs()
        assert kwargs["lr"] == 2e-4
        assert kwargs["betas"] == (0.8, 0.99)
        assert kwargs["amsgrad"] is True
        assert kwargs["eps"] == 1e-8


class TestAdamWConfig:
    def test_defaults(self):
        cfg = AdamWConfig()
        assert cfg.name == "torch.optim.AdamW"
        assert cfg.fused is False

    def test_to_kwargs_fused(self):
        cfg = AdamWConfig(fused=True)
        assert cfg.to_kwargs()["fused"] is True


class TestFusedAdamConfig:
    def test_defaults(self):
        cfg = FusedAdamConfig()
        assert "transformer_engine" in cfg.name
        assert cfg.adam_w_mode is True
        assert cfg.master_weights is True

    def test_to_kwargs_no_master_dtype(self):
        cfg = FusedAdamConfig()
        kwargs = cfg.to_kwargs()
        assert "master_weight_dtype" not in kwargs

    def test_to_kwargs_with_master_dtype(self):
        cfg = FusedAdamConfig(master_weight_dtype="torch.bfloat16")
        kwargs = cfg.to_kwargs()
        assert kwargs["master_weight_dtype"] == "torch.bfloat16"


class TestFlashAdamWConfig:
    def test_defaults(self):
        cfg = FlashAdamWConfig()
        assert cfg.name == "flashoptim.FlashAdamW"
        assert cfg.master_weight_bits == 24

    def test_to_kwargs(self):
        cfg = FlashAdamWConfig(master_weight_bits=16)
        assert cfg.to_kwargs()["master_weight_bits"] == 16


class TestMuonConfig:
    def test_defaults(self):
        cfg = MuonConfig()
        assert cfg.name == "dion.Muon"
        assert cfg.mu == 0.95
        assert cfg.scalar_opt == "adamw"

    def test_to_kwargs(self):
        cfg = MuonConfig(lr=1e-3, mu=0.9, scalar_betas=(0.8, 0.99))
        kwargs = cfg.to_kwargs()
        assert kwargs["lr"] == 1e-3
        assert kwargs["mu"] == 0.9
        assert kwargs["scalar_betas"] == (0.8, 0.99)
        assert kwargs["adjust_lr"] == "spectral_norm"


# ---------------------------------------------------------------------------
# from_name registry dispatch
# ---------------------------------------------------------------------------


class TestFromName:
    def test_known_adamw(self):
        cfg = OptimizerConfig.from_name("torch.optim.AdamW", lr=1e-3, betas=(0.9, 0.95))
        assert isinstance(cfg, AdamWConfig)
        assert cfg.lr == 1e-3
        assert cfg.betas == (0.9, 0.95)

    def test_known_adam(self):
        cfg = OptimizerConfig.from_name("torch.optim.Adam", lr=2e-4)
        assert isinstance(cfg, AdamConfig)
        assert cfg.lr == 2e-4

    def test_known_fused_adam_long_path(self):
        cfg = OptimizerConfig.from_name("transformer_engine.pytorch.optimizers.fused_adam.FusedAdam")
        assert isinstance(cfg, FusedAdamConfig)

    def test_known_fused_adam_short_path(self):
        cfg = OptimizerConfig.from_name("transformer_engine.pytorch.optimizers.FusedAdam")
        assert isinstance(cfg, FusedAdamConfig)

    def test_known_muon(self):
        cfg = OptimizerConfig.from_name("dion.Muon", lr=5e-4, mu=0.9)
        assert isinstance(cfg, MuonConfig)
        assert cfg.mu == 0.9

    def test_known_flash_adamw(self):
        cfg = OptimizerConfig.from_name("flashoptim.FlashAdamW", master_weight_bits=16)
        assert isinstance(cfg, FlashAdamWConfig)
        assert cfg.master_weight_bits == 16

    def test_unknown_optimizer_fallback(self):
        cfg = OptimizerConfig.from_name("some.custom.Optimizer", lr=1e-4, momentum=0.9, nesterov=True)
        assert type(cfg) is OptimizerConfig
        assert cfg.name == "some.custom.Optimizer"
        assert cfg.lr == 1e-4
        assert cfg.extra_kwargs == {"momentum": 0.9, "nesterov": True}

    def test_unknown_optimizer_weight_decay(self):
        cfg = OptimizerConfig.from_name("some.custom.Optimizer", lr=1e-4, weight_decay=0.05)
        assert cfg.weight_decay == 0.05
        assert "weight_decay" not in cfg.extra_kwargs


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

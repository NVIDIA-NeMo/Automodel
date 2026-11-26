# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""``DraftOptimizerSettings``: the typed ``optimizer:`` block of the speculative-decoding recipes."""

from __future__ import annotations

import pytest
import torch

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.optim.optimizer import (
    AdamConfig,
    MuonConfig,
    MuownConfig,
    OptimizerFromFactoryConfig,
)
from nemo_automodel.recipes._typed_config import DraftOptimizerSettings
from nemo_automodel.shared.import_utils import safe_import_te


def _require_te() -> None:
    has_te, _ = safe_import_te()
    if not has_te:
        pytest.skip("requires the Transformer Engine PyTorch extension")


def test_defaults_to_adamw_when_no_target():
    settings = DraftOptimizerSettings.from_config(ConfigNode({"lr": 6e-4, "warmup_ratio": 0.04, "min_lr_ratio": 0.2}))
    assert isinstance(settings.optimizer, OptimizerFromFactoryConfig)
    assert settings.optimizer.factory is torch.optim.AdamW
    assert settings.optimizer.kwargs == {"lr": 6e-4, "betas": (0.9, 0.95), "weight_decay": 0.0}
    assert (settings.peak_lr, settings.warmup_ratio, settings.min_lr_ratio) == (6e-4, 0.04, 0.2)


def test_schedule_keys_default_when_absent():
    settings = DraftOptimizerSettings.from_config(ConfigNode({"lr": 1e-4}))
    assert (settings.warmup_ratio, settings.min_lr_ratio) == (0.05, 0.1)
    assert "warmup_ratio" not in settings.optimizer.kwargs and "min_lr_ratio" not in settings.optimizer.kwargs


def test_explicit_target_and_extra_kwargs_reach_the_optimizer_config():
    _require_te()
    settings = DraftOptimizerSettings.from_config(
        ConfigNode(
            {
                "_target_": "transformer_engine.pytorch.optimizers.FusedAdam",
                "lr": 1e-5,
                "master_weights": True,
                "master_weight_dtype": "torch.float32",
                "exp_avg_dtype": "torch.float32",
                "exp_avg_sq_dtype": "torch.float32",
                "store_param_remainders": True,
                "warmup_ratio": 0.04,
            }
        )
    )
    assert settings.optimizer.factory.__name__ == "FusedAdam"
    # dtype names are already dtypes in the constructed config, not YAML strings.
    assert settings.optimizer.kwargs == {
        "lr": 1e-5,
        "master_weights": True,
        "master_weight_dtype": torch.float32,
        "exp_avg_dtype": torch.float32,
        "exp_avg_sq_dtype": torch.float32,
        "store_param_remainders": True,
    }
    assert settings.warmup_ratio == 0.04


def test_explicit_betas_and_weight_decay_are_kept_and_lr_is_coerced():
    settings = DraftOptimizerSettings.from_config(
        ConfigNode({"lr": "6e-4", "betas": [0.9, 0.999], "weight_decay": 0.01})
    )
    kwargs = settings.optimizer.kwargs
    assert kwargs["lr"] == pytest.approx(6e-4) and isinstance(kwargs["lr"], float)
    assert list(kwargs["betas"]) == [0.9, 0.999]
    assert kwargs["weight_decay"] == 0.01
    assert isinstance(settings.peak_lr, float)


def test_adamw_defaults_are_not_forced_onto_an_explicit_target():
    settings = DraftOptimizerSettings.from_config(
        ConfigNode({"_target_": "torch.optim.SGD", "lr": 0.1, "momentum": 0.9})
    )
    assert settings.optimizer.factory is torch.optim.SGD
    assert settings.optimizer.kwargs == {"lr": 0.1, "momentum": 0.9}


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("torch.optim.SGD", torch.optim.SGD),
        ("nemo_automodel.components.optim.optimizer.AdamConfig", torch.optim.Adam),
    ],
)
def test_explicit_target_builds_with_configured_decay(target: str, expected: type[torch.optim.Optimizer]) -> None:
    settings = DraftOptimizerSettings.from_config(ConfigNode({"_target_": target, "lr": 3e-4, "weight_decay": 0.2}))
    (optimizer,) = settings.optimizer.build(torch.nn.Linear(2, 2))
    assert type(optimizer) is expected
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.2)


def test_optimizer_config_subclass_target_is_constructed_there():
    settings = DraftOptimizerSettings.from_config(
        ConfigNode({"_target_": "nemo_automodel.components.optim.optimizer.AdamConfig", "lr": 3e-4, "eps": 1e-6})
    )
    assert isinstance(settings.optimizer, AdamConfig)
    assert settings.optimizer.eps == 1e-6


def _matrices() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False), torch.nn.Linear(8, 8, bias=False))


def _native_step(optimizer: torch.optim.Optimizer, model: torch.nn.Module) -> None:
    model(torch.randn(4, 8)).sum().backward()
    optimizer.step()


@pytest.mark.skipif(not hasattr(torch.optim, "Muon"), reason="torch.optim.Muon needs PyTorch 2.9+")
def test_torch_muon_target_builds_torch_muon_and_takes_a_step_with_momentum():
    settings = DraftOptimizerSettings.from_config(
        ConfigNode({"_target_": "torch.optim.Muon", "lr": 1e-3, "momentum": 0.8})
    )
    assert isinstance(settings.optimizer, OptimizerFromFactoryConfig)
    assert settings.optimizer.factory is torch.optim.Muon
    model = _matrices()
    (optimizer,) = settings.optimizer.build(model)
    assert type(optimizer) is torch.optim.Muon
    assert optimizer.param_groups[0]["momentum"] == 0.8
    _native_step(optimizer, model)


def test_dion_muon_target_keeps_the_dion_grouping_config():
    dion = pytest.importorskip("dion")
    settings = DraftOptimizerSettings.from_config(ConfigNode({"_target_": "dion.Muon", "lr": 1e-3}))
    assert isinstance(settings.optimizer, MuonConfig)
    assert dion.Muon.__name__ == "Muon"


def test_empty_param_group_overrides_is_a_no_op():
    settings = DraftOptimizerSettings.from_config(ConfigNode({"lr": 1e-3, "param_group_overrides": []}))
    assert "param_group_overrides" not in settings.optimizer.kwargs
    (optimizer,) = settings.optimizer.build(torch.nn.Linear(2, 2))
    assert len(optimizer.param_groups) == 1


@pytest.mark.parametrize("target", [None, "torch.optim.SGD", "nemo_automodel.components.optim.optimizer.AdamWConfig"])
def test_param_group_overrides_are_rejected_where_the_schedule_would_not_apply_them(target):
    block = {"lr": 1.0e-3, "param_group_overrides": [{"pattern": "bias", "lr_mult": 0.0}]}
    if target is not None:
        block["_target_"] = target
    with pytest.raises(ValueError, match="optimizer.param_group_overrides is not supported"):
        DraftOptimizerSettings.from_config(ConfigNode(block))


def test_param_group_overrides_are_accepted_for_the_dion_family():
    pytest.importorskip("dion")
    settings = DraftOptimizerSettings.from_config(
        ConfigNode(
            {
                "_target_": "nemo_automodel.components.optim.optimizer.MuownConfig",
                "lr": 5e-5,
                "param_group_overrides": [{"pattern": "bias", "lr_mult": 0.25, "wd_mult": 0.5}],
            }
        )
    )
    assert isinstance(settings.optimizer, MuownConfig)
    assert settings.optimizer.param_group_overrides[0].lr_mult == 0.25


def test_from_config_does_not_modify_the_node():
    node = ConfigNode({"_target_": "torch.optim.SGD", "lr": 0.1, "momentum": 0.9, "warmup_ratio": 0.1})
    before = node.to_dict()
    DraftOptimizerSettings.from_config(node)
    assert node.to_dict() == before


def test_settings_are_frozen():
    settings = DraftOptimizerSettings.from_config(ConfigNode({"lr": 1e-4}))
    with pytest.raises(AttributeError):
        settings.peak_lr = 1.0

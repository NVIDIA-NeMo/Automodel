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
from nemo_automodel.components.optim.optimizer import OptimizerFromFactoryConfig
from nemo_automodel.recipes._typed_config import DraftOptimizerSettings


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
    pytest.importorskip("transformer_engine")
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
    assert settings.optimizer.kwargs == {
        "lr": 1e-5,
        "master_weights": True,
        "master_weight_dtype": "torch.float32",
        "exp_avg_dtype": "torch.float32",
        "exp_avg_sq_dtype": "torch.float32",
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


def test_param_group_overrides_are_rejected():
    with pytest.raises(ValueError, match="optimizer.param_group_overrides is not supported"):
        DraftOptimizerSettings.from_config(
            ConfigNode({"lr": 1.0e-3, "param_group_overrides": [{"pattern": "bias", "lr_mult": 0.0}]})
        )


def test_from_config_does_not_modify_the_node():
    node = ConfigNode({"_target_": "torch.optim.SGD", "lr": 0.1, "momentum": 0.9, "warmup_ratio": 0.1})
    before = node.to_dict()
    DraftOptimizerSettings.from_config(node)
    assert node.to_dict() == before


def test_settings_are_frozen():
    settings = DraftOptimizerSettings.from_config(ConfigNode({"lr": 1e-4}))
    with pytest.raises(AttributeError):
        settings.peak_lr = 1.0

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

"""Optimizer capability dispatch through the existing recipe sharding hook."""

from dataclasses import asdict
from unittest.mock import MagicMock

import pytest
import torch

from nemo_automodel.components.distributed import megatron_fsdp
from nemo_automodel.components.distributed.config import FSDP2Config, MegatronFSDPConfig
from nemo_automodel.components.optim.optimizer import (
    AdamConfig,
    AdamWConfig,
    Dion2Config,
    DionConfig,
    FusedAdamConfig,
    MuonConfig,
    NorMuonConfig,
    OptimizerConfig,
)
from nemo_automodel.components.optim.weight_rounding import WeightRounding
from nemo_automodel.recipes._dist_utils import shard_optimizers_for_megatron_fsdp

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("config", [OptimizerConfig(), AdamConfig(), AdamWConfig(), FusedAdamConfig()])
def test_default_capability_is_not_a_constructor_field(config: OptimizerConfig) -> None:
    assert config.supports_megatron_fsdp_sharding is True
    assert "supports_megatron_fsdp_sharding" not in asdict(config)


@pytest.mark.parametrize("config", [DionConfig(), Dion2Config(), MuonConfig(), NorMuonConfig()])
def test_dion_capability_remains_disabled(config: OptimizerConfig) -> None:
    assert config.supports_megatron_fsdp_sharding is False
    assert "supports_megatron_fsdp_sharding" not in asdict(config)


@pytest.mark.parametrize("rounding", [None, "nearest", "stochastic"])
def test_recipe_checks_config_before_megatron_sharding(
    monkeypatch: pytest.MonkeyPatch, rounding: WeightRounding | None
) -> None:
    model = torch.nn.Linear(2, 2)
    config = AdamWConfig(weight_rounding=rounding)
    optimizer = config.build(model)[0]
    sharded_optimizer = MagicMock()
    shard = MagicMock(return_value=sharded_optimizer)
    monkeypatch.setattr(megatron_fsdp.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(megatron_fsdp, "HAS_MEGATRON_FSDP", True)
    monkeypatch.setattr(megatron_fsdp, "fully_shard_optimizer", shard)

    if rounding is None:
        actual = shard_optimizers_for_megatron_fsdp(
            model, [optimizer], MegatronFSDPConfig(), allow=config.supports_megatron_fsdp_sharding
        )
        assert actual == [sharded_optimizer]
        shard.assert_called_once_with(model, optimizer)
    else:
        with pytest.raises(AssertionError, match="Configured optimizer does not support Megatron-FSDP"):
            shard_optimizers_for_megatron_fsdp(
                model, [optimizer], MegatronFSDPConfig(), allow=config.supports_megatron_fsdp_sharding
            )
        shard.assert_not_called()


@pytest.mark.parametrize("rounding", ["nearest", "stochastic"])
def test_fsdp2_does_not_use_megatron_sharding(monkeypatch: pytest.MonkeyPatch, rounding: WeightRounding) -> None:
    model = torch.nn.Linear(2, 2)
    config = AdamWConfig(weight_rounding=rounding)
    optimizer = config.build(model)[0]
    shard = MagicMock()
    monkeypatch.setattr(megatron_fsdp, "fully_shard_optimizer", shard)

    actual = shard_optimizers_for_megatron_fsdp(
        model, [optimizer], FSDP2Config(), allow=config.supports_megatron_fsdp_sharding
    )
    assert actual == [optimizer]
    shard.assert_not_called()

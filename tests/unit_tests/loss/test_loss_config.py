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

"""Tests for nemo_automodel.components.loss.config — LossConfig hierarchy."""

import pytest

from nemo_automodel.components.loss.config import (
    FusedLinearCEConfig,
    KDLossConfig,
    LossConfig,
    MaskedCrossEntropyConfig,
    TEParallelCEConfig,
)

# ---------------------------------------------------------------------------
# LossConfig base
# ---------------------------------------------------------------------------


class TestLossConfig:
    def test_defaults(self):
        cfg = LossConfig()
        assert "MaskedCrossEntropy" in cfg.name
        assert cfg.extra_kwargs == {}

    def test_to_kwargs(self):
        cfg = LossConfig(extra_kwargs={"alpha": 0.5})
        assert cfg.to_kwargs() == {"alpha": 0.5}


# ---------------------------------------------------------------------------
# Typed subclasses
# ---------------------------------------------------------------------------


class TestMaskedCrossEntropyConfig:
    def test_defaults(self):
        cfg = MaskedCrossEntropyConfig()
        assert cfg.fp32_upcast is True
        assert cfg.ignore_index == -100
        assert cfg.reduction == "sum"

    def test_to_kwargs(self):
        cfg = MaskedCrossEntropyConfig(fp32_upcast=False)
        kwargs = cfg.to_kwargs()
        assert kwargs["fp32_upcast"] is False
        assert kwargs["ignore_index"] == -100
        assert kwargs["reduction"] == "sum"


class TestFusedLinearCEConfig:
    def test_defaults(self):
        cfg = FusedLinearCEConfig()
        assert cfg.logit_softcapping == 0.0

    def test_to_kwargs(self):
        cfg = FusedLinearCEConfig(logit_softcapping=30.0)
        assert cfg.to_kwargs()["logit_softcapping"] == 30.0


class TestTEParallelCEConfig:
    def test_defaults(self):
        cfg = TEParallelCEConfig()
        assert cfg.ignore_index == -100
        assert cfg.reduction == "sum"

    def test_to_kwargs(self):
        cfg = TEParallelCEConfig(reduction="mean")
        assert cfg.to_kwargs()["reduction"] == "mean"


class TestKDLossConfig:
    def test_defaults(self):
        cfg = KDLossConfig()
        assert cfg.temperature == 1.0
        assert cfg.fp32_upcast is True

    def test_to_kwargs(self):
        cfg = KDLossConfig(temperature=2.0, fp32_upcast=False)
        kwargs = cfg.to_kwargs()
        assert kwargs["temperature"] == 2.0
        assert kwargs["fp32_upcast"] is False


# ---------------------------------------------------------------------------
# build_loss_fn with config
# ---------------------------------------------------------------------------


class TestBuildLossFnWithConfig:
    def test_build_masked_ce_from_config(self):
        from nemo_automodel.components.loss.api import build_loss_fn
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

        loss = build_loss_fn(config=MaskedCrossEntropyConfig(fp32_upcast=False))
        assert isinstance(loss, MaskedCrossEntropy)
        assert loss.fp32_upcast is False

    def test_build_kd_loss_from_config(self):
        from nemo_automodel.components.loss.api import build_loss_fn
        from nemo_automodel.components.loss.kd_loss import KDLoss

        loss = build_loss_fn(config=KDLossConfig(temperature=2.0))
        assert isinstance(loss, KDLoss)
        assert loss.temperature == 2.0

    def test_build_requires_config_or_factory(self):
        from nemo_automodel.components.loss.api import build_loss_fn

        with pytest.raises(ValueError, match="Either config or loss_factory"):
            build_loss_fn()

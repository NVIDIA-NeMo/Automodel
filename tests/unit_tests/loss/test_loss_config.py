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

"""Tests for nemo_automodel.components.loss.loss — typed configs + build_loss_module."""

import pytest

from nemo_automodel.components.loss.loss import (
    FusedLinearCEConfig,
    KDLossConfig,
    LossConfig,
    LossFromFactoryConfig,
    MaskedCrossEntropyConfig,
    TEParallelCEConfig,
    build_loss_config,
    build_loss_module,
)

# ---------------------------------------------------------------------------
# Typed config fields + build()
# ---------------------------------------------------------------------------


class TestMaskedCrossEntropyConfig:
    def test_defaults(self):
        cfg = MaskedCrossEntropyConfig()
        assert cfg.fp32_upcast is True
        assert cfg.ignore_index == -100
        assert cfg.reduction == "sum"

    def test_build(self):
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

        loss = MaskedCrossEntropyConfig(fp32_upcast=False).build()
        assert isinstance(loss, MaskedCrossEntropy)
        assert loss.fp32_upcast is False


class TestFusedLinearCEConfig:
    def test_defaults(self):
        cfg = FusedLinearCEConfig()
        assert cfg.logit_softcapping == 0.0
        assert cfg.ignore_index == -100
        assert cfg.reduction == "sum"


class TestTEParallelCEConfig:
    def test_defaults(self):
        cfg = TEParallelCEConfig()
        assert cfg.ignore_index == -100
        assert cfg.reduction == "sum"


class TestKDLossConfig:
    def test_defaults(self):
        cfg = KDLossConfig()
        assert cfg.temperature == 1.0
        assert cfg.fp32_upcast is True
        assert cfg.ignore_index == -100

    def test_build(self):
        from nemo_automodel.components.loss.kd_loss import KDLoss

        loss = KDLossConfig(temperature=2.0).build()
        assert isinstance(loss, KDLoss)
        assert loss.temperature == 2.0


class TestLossConfigBase:
    def test_base_build_not_implemented(self):
        with pytest.raises(NotImplementedError):
            LossConfig().build()


# ---------------------------------------------------------------------------
# build_loss_module — typed config (native path)
# ---------------------------------------------------------------------------


class TestBuildLossFnTypedConfig:
    def test_masked_ce(self):
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

        loss = build_loss_module(MaskedCrossEntropyConfig(fp32_upcast=False))
        assert isinstance(loss, MaskedCrossEntropy)
        assert loss.fp32_upcast is False

    def test_kd_loss(self):
        from nemo_automodel.components.loss.kd_loss import KDLoss

        loss = build_loss_module(KDLossConfig(temperature=2.0))
        assert isinstance(loss, KDLoss)
        assert loss.temperature == 2.0

    def test_config_with_kwargs_raises(self):
        with pytest.raises(ValueError, match="must be set on the config"):
            build_loss_module(MaskedCrossEntropyConfig(), fp32_upcast=False)

    def test_config_class_instead_of_instance_raises(self):
        with pytest.raises(TypeError, match="instance, not the class"):
            build_loss_module(MaskedCrossEntropyConfig)


# ---------------------------------------------------------------------------
# build_loss_module — class / callable form (integration / YAML escape hatch)
# ---------------------------------------------------------------------------


class TestBuildLossFnEscapeHatch:
    def test_resolved_class(self):
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

        loss = build_loss_module(MaskedCrossEntropy, fp32_upcast=False)
        assert isinstance(loss, MaskedCrossEntropy)
        assert loss.fp32_upcast is False

    def test_arbitrary_factory_kwargs(self):
        # Any callable + kwargs works; no typed config required.
        captured = {}

        def fake_loss(**kwargs):
            captured.update(kwargs)
            return "loss_module"

        result = build_loss_module(fake_loss, alpha=0.5, beta=0.3)
        assert result == "loss_module"
        assert captured == {"alpha": 0.5, "beta": 0.3}

    def test_non_callable_raises(self):
        with pytest.raises(TypeError, match="class/callable"):
            build_loss_module("nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy")


# ---------------------------------------------------------------------------
# build_loss_config — normalization to a LossConfig
# ---------------------------------------------------------------------------


class TestBuildLossConfig:
    def test_typed_instance_returned_as_is(self):
        cfg = MaskedCrossEntropyConfig(fp32_upcast=False)
        assert build_loss_config(cfg) is cfg

    def test_typed_instance_with_kwargs_raises(self):
        with pytest.raises(ValueError, match="must be set on the config"):
            build_loss_config(MaskedCrossEntropyConfig(), fp32_upcast=False)

    def test_config_class_instead_of_instance_raises(self):
        with pytest.raises(TypeError, match="instance, not the class"):
            build_loss_config(MaskedCrossEntropyConfig)

    def test_unregistered_callable_wrapped_in_factory_config(self):
        def fake_loss(**kwargs):
            return "loss_module"

        cfg = build_loss_config(fake_loss, alpha=0.5)
        assert isinstance(cfg, LossFromFactoryConfig)
        assert cfg.factory is fake_loss
        assert cfg.kwargs == {"alpha": 0.5}

    def test_non_callable_raises(self):
        with pytest.raises(TypeError, match="class/callable"):
            build_loss_config("nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy")


class TestLossConfigRegistry:
    def test_registry_maps_loss_classes_to_configs(self):
        from nemo_automodel.components.loss.loss import LOSS_CONFIG_REGISTRY

        assert LOSS_CONFIG_REGISTRY["MaskedCrossEntropy"] is MaskedCrossEntropyConfig
        assert LOSS_CONFIG_REGISTRY["FusedLinearCrossEntropy"] is FusedLinearCEConfig

    def test_string_name_resolves_to_typed_config(self):
        cfg = build_loss_config("MaskedCrossEntropy", fp32_upcast=False)
        assert isinstance(cfg, MaskedCrossEntropyConfig)
        assert cfg.fp32_upcast is False

    def test_unknown_string_name_raises(self):
        with pytest.raises(TypeError, match="Unknown loss name"):
            build_loss_config("NotARealLoss")

    def test_loss_class_upgraded_to_typed_config(self):
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

        cfg = build_loss_config(MaskedCrossEntropy, fp32_upcast=False, reduction="mean")
        assert isinstance(cfg, MaskedCrossEntropyConfig)
        assert cfg.fp32_upcast is False
        assert cfg.reduction == "mean"

        loss = cfg.build()
        assert isinstance(loss, MaskedCrossEntropy)
        assert loss.fp32_upcast is False

    def test_kd_loss_class_upgraded_with_full_kwargs(self):
        # KDLossConfig now exposes the full KDLoss surface (incl. chunk_size).
        from nemo_automodel.components.loss.kd_loss import KDLoss

        cfg = build_loss_config(KDLoss, temperature=2.0, chunk_size=8)
        assert isinstance(cfg, KDLossConfig)
        assert cfg.temperature == 2.0
        assert cfg.chunk_size == 8

    def test_loss_class_with_unconfigurable_kwargs_falls_back_to_factory(self):
        # A kwarg the typed config doesn't expose must fall back to the factory
        # wrapper rather than raising at normalization time.
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

        cfg = build_loss_config(MaskedCrossEntropy, not_a_field=123)
        assert isinstance(cfg, LossFromFactoryConfig)
        assert cfg.factory is MaskedCrossEntropy
        assert cfg.kwargs == {"not_a_field": 123}


class TestLossFromFactoryConfig:
    def test_build_calls_factory_with_kwargs(self):
        captured = {}

        def fake_loss(**kwargs):
            captured.update(kwargs)
            return "loss_module"

        cfg = LossFromFactoryConfig(factory=fake_loss, kwargs={"alpha": 0.5})
        assert cfg.build() == "loss_module"
        assert captured == {"alpha": 0.5}

    def test_build_without_factory_raises(self):
        with pytest.raises(AssertionError, match="must be a callable"):
            LossFromFactoryConfig().build()

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

"""Tests for nemo_automodel.components.loss.loss — typed configs + build_loss_fn."""

import pytest

from nemo_automodel.components.loss.loss import (
    FusedLinearCEConfig,
    KDLossConfig,
    LossConfig,
    MaskedCrossEntropyConfig,
    TEParallelCEConfig,
    _resolve_dotted_path,
    build_loss_fn,
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
# build_loss_fn — typed config (native path)
# ---------------------------------------------------------------------------


class TestBuildLossFnTypedConfig:
    def test_masked_ce(self):
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

        loss = build_loss_fn(MaskedCrossEntropyConfig(fp32_upcast=False))
        assert isinstance(loss, MaskedCrossEntropy)
        assert loss.fp32_upcast is False

    def test_kd_loss(self):
        from nemo_automodel.components.loss.kd_loss import KDLoss

        loss = build_loss_fn(KDLossConfig(temperature=2.0))
        assert isinstance(loss, KDLoss)
        assert loss.temperature == 2.0

    def test_config_with_kwargs_raises(self):
        with pytest.raises(ValueError, match="must be set on the config"):
            build_loss_fn(MaskedCrossEntropyConfig(), fp32_upcast=False)

    def test_config_class_instead_of_instance_raises(self):
        with pytest.raises(TypeError, match="instance, not the class"):
            build_loss_fn(MaskedCrossEntropyConfig)


# ---------------------------------------------------------------------------
# build_loss_fn — dotted-path / class form (integration / YAML escape hatch)
# ---------------------------------------------------------------------------


class TestBuildLossFnEscapeHatch:
    def test_dotted_path_string(self):
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

        loss = build_loss_fn(
            "nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy",
            fp32_upcast=False,
        )
        assert isinstance(loss, MaskedCrossEntropy)
        assert loss.fp32_upcast is False

    def test_resolved_class(self):
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

        loss = build_loss_fn(MaskedCrossEntropy, reduction="mean")
        assert isinstance(loss, MaskedCrossEntropy)

    def test_arbitrary_factory_kwargs(self):
        # Any callable + kwargs works; no typed config required.
        captured = {}

        def fake_loss(**kwargs):
            captured.update(kwargs)
            return "loss_module"

        result = build_loss_fn(fake_loss, alpha=0.5, beta=0.3)
        assert result == "loss_module"
        assert captured == {"alpha": 0.5, "beta": 0.3}


class TestResolveDottedPath:
    def test_resolve(self):
        from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy

        assert _resolve_dotted_path("nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy") is MaskedCrossEntropy

    def test_bad_path_no_dot(self):
        with pytest.raises(ValueError, match="Expected a dotted path"):
            _resolve_dotted_path("MaskedCrossEntropy")

    def test_bad_class(self):
        with pytest.raises(ImportError, match="Cannot find"):
            _resolve_dotted_path("nemo_automodel.components.loss.masked_ce.NonExistent")

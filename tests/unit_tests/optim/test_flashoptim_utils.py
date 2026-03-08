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

from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class DummyCfgOpt:
    """Minimal config shim compatible with build_flashoptim_optimizer()."""

    def __init__(self, target, d: dict):
        self._target_ = target
        self._d = dict(d)

    def to_dict(self):
        return dict(self._d)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 4)
        self.linear = nn.Linear(4, 4, bias=True)
        self.lm_head = nn.Linear(4, 10, bias=False)


# ---------------------------------------------------------------------------
# Tests for is_flashoptim_optimizer()
# ---------------------------------------------------------------------------

class TestIsFlashoptimOptimizer:
    def test_returns_true_for_flashoptim_module(self):
        from nemo_automodel.components.optim.utils import is_flashoptim_optimizer

        class _Cfg:
            class _target_:
                __name__ = "SomeOpt"
                __module__ = "flashoptim.optimizers"

        assert is_flashoptim_optimizer(_Cfg()) is True

    def test_returns_true_for_known_names(self):
        from nemo_automodel.components.optim.utils import is_flashoptim_optimizer

        for name in ("FlashAdamW", "FlashAdam", "FlashSGD", "FlashSGDW", "FlashLion"):

            class _Cfg:
                pass

            _Cfg._target_ = type(name, (), {"__name__": name, "__module__": "some.module"})
            assert is_flashoptim_optimizer(_Cfg()) is True, f"Expected True for {name}"

    def test_returns_false_for_non_flashoptim(self):
        from nemo_automodel.components.optim.utils import is_flashoptim_optimizer

        class _Cfg:
            class _target_:
                __name__ = "Adam"
                __module__ = "torch.optim"

        assert is_flashoptim_optimizer(_Cfg()) is False

    def test_returns_false_when_no_target(self):
        from nemo_automodel.components.optim.utils import is_flashoptim_optimizer

        class _Cfg:
            pass

        assert is_flashoptim_optimizer(_Cfg()) is False


# ---------------------------------------------------------------------------
# Tests for build_flashoptim_optimizer()
# ---------------------------------------------------------------------------

class TestBuildFlashoptimOptimizer:
    def _build(self, monkeypatch, target_cls, cfg_dict, model=None):
        from nemo_automodel.components.optim import utils as optim_utils

        monkeypatch.setattr(optim_utils, "_flashoptim_import_error", None, raising=False)
        if model is None:
            model = TinyModel()
        cfg = DummyCfgOpt(target_cls, cfg_dict)
        return optim_utils.build_flashoptim_optimizer(cfg_opt=cfg, model=model)

    def test_passes_trainable_params_and_kwargs(self, monkeypatch):
        captured = {}

        class Target:
            __name__ = "FlashAdamW"

            def __init__(self, params, lr=None, master_weight_bits=None):
                captured["params"] = params
                captured["lr"] = lr
                captured["master_weight_bits"] = master_weight_bits

        self._build(monkeypatch, Target, {"lr": 1e-4, "master_weight_bits": 24})

        assert captured["lr"] == pytest.approx(1e-4)
        assert captured["master_weight_bits"] == 24
        assert len(captured["params"]) > 0
        assert all(p.requires_grad for p in captured["params"])

    def test_unknown_keys_filtered_out(self, monkeypatch):
        captured = {}

        class Target:
            __name__ = "FlashAdamW"

            def __init__(self, params, lr=None):
                captured["lr"] = lr

        self._build(monkeypatch, Target, {"lr": 1e-4, "totally_unknown": 42})
        assert captured["lr"] == pytest.approx(1e-4)

    def test_import_error_raises(self, monkeypatch):
        from nemo_automodel.components.optim import utils as optim_utils

        monkeypatch.setattr(optim_utils, "_flashoptim_import_error", ImportError("no flashoptim"), raising=False)

        class Target:
            __name__ = "FlashAdamW"

            def __init__(self, params):
                pass

        cfg = DummyCfgOpt(Target, {"lr": 1e-4})
        with pytest.raises(RuntimeError, match="Failed to import flashoptim"):
            optim_utils.build_flashoptim_optimizer(cfg_opt=cfg, model=TinyModel())

    def test_frozen_params_excluded(self, monkeypatch):
        captured = {}

        class Target:
            __name__ = "FlashAdamW"

            def __init__(self, params, lr=None):
                captured["params"] = params

        model = TinyModel()
        # Freeze embed
        for p in model.embed.parameters():
            p.requires_grad = False

        self._build(monkeypatch, Target, {"lr": 1e-4}, model=model)
        # Frozen params should be excluded
        total_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert len(captured["params"]) == total_trainable


# ---------------------------------------------------------------------------
# Tests for build_optimizer in train_ft.py with FlashOptim branch
# ---------------------------------------------------------------------------

def _patch_train_ft_for_cpu(monkeypatch, train_ft_mod, model):
    _sentinel = object()
    monkeypatch.setattr(
        train_ft_mod, "NeMoAutoModelForCausalLM",
        type("_FakeAutoModel", (), {"from_pretrained": _sentinel, "from_config": object()})(),
    )
    monkeypatch.setattr(
        train_ft_mod, "NeMoAutoModelForSequenceClassification",
        type("_FakeAutoModel2", (), {"from_pretrained": object(), "from_config": object()})(),
    )
    monkeypatch.setattr(train_ft_mod, "_supports_logits_to_keep", lambda m: True)
    monkeypatch.setattr(train_ft_mod, "ScopedRNG", lambda seed, ranked: nullcontext())
    return _sentinel


class TestBuildOptimizerFlashoptimBranch:
    @staticmethod
    def _make_simple_model():
        return nn.Linear(4, 4)

    @staticmethod
    def _make_parts_model():
        class PartsModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.part1 = nn.Linear(4, 4)
                self.part2 = nn.Linear(4, 4)
                self.parts = [self.part1, self.part2]

            def forward(self, x):
                return self.part2(self.part1(x))

        return PartsModel()

    def test_flashoptim_optimizer_single_model(self, monkeypatch):
        import nemo_automodel.recipes.llm.train_ft as train_ft_mod

        model = self._make_simple_model()
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

        monkeypatch.setattr(train_ft_mod, "is_dion_optimizer", lambda cfg: False)
        monkeypatch.setattr(train_ft_mod, "is_flashoptim_optimizer", lambda cfg: True)
        monkeypatch.setattr(
            train_ft_mod, "build_flashoptim_optimizer",
            lambda cfg_opt, model: (
                build_calls.append(model) or "fake_flashoptim_opt"
            ),
        )

        result_model = train_ft_mod.build_model(
            cfg_model=FakeCfgModel(),
            cfg_peft=None,
            seed=42,
        )
        optimizers = train_ft_mod.build_optimizer(result_model, FakeCfgOpt(), None, None)

        assert optimizers == ["fake_flashoptim_opt"]
        assert len(build_calls) == 1
        assert build_calls[0] is model

    def test_flashoptim_optimizer_with_parts(self, monkeypatch):
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

        monkeypatch.setattr(train_ft_mod, "is_dion_optimizer", lambda cfg: False)
        monkeypatch.setattr(train_ft_mod, "is_flashoptim_optimizer", lambda cfg: True)
        monkeypatch.setattr(
            train_ft_mod, "build_flashoptim_optimizer",
            lambda cfg_opt, model: (
                build_calls.append(model) or f"opt_for_{id(model)}"
            ),
        )

        result_model = train_ft_mod.build_model(
            cfg_model=FakeCfgModel(),
            cfg_peft=None,
            seed=42,
        )
        optimizers = train_ft_mod.build_optimizer(result_model, FakeCfgOpt(), None, None)

        assert len(build_calls) == 2
        assert build_calls[0] is model.parts[0]
        assert build_calls[1] is model.parts[1]
        assert len(optimizers) == 2


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

"""Tests verifying that mlflow and wandb are optional extras.

Confirms:
1. WandbConfig / MLflowConfig can be *instantiated* without the extras installed.
2. WandbConfig.build() / MLflowConfig.build() raise when the extra is absent.
3. pyproject.toml declares [mlflow], [mlflow-full], [wandb], [tracking] extras
   and neither mlflow nor wandb appears in core dependencies.
4. Recipe modules expose the _HAS_WANDB / _HAS_MLFLOW sentinels from safe_import,
   proving they import cleanly without crashing at module load time.
"""

import sys
import types
from pathlib import Path

import pytest

from nemo_automodel.components.loggers.loggers import MLflowConfig, WandbConfig
from nemo_automodel.shared.import_utils import UnavailableError


# ---------------------------------------------------------------------------
# WandbConfig
# ---------------------------------------------------------------------------


class TestWandbConfigOptional:
    def test_instantiation_does_not_require_wandb(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "wandb", raising=False)
        cfg = WandbConfig(project="test", entity="acme")
        assert cfg.project == "test"

    def test_build_raises_when_wandb_absent(self, monkeypatch):
        """build() propagates UnavailableError when wandb is not installed."""
        monkeypatch.delitem(sys.modules, "wandb", raising=False)
        cfg = WandbConfig()
        with pytest.raises(UnavailableError, match="wandb is not installed"):
            cfg.build()

    def test_build_succeeds_with_mocked_wandb(self, monkeypatch):
        captured = {}
        fake_wandb = types.ModuleType("wandb")
        fake_wandb.init = lambda **kw: captured.update(kw) or "run"
        fake_wandb.Settings = lambda **kw: None
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        cfg = WandbConfig(project="p", entity="e")
        run = cfg.build(run_config={"lr": 0.001}, model_name="nvidia/llama-3")
        assert run == "run"
        assert captured["project"] == "p"


# ---------------------------------------------------------------------------
# MLflowConfig
# ---------------------------------------------------------------------------


class TestMLflowConfigOptional:
    def test_instantiation_does_not_require_mlflow(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "mlflow", raising=False)
        cfg = MLflowConfig(experiment_name="my-exp")
        assert cfg.experiment_name == "my-exp"

    def test_build_raises_when_mlflow_absent_on_rank0(self, monkeypatch):
        import torch.distributed as dist

        monkeypatch.delitem(sys.modules, "mlflow", raising=False)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_rank", lambda: 0)

        cfg = MLflowConfig()
        with pytest.raises(UnavailableError, match="mlflow is not installed"):
            cfg.build()


# ---------------------------------------------------------------------------
# pyproject.toml extras
# ---------------------------------------------------------------------------


class TestPyprojectExtras:
    def _load_optional_deps(self):
        import tomllib

        pyproject = Path(__file__).parents[3] / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        return data["project"]

    def test_mlflow_extra_uses_skinny(self):
        proj = self._load_optional_deps()
        opt = proj["optional-dependencies"]
        assert "mlflow" in opt, "Missing [mlflow] extra"
        assert any("mlflow-skinny" in d for d in opt["mlflow"]), "[mlflow] must use mlflow-skinny"

    def test_mlflow_full_extra_uses_full_mlflow(self):
        proj = self._load_optional_deps()
        opt = proj["optional-dependencies"]
        assert "mlflow-full" in opt, "Missing [mlflow-full] extra"
        assert any("mlflow" in d and "skinny" not in d for d in opt["mlflow-full"])

    def test_wandb_extra_declared(self):
        proj = self._load_optional_deps()
        opt = proj["optional-dependencies"]
        assert "wandb" in opt, "Missing [wandb] extra"
        assert any("wandb" in d for d in opt["wandb"])

    def test_tracking_meta_extra(self):
        proj = self._load_optional_deps()
        opt = proj["optional-dependencies"]
        assert "tracking" in opt, "Missing [tracking] meta-extra"
        assert any("mlflow" in d for d in opt["tracking"])
        assert any("wandb" in d for d in opt["tracking"])

    def test_mlflow_and_wandb_absent_from_core_deps(self):
        proj = self._load_optional_deps()
        core = proj["dependencies"]
        assert not any(d.startswith("mlflow") for d in core), "mlflow must not be a core dep"
        assert not any(d.startswith("wandb") for d in core), "wandb must not be a core dep"


# ---------------------------------------------------------------------------
# Recipe module import smoke-test
# ---------------------------------------------------------------------------


class TestRecipeImportSentinels:
    """Recipe modules must expose _HAS_WANDB / _HAS_MLFLOW from safe_import (not raw None)."""

    @pytest.mark.parametrize(
        "module_path,expected_sentinels",
        [
            ("nemo_automodel.recipes.llm.train_ft", ["_HAS_WANDB", "_HAS_MLFLOW"]),
            ("nemo_automodel.recipes.llm.kd", ["_HAS_WANDB"]),
            ("nemo_automodel.recipes.llm.train_seq_cls", ["_HAS_WANDB"]),
            ("nemo_automodel.recipes.dllm.train_ft", ["_HAS_WANDB", "_HAS_MLFLOW"]),
            ("nemo_automodel.recipes.diffusion.train", ["_HAS_WANDB"]),
            ("nemo_automodel.recipes.vlm.finetune", ["_HAS_WANDB", "_HAS_MLFLOW"]),
            ("nemo_automodel.recipes.vlm.kd", ["_HAS_WANDB"]),
            ("nemo_automodel.recipes.multimodal.finetune", ["_HAS_WANDB"]),
        ],
    )
    def test_sentinel_present_and_is_bool(self, module_path, expected_sentinels):
        import importlib

        mod = importlib.import_module(module_path)
        for sentinel in expected_sentinels:
            val = getattr(mod, sentinel, None)
            assert val is not None, f"{module_path} missing {sentinel}"
            assert isinstance(val, bool), f"{module_path}.{sentinel} must be bool, got {type(val)}"

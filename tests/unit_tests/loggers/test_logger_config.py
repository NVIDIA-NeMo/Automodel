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

"""Tests for nemo_automodel.components.loggers.loggers — WandbConfig, MLflowConfig, CometConfig."""

import sys
import types

from nemo_automodel.components.loggers.loggers import CometConfig, MLflowConfig, WandbConfig


class TestWandbConfig:
    def test_defaults(self):
        cfg = WandbConfig()
        assert cfg.project == "automodel"
        assert cfg.entity is None
        assert cfg.name == ""
        assert cfg.tags == []
        assert cfg.extra == {}

    def test_custom_values(self):
        cfg = WandbConfig(project="my-project", entity="my-team", name="run-1", tags=["exp", "v2"])
        assert cfg.project == "my-project"
        assert cfg.entity == "my-team"
        assert cfg.tags == ["exp", "v2"]

    def test_from_kwargs_routes_passthrough_keys_to_extra(self):
        # ``mode``/``dir`` are valid wandb.init kwargs but not named fields; they must not raise
        # (regression: the closed dataclass used to crash on any non-field key).
        cfg = WandbConfig.from_kwargs(project="flux", mode="online", dir="/tmp/wandb")
        assert cfg.project == "flux"
        assert cfg.extra == {"mode": "online", "dir": "/tmp/wandb"}

    def test_from_kwargs_known_keys_assigned_directly(self):
        cfg = WandbConfig.from_kwargs(project="p", entity="e", tags=["a"], notes="n")
        assert (cfg.project, cfg.entity, cfg.tags, cfg.notes) == ("p", "e", ["a"], "n")
        assert cfg.extra == {}

    def test_build_forwards_extra_to_wandb_init(self, monkeypatch):
        captured = {}
        fake_wandb = types.ModuleType("wandb")
        fake_wandb.init = lambda **kw: captured.update(kw) or "run"
        fake_wandb.Settings = lambda **kw: None
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        cfg = WandbConfig.from_kwargs(project="flux", mode="online", dir="/tmp/w")
        run = cfg.build()
        assert run == "run"
        assert captured["project"] == "flux"
        assert captured["mode"] == "online"
        assert captured["dir"] == "/tmp/w"


class TestMLflowConfig:
    def test_defaults(self):
        cfg = MLflowConfig()
        assert cfg.experiment_name == "automodel-experiment"
        assert cfg.run_name == ""
        assert cfg.tracking_uri is None
        assert cfg.tags == {}
        assert cfg.resume is True
        assert cfg.flatten_depth == 1

    def test_custom_values(self):
        cfg = MLflowConfig(
            experiment_name="my-exp",
            tracking_uri="http://localhost:5000",
            tags={"model": "llama"},
            resume=False,
            description="Test run",
        )
        assert cfg.experiment_name == "my-exp"
        assert cfg.tracking_uri == "http://localhost:5000"
        assert cfg.tags["model"] == "llama"
        assert cfg.resume is False
        assert cfg.description == "Test run"


class TestCometConfig:
    def test_defaults(self):
        cfg = CometConfig()
        assert cfg.project_name == "automodel"
        assert cfg.workspace is None
        assert cfg.api_key is None
        assert cfg.tags == []
        assert cfg.auto_metric_logging is False

    def test_custom_values(self):
        cfg = CometConfig(project_name="my-project", experiment_name="exp-1", tags=["a", "b"])
        assert cfg.project_name == "my-project"
        assert cfg.experiment_name == "exp-1"
        assert cfg.tags == ["a", "b"]

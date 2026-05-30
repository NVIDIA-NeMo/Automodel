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

from nemo_automodel.components.loggers.loggers import CometConfig, MLflowConfig, WandbConfig


class TestWandbConfig:
    def test_defaults(self):
        cfg = WandbConfig()
        assert cfg.project == "automodel"
        assert cfg.entity is None
        assert cfg.name == ""
        assert cfg.tags == []
        assert cfg.save_dir is None

    def test_custom_values(self):
        cfg = WandbConfig(project="my-project", entity="my-team", name="run-1", tags=["exp", "v2"])
        assert cfg.project == "my-project"
        assert cfg.entity == "my-team"
        assert cfg.tags == ["exp", "v2"]


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

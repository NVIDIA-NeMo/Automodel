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

"""Tests for nemo_automodel.components.loggers.loggers — WandbConfig, TrackioConfig, MLflowConfig, CometConfig."""

import importlib.util
import sys
import types

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.loggers.loggers import CometConfig, MLflowConfig, TrackioConfig, WandbConfig
from nemo_automodel.components.loggers.trackio_utils import TrackioLogger


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

    def test_build_forwards_environment_resolved_strings(self, monkeypatch):
        captured = {}
        fake_wandb = types.ModuleType("wandb")
        fake_wandb.init = lambda **kw: captured.update(kw) or "run"
        fake_wandb.Settings = lambda **kw: None
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
        monkeypatch.setenv("TEST_WANDB_PROJECT", "resolved-project")
        monkeypatch.setenv("TEST_WANDB_DIR", "/tmp/resolved-wandb")

        wandb_node = ConfigNode(
            {
                "project": "${TEST_WANDB_PROJECT}",
                "dir": "${TEST_WANDB_DIR}",
            }
        )
        cfg = WandbConfig.from_kwargs(**wandb_node.to_dict())

        assert cfg.build() == "run"
        assert captured["project"] == "resolved-project"
        assert captured["dir"] == "/tmp/resolved-wandb"


def _fake_trackio(monkeypatch, captured):
    fake_trackio = types.ModuleType("trackio")
    # importlib.util.find_spec("trackio") raises if __spec__ is None
    fake_trackio.__spec__ = importlib.util.spec_from_loader("trackio", loader=None)
    fake_trackio.init = lambda **kw: captured.update(kw) or "run"
    monkeypatch.setitem(sys.modules, "trackio", fake_trackio)


class TestTrackioConfig:
    def test_defaults(self):
        cfg = TrackioConfig()
        assert cfg.project == "automodel"
        assert cfg.name == ""
        assert cfg.group is None
        assert cfg.space_id is None
        assert cfg.extra == {}

    def test_from_kwargs_routes_passthrough_keys_to_extra(self):
        cfg = TrackioConfig.from_kwargs(project="p", space_id="me/runs", resume="allow", private=True)
        assert (cfg.project, cfg.space_id) == ("p", "me/runs")
        assert cfg.extra == {"resume": "allow", "private": True}

    def test_build_forwards_fields_extra_and_run_config(self, monkeypatch):
        captured = {}
        _fake_trackio(monkeypatch, captured)

        cfg = TrackioConfig.from_kwargs(project="p", name="run-1", group="g", resume="allow")
        logger = cfg.build(run_config={"lr": 1e-3})

        assert isinstance(logger, TrackioLogger) and logger.run == "run"
        assert captured == {"project": "p", "name": "run-1", "group": "g", "resume": "allow", "config": {"lr": 1e-3}}

    def test_build_records_the_yaml_run_config_from_the_recipe_boundary(self, monkeypatch):
        """Recipes pass ``to_yaml_dict(use_orig_values=True)``: targets and env placeholders stay as written.

        ``to_dict()`` would record the inherited ``_BaseNeMoAutoModelClass.from_pretrained`` and the resolved path.
        """
        captured = {}
        _fake_trackio(monkeypatch, captured)
        monkeypatch.setenv("TEST_TRACKIO_CKPT_DIR", "/resolved/ckpts")

        recipe_cfg = ConfigNode(
            {
                "model": {"_target_": "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained"},
                "checkpoint": {"checkpoint_dir": "${TEST_TRACKIO_CKPT_DIR}"},
            }
        )
        TrackioConfig().build(run_config=recipe_cfg.to_yaml_dict(use_orig_values=True))

        assert captured["config"] == {
            "model": {"_target_": "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained"},
            "checkpoint": {"checkpoint_dir": "${TEST_TRACKIO_CKPT_DIR}"},
        }

    def test_build_leaves_out_top_level_underscore_sections(self, monkeypatch):
        """``trackio.init`` rejects top-level config keys starting with "_"; nested ``_target_`` keys are kept."""
        captured = {}
        _fake_trackio(monkeypatch, captured)

        TrackioConfig().build(run_config={"model": {"_target_": "m.f"}, "_validation_dataset": {"split": "val"}})

        assert captured["config"] == {"model": {"_target_": "m.f"}}

    def test_build_derives_the_run_name_from_the_model(self, monkeypatch):
        captured = {}
        _fake_trackio(monkeypatch, captured)

        TrackioConfig(project="p").build(model_name="Qwen/Qwen3-0.6B")

        assert captured["name"] == "Qwen_Qwen3-0.6B"

    def test_build_returns_none_on_non_zero_ranks(self, monkeypatch):
        import torch.distributed as dist

        captured = {}
        _fake_trackio(monkeypatch, captured)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_rank", lambda: 1)

        assert TrackioConfig().build(run_config={"lr": 1e-3}) is None
        assert captured == {}


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

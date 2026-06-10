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

from argparse import Namespace

import pytest


def test_infer_config_path_uses_checkpoint_dir():
    from scripts.export_llm_dcp_to_hf import infer_config_path

    config_path = infer_config_path("/tmp/run/checkpoints/epoch_1_step_23")

    assert str(config_path) == "/tmp/run/checkpoints/epoch_1_step_23/config.yaml"


def test_infer_epoch_step_parses_checkpoint_name():
    from scripts.export_llm_dcp_to_hf import infer_epoch_step

    assert infer_epoch_step("/tmp/run/checkpoints/epoch_3_step_456") == (3, 456)


def test_infer_epoch_step_requires_standard_checkpoint_name():
    from scripts.export_llm_dcp_to_hf import infer_epoch_step

    with pytest.raises(ValueError):
        infer_epoch_step("/tmp/run/checkpoints/latest")


def test_resolve_epoch_step_prefers_explicit_values():
    from scripts.export_llm_dcp_to_hf import resolve_epoch_step

    assert resolve_epoch_step("/tmp/run/checkpoints/epoch_3_step_456", 9, 10) == (9, 10)


def test_infer_export_root_uses_epoch_and_step():
    from scripts.export_llm_dcp_to_hf import infer_export_root

    export_root = infer_export_root("/tmp/export", 7, 99)

    assert str(export_root) == "/tmp/export/epoch_7_step_99"


def test_disable_tracking_loggers_strips_remote_logger_sections():
    from nemo_automodel.components.config.loader import ConfigNode
    from scripts.export_llm_dcp_to_hf import disable_tracking_loggers

    cfg = ConfigNode(
        {
            "wandb": {"project": "demo"},
            "mlflow": {"experiment": "demo"},
            "comet": {"project": "demo"},
            "checkpoint": {"enabled": True},
        }
    )

    disable_tracking_loggers(cfg)

    assert not hasattr(cfg, "wandb")
    assert not hasattr(cfg, "mlflow")
    assert not hasattr(cfg, "comet")
    assert cfg.checkpoint.enabled is True


def test_build_export_config_applies_export_overrides(monkeypatch):
    from nemo_automodel.components.config.loader import ConfigNode
    from scripts import export_llm_dcp_to_hf as script

    captured = {}

    def fake_parse_args_and_load_config(config_path, argv):
        captured["config_path"] = config_path
        captured["argv"] = argv
        return ConfigNode({"checkpoint": {"enabled": True}, "wandb": {"project": "demo"}})

    monkeypatch.setattr(script, "parse_args_and_load_config", fake_parse_args_and_load_config)

    cfg = script.build_export_config(
        Namespace(
            checkpoint_dir="/tmp/run/checkpoints/epoch_0_step_42",
            output_dir="/tmp/export",
            config=None,
            model_name_or_path="/models/gemma4",
            save_consolidated="final",
            epoch=None,
            step=None,
        )
    )

    assert captured["config_path"] == "/tmp/run/checkpoints/epoch_0_step_42/config.yaml"
    assert captured["argv"] == [
        "--checkpoint.restore_from",
        "None",
        "--checkpoint.checkpoint_dir",
        "/tmp/export/.export_workdir",
        "--checkpoint.model_save_format",
        "safetensors",
        "--checkpoint.save_consolidated",
        "final",
    ]
    assert cfg.model.pretrained_model_name_or_path == "/models/gemma4"
    assert not hasattr(cfg, "wandb")


def test_resolve_model_for_export_unwraps_single_ddp(monkeypatch):
    from scripts import export_llm_dcp_to_hf as script

    model = object()

    class FakeDDP:
        def __init__(self, wrapped_module):
            self.module = wrapped_module

    monkeypatch.setattr(script, "DistributedDataParallel", FakeDDP)

    trainer = Namespace(model_parts=[FakeDDP(model)])

    assert script.resolve_model_for_export(trainer) is model

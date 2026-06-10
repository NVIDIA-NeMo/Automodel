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
from unittest.mock import MagicMock

import pytest


def test_parse_args_uses_defaults_for_optional_flags():
    from scripts.export_llm_dcp_to_hf import parse_args

    args = parse_args(["--checkpoint-dir", "/tmp/ckpt/epoch_0_step_1", "--output-dir", "/tmp/export"])

    assert args.checkpoint_dir == "/tmp/ckpt/epoch_0_step_1"
    assert args.output_dir == "/tmp/export"
    assert args.config is None
    assert args.model_name_or_path is None
    assert args.epoch is None
    assert args.step is None


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


def test_resolve_epoch_step_with_explicit_values_skips_name_parsing():
    from scripts.export_llm_dcp_to_hf import resolve_epoch_step

    # A non-standard directory name must be fine when both flags are given,
    # which is exactly the workaround the infer_epoch_step error suggests.
    assert resolve_epoch_step("/tmp/run/checkpoints/latest", 0, 500) == (0, 500)


def test_resolve_epoch_step_falls_back_to_inferred_values():
    from scripts.export_llm_dcp_to_hf import resolve_epoch_step

    assert resolve_epoch_step("/tmp/run/checkpoints/epoch_3_step_456", None, None) == (3, 456)
    assert resolve_epoch_step("/tmp/run/checkpoints/epoch_3_step_456", 9, None) == (9, 456)


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
        "--wandb",
        "None",
        "--mlflow",
        "None",
        "--comet",
        "None",
    ]
    assert cfg.model.pretrained_model_name_or_path == "/models/gemma4"


def test_build_export_config_uses_explicit_config_path_without_model_override(monkeypatch):
    from nemo_automodel.components.config.loader import ConfigNode
    from scripts import export_llm_dcp_to_hf as script

    captured = {}

    def fake_parse_args_and_load_config(config_path, argv):
        captured["config_path"] = config_path
        return ConfigNode({"checkpoint": {"enabled": True}, "model": {"pretrained_model_name_or_path": "/orig"}})

    monkeypatch.setattr(script, "parse_args_and_load_config", fake_parse_args_and_load_config)

    cfg = script.build_export_config(
        Namespace(
            checkpoint_dir="/tmp/run/checkpoints/epoch_0_step_42",
            output_dir="/tmp/export",
            config="/custom/config.yaml",
            model_name_or_path=None,
            epoch=None,
            step=None,
        )
    )

    assert captured["config_path"] == "/custom/config.yaml"
    # No --model-name-or-path override, so the recorded base model path is untouched.
    assert cfg.model.pretrained_model_name_or_path == "/orig"


def test_build_export_config_rejects_peft_configs(monkeypatch):
    from nemo_automodel.components.config.loader import ConfigNode
    from scripts import export_llm_dcp_to_hf as script

    monkeypatch.setattr(
        script,
        "parse_args_and_load_config",
        lambda config_path, argv: ConfigNode({"checkpoint": {"enabled": True}, "peft": {"match_all_linear": True}}),
    )

    with pytest.raises(ValueError, match="PEFT checkpoints"):
        script.build_export_config(
            Namespace(
                checkpoint_dir="/tmp/run/checkpoints/epoch_0_step_42",
                output_dir="/tmp/export",
                config=None,
                model_name_or_path=None,
                epoch=None,
                step=None,
            )
        )


def test_resolve_model_for_export_unwraps_single_ddp(monkeypatch):
    from scripts import export_llm_dcp_to_hf as script

    model = object()

    class FakeDDP:
        def __init__(self, wrapped_module):
            self.module = wrapped_module

    monkeypatch.setattr(script, "DistributedDataParallel", FakeDDP)

    trainer = Namespace(model_parts=[FakeDDP(model)])

    assert script.resolve_model_for_export(trainer) is model


def test_resolve_model_for_export_returns_single_non_ddp_model():
    from scripts import export_llm_dcp_to_hf as script

    model = object()
    trainer = Namespace(model_parts=[model])

    assert script.resolve_model_for_export(trainer) is model


def test_resolve_model_for_export_returns_multipart_list_unchanged():
    from scripts import export_llm_dcp_to_hf as script

    parts = [object(), object()]
    trainer = Namespace(model_parts=parts)

    assert script.resolve_model_for_export(trainer) is parts


def test_barrier_invokes_distributed_barrier_only_when_initialized(monkeypatch):
    import torch.distributed as dist

    from scripts import export_llm_dcp_to_hf as script

    mock_barrier = MagicMock()
    monkeypatch.setattr(dist, "barrier", mock_barrier)

    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    script.barrier()
    assert mock_barrier.call_count == 0

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    script.barrier()
    assert mock_barrier.call_count == 1


def test_close_trainer_is_noop_for_none():
    from scripts.export_llm_dcp_to_hf import close_trainer

    # Should not raise.
    close_trainer(None)


def test_close_trainer_closes_all_recipe_resources():
    from scripts.export_llm_dcp_to_hf import close_trainer

    trainer = MagicMock()
    valid_logger = MagicMock()
    trainer.metric_logger_valid = {"val": valid_logger}

    close_trainer(trainer)

    trainer.metric_logger_train.close.assert_called_once()
    valid_logger.close.assert_called_once()
    trainer.checkpointer.close.assert_called_once()


def test_main_exports_checkpoint_end_to_end(monkeypatch, tmp_path):
    from scripts import export_llm_dcp_to_hf as script

    checkpoint_dir = tmp_path / "checkpoints" / "epoch_2_step_7"
    checkpoint_dir.mkdir(parents=True)
    output_dir = tmp_path / "export"

    trainer = MagicMock()
    model = object()
    exported_yaml = {"checkpoint": {"model_save_format": "safetensors"}}
    monkeypatch.setattr(script, "build_export_config", lambda args: MagicMock(to_yaml_dict=lambda: exported_yaml))
    monkeypatch.setattr(script, "TrainFinetuneRecipeForNextTokenPrediction", lambda cfg: trainer)
    monkeypatch.setattr(script, "resolve_model_for_export", lambda t: model)
    save_config_calls = {}
    monkeypatch.setattr(
        script, "save_config", lambda config, path: save_config_calls.update({"config": config, "path": path})
    )

    script.main(["--checkpoint-dir", str(checkpoint_dir), "--output-dir", str(output_dir)])

    export_root = output_dir / "epoch_2_step_7"
    assert export_root.is_dir()

    load_kwargs = trainer.checkpointer.load_model.call_args.kwargs
    assert load_kwargs["model_path"] == str(checkpoint_dir / "model")
    assert load_kwargs["allow_checkpoint_key_subset"] is True
    assert trainer.checkpointer.load_model.call_args.args[0] is model

    save_kwargs = trainer.checkpointer.save_model.call_args.kwargs
    assert save_kwargs["model"] is model
    assert save_kwargs["weights_path"] == str(export_root)
    assert save_kwargs["is_final_checkpoint"] is True
    assert save_kwargs["tokenizer"] is trainer.tokenizer

    # The exported config must come from to_yaml_dict(): to_dict() would serialize
    # resolved _target_ objects as !!python tags that yaml.safe_load rejects.
    assert save_config_calls["config"] is exported_yaml
    assert save_config_calls["path"] == str(export_root)
    trainer.checkpointer.close.assert_called_once()


def test_main_fails_fast_when_export_root_exists(monkeypatch, tmp_path):
    from scripts import export_llm_dcp_to_hf as script

    checkpoint_dir = tmp_path / "checkpoints" / "epoch_2_step_7"
    checkpoint_dir.mkdir(parents=True)
    export_root = tmp_path / "export" / "epoch_2_step_7"
    export_root.mkdir(parents=True)

    build_config = MagicMock()
    monkeypatch.setattr(script, "build_export_config", build_config)

    with pytest.raises(FileExistsError, match="already exists"):
        script.main(["--checkpoint-dir", str(checkpoint_dir), "--output-dir", str(tmp_path / "export")])

    # The check runs before any expensive recipe construction.
    build_config.assert_not_called()


def test_main_closes_trainer_even_when_setup_fails(monkeypatch, tmp_path):
    from scripts import export_llm_dcp_to_hf as script

    checkpoint_dir = tmp_path / "checkpoints" / "epoch_0_step_1"
    checkpoint_dir.mkdir(parents=True)
    output_dir = tmp_path / "export"

    trainer = MagicMock()
    trainer.setup.side_effect = RuntimeError("boom")
    monkeypatch.setattr(script, "build_export_config", lambda args: MagicMock(to_yaml_dict=lambda: {}))
    monkeypatch.setattr(script, "TrainFinetuneRecipeForNextTokenPrediction", lambda cfg: trainer)

    with pytest.raises(RuntimeError, match="boom"):
        script.main(["--checkpoint-dir", str(checkpoint_dir), "--output-dir", str(output_dir)])

    trainer.checkpointer.close.assert_called_once()

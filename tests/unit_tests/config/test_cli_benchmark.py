# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

from pathlib import Path

import pytest
import yaml

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def canonical_yaml(tmp_path: Path) -> Path:
    """Provide a real YAML boundary without importing a training model."""
    path = tmp_path / "canonical.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "recipe": "TrainFinetuneRecipeForNextTokenPrediction",
                "step_scheduler": {
                    "global_batch_size": 256,
                    "local_batch_size": 4,
                    "max_steps": 1000,
                    "num_epochs": 1,
                    "val_every_steps": 10,
                    "validate_on_checkpoint": True,
                },
                "checkpoint": {"enabled": True},
                "model": {"backend": {"attn": "sdpa", "fake_balanced_gate": False}},
                "dataset": {"dataset_name": "real-text"},
                "optimizer": {"lr": 0.0001},
                "distributed": {"ep_size": 8},
                "ci": {"perf_testing": {"schedule": "disabled", "time": "00:05:00"}},
            }
        )
    )
    return path


def test_benchmark_defaults_preserve_canonical_workload(canonical_yaml: Path) -> None:
    """Measurement controls must not substitute synthetic data or retune the workload."""
    original = canonical_yaml.read_text()
    cfg = parse_args_and_load_config(str(canonical_yaml), argv=["--benchmark"])
    assert cfg.step_scheduler.max_steps == 50
    assert cfg.step_scheduler.num_epochs is None
    assert cfg.checkpoint.enabled is False
    assert cfg.step_scheduler.val_every_steps is None
    assert cfg.step_scheduler.validate_on_checkpoint is False
    assert cfg.model.backend.fake_balanced_gate is False
    assert cfg.step_scheduler.global_batch_size == 256
    assert cfg.step_scheduler.local_batch_size == 4
    assert cfg.model.backend.attn == "sdpa"
    assert cfg.dataset.dataset_name == "real-text"
    assert cfg.optimizer.lr == 0.0001
    assert cfg.distributed.ep_size == 8
    assert "benchmark" not in cfg
    assert canonical_yaml.read_text() == original


@pytest.mark.parametrize("validation", [True, False])
@pytest.mark.parametrize("fake_gate", [True, False])
def test_benchmark_metadata_controls_validation_and_gate(
    canonical_yaml: Path, validation: bool, fake_gate: bool
) -> None:
    """Apply both strict boolean choices through the same CLI path used by workers."""
    data = yaml.safe_load(canonical_yaml.read_text())
    data["model"]["backend"]["fake_balanced_gate"] = not fake_gate
    data["ci"]["perf_testing"].update(validation=validation, fake_balanced_gate=fake_gate)
    canonical_yaml.write_text(yaml.safe_dump(data))
    cfg = parse_args_and_load_config(argv=["--config", str(canonical_yaml), "--benchmark"])
    assert cfg.model.backend.fake_balanced_gate is fake_gate
    assert cfg.step_scheduler.val_every_steps == (10 if validation else None)
    assert cfg.step_scheduler.validate_on_checkpoint is validation
    assert cfg.checkpoint.enabled is False


@pytest.mark.parametrize("flag_first", [True, False])
def test_explicit_cli_settings_win_over_benchmark_defaults(canonical_yaml: Path, flag_first: bool) -> None:
    """Explicit budgets and metadata overrides work regardless of flag order."""
    overrides = [
        "--step_scheduler.max_steps",
        "7",
        "--step_scheduler.num_epochs",
        "3",
        "--checkpoint.enabled",
        "true",
        "--ci.perf_testing.validation",
        "true",
        "--ci.perf_testing.fake_balanced_gate",
        "true",
        "--model.backend.fake_balanced_gate",
        "false",
    ]
    argv = ["--benchmark", *overrides] if flag_first else [*overrides, "--benchmark"]
    cfg = parse_args_and_load_config(str(canonical_yaml), argv=argv)
    assert cfg.step_scheduler.max_steps == 7
    assert cfg.step_scheduler.num_epochs == 3
    assert cfg.checkpoint.enabled is True
    assert cfg.step_scheduler.val_every_steps == 10
    assert cfg.model.backend.fake_balanced_gate is False


def test_normal_launch_retains_training_settings(canonical_yaml: Path) -> None:
    """Performance metadata alone must not opt normal training into benchmark mode."""
    cfg = parse_args_and_load_config(str(canonical_yaml), argv=[])
    assert cfg.to_yaml_dict() == yaml.safe_load(canonical_yaml.read_text())


@pytest.mark.parametrize(
    "perf, message",
    [
        (None, "requires a ci.perf_testing mapping"),
        (False, "requires a ci.perf_testing mapping"),
        ({"validation": 1}, "validation must be a boolean"),
        ({"fake_balanced_gate": None}, "fake_balanced_gate must be a boolean"),
        ({"fake_balanced_gate": "invalid"}, "fake_balanced_gate must be a boolean"),
        ({"overrides": {"dataset": "synthetic"}}, "unsupported settings"),
    ],
)
def test_benchmark_rejects_invalid_metadata(canonical_yaml: Path, perf: object, message: str) -> None:
    """Malformed or general-purpose override sections must fail before training starts."""
    data = yaml.safe_load(canonical_yaml.read_text())
    data["ci"]["perf_testing"] = perf
    canonical_yaml.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=message):
        parse_args_and_load_config(str(canonical_yaml), argv=["--benchmark"])

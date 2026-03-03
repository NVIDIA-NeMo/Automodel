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

import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import yaml

sys.modules["nemo_run"] = mock.MagicMock()
sys.modules["torch.distributed.run"] = mock.MagicMock()

import cli.app as module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_yaml_file():
    config_data = {"dummy_key": "dummy_value"}
    tmp_dir = tempfile.mkdtemp()
    tmp_file = Path(tmp_dir) / "config.yaml"
    with open(tmp_file, "w") as f:
        yaml.dump(config_data, f)
    yield tmp_file
    shutil.rmtree(tmp_dir)


@pytest.fixture
def recipe_yaml(tmp_path):
    """YAML with a recipe._target_ (no launcher section)."""
    cfg = tmp_path / "recipe.yaml"
    cfg.write_text(yaml.dump({
        "recipe": {
            "_target_": "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction",
        },
        "step_scheduler": {"num_epochs": 1},
    }))
    return cfg


@pytest.fixture
def slurm_yaml(tmp_path):
    """YAML with recipe + slurm section."""
    cfg = tmp_path / "slurm.yaml"
    cfg.write_text(yaml.dump({
        "recipe": {
            "_target_": "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction",
        },
        "step_scheduler": {"num_epochs": 1},
        "slurm": {
            "job_name": "test_job",
            "nodes": 1,
            "ntasks_per_node": 8,
            "job_dir": str(tmp_path / "slurm_jobs"),
        },
    }))
    return cfg


@pytest.fixture
def k8s_yaml(tmp_path):
    """YAML with recipe + k8s section."""
    cfg = tmp_path / "k8s.yaml"
    cfg.write_text(yaml.dump({
        "recipe": {
            "_target_": "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction",
        },
        "step_scheduler": {"num_epochs": 1},
        "k8s": {
            "num_nodes": 2,
            "gpus_per_node": 8,
            "image": "nvcr.io/nvidia/nemo-automodel:latest",
        },
    }))
    return cfg


@pytest.fixture
def nemo_run_yaml(tmp_path):
    """YAML with recipe + nemo_run section."""
    cfg = tmp_path / "nemo_run.yaml"
    cfg.write_text(yaml.dump({
        "recipe": {
            "_target_": "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction",
        },
        "step_scheduler": {"num_epochs": 1},
        "nemo_run": {
            "executor": "local",
            "num_gpus_per_node": 1,
        },
    }))
    return cfg


# ---------------------------------------------------------------------------
# load_yaml tests
# ---------------------------------------------------------------------------

def test_load_yaml_valid(tmp_yaml_file):
    data = module.load_yaml(tmp_yaml_file)
    assert isinstance(data, dict)
    assert "dummy_key" in data


def test_load_yaml_missing():
    with pytest.raises(FileNotFoundError):
        module.load_yaml("non_existent.yaml")


def test_load_yaml_bad_format(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(":\n  -")
    with pytest.raises(Exception):
        module.load_yaml(bad_yaml)


# ---------------------------------------------------------------------------
# build_parser tests
# ---------------------------------------------------------------------------

def test_build_parser_requires_config():
    parser = module.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_build_parser_accepts_config(tmp_path):
    cfg = tmp_path / "test.yaml"
    cfg.write_text("foo: bar")
    parser = module.build_parser()
    args, extra = parser.parse_known_args([str(cfg)])
    assert args.config == cfg
    assert args.nproc_per_node is None


def test_build_parser_nproc_per_node(tmp_path):
    cfg = tmp_path / "test.yaml"
    cfg.write_text("foo: bar")
    parser = module.build_parser()
    args, _ = parser.parse_known_args([str(cfg), "--nproc-per-node=4"])
    assert args.nproc_per_node == 4


# ---------------------------------------------------------------------------
# main() dispatch tests
# ---------------------------------------------------------------------------

def test_main_missing_recipe(monkeypatch, tmp_yaml_file):
    """YAML without recipe._target_ should exit with error."""
    monkeypatch.setattr("sys.argv", ["automodel", str(tmp_yaml_file)])
    with pytest.raises(SystemExit) as exc_info:
        module.main()
    assert exc_info.value.code == 1


def test_main_dispatches_to_interactive(monkeypatch, recipe_yaml):
    """No launcher section -> InteractiveLauncher."""
    monkeypatch.setattr("sys.argv", ["automodel", str(recipe_yaml)])

    launched = {}

    class FakeInteractiveLauncher:
        def launch(self, config, config_path, recipe_target, nproc, extra):
            launched["recipe_target"] = recipe_target
            launched["nproc"] = nproc
            return 0

    monkeypatch.setattr(
        "nemo_automodel.components.launcher.interactive.InteractiveLauncher",
        FakeInteractiveLauncher,
    )
    result = module.main()
    assert result == 0
    assert "TrainFinetuneRecipeForNextTokenPrediction" in launched["recipe_target"]


def test_main_dispatches_to_slurm(monkeypatch, slurm_yaml):
    """slurm: section -> SlurmLauncher."""
    monkeypatch.setattr("sys.argv", ["automodel", str(slurm_yaml)])

    launched = {}

    class FakeSlurmLauncher:
        def launch(self, config, config_path, recipe_target, launcher_config, extra):
            launched["launcher_config"] = launcher_config
            launched["recipe_target"] = recipe_target
            return 0

    monkeypatch.setattr(
        "nemo_automodel.components.launcher.slurm.launcher.SlurmLauncher",
        FakeSlurmLauncher,
    )
    result = module.main()
    assert result == 0
    assert launched["launcher_config"]["job_name"] == "test_job"


def test_main_dispatches_to_k8s(monkeypatch, k8s_yaml):
    """k8s: section -> K8sLauncher."""
    monkeypatch.setattr("sys.argv", ["automodel", str(k8s_yaml)])

    launched = {}

    class FakeK8sLauncher:
        def launch(self, config, config_path, recipe_target, launcher_config, extra):
            launched["launcher_config"] = launcher_config
            return 0

    monkeypatch.setattr(
        "nemo_automodel.components.launcher.k8s.launcher.K8sLauncher",
        FakeK8sLauncher,
    )
    result = module.main()
    assert result == 0
    assert launched["launcher_config"]["num_nodes"] == 2


def test_main_dispatches_to_nemo_run(monkeypatch, nemo_run_yaml):
    """nemo_run: section -> NemoRunLauncher."""
    monkeypatch.setattr("sys.argv", ["automodel", str(nemo_run_yaml)])

    launched = {}

    class FakeNemoRunLauncher:
        def launch(self, config, config_path, recipe_target, launcher_config, extra):
            launched["launcher_config"] = launcher_config
            return 0

    monkeypatch.setattr(
        "nemo_automodel.components.launcher.nemo_run.launcher.NemoRunLauncher",
        FakeNemoRunLauncher,
    )
    result = module.main()
    assert result == 0
    assert launched["launcher_config"]["executor"] == "local"


def test_main_passes_extra_args(monkeypatch, recipe_yaml):
    """Extra CLI args should be forwarded to the launcher."""
    monkeypatch.setattr(
        "sys.argv",
        ["automodel", str(recipe_yaml), "--model.pretrained_model_name_or_path=foo"],
    )

    launched = {}

    class FakeInteractiveLauncher:
        def launch(self, config, config_path, recipe_target, nproc, extra):
            launched["extra"] = extra
            return 0

    monkeypatch.setattr(
        "nemo_automodel.components.launcher.interactive.InteractiveLauncher",
        FakeInteractiveLauncher,
    )
    module.main()
    assert "--model.pretrained_model_name_or_path=foo" in launched["extra"]


# ---------------------------------------------------------------------------
# Repo structure test (unchanged)
# ---------------------------------------------------------------------------

def test_repo_structure():
    repo_root = Path(__file__).parents[3]
    assert (repo_root / "cli").exists()
    assert (repo_root / "cli" / "app.py").exists()
    assert (repo_root / "nemo_automodel").exists()
    assert (repo_root / "nemo_automodel" / "components").exists()
    assert (repo_root / "nemo_automodel" / "recipes").exists()
    assert (repo_root / "nemo_automodel" / "shared").exists()
    assert (repo_root / "docs").exists()
    assert (repo_root / "examples").exists()

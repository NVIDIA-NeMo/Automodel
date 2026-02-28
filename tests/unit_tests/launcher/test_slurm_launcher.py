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

import os
from pathlib import Path
from unittest import mock

import pytest
import yaml

from nemo_automodel.components.launcher.slurm.launcher import (
    SlurmLauncher,
    _get_automodel_repo_root,
    _recipe_module_path,
)


# ---------------------------------------------------------------------------
# _get_automodel_repo_root
# ---------------------------------------------------------------------------
def test_get_automodel_repo_root_detected(tmp_path):
    (tmp_path / "nemo_automodel" / "components").mkdir(parents=True)
    (tmp_path / "examples").mkdir()
    with mock.patch("nemo_automodel.components.launcher.slurm.launcher.Path") as MockPath:
        MockPath.cwd.return_value = tmp_path
        result = _get_automodel_repo_root()
    assert result == tmp_path


def test_get_automodel_repo_root_not_detected(tmp_path):
    with mock.patch("nemo_automodel.components.launcher.slurm.launcher.Path") as MockPath:
        MockPath.cwd.return_value = tmp_path
        result = _get_automodel_repo_root()
    assert result is None


# ---------------------------------------------------------------------------
# _recipe_module_path
# ---------------------------------------------------------------------------
def test_recipe_module_path_basic():
    target = "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipe"
    result = _recipe_module_path(target, "/opt/Automodel")
    assert result == "/opt/Automodel/nemo_automodel/recipes/llm/train_ft.py"


def test_recipe_module_path_short():
    target = "some.module.Recipe"
    result = _recipe_module_path(target, "/workspace")
    assert result == "/workspace/some/module.py"


# ---------------------------------------------------------------------------
# SlurmLauncher.launch
# ---------------------------------------------------------------------------
@pytest.fixture
def base_slurm_config():
    return {
        "nodes": 1,
        "ntasks_per_node": 8,
        "container_image": "nvcr.io/test:latest",
        "partition": "batch",
        "time": "01:00:00",
    }


@pytest.fixture
def recipe_config():
    return {
        "recipe": {"_target_": "nemo_automodel.recipes.llm.train_ft.TrainRecipe"},
        "trainer": {"max_steps": 100},
    }


RECIPE_TARGET = "nemo_automodel.recipes.llm.train_ft.TrainRecipe"


def _mock_submit(return_code=0):
    return mock.patch(
        "nemo_automodel.components.launcher.slurm.utils.submit_slurm_job",
        return_value=return_code,
    )


def _mock_slurm_config():
    return mock.patch(
        "nemo_automodel.components.launcher.slurm.config.SlurmConfig",
        side_effect=lambda **kwargs: mock.MagicMock(**kwargs),
    )


class TestSlurmLauncherLaunch:
    def test_basic_launch(self, tmp_path, base_slurm_config, recipe_config):
        base_slurm_config["job_dir"] = str(tmp_path / "slurm_jobs")
        with _mock_submit(0) as mock_sub, _mock_slurm_config():
            launcher = SlurmLauncher()
            rc = launcher.launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=base_slurm_config,
            )
        assert rc == 0
        mock_sub.assert_called_once()

    def test_writes_config_yaml(self, tmp_path, base_slurm_config, recipe_config):
        base_slurm_config["job_dir"] = str(tmp_path / "slurm_jobs")
        with _mock_submit(0), _mock_slurm_config():
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=base_slurm_config,
            )
        timestamp_dirs = [
            d for d in (tmp_path / "slurm_jobs").iterdir()
            if d.is_dir() and d.name.isdigit()
        ]
        assert len(timestamp_dirs) == 1
        cfg_file = timestamp_dirs[0] / "job_config.yaml"
        assert cfg_file.exists()
        loaded = yaml.safe_load(cfg_file.read_text())
        assert loaded["trainer"]["max_steps"] == 100

    def test_empty_job_name_defaults(self, tmp_path, base_slurm_config, recipe_config):
        base_slurm_config["job_dir"] = str(tmp_path / "slurm_jobs")
        base_slurm_config["job_name"] = ""
        with _mock_submit(0) as mock_sub, _mock_slurm_config() as mock_cfg:
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=base_slurm_config,
            )
        call_kwargs = mock_cfg.call_args.kwargs
        assert call_kwargs.get("job_name") == "automodel_job"

    def test_repo_root_from_config(self, tmp_path, base_slurm_config, recipe_config):
        base_slurm_config["job_dir"] = str(tmp_path / "slurm_jobs")
        base_slurm_config["repo_root"] = "/custom/repo"
        with _mock_submit(0) as mock_sub, _mock_slurm_config() as mock_cfg:
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=base_slurm_config,
            )
        call_kwargs = mock_cfg.call_args.kwargs
        assert call_kwargs.get("chdir") == "/custom/repo"
        assert "repo_root" not in call_kwargs

    def test_repo_root_detected_from_cwd(self, tmp_path, base_slurm_config, recipe_config):
        base_slurm_config["job_dir"] = str(tmp_path / "slurm_jobs")
        with (
            _mock_submit(0),
            _mock_slurm_config() as mock_cfg,
            mock.patch(
                "nemo_automodel.components.launcher.slurm.launcher._get_automodel_repo_root",
                return_value=Path("/detected/root"),
            ),
        ):
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=base_slurm_config,
            )
        call_kwargs = mock_cfg.call_args.kwargs
        assert call_kwargs.get("chdir") == "/detected/root"

    def test_repo_root_fallback(self, tmp_path, base_slurm_config, recipe_config):
        base_slurm_config["job_dir"] = str(tmp_path / "slurm_jobs")
        with (
            _mock_submit(0),
            _mock_slurm_config() as mock_cfg,
            mock.patch(
                "nemo_automodel.components.launcher.slurm.launcher._get_automodel_repo_root",
                return_value=None,
            ),
        ):
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=base_slurm_config,
            )
        call_kwargs = mock_cfg.call_args.kwargs
        assert call_kwargs.get("chdir") == "/opt/Automodel"

    def test_nsys_enabled(self, tmp_path, base_slurm_config, recipe_config):
        base_slurm_config["job_dir"] = str(tmp_path / "slurm_jobs")
        base_slurm_config["nsys_enabled"] = True
        with _mock_submit(0) as mock_sub, _mock_slurm_config() as mock_cfg:
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=base_slurm_config,
            )
        call_kwargs = mock_cfg.call_args.kwargs
        assert "nsys profile" in call_kwargs["command"]
        assert "--capture-range=cudaProfilerApi" in call_kwargs["command"]

    def test_nsys_disabled(self, tmp_path, base_slurm_config, recipe_config):
        base_slurm_config["job_dir"] = str(tmp_path / "slurm_jobs")
        base_slurm_config["nsys_enabled"] = False
        with _mock_submit(0), _mock_slurm_config() as mock_cfg:
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=base_slurm_config,
            )
        call_kwargs = mock_cfg.call_args.kwargs
        assert "nsys profile" not in call_kwargs["command"]

    def test_extra_args_forwarded(self, tmp_path, base_slurm_config, recipe_config):
        base_slurm_config["job_dir"] = str(tmp_path / "slurm_jobs")
        with _mock_submit(0), _mock_slurm_config() as mock_cfg:
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=base_slurm_config,
                extra_args=["--lr=0.001", "--warmup=100"],
            )
        call_kwargs = mock_cfg.call_args.kwargs
        assert "--lr=0.001" in call_kwargs["command"]
        assert "--warmup=100" in call_kwargs["command"]

    def test_hf_home_created(self, tmp_path, base_slurm_config, recipe_config):
        base_slurm_config["job_dir"] = str(tmp_path / "slurm_jobs")
        with _mock_submit(0), _mock_slurm_config():
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=base_slurm_config,
            )
        hf_home = tmp_path / "slurm_jobs" / ".hf_home"
        assert hf_home.exists()

    def test_hf_home_from_config(self, tmp_path, base_slurm_config, recipe_config):
        base_slurm_config["job_dir"] = str(tmp_path / "slurm_jobs")
        hf_dir = tmp_path / "custom_hf"
        hf_dir.mkdir()
        base_slurm_config["hf_home"] = str(hf_dir)
        with _mock_submit(0), _mock_slurm_config() as mock_cfg:
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=base_slurm_config,
            )
        call_kwargs = mock_cfg.call_args.kwargs
        assert call_kwargs["hf_home"] == str(hf_dir)

    def test_extra_mounts_with_existing_repo(self, tmp_path, base_slurm_config, recipe_config):
        base_slurm_config["job_dir"] = str(tmp_path / "slurm_jobs")
        repo = tmp_path / "my_repo"
        repo.mkdir()
        base_slurm_config["repo_root"] = str(repo)
        with _mock_submit(0), _mock_slurm_config() as mock_cfg:
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=base_slurm_config,
            )
        call_kwargs = mock_cfg.call_args.kwargs
        assert len(call_kwargs["extra_mounts"]) == 1

    def test_submit_returns_nonzero(self, tmp_path, base_slurm_config, recipe_config):
        base_slurm_config["job_dir"] = str(tmp_path / "slurm_jobs")
        with _mock_submit(1), _mock_slurm_config():
            rc = SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=base_slurm_config,
            )
        assert rc == 1

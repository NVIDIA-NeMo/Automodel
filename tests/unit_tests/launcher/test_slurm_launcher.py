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

from pathlib import Path
from unittest import mock

import pytest
import yaml

from nemo_automodel.components.launcher.slurm.launcher import (
    SlurmLauncher,
    _get_automodel_repo_root,
    _recipe_module_path,
)
from nemo_automodel.components.launcher.slurm.utils import submit_slurm_job


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
def slurm_script(tmp_path):
    script = tmp_path / "my_cluster.sub"
    script.write_text('#!/bin/bash\nsrun bash -c "$AUTOMODEL_COMMAND"\n')
    return script


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


class TestSlurmLauncherLaunch:
    def test_basic_launch(self, tmp_path, slurm_script, recipe_config):
        slurm_config = {
            "script": str(slurm_script),
            "job_dir": str(tmp_path / "slurm_jobs"),
        }
        with _mock_submit(0) as mock_sub:
            rc = SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=slurm_config,
            )
        assert rc == 0
        mock_sub.assert_called_once()

    def test_writes_config_yaml(self, tmp_path, slurm_script, recipe_config):
        slurm_config = {
            "script": str(slurm_script),
            "job_dir": str(tmp_path / "slurm_jobs"),
        }
        with _mock_submit(0):
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=slurm_config,
            )
        timestamp_dirs = [d for d in (tmp_path / "slurm_jobs").iterdir() if d.is_dir() and d.name.isdigit()]
        assert len(timestamp_dirs) == 1
        cfg_file = timestamp_dirs[0] / "job_config.yaml"
        assert cfg_file.exists()
        loaded = yaml.safe_load(cfg_file.read_text())
        assert loaded["trainer"]["max_steps"] == 100

    def test_copies_script_to_job_dir(self, tmp_path, slurm_script, recipe_config):
        slurm_config = {
            "script": str(slurm_script),
            "job_dir": str(tmp_path / "slurm_jobs"),
        }
        with _mock_submit(0) as mock_sub:
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=slurm_config,
            )
        submitted_path = mock_sub.call_args.args[0]
        assert "my_cluster.sub" in submitted_path
        assert "slurm_jobs" in submitted_path

    def test_exports_env_vars(self, tmp_path, slurm_script, recipe_config):
        slurm_config = {
            "script": str(slurm_script),
            "job_dir": str(tmp_path / "slurm_jobs"),
        }
        with _mock_submit(0) as mock_sub:
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=slurm_config,
            )
        _, env_vars, _ = mock_sub.call_args.args
        assert "AUTOMODEL_COMMAND" in env_vars
        assert "AUTOMODEL_CONFIG" in env_vars
        assert "AUTOMODEL_JOB_DIR" in env_vars
        assert "AUTOMODEL_REPO_ROOT" in env_vars

    def test_command_uses_slurm_env_vars(self, tmp_path, slurm_script, recipe_config):
        slurm_config = {
            "script": str(slurm_script),
            "job_dir": str(tmp_path / "slurm_jobs"),
        }
        with _mock_submit(0) as mock_sub:
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=slurm_config,
            )
        _, env_vars, _ = mock_sub.call_args.args
        cmd = env_vars["AUTOMODEL_COMMAND"]
        assert "${SLURM_GPUS_PER_NODE:-8}" in cmd
        assert "${SLURM_NNODES:-1}" in cmd
        assert "${MASTER_ADDR}:${MASTER_PORT}" in cmd

    def test_script_required(self, tmp_path, recipe_config):
        slurm_config = {"job_dir": str(tmp_path / "slurm_jobs")}
        with pytest.raises(ValueError, match="slurm.script is required"):
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=slurm_config,
            )

    def test_script_not_found(self, tmp_path, recipe_config):
        slurm_config = {
            "script": "/nonexistent/path.sub",
            "job_dir": str(tmp_path / "slurm_jobs"),
        }
        with pytest.raises(FileNotFoundError, match="SLURM script not found"):
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=slurm_config,
            )

    def test_legacy_custom_script_field(self, tmp_path, slurm_script, recipe_config):
        """Backward compat: ``custom_script`` still works."""
        slurm_config = {
            "custom_script": str(slurm_script),
            "job_dir": str(tmp_path / "slurm_jobs"),
        }
        with _mock_submit(0) as mock_sub:
            rc = SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=slurm_config,
            )
        assert rc == 0
        mock_sub.assert_called_once()

    def test_repo_root_from_config(self, tmp_path, slurm_script, recipe_config):
        slurm_config = {
            "script": str(slurm_script),
            "job_dir": str(tmp_path / "slurm_jobs"),
            "repo_root": "/custom/repo",
        }
        with _mock_submit(0) as mock_sub:
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=slurm_config,
            )
        _, env_vars, _ = mock_sub.call_args.args
        assert env_vars["AUTOMODEL_REPO_ROOT"] == "/custom/repo"
        assert "PYTHONPATH=/custom/repo" in env_vars["AUTOMODEL_COMMAND"]

    def test_repo_root_detected_from_cwd(self, tmp_path, slurm_script, recipe_config):
        slurm_config = {
            "script": str(slurm_script),
            "job_dir": str(tmp_path / "slurm_jobs"),
        }
        with (
            _mock_submit(0) as mock_sub,
            mock.patch(
                "nemo_automodel.components.launcher.slurm.launcher._get_automodel_repo_root",
                return_value=Path("/detected/root"),
            ),
        ):
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=slurm_config,
            )
        _, env_vars, _ = mock_sub.call_args.args
        assert env_vars["AUTOMODEL_REPO_ROOT"] == "/detected/root"

    def test_repo_root_fallback(self, tmp_path, slurm_script, recipe_config):
        slurm_config = {
            "script": str(slurm_script),
            "job_dir": str(tmp_path / "slurm_jobs"),
        }
        with (
            _mock_submit(0) as mock_sub,
            mock.patch(
                "nemo_automodel.components.launcher.slurm.launcher._get_automodel_repo_root",
                return_value=None,
            ),
        ):
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=slurm_config,
            )
        _, env_vars, _ = mock_sub.call_args.args
        assert env_vars["AUTOMODEL_REPO_ROOT"] == "/opt/Automodel"

    def test_nsys_enabled(self, tmp_path, slurm_script, recipe_config):
        slurm_config = {
            "script": str(slurm_script),
            "job_dir": str(tmp_path / "slurm_jobs"),
            "nsys_enabled": True,
        }
        with _mock_submit(0) as mock_sub:
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=slurm_config,
            )
        _, env_vars, _ = mock_sub.call_args.args
        assert "nsys profile" in env_vars["AUTOMODEL_COMMAND"]
        assert "--capture-range=cudaProfilerApi" in env_vars["AUTOMODEL_COMMAND"]

    def test_nsys_disabled(self, tmp_path, slurm_script, recipe_config):
        slurm_config = {
            "script": str(slurm_script),
            "job_dir": str(tmp_path / "slurm_jobs"),
            "nsys_enabled": False,
        }
        with _mock_submit(0) as mock_sub:
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=slurm_config,
            )
        _, env_vars, _ = mock_sub.call_args.args
        assert "nsys profile" not in env_vars["AUTOMODEL_COMMAND"]

    def test_extra_args_forwarded(self, tmp_path, slurm_script, recipe_config):
        slurm_config = {
            "script": str(slurm_script),
            "job_dir": str(tmp_path / "slurm_jobs"),
        }
        with _mock_submit(0) as mock_sub:
            SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=slurm_config,
                extra_args=["--lr=0.001", "--warmup=100"],
            )
        _, env_vars, _ = mock_sub.call_args.args
        assert "--lr=0.001" in env_vars["AUTOMODEL_COMMAND"]
        assert "--warmup=100" in env_vars["AUTOMODEL_COMMAND"]

    def test_submit_returns_nonzero(self, tmp_path, slurm_script, recipe_config):
        slurm_config = {
            "script": str(slurm_script),
            "job_dir": str(tmp_path / "slurm_jobs"),
        }
        with _mock_submit(1):
            rc = SlurmLauncher().launch(
                config=recipe_config,
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config=slurm_config,
            )
        assert rc == 1


class TestSubmitSlurmJob:
    def test_submits_with_env(self, tmp_path):
        script = tmp_path / "test.sbatch"
        script.write_text("#!/bin/bash\necho ok\n")
        job_dir = str(tmp_path / "job")

        with mock.patch("subprocess.Popen") as mock_popen:
            mock_proc = mock.MagicMock()
            mock_proc.communicate.return_value = (b"Submitted batch job 12345\n", b"")
            mock_proc.returncode = 0
            mock_popen.return_value = mock_proc

            rc = submit_slurm_job(
                str(script),
                {"AUTOMODEL_COMMAND": "torchrun foo.py", "AUTOMODEL_CONFIG": "/tmp/cfg.yaml"},
                job_dir,
            )

        assert rc == 0
        call_kwargs = mock_popen.call_args
        assert call_kwargs.kwargs["env"]["AUTOMODEL_COMMAND"] == "torchrun foo.py"
        assert call_kwargs.kwargs["env"]["AUTOMODEL_CONFIG"] == "/tmp/cfg.yaml"
        assert (Path(job_dir) / "subproc_sbatch.stdout").exists()

    def test_returns_nonzero_on_failure(self, tmp_path):
        script = tmp_path / "test.sbatch"
        script.write_text("#!/bin/bash\necho ok\n")
        job_dir = str(tmp_path / "job")

        with mock.patch("subprocess.Popen") as mock_popen:
            mock_proc = mock.MagicMock()
            mock_proc.communicate.return_value = (b"", b"sbatch: error: something\n")
            mock_proc.returncode = 1
            mock_popen.return_value = mock_proc

            rc = submit_slurm_job(str(script), {}, job_dir)

        assert rc == 1

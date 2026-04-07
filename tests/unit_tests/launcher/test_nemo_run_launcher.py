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

import sys
from pathlib import Path
from unittest import mock

import pytest
import yaml

from nemo_automodel.components.launcher.nemo_run.config import NemoRunConfig
from nemo_automodel.components.launcher.nemo_run.launcher import NemoRunLauncher


RECIPE_TARGET = "nemo_automodel.recipes.llm.train_ft.TrainRecipe"


# ---------------------------------------------------------------------------
# Stub nemo_run module
# ---------------------------------------------------------------------------


def _make_mock_nemo_run():
    """Create a mock nemo_run module with the necessary attributes."""
    mock_run = mock.MagicMock()
    mock_run.LocalExecutor = mock.MagicMock
    mock_run.Script = mock.MagicMock()
    mock_exp = mock.MagicMock()
    mock_exp.__enter__ = mock.MagicMock(return_value=mock_exp)
    mock_exp.__exit__ = mock.MagicMock(return_value=False)
    mock_exp._exp_dir = "/tmp/fake_exp_dir"
    mock_run.Experiment.return_value = mock_exp
    return mock_run


# ---------------------------------------------------------------------------
# _configure_torchrun
# ---------------------------------------------------------------------------


class TestConfigureTorchrun:
    def test_sets_launcher_to_torchrun(self):
        executor = mock.MagicMock()
        executor.torchrun_nproc_per_node = None
        NemoRunLauncher._configure_torchrun(executor, devices=8)
        assert executor.launcher == "torchrun"
        assert executor.torchrun_nproc_per_node == 8

    def test_sets_nproc_per_node(self):
        executor = mock.MagicMock()
        executor.torchrun_nproc_per_node = None
        NemoRunLauncher._configure_torchrun(executor, devices=4)
        assert executor.torchrun_nproc_per_node == 4

    def test_skips_nproc_if_attr_missing(self):
        executor = mock.MagicMock(spec=[])  # no attributes
        NemoRunLauncher._configure_torchrun(executor, devices=8)
        assert executor.launcher == "torchrun"
        assert not hasattr(executor, "torchrun_nproc_per_node")


# ---------------------------------------------------------------------------
# _resolve_executor
# ---------------------------------------------------------------------------


class TestResolveExecutor:
    def test_local_executor(self, monkeypatch):
        mock_run = _make_mock_nemo_run()
        local_executor_instance = mock.MagicMock()
        mock_run.LocalExecutor = mock.MagicMock(return_value=local_executor_instance)
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)

        launcher = NemoRunLauncher()
        nr_config = NemoRunConfig(executor="local")
        executor = launcher._resolve_executor(nr_config)

        mock_run.LocalExecutor.assert_called_once()
        assert executor is local_executor_instance

    def test_local_executor_with_overrides(self, monkeypatch):
        mock_run = _make_mock_nemo_run()
        local_executor_instance = mock.MagicMock()
        mock_run.LocalExecutor = mock.MagicMock(return_value=local_executor_instance)
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)

        launcher = NemoRunLauncher()
        nr_config = NemoRunConfig(executor="local", overrides={"ntasks_per_node": 4})
        executor = launcher._resolve_executor(nr_config)
        # overrides are applied via apply_overrides -> setattr
        assert executor.ntasks_per_node == 4

    def test_named_executor_loads_from_file(self, monkeypatch):
        mock_run = _make_mock_nemo_run()
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)

        fake_executor = mock.MagicMock()

        with mock.patch(
            "nemo_automodel.components.launcher.nemo_run.launcher.load_executor_from_file",
            return_value=fake_executor,
        ) as mock_load, mock.patch(
            "nemo_automodel.components.launcher.nemo_run.launcher.apply_overrides",
        ) as mock_apply:
            launcher = NemoRunLauncher()
            nr_config = NemoRunConfig(
                executor="my_cluster",
                executors_file="/custom/executors.py",
                overrides={"nodes": 2},
            )
            executor = launcher._resolve_executor(nr_config)

        mock_load.assert_called_once_with("my_cluster", "/custom/executors.py")
        mock_apply.assert_called_once_with(fake_executor, {"nodes": 2})
        assert executor is fake_executor

    def test_missing_nemo_run_raises_system_exit(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "nemo_run", None)

        launcher = NemoRunLauncher()
        nr_config = NemoRunConfig(executor="local")

        with pytest.raises(SystemExit):
            launcher._resolve_executor(nr_config)

        monkeypatch.delitem(sys.modules, "nemo_run", raising=False)


# ---------------------------------------------------------------------------
# NemoRunLauncher.launch
# ---------------------------------------------------------------------------


def _make_patch_submit(monkeypatch, captured=None):
    """Patch submit_nemo_run_job to capture call kwargs."""
    def _submit(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return 0

    monkeypatch.setattr(
        "nemo_automodel.components.launcher.nemo_run.launcher.submit_nemo_run_job",
        lambda **kw: _submit(**kw),
    )


class TestLaunch:
    def test_launch_returns_zero(self, tmp_path, monkeypatch):
        mock_run = _make_mock_nemo_run()
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)
        _make_patch_submit(monkeypatch)

        launcher = NemoRunLauncher()
        result = launcher.launch(
            config={"model": {"name": "gpt2"}},
            config_path=Path("/tmp/config.yaml"),
            recipe_target=RECIPE_TARGET,
            launcher_config={
                "executor": "local",
                "ntasks_per_node": 4,
                "job_dir": str(tmp_path / "nemo_jobs"),
            },
        )
        assert result == 0

    def test_launch_strips_nemo_run_from_config(self, tmp_path, monkeypatch):
        mock_run = _make_mock_nemo_run()
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)
        _make_patch_submit(monkeypatch)

        launcher = NemoRunLauncher()
        config = {"model": {"name": "gpt2"}, "trainer": {"max_steps": 100}}
        job_dir = str(tmp_path / "nemo_jobs")
        launcher.launch(
            config=config,
            config_path=Path("/tmp/config.yaml"),
            recipe_target=RECIPE_TARGET,
            launcher_config={"executor": "local", "job_dir": job_dir},
        )

        import os
        import glob

        job_dirs = glob.glob(os.path.join(job_dir, "*"))
        assert len(job_dirs) == 1
        conf_path = os.path.join(job_dirs[0], "automodel_config.yaml")
        with open(conf_path) as f:
            written_config = yaml.safe_load(f)

        assert "nemo_run" not in written_config
        assert written_config["model"]["name"] == "gpt2"
        assert written_config["trainer"]["max_steps"] == 100

    def test_launch_creates_job_dir(self, tmp_path, monkeypatch):
        mock_run = _make_mock_nemo_run()
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)
        _make_patch_submit(monkeypatch)

        launcher = NemoRunLauncher()
        job_dir = str(tmp_path / "nemo_jobs")
        launcher.launch(
            config={"k": "v"},
            config_path=Path("/tmp/cfg.yaml"),
            recipe_target=RECIPE_TARGET,
            launcher_config={"executor": "local", "job_dir": job_dir},
        )

        import os
        import glob

        job_dirs = glob.glob(os.path.join(job_dir, "*"))
        assert len(job_dirs) == 1
        assert os.path.isfile(os.path.join(job_dirs[0], "automodel_config.yaml"))

    def test_launch_creates_script_with_module_path(self, tmp_path, monkeypatch):
        mock_run = _make_mock_nemo_run()
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)
        _make_patch_submit(monkeypatch)

        launcher = NemoRunLauncher()
        launcher.launch(
            config={"model": {"name": "llama"}},
            config_path=Path("/tmp/cfg.yaml"),
            recipe_target=RECIPE_TARGET,
            launcher_config={
                "executor": "local",
                "ntasks_per_node": 8,
                "job_dir": str(tmp_path / "jobs"),
            },
        )

        mock_run.Script.assert_called_once()
        call_kwargs = mock_run.Script.call_args.kwargs
        assert call_kwargs["m"] is True
        assert call_kwargs["entrypoint"] == "python"
        assert call_kwargs["path"] == "nemo_automodel.recipes.llm.train_ft"

    def test_launch_passes_config_path_in_args(self, tmp_path, monkeypatch):
        mock_run = _make_mock_nemo_run()
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)
        _make_patch_submit(monkeypatch)

        launcher = NemoRunLauncher()
        launcher.launch(
            config={},
            config_path=Path("/tmp/cfg.yaml"),
            recipe_target=RECIPE_TARGET,
            launcher_config={"executor": "local", "job_dir": str(tmp_path / "jobs")},
        )

        call_kwargs = mock_run.Script.call_args.kwargs
        assert "-c" in call_kwargs["args"]
        assert "/nemo_run/code/automodel_config.yaml" in call_kwargs["args"]

    def test_launch_configures_torchrun_on_executor(self, tmp_path, monkeypatch):
        mock_run = _make_mock_nemo_run()
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)

        captured = {}
        _make_patch_submit(monkeypatch, captured)

        launcher = NemoRunLauncher()
        launcher.launch(
            config={},
            config_path=Path("/tmp/cfg.yaml"),
            recipe_target=RECIPE_TARGET,
            launcher_config={
                "executor": "local",
                "ntasks_per_node": 8,
                "job_dir": str(tmp_path / "jobs"),
            },
        )

        submitted_executor = captured["executor"]
        assert submitted_executor.launcher == "torchrun"

    def test_launch_missing_nemo_run_exits(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "nemo_run", None)

        launcher = NemoRunLauncher()
        with pytest.raises(SystemExit):
            launcher.launch(
                config={},
                config_path=Path("/tmp/cfg.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config={"executor": "local"},
            )

        monkeypatch.delitem(sys.modules, "nemo_run", raising=False)

    def test_launch_job_name_from_recipe_target(self, tmp_path, monkeypatch):
        mock_run = _make_mock_nemo_run()
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)

        captured = {}
        _make_patch_submit(monkeypatch, captured)

        launcher = NemoRunLauncher()
        launcher.launch(
            config={},
            config_path=Path("/tmp/cfg.yaml"),
            recipe_target="nemo_automodel.recipes.llm.train_ft.TrainRecipe",
            launcher_config={"executor": "local", "job_dir": str(tmp_path / "jobs")},
        )
        assert captured["job_name"] == "TrainRecipe"

    def test_launch_custom_job_name(self, tmp_path, monkeypatch):
        mock_run = _make_mock_nemo_run()
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)

        captured = {}
        _make_patch_submit(monkeypatch, captured)

        launcher = NemoRunLauncher()
        launcher.launch(
            config={},
            config_path=Path("/tmp/cfg.yaml"),
            recipe_target=RECIPE_TARGET,
            launcher_config={
                "executor": "local",
                "job_name": "my_experiment",
                "job_dir": str(tmp_path / "jobs"),
            },
        )
        assert captured["job_name"] == "my_experiment"

    def test_launch_extra_args_forwarded(self, tmp_path, monkeypatch):
        mock_run = _make_mock_nemo_run()
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)
        _make_patch_submit(monkeypatch)

        launcher = NemoRunLauncher()
        launcher.launch(
            config={},
            config_path=Path("/tmp/cfg.yaml"),
            recipe_target=RECIPE_TARGET,
            launcher_config={"executor": "local", "job_dir": str(tmp_path / "jobs")},
            extra_args=["--override", "lr=0.001"],
        )

        call_kwargs = mock_run.Script.call_args.kwargs
        assert "--override" in call_kwargs["args"]
        assert "lr=0.001" in call_kwargs["args"]

    def test_launch_sets_pattern_packager(self, tmp_path, monkeypatch):
        mock_run = _make_mock_nemo_run()
        mock_run.PatternPackager = mock.MagicMock()
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)

        captured = {}
        _make_patch_submit(monkeypatch, captured)

        launcher = NemoRunLauncher()
        launcher.launch(
            config={"model": "test"},
            config_path=Path("/tmp/cfg.yaml"),
            recipe_target=RECIPE_TARGET,
            launcher_config={"executor": "local", "job_dir": str(tmp_path / "jobs")},
        )

        # PatternPackager should have been called with the config file path
        mock_run.PatternPackager.assert_called_once()
        call_kwargs = mock_run.PatternPackager.call_args.kwargs
        assert "automodel_config.yaml" in call_kwargs["include_pattern"]

    def test_launch_executor_overrides_from_yaml(self, tmp_path, monkeypatch):
        """Arbitrary YAML keys become executor overrides via from_dict."""
        mock_run = _make_mock_nemo_run()
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)
        _make_patch_submit(monkeypatch)

        launcher = NemoRunLauncher()
        launcher.launch(
            config={},
            config_path=Path("/tmp/cfg.yaml"),
            recipe_target=RECIPE_TARGET,
            launcher_config={
                "executor": "local",
                "ntasks_per_node": 4,
                "partition": "interactive",
                "container_image": "test.sqsh",
                "job_dir": str(tmp_path / "jobs"),
            },
        )

        # The Script call should succeed — these fields go to overrides
        mock_run.Script.assert_called_once()

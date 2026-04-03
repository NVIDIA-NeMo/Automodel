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
from nemo_automodel.components.launcher.nemo_run.utils import REMOTE_CONFIG_PATH


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
    mock_run.Experiment.return_value = mock_exp
    return mock_run


# ---------------------------------------------------------------------------
# _build_inline_script
# ---------------------------------------------------------------------------


class TestBuildInlineScript:
    def test_single_node_script(self):
        launcher = NemoRunLauncher()
        script = launcher._build_inline_script(
            config_yaml="model: gpt2\n",
            recipe_target=RECIPE_TARGET,
            devices=4,
            num_nodes=1,
        )
        assert "#!/bin/bash" in script
        assert "set -euo pipefail" in script
        assert f"cat > {REMOTE_CONFIG_PATH}" in script
        assert "model: gpt2" in script
        assert "AUTOMODEL_CONFIG_EOF" in script
        assert "torchrun" in script
        assert "--nproc-per-node=4" in script
        # Single-node should NOT include multi-node flags
        assert "--nnodes=" not in script
        assert "$NODE_RANK" not in script
        assert "--rdzv-backend" not in script
        assert "--master-addr" not in script
        assert "--master-port" not in script

    def test_multi_node_script(self):
        launcher = NemoRunLauncher()
        script = launcher._build_inline_script(
            config_yaml="model: llama\n",
            recipe_target=RECIPE_TARGET,
            devices=8,
            num_nodes=4,
        )
        assert "--nproc-per-node=8" in script
        assert "--nnodes=4" in script
        assert "--node-rank=$NODE_RANK" in script
        assert "--rdzv-backend=c10d" in script
        assert "--master-addr=$MASTER_ADDR" in script
        assert "--master-port=$MASTER_PORT" in script

    def test_extra_args_appended(self):
        launcher = NemoRunLauncher()
        script = launcher._build_inline_script(
            config_yaml="k: v\n",
            recipe_target=RECIPE_TARGET,
            devices=1,
            num_nodes=1,
            extra_args=["--my-flag", "val"],
        )
        assert "--my-flag" in script
        assert "val" in script

    def test_config_yaml_embedded(self):
        long_yaml = "trainer:\n  max_steps: 1000\nmodel:\n  name: gpt2\n"
        launcher = NemoRunLauncher()
        script = launcher._build_inline_script(
            config_yaml=long_yaml,
            recipe_target=RECIPE_TARGET,
            devices=1,
            num_nodes=1,
        )
        assert "trainer:" in script
        assert "max_steps: 1000" in script
        assert "name: gpt2" in script

    def test_recipe_target_converted_to_path(self):
        launcher = NemoRunLauncher()
        script = launcher._build_inline_script(
            config_yaml="k: v\n",
            recipe_target="nemo_automodel.recipes.llm.train_ft.TrainRecipe",
            devices=1,
            num_nodes=1,
        )
        # The recipe module path should be converted to a filesystem path
        assert "nemo_automodel/recipes/llm/train_ft.py" in script

    def test_script_includes_config_flag(self):
        launcher = NemoRunLauncher()
        script = launcher._build_inline_script(
            config_yaml="k: v\n",
            recipe_target=RECIPE_TARGET,
            devices=1,
            num_nodes=1,
        )
        assert f"-c {REMOTE_CONFIG_PATH}" in script


# ---------------------------------------------------------------------------
# _resolve_executor
# ---------------------------------------------------------------------------


class TestResolveExecutor:
    def test_local_executor(self, monkeypatch):
        mock_run = _make_mock_nemo_run()
        # Make LocalExecutor a real callable that records its calls
        local_executor_instance = mock.MagicMock()
        mock_run.LocalExecutor = mock.MagicMock(return_value=local_executor_instance)
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)

        launcher = NemoRunLauncher()
        nr_config = NemoRunConfig(executor="local", devices=4)
        executor = launcher._resolve_executor(nr_config)

        mock_run.LocalExecutor.assert_called_once_with(ntasks_per_node=4)
        assert executor is local_executor_instance

    def test_local_executor_default_devices(self, monkeypatch):
        mock_run = _make_mock_nemo_run()
        local_executor_instance = mock.MagicMock()
        mock_run.LocalExecutor = mock.MagicMock(return_value=local_executor_instance)
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)

        launcher = NemoRunLauncher()
        nr_config = NemoRunConfig(executor="local")
        launcher._resolve_executor(nr_config)

        mock_run.LocalExecutor.assert_called_once_with(ntasks_per_node=1)

    def test_local_executor_env_vars(self, monkeypatch):
        mock_run = _make_mock_nemo_run()
        local_executor_instance = mock.MagicMock()
        mock_run.LocalExecutor = mock.MagicMock(return_value=local_executor_instance)
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)

        launcher = NemoRunLauncher()
        nr_config = NemoRunConfig(
            executor="local",
            env_vars={"MY_VAR": "val"},
        )
        executor = launcher._resolve_executor(nr_config)
        assert executor.env_vars == {"MY_VAR": "val"}

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
                nodes=2,
                devices=8,
                executors_file="/custom/executors.py",
            )
            executor = launcher._resolve_executor(nr_config)

        mock_load.assert_called_once_with("my_cluster", "/custom/executors.py")
        mock_apply.assert_called_once()
        assert executor is fake_executor

    def test_missing_nemo_run_raises_system_exit(self, monkeypatch):
        # Ensure nemo_run cannot be imported
        monkeypatch.setitem(sys.modules, "nemo_run", None)

        launcher = NemoRunLauncher()
        nr_config = NemoRunConfig(executor="local")

        with pytest.raises(SystemExit):
            launcher._resolve_executor(nr_config)

        # Restore
        monkeypatch.delitem(sys.modules, "nemo_run", raising=False)


# ---------------------------------------------------------------------------
# NemoRunLauncher.launch
# ---------------------------------------------------------------------------


class TestLaunch:
    def test_launch_returns_zero(self, tmp_path, monkeypatch):
        mock_run = _make_mock_nemo_run()
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)

        captured = {}

        def fake_submit(script, executor, job_name, detach, tail_logs):
            captured["job_name"] = job_name
            captured["detach"] = detach
            captured["tail_logs"] = tail_logs
            return 0

        monkeypatch.setattr(
            "nemo_automodel.components.launcher.nemo_run.launcher.submit_nemo_run_job",
            fake_submit,
        )

        launcher = NemoRunLauncher()
        result = launcher.launch(
            config={"model": {"name": "gpt2"}},
            config_path=Path("/tmp/config.yaml"),
            recipe_target=RECIPE_TARGET,
            launcher_config={
                "executor": "local",
                "devices": 4,
                "job_dir": str(tmp_path / "nemo_jobs"),
            },
        )
        assert result == 0

    def test_launch_strips_nemo_run_from_config(self, tmp_path, monkeypatch):
        mock_run = _make_mock_nemo_run()
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)

        written_config = {}

        def fake_submit(script, executor, job_name, detach, tail_logs):
            return 0

        monkeypatch.setattr(
            "nemo_automodel.components.launcher.nemo_run.launcher.submit_nemo_run_job",
            fake_submit,
        )

        launcher = NemoRunLauncher()
        # The config passed to launch() should not contain the nemo_run section
        # (it is stripped by the CLI before calling launch).
        config = {"model": {"name": "gpt2"}, "trainer": {"max_steps": 100}}
        job_dir = str(tmp_path / "nemo_jobs")
        launcher.launch(
            config=config,
            config_path=Path("/tmp/config.yaml"),
            recipe_target=RECIPE_TARGET,
            launcher_config={
                "executor": "local",
                "job_dir": job_dir,
            },
        )

        # Find and read back the written job_config.yaml
        import os
        import glob

        job_dirs = glob.glob(os.path.join(job_dir, "*"))
        assert len(job_dirs) == 1
        conf_path = os.path.join(job_dirs[0], "job_config.yaml")
        with open(conf_path) as f:
            written_config = yaml.safe_load(f)

        assert "nemo_run" not in written_config
        assert written_config["model"]["name"] == "gpt2"
        assert written_config["trainer"]["max_steps"] == 100

    def test_launch_creates_job_dir(self, tmp_path, monkeypatch):
        mock_run = _make_mock_nemo_run()
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)

        monkeypatch.setattr(
            "nemo_automodel.components.launcher.nemo_run.launcher.submit_nemo_run_job",
            lambda script, executor, job_name, detach, tail_logs: 0,
        )

        launcher = NemoRunLauncher()
        job_dir = str(tmp_path / "nemo_jobs")
        launcher.launch(
            config={"k": "v"},
            config_path=Path("/tmp/cfg.yaml"),
            recipe_target=RECIPE_TARGET,
            launcher_config={
                "executor": "local",
                "job_dir": job_dir,
            },
        )

        import os
        import glob

        job_dirs = glob.glob(os.path.join(job_dir, "*"))
        assert len(job_dirs) == 1
        assert os.path.isfile(os.path.join(job_dirs[0], "job_config.yaml"))

    def test_launch_builds_script(self, tmp_path, monkeypatch):
        mock_run = _make_mock_nemo_run()
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)

        captured = {}

        def fake_submit(script, executor, job_name, detach, tail_logs):
            captured["script"] = script
            return 0

        monkeypatch.setattr(
            "nemo_automodel.components.launcher.nemo_run.launcher.submit_nemo_run_job",
            fake_submit,
        )

        launcher = NemoRunLauncher()
        launcher.launch(
            config={"model": {"name": "llama"}},
            config_path=Path("/tmp/cfg.yaml"),
            recipe_target=RECIPE_TARGET,
            launcher_config={
                "executor": "local",
                "devices": 8,
                "nodes": 2,
                "job_dir": str(tmp_path / "jobs"),
            },
        )

        # run.Script was called with the inline script
        mock_run.Script.assert_called_once()
        call_kwargs = mock_run.Script.call_args
        inline_arg = call_kwargs.kwargs.get("inline") or call_kwargs[1].get("inline")
        assert inline_arg is not None
        assert "torchrun" in inline_arg
        assert "--nnodes=2" in inline_arg
        assert "--nproc-per-node=8" in inline_arg

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

        def fake_submit(script, executor, job_name, detach, tail_logs):
            captured["job_name"] = job_name
            return 0

        monkeypatch.setattr(
            "nemo_automodel.components.launcher.nemo_run.launcher.submit_nemo_run_job",
            fake_submit,
        )

        launcher = NemoRunLauncher()
        launcher.launch(
            config={},
            config_path=Path("/tmp/cfg.yaml"),
            recipe_target="nemo_automodel.recipes.llm.train_ft.TrainRecipe",
            launcher_config={
                "executor": "local",
                "job_dir": str(tmp_path / "jobs"),
            },
        )

        # When no job_name is set, it should use the last part of the recipe target
        assert captured["job_name"] == "TrainRecipe"

    def test_launch_custom_job_name(self, tmp_path, monkeypatch):
        mock_run = _make_mock_nemo_run()
        monkeypatch.setitem(sys.modules, "nemo_run", mock_run)

        captured = {}

        def fake_submit(script, executor, job_name, detach, tail_logs):
            captured["job_name"] = job_name
            return 0

        monkeypatch.setattr(
            "nemo_automodel.components.launcher.nemo_run.launcher.submit_nemo_run_job",
            fake_submit,
        )

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

        monkeypatch.setattr(
            "nemo_automodel.components.launcher.nemo_run.launcher.submit_nemo_run_job",
            lambda script, executor, job_name, detach, tail_logs: 0,
        )

        launcher = NemoRunLauncher()
        launcher.launch(
            config={},
            config_path=Path("/tmp/cfg.yaml"),
            recipe_target=RECIPE_TARGET,
            launcher_config={
                "executor": "local",
                "job_dir": str(tmp_path / "jobs"),
            },
            extra_args=["--override", "lr=0.001"],
        )

        inline_arg = mock_run.Script.call_args.kwargs.get("inline") or mock_run.Script.call_args[1].get("inline")
        assert "--override" in inline_arg
        assert "lr=0.001" in inline_arg

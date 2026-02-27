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
from types import SimpleNamespace
from unittest import mock

import pytest

from nemo_automodel.components.launcher.nemo_run.config import NemoRunConfig
from nemo_automodel.components.launcher.nemo_run.launcher import NemoRunLauncher


RECIPE_TARGET = "nemo_automodel.recipes.llm.train_ft.TrainRecipe"


def _make_mock_nemo_run():
    mock_run = mock.MagicMock()
    mock_run.LocalExecutor = mock.MagicMock
    mock_run.SlurmExecutor = mock.MagicMock
    mock_run.K8sExecutor = mock.MagicMock
    mock_run.Partial = mock.MagicMock()
    mock_exp = mock.MagicMock()
    mock_exp.__enter__ = mock.MagicMock(return_value=mock_exp)
    mock_exp.__exit__ = mock.MagicMock(return_value=False)
    mock_run.Experiment.return_value = mock_exp
    return mock_run


# ---------------------------------------------------------------------------
# _build_executor
# ---------------------------------------------------------------------------
class TestBuildExecutor:
    def test_local_executor(self):
        mock_run = mock.MagicMock()
        cfg = NemoRunConfig(executor="local", num_gpus_per_node=4)
        NemoRunLauncher._build_executor(mock_run, cfg)
        mock_run.LocalExecutor.assert_called_once_with(ntasks_per_node=4)

    def test_slurm_executor(self):
        mock_run = mock.MagicMock()
        cfg = NemoRunConfig(
            executor="slurm",
            account="my-acct",
            partition="batch",
            time="02:00:00",
            num_nodes=4,
            num_gpus_per_node=8,
            container_image="nvcr.io/test:latest",
        )
        NemoRunLauncher._build_executor(mock_run, cfg)
        mock_run.SlurmExecutor.assert_called_once_with(
            account="my-acct",
            partition="batch",
            time="02:00:00",
            nodes=4,
            ntasks_per_node=8,
            container_image="nvcr.io/test:latest",
        )

    def test_k8s_executor(self):
        mock_run = mock.MagicMock()
        cfg = NemoRunConfig(
            executor="k8s",
            num_nodes=2,
            num_gpus_per_node=8,
            container_image="nvcr.io/test:latest",
        )
        NemoRunLauncher._build_executor(mock_run, cfg)
        mock_run.K8sExecutor.assert_called_once_with(
            num_nodes=2,
            num_gpus_per_node=8,
            container_image="nvcr.io/test:latest",
        )

    def test_unknown_executor_raises(self):
        mock_run = mock.MagicMock()
        cfg = NemoRunConfig(executor="unknown_backend")
        with pytest.raises(ValueError, match="Unknown nemo_run executor"):
            NemoRunLauncher._build_executor(mock_run, cfg)

    def test_slurm_executor_with_extra_kwargs(self):
        mock_run = mock.MagicMock()
        cfg = NemoRunConfig(
            executor="slurm",
            account="acct",
            partition="gpu",
            time="01:00:00",
            num_nodes=1,
            num_gpus_per_node=8,
            container_image="img:latest",
            executor_kwargs={"mem": "64G", "qos": "high"},
        )
        NemoRunLauncher._build_executor(mock_run, cfg)
        call_kwargs = mock_run.SlurmExecutor.call_args.kwargs
        assert call_kwargs["mem"] == "64G"
        assert call_kwargs["qos"] == "high"

    def test_local_executor_with_extra_kwargs(self):
        mock_run = mock.MagicMock()
        cfg = NemoRunConfig(
            executor="local",
            num_gpus_per_node=2,
            executor_kwargs={"env_vars": {"CUDA_VISIBLE_DEVICES": "0,1"}},
        )
        NemoRunLauncher._build_executor(mock_run, cfg)
        call_kwargs = mock_run.LocalExecutor.call_args.kwargs
        assert "env_vars" in call_kwargs

    def test_k8s_executor_with_extra_kwargs(self):
        mock_run = mock.MagicMock()
        cfg = NemoRunConfig(
            executor="k8s",
            num_nodes=1,
            num_gpus_per_node=4,
            container_image="img:v1",
            executor_kwargs={"namespace": "custom-ns"},
        )
        NemoRunLauncher._build_executor(mock_run, cfg)
        call_kwargs = mock_run.K8sExecutor.call_args.kwargs
        assert call_kwargs["namespace"] == "custom-ns"


# ---------------------------------------------------------------------------
# NemoRunLauncher.launch
# ---------------------------------------------------------------------------
class TestNemoRunLauncherLaunch:
    def test_launch_success(self, tmp_path):
        mock_run = _make_mock_nemo_run()
        mock_recipe_cls = mock.MagicMock()

        with (
            mock.patch.dict(sys.modules, {"nemo_run": mock_run}),
            mock.patch(
                "nemo_automodel.components.launcher.nemo_run.launcher.resolve_recipe_cls",
                return_value=mock_recipe_cls,
            ),
        ):
            launcher = NemoRunLauncher()
            rc = launcher.launch(
                config={"trainer": {"max_steps": 100}},
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config={"executor": "local", "num_gpus_per_node": 4},
            )
        assert rc == 0
        mock_run.Partial.assert_called_once()
        mock_run.Experiment.assert_called_once_with("automodel_job")

    def test_launch_nemo_run_not_installed(self):
        with mock.patch(
            "nemo_automodel.components.launcher.nemo_run.launcher.resolve_recipe_cls",
        ):
            original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            def fake_import(name, *args, **kwargs):
                if name == "nemo_run":
                    raise ImportError("No module named 'nemo_run'")
                return original_import(name, *args, **kwargs)

            with (
                mock.patch("builtins.__import__", side_effect=fake_import),
                pytest.raises(SystemExit),
            ):
                NemoRunLauncher().launch(
                    config={},
                    config_path=Path("/tmp/config.yaml"),
                    recipe_target=RECIPE_TARGET,
                    launcher_config={"executor": "local"},
                )

    def test_launch_calls_experiment_run(self, tmp_path):
        mock_run = _make_mock_nemo_run()
        mock_exp = mock_run.Experiment.return_value.__enter__.return_value
        mock_recipe_cls = mock.MagicMock()

        with (
            mock.patch.dict(sys.modules, {"nemo_run": mock_run}),
            mock.patch(
                "nemo_automodel.components.launcher.nemo_run.launcher.resolve_recipe_cls",
                return_value=mock_recipe_cls,
            ),
        ):
            NemoRunLauncher().launch(
                config={"key": "val"},
                config_path=Path("/tmp/config.yaml"),
                recipe_target=RECIPE_TARGET,
                launcher_config={"executor": "local", "num_gpus_per_node": 2},
            )
        mock_exp.add.assert_called_once()
        mock_exp.run.assert_called_once_with(sequential=True)


# ---------------------------------------------------------------------------
# NemoRunConfig defaults
# ---------------------------------------------------------------------------
class TestNemoRunConfig:
    def test_default_values(self):
        cfg = NemoRunConfig()
        assert cfg.executor == "local"
        assert cfg.num_nodes == 1
        assert cfg.num_gpus_per_node == 8
        assert cfg.container_image == ""
        assert cfg.account == ""
        assert cfg.partition == ""
        assert cfg.time == "01:00:00"
        assert cfg.executor_kwargs == {}

    def test_custom_values(self):
        cfg = NemoRunConfig(
            executor="slurm",
            num_nodes=4,
            num_gpus_per_node=8,
            container_image="img:v1",
            account="my-acct",
            partition="gpu",
            time="04:00:00",
            executor_kwargs={"mem": "128G"},
        )
        assert cfg.executor == "slurm"
        assert cfg.num_nodes == 4
        assert cfg.executor_kwargs["mem"] == "128G"

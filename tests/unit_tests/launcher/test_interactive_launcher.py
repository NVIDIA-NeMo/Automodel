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
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from nemo_automodel.components.launcher.interactive import (
    InteractiveLauncher,
    _get_repo_root,
    _recipe_module_path,
    resolve_recipe_cls,
)


# ---------------------------------------------------------------------------
# resolve_recipe_cls
# ---------------------------------------------------------------------------
def test_resolve_recipe_cls_invalid():
    with pytest.raises(ModuleNotFoundError):
        resolve_recipe_cls("nonexistent.module.ClassName")


def test_resolve_recipe_cls_valid():
    cls = resolve_recipe_cls("pathlib.Path")
    assert cls is Path


def test_resolve_recipe_cls_builtin_module():
    cls = resolve_recipe_cls("os.path.join")
    assert cls is os.path.join


# ---------------------------------------------------------------------------
# _recipe_module_path
# ---------------------------------------------------------------------------
def test_recipe_module_path():
    target = "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction"
    repo_root = Path("/opt/Automodel")
    result = _recipe_module_path(target, repo_root)
    assert result == Path("/opt/Automodel/nemo_automodel/recipes/llm/train_ft.py")


def test_recipe_module_path_vlm():
    target = "nemo_automodel.recipes.vlm.finetune.FinetuneRecipeForVLM"
    repo_root = Path("/workspace")
    result = _recipe_module_path(target, repo_root)
    assert result == Path("/workspace/nemo_automodel/recipes/vlm/finetune.py")


def test_recipe_module_path_short_target():
    target = "pkg.mod.Cls"
    result = _recipe_module_path(target, Path("/root"))
    assert result == Path("/root/pkg/mod.py")


# ---------------------------------------------------------------------------
# _get_repo_root
# ---------------------------------------------------------------------------
def test_get_repo_root_editable_checkout(tmp_path, monkeypatch):
    (tmp_path / "nemo_automodel" / "components").mkdir(parents=True)
    (tmp_path / "examples").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PYTHONPATH", raising=False)
    result = _get_repo_root()
    assert result == tmp_path
    assert os.environ["PYTHONPATH"].startswith(str(tmp_path))


def test_get_repo_root_editable_checkout_appends_pythonpath(tmp_path, monkeypatch):
    (tmp_path / "nemo_automodel" / "components").mkdir(parents=True)
    (tmp_path / "examples").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", "/existing/path")
    result = _get_repo_root()
    assert result == tmp_path
    assert os.environ["PYTHONPATH"] == f"{tmp_path}:/existing/path"


def test_get_repo_root_not_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = _get_repo_root()
    expected = Path(__file__).parents[3] / "nemo_automodel" / "components" / "launcher"
    assert result == Path(
        nemo_automodel.components.launcher.interactive.__file__
    ).parents[3]


# ---------------------------------------------------------------------------
# InteractiveLauncher.launch – torch missing
# ---------------------------------------------------------------------------
def test_interactive_launcher_returns_1_when_torch_missing(monkeypatch, tmp_path):
    """Simulates a cli-only install where torch is not available."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("recipe:\n  _target_: some.Recipe\n")

    import nemo_automodel.components.launcher.interactive as mod

    original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

    def fake_import(name, *args, **kwargs):
        if name == "torch.distributed.run":
            raise ImportError("No module named 'torch'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    launcher = InteractiveLauncher()
    result = launcher.launch(
        config={"recipe": {"_target_": "some.Recipe"}},
        config_path=cfg_file,
        recipe_target="some.Recipe",
        launcher_config=None,
    )
    assert result == 1


# ---------------------------------------------------------------------------
# InteractiveLauncher.launch – single device path
# ---------------------------------------------------------------------------
def _make_torch_distributed_mock(world_size=1, run_return=0):
    """Build a mock ``torch.distributed.run`` module for patching the lazy import."""
    mock_module = mock.MagicMock()
    mock_module.determine_local_world_size.return_value = world_size
    mock_module.run.return_value = run_return
    mock_parser = mock.MagicMock()
    mock_args = SimpleNamespace(
        training_script="", training_script_args=[], nproc_per_node=0
    )
    mock_parser.parse_known_args.return_value = (mock_args, [])
    mock_module.get_args_parser.return_value = mock_parser
    return mock_module, mock_args


def test_interactive_launcher_single_device(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("recipe:\n  _target_: some.Recipe\n")

    mock_recipe_instance = mock.MagicMock()
    mock_recipe_instance.run_train_validation_loop.return_value = 0
    mock_recipe_cls = mock.MagicMock(return_value=mock_recipe_instance)
    mock_dist, _ = _make_torch_distributed_mock(world_size=1)

    with (
        mock.patch.dict("sys.modules", {"torch.distributed.run": mock_dist}),
        mock.patch(
            "nemo_automodel.components.launcher.interactive.resolve_recipe_cls",
            return_value=mock_recipe_cls,
        ),
        mock.patch(
            "nemo_automodel.components.launcher.interactive._get_repo_root",
            return_value=Path("/opt/Automodel"),
        ),
    ):
        launcher = InteractiveLauncher()
        rc = launcher.launch(
            config={"key": "val"},
            config_path=cfg_file,
            recipe_target="some.module.Recipe",
            launcher_config=None,
        )
    assert rc == 0
    mock_recipe_cls.assert_called_once_with({"key": "val"})
    mock_recipe_instance.setup.assert_called_once()
    mock_recipe_instance.run_train_validation_loop.assert_called_once()


def test_interactive_launcher_single_device_nproc_1(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("")

    mock_recipe_instance = mock.MagicMock()
    mock_recipe_instance.run_train_validation_loop.return_value = 0
    mock_recipe_cls = mock.MagicMock(return_value=mock_recipe_instance)
    mock_dist, _ = _make_torch_distributed_mock(world_size=8)

    with (
        mock.patch.dict("sys.modules", {"torch.distributed.run": mock_dist}),
        mock.patch(
            "nemo_automodel.components.launcher.interactive.resolve_recipe_cls",
            return_value=mock_recipe_cls,
        ),
        mock.patch(
            "nemo_automodel.components.launcher.interactive._get_repo_root",
            return_value=Path("/opt/Automodel"),
        ),
    ):
        launcher = InteractiveLauncher()
        rc = launcher.launch(
            config={"key": "val"},
            config_path=cfg_file,
            recipe_target="some.module.Recipe",
            launcher_config=1,
        )
    assert rc == 0
    mock_recipe_instance.setup.assert_called_once()


# ---------------------------------------------------------------------------
# InteractiveLauncher.launch – multi-device torchrun path
# ---------------------------------------------------------------------------
def test_interactive_launcher_multi_device(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("")

    mock_dist, mock_args = _make_torch_distributed_mock(world_size=4, run_return=0)

    with (
        mock.patch.dict("sys.modules", {"torch.distributed.run": mock_dist}),
        mock.patch(
            "nemo_automodel.components.launcher.interactive._get_repo_root",
            return_value=Path("/opt/Automodel"),
        ),
    ):
        launcher = InteractiveLauncher()
        rc = launcher.launch(
            config={"key": "val"},
            config_path=cfg_file,
            recipe_target="some.module.Recipe",
            launcher_config=None,
        )
    assert rc == 0
    mock_dist.run.assert_called_once()
    assert mock_args.nproc_per_node == 4
    assert mock_args.training_script_args == ["-c", str(cfg_file)]


def test_interactive_launcher_multi_device_with_extra_args(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("")

    mock_dist, mock_args = _make_torch_distributed_mock(world_size=4, run_return=0)

    with (
        mock.patch.dict("sys.modules", {"torch.distributed.run": mock_dist}),
        mock.patch(
            "nemo_automodel.components.launcher.interactive._get_repo_root",
            return_value=Path("/opt/Automodel"),
        ),
    ):
        launcher = InteractiveLauncher()
        rc = launcher.launch(
            config={"key": "val"},
            config_path=cfg_file,
            recipe_target="some.module.Recipe",
            launcher_config=2,
            extra_args=["--lr=0.001"],
        )
    assert rc == 0
    assert "--lr=0.001" in mock_args.training_script_args
    assert mock_args.nproc_per_node == 2


# ---------------------------------------------------------------------------
# InteractiveLauncher.launch – torchrun worker detection (in-process path)
# ---------------------------------------------------------------------------
def test_interactive_launcher_torchrun_worker_runs_in_process(tmp_path, monkeypatch):
    """When LOCAL_RANK is set (i.e. already inside torchrun), run in-process."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("")
    monkeypatch.setenv("LOCAL_RANK", "3")

    mock_recipe_instance = mock.MagicMock()
    mock_recipe_instance.run_train_validation_loop.return_value = 0
    mock_recipe_cls = mock.MagicMock(return_value=mock_recipe_instance)
    mock_dist, _ = _make_torch_distributed_mock(world_size=8)

    with (
        mock.patch.dict("sys.modules", {"torch.distributed.run": mock_dist}),
        mock.patch(
            "nemo_automodel.components.launcher.interactive.resolve_recipe_cls",
            return_value=mock_recipe_cls,
        ),
    ):
        launcher = InteractiveLauncher()
        rc = launcher.launch(
            config={"key": "val"},
            config_path=cfg_file,
            recipe_target="some.module.Recipe",
            launcher_config=None,
        )
    assert rc == 0
    mock_recipe_cls.assert_called_once_with({"key": "val"})
    mock_recipe_instance.setup.assert_called_once()
    mock_recipe_instance.run_train_validation_loop.assert_called_once()
    mock_dist.run.assert_not_called()


def test_interactive_launcher_torchrun_worker_ignores_nproc(tmp_path, monkeypatch):
    """nproc_per_node should be ignored when already inside torchrun."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("")
    monkeypatch.setenv("LOCAL_RANK", "0")

    mock_recipe_instance = mock.MagicMock()
    mock_recipe_instance.run_train_validation_loop.return_value = 0
    mock_recipe_cls = mock.MagicMock(return_value=mock_recipe_instance)
    mock_dist, _ = _make_torch_distributed_mock(world_size=8)

    with (
        mock.patch.dict("sys.modules", {"torch.distributed.run": mock_dist}),
        mock.patch(
            "nemo_automodel.components.launcher.interactive.resolve_recipe_cls",
            return_value=mock_recipe_cls,
        ),
    ):
        launcher = InteractiveLauncher()
        rc = launcher.launch(
            config={"key": "val"},
            config_path=cfg_file,
            recipe_target="some.module.Recipe",
            launcher_config=4,
        )
    assert rc == 0
    mock_recipe_instance.setup.assert_called_once()
    mock_dist.run.assert_not_called()


def test_is_torchrun_worker_false(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert InteractiveLauncher._is_torchrun_worker() is False


def test_is_torchrun_worker_true(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert InteractiveLauncher._is_torchrun_worker() is True


import nemo_automodel.components.launcher.interactive

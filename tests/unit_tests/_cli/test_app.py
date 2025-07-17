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

# Patch the script module
import sys, importlib
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml
import os

sys.modules["nemo_run"] = mock.MagicMock()

# Simulate torch distributed imports
sys.modules["torch.distributed.run"] = mock.MagicMock()

# Import the script under test
import argparse

import nemo_automodel._cli.app as module


@pytest.fixture
def tmp_yaml_file():
    config_data = {"dummy_key": "dummy_value"}
    tmp_dir = tempfile.mkdtemp()
    tmp_file = Path(tmp_dir) / "config.yaml"
    with open(tmp_file, "w") as f:
        yaml.dump(config_data, f)
    yield tmp_file
    shutil.rmtree(tmp_dir)


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


def test_load_function_success(tmp_path):
    # Create a mock Python file with a function
    file_path = tmp_path / "mock_module.py"
    file_path.write_text("def mock_func(): return 'success'")
    func = module.load_function(file_path, "mock_func")
    assert callable(func)
    assert func() == "success"


def test_load_function_missing_file():
    with pytest.raises(FileNotFoundError):
        module.load_function("missing.py", "func")


def test_load_function_missing_func(tmp_path):
    file_path = tmp_path / "mock_module2.py"
    file_path.write_text("x = 5")
    with pytest.raises(ImportError):
        module.load_function(file_path, "missing_func")


def test_build_parser(monkeypatch):
    recipes_path = Path(__file__).parents[3]  # / "recipes"
    # Simulate recipes path structure
    monkeypatch.setattr(module.Path, "parents", [None, None, recipes_path])
    parser = module.build_parser()
    assert parser is not None


def test_launch_with_slurm(monkeypatch):
    mock_script = "some_script.py"
    mock_config = "some_config.yaml"
    mock_slurm_config = {"nodes": 1}

    fake_executor = mock.MagicMock()
    fake_exp = mock.MagicMock()
    dummy_args = SimpleNamespace(
        domain="llm",
        command="finetune",
    )
    monkeypatch.setattr(module, "load_yaml", lambda x: {"slurm": mock_slurm_config})
    monkeypatch.setitem(
        sys.modules,
        "nemo_run",
        mock.MagicMock(
            SlurmExecutor=lambda **kwargs: fake_executor,
            LocalTunnel=lambda: "tunnel",
            Experiment=mock.MagicMock(return_value=fake_exp),
            Script=mock.MagicMock(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "nemo_run.config",
        mock.MagicMock(
            job_dir='',
        )
    )
    # fake implementation
    def fake_submit_slurm_job(cfg, job_dir):
        parts = cfg.command.split()
        assert len(parts) == 5
        assert parts[0].startswith("PYTHONPATH=")
        assert parts[0].endswith(":$PYTHONPATH")
        assert parts[1] == "python3"
        assert parts[2].endswith("recipes/llm/finetune.py")
        assert parts[3] == "-c"
        assert parts[4] == "/tmp/a/0123456789/y.conf"


        # whatever you want to check
        return "FAKE_JOB_ID"

    import nemo_automodel.components.launcher.slurm.utils as slurm_utils
    monkeypatch.setattr(slurm_utils, "submit_slurm_job", fake_submit_slurm_job)
    job_dir = '/tmp/a/0123456789/'
    module.launch_with_slurm(dummy_args, job_dir +'y.conf', job_dir, slurm_config={})

    # maybe separate test?
    with pytest.raises(AssertionError, match='Expected last dir to be unix timestamp'):
        job_dir = '/tmp/a/123456789/'
        module.launch_with_slurm(dummy_args, job_dir +'y.conf', job_dir, slurm_config={})

def test_main_single_node(monkeypatch, tmp_yaml_file):
    config_path = tmp_yaml_file

    monkeypatch.setattr("sys.argv", ["prog", "finetune", "llm", "-c", str(config_path)])
    monkeypatch.setattr(module, "load_yaml", lambda x: {})
    import torch.distributed.run as thrun
    monkeypatch.setattr(thrun, "determine_local_world_size", lambda **kwargs: 1)

    def dummy_main(config_path_arg):
        assert config_path_arg == config_path
        return 0

    monkeypatch.setattr(module, "load_function", lambda f, n: dummy_main)
    monkeypatch.setattr(module.Path, "parents", [None, None, Path(__file__).parent])

    result = module.main()
    assert result == 0


def test_main_multi_node(monkeypatch, tmp_yaml_file):
    config_path = tmp_yaml_file

    # Simulate CLI args using sys.argv
    monkeypatch.setattr("sys.argv", ["prog", "finetune", "llm", "-c", str(config_path)])

    monkeypatch.setattr(module, "load_yaml", lambda x: {})
    run_mod = importlib.import_module("torch.distributed.run")
    monkeypatch.setattr(run_mod, "run", lambda *a, **kw: 0)
    import torch.distributed.run as trn
    monkeypatch.setattr(trn, "get_args_parser", lambda: argparse.Namespace(parse_args=lambda: DummyArgs()))
    monkeypatch.setattr(trn, "determine_local_world_size", lambda **kwargs: 4)

    # Simulate torchrun parser and arguments
    class DummyArgs:
        def __init__(self):
            self.training_script = None
            self.training_script_args = ["finetune", "--config", str(config_path)]
            self.nproc_per_node = None

    # Dummy load_function and Path.parents
    monkeypatch.setattr(module, "load_function", lambda f, n: None)
    monkeypatch.setattr(module.Path, "parents", [None, None, Path(__file__).parent])

    result = module.main()
    assert result == 0


def test_main_k8s_not_implemented(monkeypatch, tmp_yaml_file):
    config_path = tmp_yaml_file

    # Get the original parser builder
    original_build_parser = module.build_parser

    def custom_parser():
        parser = original_build_parser()
        parser.set_defaults(config=str(config_path), domain="llm", command="finetune")
        return parser

    monkeypatch.setattr("sys.argv", ["automodel", "finetune", "llm", "-c", str(config_path)])
    monkeypatch.setattr(module, "build_parser", custom_parser)
    monkeypatch.setattr(module, "load_yaml", lambda x: {"k8s": {}})
    monkeypatch.setattr(module.Path, "parents", [None, None, Path(__file__).parent])

    with pytest.raises(NotImplementedError):
        module.main()


def argparse_mock(args_list):
    parser = module.build_parser()
    return parser.parse_args(args_list)


def argparse_mock_parser():
    class DummyParser:
        def parse_args(self):
            return SimpleNamespace(
                training_script="dummy.py",
                training_script_args=["finetune", "--config", "dummy.yaml"],
                nproc_per_node=None,
            )

    return DummyParser()

def test_repo_root_when_found(monkeypatch):
    dummy_root = Path("/tmp/dummy_repo").resolve()

    # Pretend an initial PYTHONPATH
    monkeypatch.setenv("PYTHONPATH", "foo:bar")

    # Force the helper to return our dummy path
    monkeypatch.setattr(
        module, "get_automodel_repo_root", lambda: dummy_root, raising=True
    )

    # Act
    result = module.get_repo_root()

    # Assert return value
    assert result == dummy_root

    # Assert PYTHONPATH was *prepended* with dummy_root and old entries kept
    assert os.environ["PYTHONPATH"].split(":") == [str(dummy_root), "foo", "bar"]


def test_repo_root_when_not_found(monkeypatch):
    # Start from a known PYTHONPATH
    monkeypatch.setenv("PYTHONPATH", "foo:bar")

    # Make helper return None
    monkeypatch.setattr(module, "get_automodel_repo_root", lambda: None, raising=True)

    # Act
    result = module.get_repo_root()

    # Expected path = two parents up from the module's file
    expected = Path(module.__file__).parents[2]
    assert result == expected

    # PYTHONPATH must remain unchanged
    assert os.environ["PYTHONPATH"] == "foo:bar"


def test_repo_structure():
    """
    inside the nemo_automodel/_cli/app.py we assume a specific directory structure.
    This test ensures the directory structure is preserved.
    """
    cwd = Path.cwd()
    with pytest.raises(AssertionError):
        assert (cwd / "nemo_automodel_abc").exists()
    assert (cwd / "nemo_automodel").exists()
    assert (cwd / "nemo_automodel" / "components").exists()
    assert (cwd / "nemo_automodel" / "_cli").exists()
    assert (cwd / "nemo_automodel" / "recipes").exists()
    assert (cwd / "nemo_automodel" / "shared").exists()
    assert (cwd / "docs").exists()
    assert (cwd / "examples").exists()

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
import pytest
import tempfile
import shutil
import yaml
from pathlib import Path
from unittest import mock
import builtins

from types import SimpleNamespace
from importlib.util import spec_from_file_location, module_from_spec

# Patch the script module
import sys
sys.modules["nemo_run"] = mock.MagicMock()

# Simulate torch distributed imports
sys.modules["torch.distributed.run"] = mock.MagicMock()
from torch.distributed.run import determine_local_world_size, run as thrun

# Import the script under test
import importlib.util
import argparse
import nemo_automodel._cli.app as module

@pytest.fixture
def tmp_yaml_file():
    config_data = {
        "dummy_key": "dummy_value"
    }
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
    recipes_path = Path(__file__).parents[3] # / "recipes"
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

    monkeypatch.setattr(module, "load_yaml", lambda x: {"slurm": mock_slurm_config})
    monkeypatch.setitem(sys.modules, "nemo_run", mock.MagicMock(
        SlurmExecutor=lambda **kwargs: fake_executor,
        LocalTunnel=lambda: "tunnel",
        Experiment=mock.MagicMock(return_value=fake_exp),
        Script=mock.MagicMock()
    ))

    module.launch_with_slurm(mock_slurm_config, mock_script, mock_config)


def test_main_single_node(monkeypatch, tmp_yaml_file):
    config_path = tmp_yaml_file

    monkeypatch.setattr("sys.argv", ["prog", "llm", "finetune", "-c", str(config_path)])
    monkeypatch.setattr(module, "load_yaml", lambda x: {})
    monkeypatch.setattr(module, "determine_local_world_size", lambda **kwargs: 1)

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
    monkeypatch.setattr("sys.argv", ["prog", "llm", "finetune", "-c", str(config_path)])

    monkeypatch.setattr(module, "load_yaml", lambda x: {})
    monkeypatch.setattr(module, "determine_local_world_size", lambda **kwargs: 4)

    # Simulate torchrun parser and arguments
    class DummyArgs:
        def __init__(self):
            self.training_script = None
            self.training_script_args = ["finetune", "--config", str(config_path)]
            self.nproc_per_node = None

    monkeypatch.setattr(module, "get_args_parser", lambda: argparse.Namespace(parse_args=lambda: DummyArgs()))
    monkeypatch.setattr(module, "thrun", lambda args: 0)

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
        parser.set_defaults(
            config=str(config_path),
            domain="llm",
            command="finetune"
        )
        return parser

    monkeypatch.setattr("sys.argv", ["automodel", "llm", "finetune", "-c", str(config_path)])
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
                nproc_per_node=None
            )
    return DummyParser()


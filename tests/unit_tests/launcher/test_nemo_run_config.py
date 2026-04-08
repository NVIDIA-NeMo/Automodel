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

import pytest

from nemo_automodel.components.launcher.nemo_run.config import (
    DEFAULT_EXECUTORS_FILE,
    NemoRunConfig,
)


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


def test_defaults():
    cfg = NemoRunConfig()
    assert cfg.executor == "local"
    assert cfg.job_name == ""
    assert cfg.detach is True
    assert cfg.tail_logs is False
    assert cfg.job_dir == ""
    assert cfg.overrides == {}


def test_executors_file_defaults_to_nemorun_home(monkeypatch):
    monkeypatch.delenv("NEMORUN_HOME", raising=False)
    import importlib
    import nemo_automodel.components.launcher.nemo_run.config as cfg_mod
    importlib.reload(cfg_mod)

    cfg = cfg_mod.NemoRunConfig()
    expected = os.path.join(os.path.expanduser("~"), ".nemo_run", "executors.py")
    assert cfg.executors_file == expected


def test_executors_file_respects_nemorun_home_env(monkeypatch, tmp_path):
    monkeypatch.setenv("NEMORUN_HOME", str(tmp_path))
    import importlib
    import nemo_automodel.components.launcher.nemo_run.config as cfg_mod
    importlib.reload(cfg_mod)

    cfg = cfg_mod.NemoRunConfig()
    assert cfg.executors_file == os.path.join(str(tmp_path), "executors.py")


# ---------------------------------------------------------------------------
# from_dict — splits launcher keys from executor overrides
# ---------------------------------------------------------------------------


def test_from_dict_launcher_keys_only():
    cfg = NemoRunConfig.from_dict({
        "executor": "my_slurm",
        "job_name": "test_job",
        "detach": False,
    })
    assert cfg.executor == "my_slurm"
    assert cfg.job_name == "test_job"
    assert cfg.detach is False
    assert cfg.overrides == {}


def test_from_dict_executor_overrides():
    cfg = NemoRunConfig.from_dict({
        "executor": "my_slurm",
        "nodes": 2,
        "ntasks_per_node": 8,
        "partition": "batch",
        "container_image": "nvcr.io/nvidia/nemo-automodel:26.02",
        "time": "04:00:00",
    })
    assert cfg.executor == "my_slurm"
    assert cfg.overrides == {
        "nodes": 2,
        "ntasks_per_node": 8,
        "partition": "batch",
        "container_image": "nvcr.io/nvidia/nemo-automodel:26.02",
        "time": "04:00:00",
    }


def test_from_dict_mixed_keys():
    cfg = NemoRunConfig.from_dict({
        "executor": "my_slurm",
        "job_name": "my_job",
        "detach": True,
        "nodes": 4,
        "container_image": "img.sqsh",
        "env_vars": {"MY_VAR": "val"},
    })
    assert cfg.executor == "my_slurm"
    assert cfg.job_name == "my_job"
    assert cfg.overrides == {
        "nodes": 4,
        "container_image": "img.sqsh",
        "env_vars": {"MY_VAR": "val"},
    }


def test_from_dict_explicit_overrides_merged():
    cfg = NemoRunConfig.from_dict({
        "executor": "my_slurm",
        "overrides": {"mem": "64G"},
        "nodes": 2,
    })
    assert cfg.overrides == {"mem": "64G", "nodes": 2}


def test_from_dict_empty():
    cfg = NemoRunConfig.from_dict({})
    assert cfg.executor == "local"
    assert cfg.overrides == {}


# ---------------------------------------------------------------------------
# overrides default not shared between instances
# ---------------------------------------------------------------------------


def test_overrides_default_not_shared():
    a = NemoRunConfig()
    b = NemoRunConfig()
    a.overrides["x"] = 1
    assert "x" not in b.overrides

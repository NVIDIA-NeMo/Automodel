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
import yaml

from nemo_automodel.components.launcher.nemo_run.config import (
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


# ---------------------------------------------------------------------------
# from_dict — validates the YAML section at the launcher boundary
# ---------------------------------------------------------------------------


def test_from_dict_parsed_yaml_section():
    section = yaml.safe_load(
        "executor: my_slurm\n"
        "job_name: sft\n"
        "detach: false\n"
        "tail_logs: true\n"
        "job_dir: /tmp/jobs\n"
        "executors_file: /tmp/executors.py\n"
        "nodes: 2\n"
        "env_vars:\n"
        "  A: b\n"
        "overrides:\n"
        "  mem: 64G\n"
    )
    cfg = NemoRunConfig.from_dict(section)
    assert cfg == NemoRunConfig(
        executor="my_slurm",
        job_name="sft",
        detach=False,
        tail_logs=True,
        executors_file="/tmp/executors.py",
        job_dir="/tmp/jobs",
        overrides={"mem": "64G", "nodes": 2, "env_vars": {"A": "b"}},
    )


def test_from_dict_null_launcher_keys_take_defaults():
    section = yaml.safe_load("executor:\njob_name:\njob_dir: null\nexecutors_file:\ntail_logs:\noverrides:\nnodes: 2\n")
    cfg = NemoRunConfig.from_dict(section)
    default = NemoRunConfig()
    assert cfg.executor == "local"
    assert cfg.job_name == ""
    assert cfg.job_dir == ""
    assert cfg.executors_file == default.executors_file
    assert cfg.tail_logs is False
    assert cfg.overrides == {"nodes": 2}


def test_from_dict_null_detach_keeps_meaning_false():
    assert NemoRunConfig.from_dict({"detach": None}).detach is False
    assert NemoRunConfig.from_dict({}).detach is True


def test_from_dict_local_executor_with_null_executors_file():
    cfg = NemoRunConfig.from_dict(yaml.safe_load("executor: local\nexecutors_file:\n"))
    assert cfg.executor == "local"
    assert cfg.executors_file == NemoRunConfig().executors_file


def test_from_dict_top_level_key_wins_over_explicit_override():
    cfg = NemoRunConfig.from_dict({"overrides": {"nodes": 1, "mem": "64G"}, "nodes": 2})
    assert cfg.overrides == {"nodes": 2, "mem": "64G"}


def test_from_dict_does_not_mutate_input():
    explicit = {"mem": "64G"}
    section = {"overrides": explicit, "nodes": 2}
    NemoRunConfig.from_dict(section)
    assert explicit == {"mem": "64G"}
    assert section == {"overrides": {"mem": "64G"}, "nodes": 2}


@pytest.mark.parametrize(
    ("section", "message"),
    [
        ({"executor": 3}, "nemo_run.executor must be a string, got int"),
        ({"executor": False}, "nemo_run.executor must be a string, got bool"),
        ({"executors_file": 1}, "nemo_run.executors_file must be a string, got int"),
        ({"job_name": 7}, "nemo_run.job_name must be a string, got int"),
        ({"job_name": 0}, "nemo_run.job_name must be a string, got int"),
        ({"job_name": False}, "nemo_run.job_name must be a string, got bool"),
        ({"job_name": {}}, "nemo_run.job_name must be a string, got dict"),
        ({"job_dir": []}, "nemo_run.job_dir must be a string, got list"),
        ({"job_dir": ["a"]}, "nemo_run.job_dir must be a string, got list"),
        ({"detach": "false"}, "nemo_run.detach must be true or false, got str"),
        ({"detach": 0}, "nemo_run.detach must be true or false, got int"),
        ({"tail_logs": "yes"}, "nemo_run.tail_logs must be true or false, got str"),
        ({"tail_logs": 1}, "nemo_run.tail_logs must be true or false, got int"),
        ({"overrides": ["mem=64G"]}, "nemo_run.overrides must be a mapping, got list"),
        ({"overrides": "mem=64G"}, "nemo_run.overrides must be a mapping, got str"),
        ({"overrides": {1: "x"}}, "nemo_run executor override keys must be strings, got int"),
        ({1: "x"}, "nemo_run executor override keys must be strings, got int"),
    ],
)
def test_from_dict_rejects_wrong_types(section, message):
    with pytest.raises(TypeError, match=message):
        NemoRunConfig.from_dict(section)

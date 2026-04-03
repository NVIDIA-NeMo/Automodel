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
    assert cfg.nodes is None
    assert cfg.devices is None
    assert cfg.container_image is None
    assert cfg.time is None
    assert cfg.mounts == []
    assert cfg.env_vars == {}
    assert cfg.job_name == ""
    assert cfg.detach is True
    assert cfg.tail_logs is False
    assert cfg.job_dir == ""


def test_executors_file_defaults_to_nemorun_home(monkeypatch):
    monkeypatch.delenv("NEMORUN_HOME", raising=False)
    # Re-import to pick up the env var change
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
# Custom values
# ---------------------------------------------------------------------------


def test_custom_values():
    cfg = NemoRunConfig(
        executor="my_slurm_cluster",
        nodes=4,
        devices=8,
        container_image="nvcr.io/nvidia/nemo:24.05",
        time="04:00:00",
        mounts=["/data:/data", "/models:/models"],
        env_vars={"MY_VAR": "value1", "OTHER_VAR": "value2"},  # pragma: allowlist secret
        job_name="pretrain_llama",
        detach=False,
        tail_logs=True,
        executors_file="/custom/path/executors.py",
        job_dir="/scratch/jobs",
    )
    assert cfg.executor == "my_slurm_cluster"
    assert cfg.nodes == 4
    assert cfg.devices == 8
    assert cfg.container_image == "nvcr.io/nvidia/nemo:24.05"
    assert cfg.time == "04:00:00"
    assert cfg.mounts == ["/data:/data", "/models:/models"]
    assert cfg.env_vars == {"MY_VAR": "value1", "OTHER_VAR": "value2"}
    assert cfg.job_name == "pretrain_llama"
    assert cfg.detach is False
    assert cfg.tail_logs is True
    assert cfg.executors_file == "/custom/path/executors.py"
    assert cfg.job_dir == "/scratch/jobs"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_nodes_zero_raises():
    with pytest.raises(ValueError, match="'nodes' must be >= 1"):
        NemoRunConfig(nodes=0)


def test_nodes_negative_raises():
    with pytest.raises(ValueError, match="'nodes' must be >= 1"):
        NemoRunConfig(nodes=-1)


def test_devices_zero_raises():
    with pytest.raises(ValueError, match="'devices' must be >= 1"):
        NemoRunConfig(devices=0)


def test_devices_negative_raises():
    with pytest.raises(ValueError, match="'devices' must be >= 1"):
        NemoRunConfig(devices=-5)


def test_nodes_one_is_valid():
    cfg = NemoRunConfig(nodes=1)
    assert cfg.nodes == 1


def test_devices_one_is_valid():
    cfg = NemoRunConfig(devices=1)
    assert cfg.devices == 1


def test_nodes_none_skips_validation():
    cfg = NemoRunConfig(nodes=None)
    assert cfg.nodes is None


def test_devices_none_skips_validation():
    cfg = NemoRunConfig(devices=None)
    assert cfg.devices is None


# ---------------------------------------------------------------------------
# Mutable default isolation
# ---------------------------------------------------------------------------


def test_mounts_default_not_shared():
    cfg1 = NemoRunConfig()
    cfg2 = NemoRunConfig()
    cfg1.mounts.append("/tmp:/tmp")
    assert cfg2.mounts == []


def test_env_vars_default_not_shared():
    cfg1 = NemoRunConfig()
    cfg2 = NemoRunConfig()
    cfg1.env_vars["KEY"] = "val"
    assert cfg2.env_vars == {}

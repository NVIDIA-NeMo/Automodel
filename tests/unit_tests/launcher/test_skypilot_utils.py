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
import yaml

# Stub out the sky module before importing utils so sky is never required.
_fake_resources = mock.MagicMock()
_fake_task_instance = mock.MagicMock()
_fake_task_class = mock.MagicMock(return_value=_fake_task_instance)

_sky_stub = mock.MagicMock()
_sky_stub.Task = _fake_task_class
_sky_stub.Resources = mock.MagicMock(return_value=_fake_resources)
_sky_stub.GCP = mock.MagicMock(return_value="gcp_obj")
_sky_stub.AWS = mock.MagicMock(return_value="aws_obj")
_sky_stub.Azure = mock.MagicMock(return_value="azure_obj")
_sky_stub.Lambda = mock.MagicMock(return_value="lambda_obj")
_sky_stub.Kubernetes = mock.MagicMock(return_value="k8s_obj")
_sky_stub.launch = mock.MagicMock(return_value=(1, mock.MagicMock()))

sys.modules["sky"] = _sky_stub

from nemo_automodel.components.launcher.skypilot.config import SkyPilotConfig
from nemo_automodel.components.launcher.skypilot.utils import REMOTE_CONFIG_PATH, submit_skypilot_job


@pytest.fixture()
def job_dir(tmp_path):
    d = tmp_path / "skypilot_jobs" / "1234567890"
    d.mkdir(parents=True)
    (d / "job_config.yaml").write_text("model: gpt2\n")
    return str(d)


@pytest.fixture(autouse=True)
def reset_sky_mocks():
    _sky_stub.launch.reset_mock()
    _fake_task_class.reset_mock()
    _fake_task_instance.reset_mock()
    _sky_stub.Resources.reset_mock()
    yield


def _make_config(**kwargs) -> SkyPilotConfig:
    defaults = dict(cloud="gcp", accelerators="T4:1", command="torchrun train.py")
    defaults.update(kwargs)
    return SkyPilotConfig(**defaults)


def test_submit_calls_sky_launch(job_dir):
    cfg = _make_config()
    ret = submit_skypilot_job(cfg, job_dir)
    assert ret == 0
    _sky_stub.launch.assert_called_once()


def test_task_receives_correct_command(job_dir):
    cfg = _make_config(command="torchrun --nproc_per_node=4 train.py -c cfg.yaml")
    submit_skypilot_job(cfg, job_dir)
    _, kwargs = _fake_task_class.call_args
    assert kwargs.get("run") == cfg.command or _fake_task_class.call_args[0]


def test_task_num_nodes_forwarded(job_dir):
    cfg = _make_config(num_nodes=4)
    submit_skypilot_job(cfg, job_dir)
    call_kwargs = _fake_task_class.call_args[1]
    assert call_kwargs.get("num_nodes") == 4


def test_resources_use_spot(job_dir):
    cfg = _make_config(use_spot=True)
    submit_skypilot_job(cfg, job_dir)
    call_kwargs = _sky_stub.Resources.call_args[1]
    assert call_kwargs["use_spot"] is True


def test_resources_disk_size(job_dir):
    cfg = _make_config(disk_size=256)
    submit_skypilot_job(cfg, job_dir)
    call_kwargs = _sky_stub.Resources.call_args[1]
    assert call_kwargs["disk_size"] == 256


def test_env_vars_forwarded(job_dir):
    cfg = _make_config(hf_token="tok123", wandb_key="wb456", env_vars={"EXTRA": "val"})
    submit_skypilot_job(cfg, job_dir)
    call_kwargs = _fake_task_class.call_args[1]
    envs = call_kwargs["envs"]
    assert envs["HF_TOKEN"] == "tok123"
    assert envs["WANDB_API_KEY"] == "wb456"
    assert envs["EXTRA"] == "val"


def test_file_mounts_config_uploaded(job_dir):
    cfg = _make_config()
    submit_skypilot_job(cfg, job_dir)
    _fake_task_instance.set_file_mounts.assert_called_once()
    mounts_arg = _fake_task_instance.set_file_mounts.call_args[0][0]
    assert REMOTE_CONFIG_PATH in mounts_arg


def test_cluster_name_uses_job_name(job_dir):
    cfg = _make_config(job_name="my_experiment")
    submit_skypilot_job(cfg, job_dir)
    launch_kwargs = _sky_stub.launch.call_args[1]
    assert launch_kwargs["cluster_name"] == "my_experiment"


def test_launch_detach_run(job_dir):
    cfg = _make_config()
    submit_skypilot_job(cfg, job_dir)
    launch_kwargs = _sky_stub.launch.call_args[1]
    assert launch_kwargs["detach_run"] is True


@pytest.mark.parametrize("cloud,expected_attr", [
    ("gcp", "GCP"),
    ("aws", "AWS"),
    ("azure", "Azure"),
    ("lambda", "Lambda"),
    ("kubernetes", "Kubernetes"),
])
def test_correct_cloud_class_used(job_dir, cloud, expected_attr):
    cfg = _make_config(cloud=cloud)
    submit_skypilot_job(cfg, job_dir)
    getattr(_sky_stub, expected_attr).assert_called()


def test_missing_sky_raises_import_error(job_dir, monkeypatch):
    monkeypatch.setitem(sys.modules, "sky", None)
    cfg = _make_config()
    with pytest.raises(ImportError, match="SkyPilot is not installed"):
        submit_skypilot_job(cfg, job_dir)
    # Restore
    monkeypatch.setitem(sys.modules, "sky", _sky_stub)

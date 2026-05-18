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

from nemo_automodel.components.launcher.skypilot.config import SUPPORTED_CLOUDS, SkyPilotConfig


@pytest.mark.parametrize("cloud", SUPPORTED_CLOUDS)
def test_all_supported_clouds_accepted(cloud):
    cfg = SkyPilotConfig(cloud=cloud)
    assert cfg.cloud == cloud


def test_unsupported_cloud_raises():
    with pytest.raises(ValueError, match="'cloud' must be one of"):
        SkyPilotConfig(cloud="oracle")


def test_defaults():
    cfg = SkyPilotConfig(cloud="gcp")
    assert cfg.accelerators == "T4:1"
    assert cfg.num_nodes == 1
    assert cfg.use_spot is True
    assert cfg.disk_size == 100
    assert cfg.instance_type is None
    assert cfg.region is None
    assert cfg.zone is None
    assert cfg.command == ""
    assert cfg.env_vars == {}


def test_invalid_num_nodes():
    with pytest.raises(ValueError, match="num_nodes"):
        SkyPilotConfig(cloud="gcp", num_nodes=0)


def test_invalid_disk_size():
    with pytest.raises(ValueError, match="disk_size"):
        SkyPilotConfig(cloud="aws", disk_size=0)


def test_custom_fields():
    cfg = SkyPilotConfig(
        cloud="aws",
        accelerators="A100:8",
        num_nodes=2,
        use_spot=False,
        disk_size=200,
        region="us-east-1",
        zone="us-east-1a",
        instance_type="p4d.24xlarge",
        job_name="my_job",
        env_vars={"MY_VAR": "value"},
    )
    assert cfg.accelerators == "A100:8"
    assert cfg.num_nodes == 2
    assert cfg.use_spot is False
    assert cfg.disk_size == 200
    assert cfg.region == "us-east-1"
    assert cfg.zone == "us-east-1a"
    assert cfg.instance_type == "p4d.24xlarge"
    assert cfg.job_name == "my_job"
    assert cfg.env_vars == {"MY_VAR": "value"}


def test_credentials_default_to_empty_when_env_unset(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    cfg = SkyPilotConfig(cloud="gcp")
    assert cfg.hf_token == ""
    assert cfg.wandb_key == ""


def test_credentials_read_from_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_abc123")
    monkeypatch.setenv("WANDB_API_KEY", "wandb_xyz")
    cfg = SkyPilotConfig(cloud="gcp")
    assert cfg.hf_token == "hf_abc123"
    assert cfg.wandb_key == "wandb_xyz"

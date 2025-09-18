# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
from pathlib import Path

import pytest

from nemo_automodel.components.launcher.slurm.config import SlurmConfig, VolumeMapping


@pytest.fixture()
def tmp_dir(tmp_path):
    """Return an existing absolute directory path on host."""
    host_dir = tmp_path / "host_mount"
    host_dir.mkdir()
    return host_dir


# VolumeMapping tests
def test_volume_mapping_to_str(tmp_dir):
    vm = VolumeMapping(source=tmp_dir, dest=Path("/container"))
    assert vm.to_str() == f"{tmp_dir}:/container"


def test_volume_mapping_requires_existing_source(tmp_path):
    missing_src = tmp_path / "does_not_exist"
    with pytest.raises(ValueError):
        VolumeMapping(source=missing_src, dest=Path("/container"))

def test_volume_mapping_requires_absolute_paths(tmp_dir):
    # relative source
    with pytest.raises(ValueError):
        VolumeMapping(source=Path("relative"), dest=Path("/container"))
    # relative dest
    with pytest.raises(ValueError):
        VolumeMapping(source=tmp_dir, dest=Path("relative"))

def test_slurm_config_extra_mounts_conversion_from_str(tmp_dir):
    mount_str = f"{tmp_dir}:/container"
    cfg = SlurmConfig(job_name="job", extra_mounts=[mount_str])
    assert isinstance(cfg.extra_mounts[0], VolumeMapping)
    assert cfg.extra_mounts[0].to_str() == mount_str


def test_slurm_config_extra_mounts_existing_volume_mapping(tmp_dir):
    vm = VolumeMapping(source=tmp_dir, dest=Path("/container"))
    cfg = SlurmConfig(job_name="job", extra_mounts=[vm])
    assert cfg.extra_mounts[0] is vm


def test_slurm_config_extra_mounts_invalid_type(tmp_dir):
    with pytest.raises(ValueError):
        SlurmConfig(job_name="job", extra_mounts=[123])

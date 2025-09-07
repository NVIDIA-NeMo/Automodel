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

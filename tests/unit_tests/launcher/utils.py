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
from types import SimpleNamespace
from unittest import mock
import pytest
import dataclasses
import types


@pytest.fixture()
def mod():
    """
    Import the target module only once and return it.  This also lets us
    monkey-patch internals (render_script, subprocess) in a single place.
    """
    import importlib
    module = importlib.import_module("nemo_automodel.components.launcher.slurm.utils")
    return module


@pytest.fixture()
def dummy_volume_mapping():
    """Replacement for nemo_automodel.components.launcher.slurm.config.VolumeMapping"""
    return SimpleNamespace


@pytest.fixture()
def tmp_job_dir(tmp_path: Path):
    return tmp_path / "job_dir"



@pytest.mark.parametrize(
    "inp, expected",
    [
        ({"source": "/a", "dest": "/b"}, "/a:/b"),
        ("/a", "/a:/a"),
        ("/a:/b", "/a:/b"),
    ],
)
def test_volume_map_to_str_happy_paths(mod, dummy_volume_mapping, inp, expected):
    vm_cls = dummy_volume_mapping  # simple object with .source / .dest attrs
    if isinstance(inp, dict):
        # already handled
        pass
    elif isinstance(inp, str) and ":" not in inp:
        pass
    elif isinstance(inp, str) and ":" in inp:
        pass
    else:
        from nemo_automodel.components.launcher.slurm.config import VolumeMapping
        # make VolumeMapping instance
        inp = VolumeMapping(source="/a", dest="/b")

    assert mod.volume_map_to_str(inp) == expected


def test_volume_map_to_str_errors(mod):
    with pytest.raises(AssertionError):
        mod.volume_map_to_str("/missing_dest:")
    with pytest.raises(AssertionError):
        mod.volume_map_to_str(":/missing_src")
    with pytest.raises(ValueError):
        mod.volume_map_to_str("a:b:c")
    with pytest.raises(ValueError):
        mod.volume_map_to_str(123)  # unsupported type

def test_make_container_mounts_all_fields(monkeypatch, mod, tmp_path):
    from nemo_automodel.components.launcher.slurm.config import VolumeMapping
    from pathlib import Path
    src_dir = tmp_path / "e"          # create a real directory
    src_dir.mkdir()
    opts = {
        "hf_home": "/hf/cache",
        "nemo_mount": "/a:/b",
        "extra_mounts": [
            {"source": "/c", "dest": "/d"},
            VolumeMapping(source=src_dir, dest=Path("/f")),  # ✅ source exists
            "/g",
        ],
    }

    mounts = mod.make_container_mounts(opts)

    assert mounts == [
        "/hf/cache:/hf/cache",
        "/a:/b",
        "/c:/d",
        f"{src_dir}:/f",               # volume created above
        "/g:/g",
    ]
    # make_container_mounts should mutate opts by popping nemo_mount & extra_mounts
    assert "nemo_mount" not in opts
    assert "extra_mounts" not in opts


def test_make_container_mounts_hf_home_excluded(mod):
    opts = {"hf_home": "~/cache"}  # does NOT get mounted
    assert mod.make_container_mounts(opts) == []


def _fake_slurm_config():
    """
    Very small stand-in for the real SlurmConfig dataclass.
    submit_slurm_job only needs:
      - job_name
      - anything else that dataclasses.asdict should return (empty is ok)
    """
    @dataclasses.dataclass
    class DummyCfg:
        job_name: str = "unit_test"
    return DummyCfg()


def _patch_render_and_popen(monkeypatch, mod, tmp_job_dir, return_code=0):
    # Fake render_script: write something simple
    def fake_render_script(opts, job_dir):
        return "#!/bin/bash\necho hello\n"

    monkeypatch.setattr(mod, "render_script", fake_render_script, raising=True)

    # Fake subprocess.Popen
    class FakePopen:
        def __init__(self, *args, **kwargs):
            self.returncode = return_code
            self._stdout = b"SUBMITTED 123\n"
            self._stderr = b"" if return_code == 0 else b"Boom!"

        def communicate(self):
            return self._stdout, self._stderr

    monkeypatch.setattr(mod.subprocess, "Popen", FakePopen, raising=True)


def test_submit_slurm_job_success(monkeypatch, tmp_job_dir, mod):
    _patch_render_and_popen(monkeypatch, mod, tmp_job_dir, return_code=0)

    cfg = _fake_slurm_config()
    rc = mod.submit_slurm_job(cfg, tmp_job_dir)

    # Return code must propagate
    assert rc == 0

    # Script + subprocess outputs must have been written
    sbatch = Path(tmp_job_dir) / f"{cfg.job_name}.sbatch"
    assert sbatch.exists() and sbatch.read_text().startswith("#!/bin/bash")

    assert (Path(tmp_job_dir) / "subproc_sbatch.stdout").exists()
    assert (Path(tmp_job_dir) / "subproc_sbatch.stderr").exists()


def test_submit_slurm_job_failure(monkeypatch, tmp_job_dir, mod):
    _patch_render_and_popen(monkeypatch, mod, tmp_job_dir, return_code=1)

    cfg = _fake_slurm_config()
    rc = mod.submit_slurm_job(cfg, tmp_job_dir)
    assert rc == 1

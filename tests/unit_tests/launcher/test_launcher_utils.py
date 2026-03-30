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

from pathlib import Path

import pytest

from nemo_automodel.components.launcher.slurm.utils import submit_slurm_job


@pytest.fixture()
def tmp_job_dir(tmp_path: Path):
    return tmp_path / "job_dir"


def test_submit_slurm_job_success(monkeypatch, tmp_job_dir):
    class FakePopen:
        def __init__(self, *args, **kwargs):
            self.returncode = 0
            self._stdout = b"SUBMITTED 123\n"
            self._stderr = b""

        def communicate(self):
            return self._stdout, self._stderr

    import nemo_automodel.components.launcher.slurm.utils as mod

    monkeypatch.setattr(mod.subprocess, "Popen", FakePopen, raising=True)

    script = tmp_job_dir / "test.sbatch"
    tmp_job_dir.mkdir(parents=True)
    script.write_text("#!/bin/bash\necho ok\n")

    rc = submit_slurm_job(
        str(script),
        {"AUTOMODEL_COMMAND": "torchrun foo.py"},
        str(tmp_job_dir),
    )
    assert rc == 0
    assert (tmp_job_dir / "subproc_sbatch.stdout").exists()
    assert (tmp_job_dir / "subproc_sbatch.stderr").exists()


def test_submit_slurm_job_failure(monkeypatch, tmp_job_dir):
    class FakePopen:
        def __init__(self, *args, **kwargs):
            self.returncode = 1
            self._stdout = b""
            self._stderr = b"Boom!"

        def communicate(self):
            return self._stdout, self._stderr

    import nemo_automodel.components.launcher.slurm.utils as mod

    monkeypatch.setattr(mod.subprocess, "Popen", FakePopen, raising=True)

    script = tmp_job_dir / "test.sbatch"
    tmp_job_dir.mkdir(parents=True)
    script.write_text("#!/bin/bash\necho ok\n")

    rc = submit_slurm_job(str(script), {}, str(tmp_job_dir))
    assert rc == 1

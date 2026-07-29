# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import yaml

_WORKFLOW_PATH = Path(".github/workflows/install-test.yml")


def _jobs() -> dict:
    return yaml.safe_load(_WORKFLOW_PATH.read_text())["jobs"]


def test_cuda_wheelhouse_registry_permissions_are_read_only_for_prs():
    jobs = _jobs()

    assert jobs["cuda-wheelhouse"]["permissions"]["packages"] == "read"
    assert jobs["publish-cuda-wheelhouse"]["permissions"]["packages"] == "write"
    assert "github.ref == 'refs/heads/main'" in jobs["publish-cuda-wheelhouse"]["if"]


def test_cuda_wheelhouse_uses_exact_oci_reference_without_actions_cache():
    jobs = _jobs()
    cuda_job = jobs["cuda-wheelhouse"]
    cuda_job_text = yaml.safe_dump(cuda_job)

    assert "actions/cache" not in cuda_job_text
    assert "regctl manifest head" in cuda_job_text
    assert "${fingerprint}" in cuda_job_text
    assert cuda_job["outputs"]["reference"] == "${{ steps.cuda-wheelhouse-ref.outputs.reference }}"

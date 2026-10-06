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

import re
import shlex
import subprocess
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # ty: ignore[unresolved-import]


_SECURITY_FLOORS = {"click": "8.3.3", "accelerate": "1.15.0", "fsspec": "2026.6.0", "cryptography": "50.0.0"}


def test_security_constraints_prevent_older_resolutions() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())
    constraints = {
        requirement.name: requirement
        for text in project["tool"]["uv"]["constraint-dependencies"]
        if (requirement := Requirement(text))
    }
    for name, floor in _SECURITY_FLOORS.items():
        assert str(constraints[name].specifier) == f">={floor}"


@pytest.mark.parametrize("lock_path", ["uv.lock", "docker/common/uv-pytorch.lock"])
def test_security_floors_in_each_installation_lock(lock_path: str) -> None:
    lock = tomllib.loads(Path(lock_path).read_text())
    for name, floor in _SECURITY_FLOORS.items():
        packages = [package for package in lock["package"] if package["name"] == name]
        assert packages, f"{name} missing from {lock_path}"
        assert all(Version(package["version"]) >= Version(floor) for package in packages)


@pytest.mark.parametrize("present", [(), (0,), (1,), (0, 1)])
def test_nsys_cleanup_removes_only_nic_samplers(tmp_path: Path, present: tuple[int, ...]) -> None:
    dockerfile = Path("docker/Dockerfile").read_text()
    version = re.search(r'NSYS_VERSION="([^"]+)"', dockerfile).group(1)
    release = ".".join(version.split(".")[:3])
    start = dockerfile.index('    rm -rf "/opt/nvidia/nsight-systems-cli/${NSYS_RELEASE}/target-linux-x64')
    command = dockerfile[start : dockerfile.index(" &&", start)].replace("\\\n", " ")
    install = dockerfile.index('apt-get install -y --no-install-recommends "nsight-systems-cli-')
    assert install < start
    targets = [
        f"/opt/nvidia/nsight-systems-cli/{release}/{arch}/plugins/efa_metrics/nic_sampler"
        for arch in ("target-linux-x64", "target-linux-sbsa-armv8")
    ]
    assert shlex.split(command.replace("${NSYS_RELEASE}", release)) == ["rm", "-rf", *targets]
    paths = [tmp_path / target.lstrip("/") for target in targets]
    for index, path in enumerate(paths):
        path.parent.mkdir(parents=True)
        (path.parent / "other-plugin").write_text("preserve")
        if index in present:
            path.mkdir()
            (path / "sampler").write_text("vulnerable fixture")
    fixture_command = command.replace("/opt/nvidia", f"{tmp_path}/opt/nvidia")
    for _ in range(2):
        subprocess.run(["sh", "-e", "-c", fixture_command], check=True, env={"NSYS_RELEASE": release})
        assert all(not path.exists() for path in paths)
        assert all((path.parent / "other-plugin").read_text() == "preserve" for path in paths)

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import hashlib
from pathlib import Path

import pytest

from scripts.cuda_wheelhouse_manifest import verify_manifest, write_manifest


@pytest.fixture
def wheelhouse(tmp_path: Path) -> tuple[Path, str]:
    (tmp_path / "mamba_ssm-2.3.0-cp312-cp312-linux_x86_64.whl").write_bytes(b"built wheel")
    write_manifest(tmp_path, fingerprint="expected-build")
    digest = hashlib.sha256((tmp_path / "manifest.json").read_bytes()).hexdigest()
    return tmp_path, digest


def test_verified_manifest_accepts_original_wheels(wheelhouse: tuple[Path, str]) -> None:
    directory, digest = wheelhouse
    verify_manifest(directory, fingerprint="expected-build", manifest_sha256=digest)


@pytest.mark.parametrize("change", ["modified", "missing", "extra"])
def test_changed_wheel_contents_are_rejected(wheelhouse: tuple[Path, str], change: str) -> None:
    directory, digest = wheelhouse
    wheel = next(directory.glob("*.whl"))
    if change == "modified":
        wheel.write_bytes(b"modified wheel")
    elif change == "missing":
        wheel.unlink()
    else:
        (directory / "unexpected-1.0-py3-none-any.whl").write_bytes(b"extra wheel")
    with pytest.raises(ValueError, match="contents|no wheels"):
        verify_manifest(directory, fingerprint="expected-build", manifest_sha256=digest)


def test_replaced_manifest_is_rejected(wheelhouse: tuple[Path, str]) -> None:
    directory, digest = wheelhouse
    next(directory.glob("*.whl")).write_bytes(b"replacement wheel")
    write_manifest(directory, fingerprint="expected-build")
    with pytest.raises(ValueError, match="verified attestation"):
        verify_manifest(directory, fingerprint="expected-build", manifest_sha256=digest)


def test_incompatible_build_is_rejected(wheelhouse: tuple[Path, str]) -> None:
    directory, digest = wheelhouse
    with pytest.raises(ValueError, match="fingerprint"):
        verify_manifest(directory, fingerprint="different-build", manifest_sha256=digest)


@pytest.mark.parametrize("entry", ["symlink", "directory", "sdist"])
def test_non_wheel_payloads_are_rejected(wheelhouse: tuple[Path, str], entry: str) -> None:
    directory, digest = wheelhouse
    if entry == "symlink":
        (directory / "link.whl").symlink_to(next(directory.glob("*.whl")))
    elif entry == "directory":
        (directory / "nested").mkdir()
    else:
        (directory / "unexpected.tar.gz").write_bytes(b"sdist")
    with pytest.raises(ValueError, match="regular file|Unexpected"):
        verify_manifest(directory, fingerprint="expected-build", manifest_sha256=digest)

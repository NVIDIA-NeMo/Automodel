#!/usr/bin/env python3

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

"""Bind the wheelhouse contents to its attested build fingerprint."""

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_contents(directory: Path, fingerprint: str) -> bytes:
    wheels = {}
    for path in sorted(directory.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Wheelhouse entry must be a regular file: {path.name}")
        if path.name in {"manifest.json", "attestation.json"}:
            continue
        if path.suffix != ".whl":
            raise ValueError(f"Unexpected wheelhouse entry: {path.name}")
        wheels[path.name] = _sha256(path)
    if not wheels:
        raise ValueError("Wheelhouse contains no wheels")
    return json.dumps({"schema": 1, "fingerprint": fingerprint, "wheels": wheels}, sort_keys=True).encode() + b"\n"


def write_manifest(directory: Path, *, fingerprint: str) -> None:
    """Write the manifest before the builder attests it."""
    (directory / "manifest.json").write_bytes(_manifest_contents(directory, fingerprint))


def verify_manifest(directory: Path, *, fingerprint: str, manifest_sha256: str) -> None:
    """Check the attested manifest digest, build inputs, and every wheel's bytes."""
    manifest = directory / "manifest.json"
    if _sha256(manifest) != manifest_sha256:
        raise ValueError("Wheelhouse manifest does not match the verified attestation")
    if manifest.read_bytes() != _manifest_contents(directory, fingerprint):
        raise ValueError("Wheelhouse contents or build fingerprint do not match the attested manifest")


def main() -> None:
    """Create a build manifest or verify one before installing wheels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("write", "verify"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    if args.operation == "write":
        write_manifest(args.directory, fingerprint=args.fingerprint)
    else:
        if not args.manifest_sha256:
            parser.error("--manifest-sha256 is required for verification")
        verify_manifest(args.directory, fingerprint=args.fingerprint, manifest_sha256=args.manifest_sha256)


if __name__ == "__main__":
    main()

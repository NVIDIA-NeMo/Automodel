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


"""Reproduce the exact FLA source corrections used for K3 packed CP validation."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

_PATCHES = (
    (
        "ops/cp/chunk_delta_h.py",
        "fla_cp_decay_head_offset.patch",
        "c479d976b76ac58ef90439a66bed339cb672e90c897fe6a6945d4b70fd1fdf41",
        "18688a8fb700f407e0854c46ffcfe97f464540ee097f3fa44b5d2c588e6ae907",
    ),
    (
        "ops/kda/chunk_intra.py",
        "fla_kda_tril_precision.patch",
        "266b3f8b6342e82c9928ba099a833274e552da61e09799d2ea7d82a028e55f9a",
        "db4435dc18d17863b3dec28c88949a1712114324bfdd1721cb29a5944d78c8a6",
    ),
)


def main() -> None:
    """Copy a matching FLA package and patch the copy without editing the installation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Existing fla package directory")
    parser.add_argument("--output", type=Path, required=True, help="New PYTHONPATH overlay directory")
    args = parser.parse_args()
    for relative, _, before, _ in _PATCHES:
        actual = hashlib.sha256((args.source / relative).read_bytes()).hexdigest()
        if actual != before:
            raise ValueError(f"Unsupported FLA source at {relative}: {actual}; expected {before}")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.source, args.output / "fla", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for relative, patch, _, after in _PATCHES:
        subprocess.run(
            ["patch", "--batch", "--forward", "-p1", "-i", str(Path(__file__).resolve().parent / patch)],
            cwd=args.output,
            check=True,
        )
        actual = hashlib.sha256((args.output / "fla" / relative).read_bytes()).hexdigest()
        if actual != after:
            raise ValueError(f"Patched FLA source hash mismatch at {relative}: {actual}")
    (args.output / "patch_manifest.json").write_text(
        json.dumps(
            {
                "source": str(args.source.resolve()),
                "source_version": "0.4.2",
                "patches": [
                    {"path": p, "patch": patch, "before": before, "after": after}
                    for p, patch, before, after in _PATCHES
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

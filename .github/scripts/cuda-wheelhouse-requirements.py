#!/usr/bin/env python3

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

"""Print the source-built CUDA requirements used by installation CI."""

import argparse
import json
import re
from pathlib import Path

import tomllib

_CACHED_REQUIREMENTS = {
    "causal-conv1d",
    "flash-attn",
    "mamba-ssm",
    "nv-grouped-gemm",
    "transformer-engine",
}
_EXTRAS = ("cuda", "fa")
_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")
_SUPPLEMENTAL_REQUIREMENTS = ("transformer-engine-torch",)


def _canonicalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def load_requirements(pyproject_path: Path) -> list[str]:
    """Return the cached requirements selected from the CUDA-related extras."""
    with pyproject_path.open("rb") as pyproject_file:
        pyproject = tomllib.load(pyproject_file)

    optional_dependencies = pyproject["project"]["optional-dependencies"]
    requirements_by_name: dict[str, str] = {}
    for extra in _EXTRAS:
        for requirement in optional_dependencies[extra]:
            match = _REQUIREMENT_NAME.match(requirement)
            if match is None:
                raise ValueError(f"Cannot parse requirement from [{extra}]: {requirement}")
            name = _canonicalize_name(match.group())
            if name in _CACHED_REQUIREMENTS:
                requirements_by_name[name] = requirement

    missing = _CACHED_REQUIREMENTS - requirements_by_name.keys()
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"Missing cached CUDA requirements: {missing_names}")

    requirements_by_name.update(
        {_canonicalize_name(requirement): requirement for requirement in _SUPPLEMENTAL_REQUIREMENTS}
    )
    return [requirements_by_name[name] for name in sorted(requirements_by_name)]


def main() -> None:
    """Print the cached CUDA requirements in the requested format."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Print a canonical JSON array")
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    args = parser.parse_args()

    requirements = load_requirements(args.pyproject)
    if args.json:
        print(json.dumps(requirements, separators=(",", ":")))
    else:
        print(*requirements, sep="\n")


if __name__ == "__main__":
    main()

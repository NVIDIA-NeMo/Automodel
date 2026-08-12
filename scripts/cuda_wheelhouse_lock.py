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

"""Resolve the exact CUDA wheelhouse inputs from uv.lock."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import TypedDict

try:
    import tomllib  # ty: ignore[unresolved-import]
except ModuleNotFoundError:
    import tomli as tomllib  # ty: ignore[unresolved-import]


_LOCKED_BUILD_TOOLS = ("numpy", "packaging", "psutil", "pybind11", "setuptools")
_PINNED_BUILD_TOOLS = ("pip==26.1.2", "wheel==0.47.0", "wheel-stub==0.5.0")
_WHEEL_REQUIREMENTS = (
    "causal-conv1d",
    "flash-attn",
    "mamba-ssm",
    "nv-grouped-gemm",
    "transformer-engine",
    "transformer-engine-torch",
)
_RUNTIME_ONLY_REQUIREMENTS = ("transformer-engine-cu13",)


class LockedInputs(TypedDict):
    """Concrete requirements that define one wheelhouse cache entry."""

    build_tools: list[str]
    constraints: list[str]
    torch: str
    torchvision: str
    wheels: list[str]


def _normalize_registry(url: str) -> str:
    return url.rstrip("/")


def _select_package(
    packages: list[dict],
    name: str,
    *,
    python_version: str,
    platform_machine: str,
    sys_platform: str,
    registry: str | None = None,
) -> dict:
    candidates = [package for package in packages if package["name"] == name]
    if registry is not None:
        normalized_registry = _normalize_registry(registry)
        candidates = [
            package
            for package in candidates
            if _normalize_registry(package.get("source", {}).get("registry", "")) == normalized_registry
        ]

    if len(candidates) > 1:
        major_minor = ".".join(python_version.split(".")[:2])
        target_tokens = (
            f"python_full_version == '{major_minor}.*'",
            f"platform_machine == '{platform_machine}'",
            f"sys_platform == '{sys_platform}'",
        )
        candidates = [
            package
            for package in candidates
            if any(all(token in marker for token in target_tokens) for marker in package.get("resolution-markers", ()))
        ]

    if len(candidates) != 1:
        source_description = f" from {registry}" if registry is not None else ""
        raise ValueError(f"Expected one locked {name} package{source_description}, found {len(candidates)}")
    return candidates[0]


def _locked_requirement(package: dict, *, extras: str | None = None) -> str:
    name = package["name"]
    if extras is not None:
        name = f"{name}[{extras}]"
    return f"{name}=={package['version']}"


def load_locked_inputs(
    lock_path: Path,
    *,
    torch_index: str,
    python_version: str,
    platform_machine: str,
    sys_platform: str,
) -> LockedInputs:
    """Load the exact build and runtime requirements for the target platform."""
    with lock_path.open("rb") as lock_file:
        packages = tomllib.load(lock_file)["package"]

    selection_args = {
        "python_version": python_version,
        "platform_machine": platform_machine,
        "sys_platform": sys_platform,
    }
    torch_package = _select_package(packages, "torch", registry=torch_index, **selection_args)
    torchvision_package = _select_package(packages, "torchvision", registry=torch_index, **selection_args)

    wheel_packages = {name: _select_package(packages, name, **selection_args) for name in _WHEEL_REQUIREMENTS}
    wheels = [
        _locked_requirement(
            wheel_packages[name],
            extras="pytorch" if name == "transformer-engine" else None,
        )
        for name in _WHEEL_REQUIREMENTS
    ]

    build_tools = [
        _locked_requirement(_select_package(packages, name, **selection_args)) for name in _LOCKED_BUILD_TOOLS
    ]
    build_tools.extend(_PINNED_BUILD_TOOLS)

    runtime_packages = {
        name: _select_package(packages, name, **selection_args)
        for name in (*_WHEEL_REQUIREMENTS, *_RUNTIME_ONLY_REQUIREMENTS)
    }
    constraints = list(build_tools)
    constraints.append(_locked_requirement(torch_package))
    constraints.append(_locked_requirement(torchvision_package))
    constraints.extend(_locked_requirement(runtime_packages[name]) for name in sorted(runtime_packages))

    return {
        "build_tools": sorted(build_tools),
        "constraints": sorted(constraints),
        "torch": _locked_requirement(torch_package),
        "torchvision": _locked_requirement(torchvision_package),
        "wheels": wheels,
    }


def cache_fingerprint(
    locked_inputs: LockedInputs,
    *,
    runner_os: str,
    python_version: str,
    cuda_container_image: str,
    torch_index: str,
    torch_cuda_arch_list: str,
    build_script: Path,
    helper_script: Path = Path(__file__),
) -> str:
    """Hash the resolved packages and every wheel ABI/build input."""
    payload = {
        "schema": 3,
        "runner_os": runner_os,
        "python_version": python_version,
        "cuda_container_image": cuda_container_image,
        "torch_index": torch_index,
        "torch_cuda_arch_list": torch_cuda_arch_list,
        "locked_inputs": locked_inputs,
        "build_script_sha256": hashlib.sha256(build_script.read_bytes()).hexdigest(),
        "helper_script_sha256": hashlib.sha256(helper_script.read_bytes()).hexdigest(),
    }
    serialized_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(serialized_payload).hexdigest()


def _print_lines(values: list[str]) -> None:
    print(*values, sep="\n")


def main() -> None:
    """Print one lock-derived wheelhouse value for GitHub Actions."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        choices=("build-tools", "constraints", "fingerprint", "manifest", "torch", "torchvision", "wheels"),
        required=True,
    )
    parser.add_argument("--lock", type=Path, default=Path("uv.lock"))
    parser.add_argument("--torch-index", required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--platform-machine", default="x86_64")
    parser.add_argument("--sys-platform", default="linux")
    parser.add_argument("--runner-os")
    parser.add_argument("--cuda-container-image")
    parser.add_argument("--torch-cuda-arch-list")
    parser.add_argument(
        "--build-script",
        type=Path,
        default=Path(".github/scripts/build-cuda-wheelhouse.sh"),
    )
    args = parser.parse_args()

    locked_inputs = load_locked_inputs(
        args.lock,
        torch_index=args.torch_index,
        python_version=args.python_version,
        platform_machine=args.platform_machine,
        sys_platform=args.sys_platform,
    )

    if args.output == "manifest":
        print(json.dumps(locked_inputs, sort_keys=True, separators=(",", ":")))
    elif args.output == "torch":
        print(locked_inputs["torch"])
    elif args.output == "torchvision":
        print(locked_inputs["torchvision"])
    elif args.output == "wheels":
        _print_lines(locked_inputs["wheels"])
    elif args.output == "build-tools":
        _print_lines(locked_inputs["build_tools"])
    elif args.output == "constraints":
        _print_lines(locked_inputs["constraints"])
    else:
        required_fingerprint_args = {
            "--runner-os": args.runner_os,
            "--cuda-container-image": args.cuda_container_image,
            "--torch-cuda-arch-list": args.torch_cuda_arch_list,
        }
        missing_args = [name for name, value in required_fingerprint_args.items() if value is None]
        if missing_args:
            parser.error(f"{', '.join(missing_args)} required with --output fingerprint")
        print(
            cache_fingerprint(
                locked_inputs,
                runner_os=args.runner_os,
                python_version=args.python_version,
                cuda_container_image=args.cuda_container_image,
                torch_index=args.torch_index,
                torch_cuda_arch_list=args.torch_cuda_arch_list,
                build_script=args.build_script,
            )
        )


if __name__ == "__main__":
    main()

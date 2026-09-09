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

from scripts.cuda_wheelhouse_lock import cache_fingerprint, load_locked_inputs

_BUILD_SCRIPT = Path(".github/scripts/build-cuda-wheelhouse.sh")
_CUDA_IMAGE = "nvcr.io/nvidia/cuda-dl-base:26.04-cuda13.2-devel-ubuntu24.04"
_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"


def _lock_contents(
    *,
    unrelated_version: str = "1.0.0",
    torch_version: str = "2.10.0+cu130",
    torchvision_version: str = "0.25.0+cu130",
    causal_conv1d_version: str = "1.6.0",
    packaging_version: str = "25.0",
) -> str:
    versions = {
        "causal-conv1d": causal_conv1d_version,
        "flash-attn": "2.8.3",
        "mamba-ssm": "2.3.0",
        "numpy": "1.26.4",
        "nv-grouped-gemm": "1.1.4.post8",
        "packaging": packaging_version,
        "psutil": "7.1.1",
        "pybind11": "3.0.1",
        "setuptools": "80.10.2",
        "torch": torch_version,
        "torchvision": torchvision_version,
        "transformer-engine": "2.15.0",
        "transformer-engine-cu13": "2.15.0",
        "transformer-engine-torch": "2.15.0",
        "unrelated-package": unrelated_version,
    }
    packages = []
    for name, version in versions.items():
        registry = _TORCH_INDEX if name in {"torch", "torchvision"} else "https://pypi.org/simple"
        packages.append(
            "\n".join(
                (
                    "[[package]]",
                    f'name = "{name}"',
                    f'version = "{version}"',
                    f'source = {{ registry = "{registry}" }}',
                )
            )
        )
    return "\n\n".join(packages)


def _fingerprint(tmp_path: Path, **lock_versions: str) -> str:
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text(_lock_contents(**lock_versions))
    locked_inputs = load_locked_inputs(
        lock_path,
        torch_index=_TORCH_INDEX,
        python_version="3.12",
        platform_machine="x86_64",
        sys_platform="linux",
    )
    return cache_fingerprint(
        locked_inputs,
        runner_os="Linux",
        python_version="3.12",
        cuda_container_image=_CUDA_IMAGE,
        torch_index=_TORCH_INDEX,
        torch_cuda_arch_list="9.0 10.0 12.0",
        build_script=_BUILD_SCRIPT,
    )


def test_lock_manifest_pins_build_and_runtime_requirements(tmp_path):
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text(_lock_contents())
    locked_inputs = load_locked_inputs(
        lock_path,
        torch_index=_TORCH_INDEX,
        python_version="3.12",
        platform_machine="x86_64",
        sys_platform="linux",
    )

    assert locked_inputs["torch"] == "torch==2.10.0+cu130"
    assert locked_inputs["torchvision"] == "torchvision==0.25.0+cu130"
    assert "causal-conv1d==1.6.0" in locked_inputs["wheels"]
    assert "numpy==1.26.4" in locked_inputs["build_tools"]
    assert set(locked_inputs["build_tools"]) <= set(locked_inputs["constraints"])
    assert "torch==2.10.0+cu130" in locked_inputs["constraints"]
    assert "torchvision==0.25.0+cu130" in locked_inputs["constraints"]


def test_unrelated_lock_change_preserves_cache_fingerprint(tmp_path):
    baseline = _fingerprint(tmp_path, unrelated_version="1.0.0")
    updated = _fingerprint(tmp_path, unrelated_version="2.0.0")

    assert updated == baseline


def test_torch_lock_change_invalidates_cache_fingerprint(tmp_path):
    baseline = _fingerprint(tmp_path, torch_version="2.10.0+cu130")
    updated = _fingerprint(tmp_path, torch_version="2.11.0+cu130")

    assert updated != baseline


def test_torchvision_lock_change_invalidates_cache_fingerprint(tmp_path):
    baseline = _fingerprint(tmp_path, torchvision_version="0.25.0+cu130")
    updated = _fingerprint(tmp_path, torchvision_version="0.26.0+cu130")

    assert updated != baseline


def test_cached_wheel_lock_change_invalidates_cache_fingerprint(tmp_path):
    baseline = _fingerprint(tmp_path, causal_conv1d_version="1.6.0")
    updated = _fingerprint(tmp_path, causal_conv1d_version="1.6.1")

    assert updated != baseline


def test_build_tool_lock_change_invalidates_cache_fingerprint(tmp_path):
    baseline = _fingerprint(tmp_path, packaging_version="25.0")
    updated = _fingerprint(tmp_path, packaging_version="26.0")

    assert updated != baseline

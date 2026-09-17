#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Build against the selected Python's existing PyTorch and CUDA toolkit.
# DEEPSELECT_PYTHON chooses the installation interpreter.
# DEEP_SELECT_CUDA_ARCHS defaults to "90a;100a;103a"; set "90a" for H100 only.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
deepselect_python="${DEEPSELECT_PYTHON:-$(command -v python)}"
deepselect_ref=0f03b68748b304863fdf0181a11458d04ae533a9
deepselect_build="$(mktemp -d -t automodel-deepselect.XXXXXXXX)"
trap 'rm -rf -- "$deepselect_build"' EXIT

git init -q "$deepselect_build/source"
git -C "$deepselect_build/source" remote add origin https://github.com/deepseek-ai/DeepSelect.git
git -C "$deepselect_build/source" fetch --depth 1 origin "$deepselect_ref"
git -C "$deepselect_build/source" checkout --detach "$deepselect_ref"
git -C "$deepselect_build/source" submodule update --init --recursive
git -C "$deepselect_build/source" apply "$script_dir/deepselect.patch"

# The extension has no runtime dependencies besides the existing PyTorch/CUDA.
# Explicit CUDA selection permits building images without a visible GPU.
DEEP_SELECT_BUILD_TARGET_PLATFORM=CUDA \
DEEP_SELECT_CUDA_ARCHS="${DEEP_SELECT_CUDA_ARCHS:-90a;100a;103a}" \
MAX_JOBS="${MAX_JOBS:-8}" NVCC_THREADS="${NVCC_THREADS:-2}" \
    uv pip install --python "$deepselect_python" --no-build-isolation --no-deps "$deepselect_build/source"

"$deepselect_python" -c 'import torch, deep_select; print("DeepSelect:", deep_select.__file__)'

#!/usr/bin/env bash

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

set -euxo pipefail

: "${TORCH_CU_INDEX:?TORCH_CU_INDEX must be set}"
: "${WHEELHOUSE_DIR:?WHEELHOUSE_DIR must be set}"
: "${PYTHON_VERSION:?PYTHON_VERSION must be set}"
: "${PLATFORM_MACHINE:?PLATFORM_MACHINE must be set}"
: "${SYS_PLATFORM:?SYS_PLATFORM must be set}"

lock_args=(
  --lock uv.lock
  --torch-index "${TORCH_CU_INDEX}"
  --python-version "${PYTHON_VERSION}"
  --platform-machine "${PLATFORM_MACHINE}"
  --sys-platform "${SYS_PLATFORM}"
)

python -m venv ./venv

. ./venv/bin/activate

build_tools="$(
  python scripts/cuda_wheelhouse_lock.py --output build-tools "${lock_args[@]}"
)"
mapfile -t build_tool_requirements <<< "${build_tools}"
python -m pip install "${build_tool_requirements[@]}"

torch_requirement="$(
  python scripts/cuda_wheelhouse_lock.py --output torch "${lock_args[@]}"
)"
python -m pip install --index-url "${TORCH_CU_INDEX}" "${torch_requirement}"

requirements="$(
  python scripts/cuda_wheelhouse_lock.py --output wheels "${lock_args[@]}"
)"
mapfile -t wheelhouse_requirements <<< "${requirements}"

mkdir -p "${WHEELHOUSE_DIR}"
python -m pip wheel \
  --no-deps \
  --no-build-isolation \
  --wheel-dir "${WHEELHOUSE_DIR}" \
  "${wheelhouse_requirements[@]}"

#!/usr/bin/env bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source_build=false
if [[ ${GITHUB_EVENT_NAME} == schedule || ${FORCE_SOURCE_BUILD:-false} == true ]]; then
  source_build=true
else
  if [[ ${GITHUB_REF_NAME} == main ]]; then
    base=${BEFORE_SHA:-HEAD^}
  else
    base=$(git merge-base HEAD origin/main || true)
  fi
  # Missing history must enable coverage, not silently skip it.
  if ! git cat-file -e "${base}^{commit}" 2>/dev/null; then
    source_build=true
  elif git diff --quiet "${base}" HEAD -- \
    pyproject.toml uv.lock .github/workflows/install-test.yml \
    .github/scripts/build-cuda-wheelhouse.sh .github/scripts/select-uv-install-modes.sh \
    scripts/cuda_wheelhouse_lock.py; then
    source_build=false
  else
    source_build=true
  fi
fi

if [[ ${source_build} == true ]]; then
  echo 'uv_modes=["wheelhouse","source"]' >> "${GITHUB_OUTPUT}"
else
  echo 'uv_modes=["wheelhouse"]' >> "${GITHUB_OUTPUT}"
fi

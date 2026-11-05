#!/usr/bin/env bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

base_branch=main
release_branch=
cache_writer=false
if [[ ${GITHUB_REF} == refs/heads/pull-request/* ]]; then
  # Copied PRs are push events, so github.base_ref is unavailable.
  base_branch=${PR_BASE_REF:?Copied PR metadata must include its target branch}
  git check-ref-format "refs/heads/${base_branch}"
  if [[ ${base_branch} =~ ^r0\.[0-9]+\.0$ ]]; then
    release_branch=${base_branch}
  fi
elif [[ ${GITHUB_REF} == refs/heads/main ]]; then
  cache_writer=true
elif [[ ${GITHUB_REF} =~ ^refs/heads/r0\.[0-9]+\.0$ ]]; then
  base_branch=${GITHUB_REF_NAME}
  release_branch=${GITHUB_REF_NAME}
  cache_writer=true
fi

{
  echo "base_branch=${base_branch}"
  echo "release_branch=${release_branch}"
  echo "cache_writer=${cache_writer}"
} >> "${GITHUB_OUTPUT}"

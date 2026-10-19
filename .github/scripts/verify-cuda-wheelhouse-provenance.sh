#!/usr/bin/env bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

identity=(
  --repo "${GITHUB_REPOSITORY}"
  --signer-workflow "${GITHUB_REPOSITORY}/.github/workflows/install-test.yml"
)
if [[ ${WHEELHOUSE_REUSED} == true ]]; then
  # Release caches may themselves have restored main's compatible wheelhouse.
  # Accept only main or this run's release target, never a sibling PR/release.
  if ! gh attestation verify "${WHEELHOUSE_DIR}/manifest.json" \
    --bundle "${WHEELHOUSE_DIR}/attestation.json" "${identity[@]}" --source-ref refs/heads/main; then
    [[ ${RELEASE_BRANCH:-} =~ ^r0\.[0-9]+\.0$ ]]
    gh attestation verify "${WHEELHOUSE_DIR}/manifest.json" \
      --bundle "${WHEELHOUSE_DIR}/attestation.json" "${identity[@]}" --source-ref "refs/heads/${RELEASE_BRANCH}"
  fi
else
  # A new PR build must belong to this exact approved source revision.
  identity+=(--source-ref "${GITHUB_REF}" --source-digest "${GITHUB_SHA}" --signer-digest "${GITHUB_SHA}")
  gh attestation verify "${WHEELHOUSE_DIR}/manifest.json" \
    --bundle "${WHEELHOUSE_DIR}/attestation.json" "${identity[@]}"
fi
digest=$(sha256sum "${WHEELHOUSE_DIR}/manifest.json")
echo "manifest_sha256=${digest%% *}" >> "${GITHUB_OUTPUT}"

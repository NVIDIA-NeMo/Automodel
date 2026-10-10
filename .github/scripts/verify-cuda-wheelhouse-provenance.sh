#!/usr/bin/env bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

identity=(
  --repo "${GITHUB_REPOSITORY}"
  --signer-workflow "${GITHUB_REPOSITORY}/.github/workflows/install-test.yml"
)
if [[ ${WHEELHOUSE_CACHE_HIT} == true ]]; then
  # A restored cache must have been built and signed on the canonical branch.
  identity+=(--source-ref refs/heads/main)
else
  # A new PR build must belong to this exact approved source revision.
  identity+=(--source-ref "${GITHUB_REF}" --source-digest "${GITHUB_SHA}" --signer-digest "${GITHUB_SHA}")
fi

gh attestation verify "${WHEELHOUSE_DIR}/manifest.json" \
  --bundle "${WHEELHOUSE_DIR}/attestation.json" "${identity[@]}"
digest=$(sha256sum "${WHEELHOUSE_DIR}/manifest.json")
echo "manifest_sha256=${digest%% *}" >> "${GITHUB_OUTPUT}"

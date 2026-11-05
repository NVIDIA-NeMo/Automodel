#!/usr/bin/env bash

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Resolve the actual base once, then use the same immutable image in both the
# compatibility key and the build. A moved NGC tag must invalidate native wheels.
base=$(sed -n 's/^ARG PYTORCH_IMAGE=//p' docker/Dockerfile)
base_digest=$(docker buildx imagetools inspect "$base" --format '{{.Manifest.Digest}}')
[[ $base_digest =~ ^sha256:[0-9a-f]{64}$ ]]
build_args="$(cat .github/container-build-args.env)
PYTORCH_IMAGE=${base%@*}@${base_digest}"

# Include every dependency COPY input, the Dockerfile (compiler flags and source
# revisions), and this key policy. Application/test edits do not change the key.
fingerprint=$(
  {
    printf '%s\n' "$PLATFORM" "$build_args"
    git ls-files -z -- docker .dockerignore pyproject.toml uv.lock \
      nemo_automodel/__init__.py nemo_automodel/package_info.py \
      scripts/setup_uccl_ep.sh .github/container-build-args.env \
      .github/scripts/prepare-container-dependencies.sh |
      LC_ALL=C sort -z | xargs -0 sha256sum --zero
  } | sha256sum
)
fingerprint=${fingerprint%% *}

# PRs publish only their own scope. A cherry-pick may also read its release
# target's compatible image; main never consumes a PR or release image.
scope=main
if [[ $CACHE_KEY != main ]]; then
  scope=$(printf '%s' "$CACHE_KEY" | sha256sum)
  scope="branch-${scope:0:16}"
fi
image="${REGISTRY}/automodel:deps-${scope}-${PLATFORM##*/}-${fingerprint}"
baseline="${REGISTRY}/automodel:deps-main-${PLATFORM##*/}-${fingerprint}"
candidates=("$image")
if [[ $CACHE_KEY != main && -n ${RELEASE_BRANCH:-} ]]; then
  [[ $RELEASE_BRANCH =~ ^r0\.[0-9]+\.0$ ]]
  release_scope=$(printf '%s' "$RELEASE_BRANCH" | sha256sum)
  candidates+=("${REGISTRY}/automodel:deps-branch-${release_scope:0:16}-${PLATFORM##*/}-${fingerprint}")
fi
candidates+=("$baseline")
digest=
error_file=$(mktemp)
trap 'rm -f "$error_file"' EXIT
for candidate in "${candidates[@]}"; do
  if candidate_digest=$(docker buildx imagetools inspect "$candidate" --format '{{.Manifest.Digest}}' 2>"$error_file"); then
    [[ $candidate_digest =~ ^sha256:[0-9a-f]{64}$ ]]
    image=$candidate
    digest=$candidate_digest
    break
  elif ! grep -Eqi 'manifest unknown|not found|MANIFEST_UNKNOWN|NAME_UNKNOWN' "$error_file"; then
    # Authentication/network failures are not a reason to silently recompile.
    cat "$error_file" >&2
    exit 1
  fi
done

{
  echo "image=$image"
  echo "digest=$digest"
  echo "cache-hit=$([[ -n $digest ]] && echo true || echo false)"
  echo 'build-args<<AUTOMODEL_BUILD_ARGS'
  printf '%s\n' "$build_args"
  echo AUTOMODEL_BUILD_ARGS
} >> "$GITHUB_OUTPUT"

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

# Only main publishes the canonical scope. PRs/releases may reuse main or their
# own previous build, but never another PR's dependency image.
scope=main
if [[ $CACHE_KEY != main ]]; then
  scope=$(printf '%s' "$CACHE_KEY" | sha256sum)
  scope="branch-${scope:0:16}"
fi
image="${REGISTRY}/automodel:deps-${scope}-${PLATFORM##*/}-${fingerprint}"
baseline="${REGISTRY}/automodel:deps-main-${PLATFORM##*/}-${fingerprint}"
digest=
error_file=$(mktemp)
trap 'rm -f "$error_file"' EXIT
for candidate in "$image" "$baseline"; do
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

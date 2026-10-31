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

# Check out the PR merged into its target branch, so static checks see the
# result of merging instead of a possibly stale PR head.
#
# copy-pr-bot pushes the vetted PR head to `pull-request/<N>`. A check that
# tightened on the target branch after the PR was branched (a new import-linter
# contract, a new lint rule) passes on that head and only fails once merged.
# GitHub keeps a test merge commit for every open PR; this script checks it out
# when its PR parent is exactly the vetted head ($GITHUB_SHA).
#
# Any lookup problem leaves the PR head checked out, which is the behaviour
# without this script: a missing tool, a conflicting PR, a pending mergeability
# computation, or a merge commit built from a newer, not yet vetted head.

set -euo pipefail

warn() {
  echo "::warning title=Checking the PR head only::$1"
  exit 0
}

if [[ ! "${GITHUB_REF:-}" =~ ^refs/heads/pull-request/([1-9][0-9]*)$ ]]; then
  warn "GITHUB_REF '${GITHUB_REF:-}' is not a pull-request/<N> branch."
fi
PR_NUMBER=${BASH_REMATCH[1]}

command -v gh >/dev/null || warn "gh is required to look up the test merge commit."

# GitHub computes mergeability lazily; reading the PR starts it and
# `mergeable` stays null (an empty field here) until it finishes. The base
# ref goes last so `read` keeps any `|` in a branch name within that field.
attempts=${MERGE_LOOKUP_ATTEMPTS:-6}
for ((i = 1; i <= attempts; i++)); do
  fields=$(gh api "repos/$GITHUB_REPOSITORY/pulls/$PR_NUMBER" --jq "[.mergeable, .merge_commit_sha, .base.ref] | map(if . == null then \"\" else tostring end) | join(\"|\")") ||
    warn "Could not read PR #$PR_NUMBER."
  IFS='|' read -r mergeable merge_sha base_ref <<<"$fields"
  [[ -n "$mergeable" ]] && break
  if ((i < attempts)); then sleep "${MERGE_LOOKUP_DELAY:-5}"; fi
done
[[ "$mergeable" == "true" ]] || warn "PR #$PR_NUMBER has no clean merge with $base_ref (mergeable=${mergeable:-pending})."
[[ "$merge_sha" =~ ^[0-9a-f]{40}$ ]] || warn "PR #$PR_NUMBER has no test merge commit."

git fetch --no-tags --depth=1 origin "refs/pull/$PR_NUMBER/merge" || warn "Could not fetch refs/pull/$PR_NUMBER/merge."
fetched=$(git rev-parse FETCH_HEAD)
[[ "$fetched" == "$merge_sha" ]] || warn "refs/pull/$PR_NUMBER/merge moved during lookup ($fetched != $merge_sha)."

# Read the parents from the raw commit: a depth-1 fetch hides them from rev-parse.
read -r -a parents <<<"$(git cat-file -p "$merge_sha" | sed -n "s/^parent //p" | tr "\n" " ")"
[[ ${#parents[@]} -eq 2 && "${parents[1]}" == "$GITHUB_SHA" ]] ||
  warn "The test merge commit was built from ${parents[1]:-no PR parent}, not the vetted head $GITHUB_SHA."

git checkout --quiet --detach "$merge_sha"
echo "Checking PR #$PR_NUMBER head $GITHUB_SHA merged into $base_ref at ${parents[0]} (merge commit $merge_sha)."

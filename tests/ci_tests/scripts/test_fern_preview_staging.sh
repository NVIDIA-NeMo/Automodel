#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Exercise the actual token-free staging step with a PR older than main.
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
FIXTURE=$(mktemp -d)
trap 'rm -rf "$FIXTURE"' EXIT
sed -n '/^      - name: Prepare preview docs$/,/^      - name: Setup Node.js$/p' \
  "$REPO_ROOT/.github/workflows/fern-docs-preview.yml" \
  | sed '1,/^        run: |$/d; /^      - name:/,$d; s/^          //' > "$FIXTURE/stage.sh"
test -s "$FIXTURE/stage.sh"
cd "$FIXTURE"
export RUNNER_TEMP="$FIXTURE/runner"
for SOURCE in trusted-source pr-source; do
  mkdir -p "$SOURCE/docs/fern/components" "$SOURCE/docs/model-coverage/diffusion/qwen"
done
printf 'trusted navigation\n' > trusted-source/docs/fern/docs.yml
printf 'PR navigation\n' > pr-source/docs/fern/docs.yml
printf 'trusted component\n' > trusted-source/docs/fern/components/example.mdx
printf 'PR component\n' > pr-source/docs/fern/components/example.mdx
printf 'trusted page\n' > trusted-source/docs/guide.mdx
printf 'PR page\n' > pr-source/docs/guide.mdx
printf 'new upstream page\n' > trusted-source/docs/model-coverage/diffusion/qwen/qwen-image-2-1.mdx
printf 'new PR page\n' > pr-source/docs/new.mdx
printf 'PR script\n' > pr-source/docs/unsafe.js
bash stage.sh
DEST="$RUNNER_TEMP/fern-preview/docs"
cmp trusted-source/docs/model-coverage/diffusion/qwen/qwen-image-2-1.mdx \
  "$DEST/model-coverage/diffusion/qwen/qwen-image-2-1.mdx"
cmp pr-source/docs/guide.mdx "$DEST/guide.mdx"
cmp pr-source/docs/new.mdx "$DEST/new.mdx"
cmp trusted-source/docs/fern/docs.yml "$DEST/fern/docs.yml"
cmp trusted-source/docs/fern/components/example.mdx "$DEST/fern/components/example.mdx"
test ! -e "$DEST/unsafe.js"
grep -Fx '{"organization":"nvidia","version":"5.139.0"}' "$DEST/fern/fern.config.json"
rm -rf "$RUNNER_TEMP/fern-preview"
ln -s ../guide.mdx pr-source/docs/link.mdx
if bash stage.sh; then
  echo 'Expected staging to reject symlinked PR content' >&2
  exit 1
fi
echo 'Fern preview staging regression passed'

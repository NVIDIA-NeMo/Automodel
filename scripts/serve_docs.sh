#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/serve_docs.sh [--generate-api] [-- <fern docs dev args>]

Build the disposable Fern docs project from docs/ and .github/fern-template/,
add a local API-reference placeholder, and start a Fern dev server.

By default this does not require Fern login. Use --generate-api to run
'fern docs md generate' and build the real Python library reference, which may
require Fern dashboard provisioning and CLI auth.

Environment:
  FERN_VERSION         Fern CLI version to run via npx (default: 4.62.4)
  FERN_PNPM_VERSION    pnpm version for the local Fern preview (default: 9.15.9)
  FERN_SITE_DIR        Generated Fern project path (default: .fern-build/fern)
  FERN_NPM_GLOBAL_DIR  Writable npm global prefix for Fern's pnpm bootstrap
                       (default: .fern-build/npm-global)

Examples:
  scripts/serve_docs.sh
  scripts/serve_docs.sh --generate-api
  scripts/serve_docs.sh -- --port 3003
EOF
}

generate_api=0
dev_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    --generate-api)
      generate_api=1
      shift
      ;;
    --skip-generate)
      # Backward-compatible no-op; local preview skips API generation by default.
      shift
      ;;
    --)
      shift
      dev_args+=("$@")
      break
      ;;
    *)
      dev_args+=("$1")
      shift
      ;;
  esac
done

dev_port=""
for ((i = 0; i < ${#dev_args[@]}; i++)); do
  case "${dev_args[$i]}" in
    --port)
      if ((i + 1 < ${#dev_args[@]})); then
        dev_port="${dev_args[$((i + 1))]}"
      fi
      ;;
    --port=*)
      dev_port="${dev_args[$i]#--port=}"
      ;;
  esac
done

if [[ -n "${dev_port}" && ! "${dev_port}" =~ ^[0-9]+$ ]]; then
  echo "Invalid Fern docs dev port: ${dev_port}" >&2
  exit 1
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
fern_site_dir="${FERN_SITE_DIR:-"${repo_root}/.fern-build/fern"}"
fern_version="${FERN_VERSION:-4.62.4}"
pnpm_version="${FERN_PNPM_VERSION:-9.15.9}"
npm_global_dir="${FERN_NPM_GLOBAL_DIR:-"${repo_root}/.fern-build/npm-global"}"
fern_cmd=(npx -y "fern-api@${fern_version}")

if ! command -v npx >/dev/null 2>&1; then
  echo "npx is required to run the pinned Fern CLI. Install Node.js/npm first." >&2
  exit 1
fi

if [[ -n "${dev_port}" ]] && ss -ltn "( sport = :${dev_port} )" | awk 'NR > 1 { found = 1 } END { exit found ? 0 : 1 }'; then
  echo "Port ${dev_port} is already in use; not rebuilding ${fern_site_dir} while a server may be watching it." >&2
  echo "Stop the existing server or choose another port, for example:" >&2
  echo "  pkill -f 'fern docs dev --port ${dev_port}'" >&2
  echo "  scripts/serve_docs.sh -- --port $((dev_port + 1))" >&2
  exit 1
fi

mkdir -p "${npm_global_dir}/bin"
export NPM_CONFIG_PREFIX="${NPM_CONFIG_PREFIX:-"${npm_global_dir}"}"
export PATH="${NPM_CONFIG_PREFIX}/bin:${PATH}"

current_pnpm_version="$(pnpm --version 2>/dev/null || true)"
if [[ "${current_pnpm_version}" != "${pnpm_version}" ]]; then
  echo "Installing pnpm@${pnpm_version} into ${NPM_CONFIG_PREFIX} for Fern local preview."
  npm install -g "pnpm@${pnpm_version}" >/dev/null
fi

bash "${repo_root}/.github/scripts/prepare-fern-docs.sh" "${repo_root}" "${fern_site_dir}"

cd "${fern_site_dir}"

mkdir -p product-docs/nemo-automodel/Full-Library-Reference
cat > product-docs/nemo-automodel/Full-Library-Reference/index.mdx <<'EOF'
---
title: "Full Library Reference"
description: ""
---

The full Python API reference is generated during authenticated Fern preview
and publish builds.

For local prose previews, this placeholder avoids requiring Fern login.
EOF

if [[ "${generate_api}" -eq 1 ]]; then
  if ! "${fern_cmd[@]}" docs md generate; then
    status=$?
    echo "" >&2
    echo "Fern library-reference generation failed (exit ${status})." >&2
    echo "If the error mentions organization permissions, run:" >&2
    echo "" >&2
    echo "  make -f .github/fern-template/Makefile docs-login" >&2
    echo "" >&2
    exit "${status}"
  fi
else
  echo "Skipping API reference generation for no-login local preview."
  echo "Run scripts/serve_docs.sh --generate-api to build the full library reference."
fi

echo ""
echo "Starting Fern dev server. Fern will print the exact local URL."
echo ""

exec "${fern_cmd[@]}" docs dev "${dev_args[@]}"

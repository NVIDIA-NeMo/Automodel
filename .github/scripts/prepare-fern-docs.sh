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

if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [repo-root] [output-fern-root]" >&2
  exit 2
fi

if [[ $# -ge 1 ]]; then
  repo_root="$1"
else
  repo_root="$(git rev-parse --show-toplevel)"
fi

if [[ $# -eq 2 ]]; then
  output_root="$2"
else
  output_root="${repo_root}/.fern-build/fern"
fi

template_root="${repo_root}/.github/fern-template"
docs_root="${repo_root}/docs"

if [[ ! -d "${template_root}" ]]; then
  echo "Fern template not found: ${template_root}" >&2
  exit 1
fi

if [[ ! -d "${docs_root}" ]]; then
  echo "Docs source not found: ${docs_root}" >&2
  exit 1
fi

case "${output_root}" in
  "" | "/" | "${repo_root}" | "${repo_root}/docs" | "${template_root}")
    echo "Refusing unsafe output path: ${output_root}" >&2
    exit 1
    ;;
esac

rm -rf "${output_root}"
mkdir -p "${output_root}"
cp -a "${template_root}/." "${output_root}/"

rm -rf "${output_root}/versions/latest/pages"
mkdir -p "${output_root}/versions/latest/pages"
cp -a "${docs_root}/." "${output_root}/versions/latest/pages/"

echo "Prepared Fern docs project at ${output_root}"

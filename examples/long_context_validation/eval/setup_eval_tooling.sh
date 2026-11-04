#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
# Phase 3 step 0 — set up the SWE-bench eval tooling (login node).
# Creates an isolated Python 3.10 venv (mini-swe-agent + swebench need >=3.10; the
# login default is 3.9) and installs the eval tools. Idempotent: re-running only
# upgrades. These are external eval tools, NOT part of the AutoModel package.
set -euxo pipefail

CACHE=/path/to/coderforge_cache
VENV="${CACHE}/eval_venv"
PY310="${PY310:-/usr/bin/python3.10}"

"${PY310}" -m venv "${VENV}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

python --version           # expect 3.10.x
pip install --upgrade pip
pip install "mini-swe-agent" "swebench"

# Sanity: import + report versions.
python - <<'PY'
import minisweagent, swebench
print("mini-swe-agent:", minisweagent.__version__)
print("swebench:", swebench.__version__)
PY
echo "=== venv ready at ${VENV} ==="

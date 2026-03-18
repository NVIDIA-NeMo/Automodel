#!/usr/bin/env bash
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

# Stage 5: Inference quality analysis.
#
# Analyzes lm_eval sample outputs for generation failure modes:
# death looping, missing EOS, empty responses, abrupt endings.
# Optionally exports structured JSON and applies threshold-based quality gate.
#
# Usage:
#   bash examples/convergence/tulu3/inference/run.sh results/
#   bash examples/convergence/tulu3/inference/run.sh results/ --threshold-file examples/convergence/tulu3/eval/thresholds.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /opt/venv/bin/activate

if [ $# -lt 1 ]; then
    echo "Usage: $0 <results_dir> [extra args...]"
    echo ""
    echo "Examples:"
    echo "  $0 results/"
    echo "  $0 results/ --threshold-file examples/convergence/tulu3/eval/thresholds.yaml"
    echo "  $0 results/ --export results/quality_report.json"
    exit 1
fi

RESULTS_DIR="$1"
shift

echo "=== Stage 5: Inference Quality Analysis ==="
echo "  Results: ${RESULTS_DIR}"
echo ""

python "${SCRIPT_DIR}/analyze_quality.py" \
    "${RESULTS_DIR}" \
    "$@"

echo ""
echo "=== Stage 5 complete ==="
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

# Check truncation rates for Tulu-3 SFT mixture at multiple sequence lengths.
#
# Wraps scripts/check_truncation.py with Tulu-3 defaults.
# Override any default via environment variables or append extra CLI args.
#
# Usage:
#   ./check_truncation.sh
#   DATASET=my/dataset MODEL=my/model ./check_truncation.sh
#   ./check_truncation.sh --num_samples 5000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults (override via env vars)
DATASET="${DATASET:-allenai/tulu-3-sft-mixture}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
SEQ_LENGTHS="${SEQ_LENGTHS:-1024 2048 4096 8192}"
SPLIT="${SPLIT:-train}"

exec python "${SCRIPT_DIR}/check_truncation.py" \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --seq_length ${SEQ_LENGTHS} \
    --split "${SPLIT}" \
    "$@"
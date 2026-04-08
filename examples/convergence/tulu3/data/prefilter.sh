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

# Pre-filter Tulu-3 SFT mixture to remove samples exceeding seq_length.
#
# Wraps scripts/prefilter_dataset.py with Tulu-3 defaults.
# Caches the filtered dataset to data/cached/ (relative to this script).
# Override any default via environment variables or append extra CLI args.
#
# Usage:
#   ./prefilter.sh
#   SEQ_LENGTH=2048 ./prefilter.sh
#   ./prefilter.sh --benchmark --num_benchmark_samples 5000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults (override via env vars)
DATASET="${DATASET:-allenai/tulu-3-sft-mixture}"
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
SEQ_LENGTH="${SEQ_LENGTH:-1024}"
SPLIT="${SPLIT:-train}"
STRATEGIES="${STRATEGIES:-text_only}"
CACHE_DIR="${CACHE_DIR:-${SCRIPT_DIR}/cached}"

exec python "${SCRIPT_DIR}/prefilter_dataset.py" \
    --dataset "${DATASET}" \
    --model "${MODEL}" \
    --seq_length "${SEQ_LENGTH}" \
    --split "${SPLIT}" \
    --strategies ${STRATEGIES} \
    --cache_dir "${CACHE_DIR}" \
    "$@"
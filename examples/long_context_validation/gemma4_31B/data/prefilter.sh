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

# Parse, clean, length-analyze and prefilter the CoderForge SFT dataset for Gemma4.
#
# Wraps prefilter_dataset.py with CoderForge defaults. Caches JSONL to data/cached/.
# Override any default via environment variables or append extra CLI args.
#
# Usage:
#   MODEL=/path/to/hf_gemma4_31b ./prefilter.sh                 # analyze + coverage curve
#   MODEL=/path/to/hf_gemma4_31b SEQ_LENGTH=32768 ./prefilter.sh # also write a training cache
#   MODEL=/path/to/hf_gemma4_31b ./prefilter.sh --max_samples 20 # smoke test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults (override via env vars). MODEL must point at the Gemma4 checkpoint dir.
MODEL="${MODEL:?Set MODEL to the Gemma4 checkpoint dir, e.g. /path/to/hf_gemma4_31b}"
DATASET="${DATASET:-togethercomputer/CoderForge-Preview}"
NAME="${NAME:-trajectories}"
SPLIT="${SPLIT:-filtered_reward1}"
CACHE_DIR="${CACHE_DIR:-${SCRIPT_DIR}/cached}"

CMD=(python "${SCRIPT_DIR}/prefilter_dataset.py"
    --dataset "${DATASET}"
    --name "${NAME}"
    --split "${SPLIT}"
    --model "${MODEL}"
    --cache_dir "${CACHE_DIR}")

# SEQ_LENGTH is optional: omit it to only run the coverage analysis.
if [[ -n "${SEQ_LENGTH:-}" ]]; then
    CMD+=(--seq_length "${SEQ_LENGTH}")
fi

exec "${CMD[@]}" "$@"

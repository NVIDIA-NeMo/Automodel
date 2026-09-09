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

# Parse, clean, length-analyze and prefilter the CoderForge SFT dataset for Qwen3-32B.
#
# Wraps prefilter_dataset.py with CoderForge + Qwen3 defaults: the Qwen3 tokenizer and the
# tools+generation chat template, so ``n_tokens`` matches the exact training render. Caches JSONL
# to data/cached/. Run inside an env with nemo_automodel + transformers + datasets (the
# nemo-automodel container or a matching venv). Override any default via environment variables.
#
# Usage:
#   MODEL=Qwen/Qwen3-32B ./prefilter.sh                         # analyze + retention curve
#   MODEL=Qwen/Qwen3-32B SEQ_LENGTH=131072 ./prefilter.sh       # also write the 128K training cache
#   MODEL=Qwen/Qwen3-32B ./prefilter.sh --max_samples 20        # smoke test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Defaults (override via env vars). MODEL points at the Qwen3-32B checkpoint dir or HF id.
MODEL="${MODEL:?Set MODEL to the Qwen3-32B checkpoint dir or HF id, e.g. Qwen/Qwen3-32B}"
DATASET="${DATASET:-togethercomputer/CoderForge-Preview}"
NAME="${NAME:-trajectories}"
SPLIT="${SPLIT:-filtered_reward1}"
# The tools+generation chat template — matches the training render so n_tokens is exact.
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${SCRIPT_DIR}/../qwen3_coderforge_chat_template.jinja}"
CACHE_DIR="${CACHE_DIR:-${SCRIPT_DIR}/cached}"

CMD=(python "${SCRIPT_DIR}/prefilter_dataset.py"
    --dataset "${DATASET}"
    --name "${NAME}"
    --split "${SPLIT}"
    --model "${MODEL}"
    --chat_template "${CHAT_TEMPLATE}"
    --cache_dir "${CACHE_DIR}")

# SEQ_LENGTH is optional: omit it to only run the coverage analysis.
if [[ -n "${SEQ_LENGTH:-}" ]]; then
    CMD+=(--seq_length "${SEQ_LENGTH}")
fi

exec "${CMD[@]}" "$@"

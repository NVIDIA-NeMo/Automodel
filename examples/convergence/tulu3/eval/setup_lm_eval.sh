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

# Setup lm-evaluation-harness for running eval benchmarks.
# Idempotent: skips clone/install if already present.
#
# Usage:
#   bash setup_lm_eval.sh [INSTALL_DIR]
#
# Default install directory: /opt/lm-evaluation-harness

set -euo pipefail

INSTALL_DIR="${1:-/opt/lm-evaluation-harness}"
VENV_DIR="$INSTALL_DIR/.venv"

# ── Clone ────────────────────────────────────────────────────────────────────
if [ -d "$INSTALL_DIR/.git" ]; then
  echo "lm-evaluation-harness already cloned at $INSTALL_DIR, skipping."
else
  echo "Cloning lm-evaluation-harness to $INSTALL_DIR ..."
  # CI nodes intermittently get an auth challenge instead of the repo ("could not read
  # Username for 'https://github.com'"); retry, removing the partial checkout so the
  # `.git` probe above cannot short-circuit it.
  cloned=0
  for attempt in 1 2 3; do
    if git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness "$INSTALL_DIR"; then
      cloned=1
      break
    fi
    echo "clone attempt ${attempt}/3 failed; retrying"
    rm -rf "$INSTALL_DIR"
    sleep $((attempt * 10))
  done
  if [ "$cloned" -ne 1 ]; then
    echo "ERROR: could not clone lm-evaluation-harness after 3 attempts" >&2
    exit 1
  fi
fi

# ── Virtual-env & dependencies ───────────────────────────────────────────────
if [ -f "$VENV_DIR/bin/activate" ]; then
  echo "Virtual environment already exists at $VENV_DIR, skipping install."
else
  echo "Creating virtual environment and installing dependencies ..."
  unset UV_PROJECT_ENVIRONMENT 2>/dev/null || true

  cd "$INSTALL_DIR"
  uv venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  uv pip install langdetect immutabledict wonderwords nltk
  # Floor the eval stack: gemma4's head_dim makes vLLM auto-select the FA4 cute kernel
  # (vllm_flash_attn.cute). On older cutlass-dsl (the 4.5.x that vllm 0.25.x resolves
  # transitively) that kernel fails to compile -> GPUModuleOp / cudaErrorIllegalAddress.
  # vllm 0.26.0 + cutlass-dsl 4.6.0 compiles it cleanly (verified: gemma4 IFEval
  # prompt_level_strict_acc=0.5360).
  # transformers 5.15.0 makes gemma4's per-layer head_dim raise
  # AmbiguousGlobalPerLayerAttributeError on the plain getattr in vLLM's get_head_size(),
  # so pin transformers too. 5.14.1 + vllm 0.26.0 is the combination we verified.
  uv pip install -e ".[vllm]" "vllm==0.26.0" "transformers==5.14.1" "nvidia-cutlass-dsl>=4.6.0"

  echo "Setup complete. Activate with:  source $VENV_DIR/bin/activate"
fi
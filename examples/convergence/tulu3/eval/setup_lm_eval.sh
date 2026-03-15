#!/bin/bash
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
  git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness "$INSTALL_DIR"
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
  uv pip install -e ".[vllm]"

  echo "Setup complete. Activate with:  source $VENV_DIR/bin/activate"
fi

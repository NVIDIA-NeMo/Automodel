#!/usr/bin/env bash

# Exit on error and undefined variables
set -euo pipefail

# Check that exactly one argument was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <folder-path>"
    exit 1
fi

AUTOMODEL_DIR="${1%/}"

# Check that the argument is an existing directory
if [ ! -d "$AUTOMODEL_DIR" ]; then
    echo "Error: '$AUTOMODEL_DIR' is not a directory or does not exist."
    exit 1
fi

UV_PYTORCH_ARGS="$AUTOMODEL_DIR/docker/common/uv-pytorch.toml"
UV_PYTORCH_LOCK="$AUTOMODEL_DIR/docker/common/uv-pytorch.lock"
SED_SCRIPT="/\\[tool\\.uv\\]/r $UV_PYTORCH_ARGS"

# Inject additional uv configurations into pyproject.toml
sed -i "$SED_SCRIPT" pyproject.toml

# Update uv lock with additonal uv configurations
cp $UV_PYTORCH_LOCK $AUTOMODEL_DIR/uv.lock

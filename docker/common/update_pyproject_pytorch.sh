# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#!/usr/bin/env bash

#######################################
##
## Script to update pyproject.toml and uv.lock.
## Updates uv configuration with override dependnecies necessary for working with PyTorch base image.
##
## Input: Absolute path to Automodel directory
##
## Example: bash docker/common/update_pyproject_pytorch.sh /opt/Automodel
##
########################################

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

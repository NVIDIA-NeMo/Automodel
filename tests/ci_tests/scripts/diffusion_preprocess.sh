# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

# Stage 1: Download modal-labs/dissolve dataset and encode into latents
# via VAE + text encoders using the preprocessing_multiprocess tool.
#
# Environment variables (set by CI template from recipe ci: section):
#   PIPELINE_DIR      - Pipeline output root directory
#   TEST_NAME         - Name of the test (used for subdirectory)
#   PROCESSOR_NAME    - Processor name: "wan" or "hunyuan"
#   TARGET_FRAMES     - Target frame count for video encoding (e.g., 17)
set -e

RAW_DIR="$PIPELINE_DIR/$TEST_NAME/raw_data"
CACHE_DIR="$PIPELINE_DIR/$TEST_NAME/data_cache"

# Download dataset from HuggingFace
echo "[Stage 1] Downloading modal-labs/dissolve dataset..."
python tests/ci_tests/scripts/download_diffusion_test_data.py \
    --output-dir "$RAW_DIR"

echo "[Stage 1] Raw data contents:"
ls -la "$RAW_DIR"

# Encode videos into latents via VAE + text encoders
echo "[Stage 1] Preprocessing with processor=${PROCESSOR_NAME}..."
python -m tools.diffusion.preprocessing_multiprocess video \
    --video_dir "$RAW_DIR" \
    --output_dir "$CACHE_DIR" \
    --processor "$PROCESSOR_NAME" \
    --resolution_preset 512p \
    --caption_format sidecar \
    --caption_field caption \
    ${TARGET_FRAMES:+--target_frames $TARGET_FRAMES}

echo "[Stage 1] Preprocessing complete. Cache dir contents:"
ls -la "$CACHE_DIR"

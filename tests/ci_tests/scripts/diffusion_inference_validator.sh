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

# Stage 3: Generate video from a finetuned diffusion checkpoint and validate output.
#
# Maps the finetune CONFIG_PATH to the corresponding generation config,
# runs inference with minimal settings for CI speed, and validates the output.
#
# Environment variables (set by CI template / diffusion_launcher.sh):
#   CONFIG_PATH   - Path to finetune recipe YAML (used to determine model type)
#   PIPELINE_DIR  - Pipeline output root directory
#   TEST_NAME     - Name of the test
set -e

CKPT_DIR="$PIPELINE_DIR/$TEST_NAME/checkpoint"
OUTPUT_DIR="$PIPELINE_DIR/$TEST_NAME/inference_outputs"

# Map finetune config to generation config
if [[ "$CONFIG_PATH" == *"wan"* ]]; then
  GEN_CONFIG="examples/diffusion/generate/configs/generate_wan.yaml"
elif [[ "$CONFIG_PATH" == *"hunyuan"* ]]; then
  GEN_CONFIG="examples/diffusion/generate/configs/generate_hunyuan.yaml"
else
  echo "ERROR: Cannot determine model type from CONFIG_PATH=$CONFIG_PATH"
  exit 1
fi

echo "[Stage 3] Running inference with config=$GEN_CONFIG checkpoint=$CKPT_DIR"

cd /opt/Automodel

# Run generation with minimal settings for CI speed
python examples/diffusion/generate/generate.py \
    -c "$GEN_CONFIG" \
    --model.checkpoint "$CKPT_DIR" \
    --inference.max_samples 1 \
    --inference.num_inference_steps 10 \
    --output.output_dir "$OUTPUT_DIR" \
    --seed 42

# Validate output files
echo "[Stage 3] Validating inference outputs..."
python tests/ci_tests/utils/validate_diffusion_inference.py \
    --output-dir "$OUTPUT_DIR"

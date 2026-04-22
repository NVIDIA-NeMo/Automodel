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

set -euo pipefail

# Environment variables expected from CI template:
#   CONFIG_PATH, TEST_LEVEL, NPROC_PER_NODE, TEST_NODE_COUNT,
#   MASTER_ADDR, MASTER_PORT, SLURM_JOB_ID, PIPELINE_DIR, TEST_NAME

DATA_DIR="$PIPELINE_DIR/$TEST_NAME/data"
CKPT_DIR="$PIPELINE_DIR/$TEST_NAME/checkpoint"
INFER_DIR="$PIPELINE_DIR/$TEST_NAME/inference_output"

cd /opt/Automodel

# ============================================
# Derive model-specific settings from config
# ============================================
RECIPE_NAME=$(basename "$CONFIG_PATH" .yaml)
case "$RECIPE_NAME" in
    wan2_1_t2v_flow*)
        PROCESSOR="wan"
        GENERATE_CONFIG="examples/diffusion/generate/configs/generate_wan.yaml"
        MODEL_NAME="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        INFER_NUM_FRAMES=9
        PREPROCESS_EXTRA_ARGS=""
        ;;
    hunyuan_t2v_flow*)
        PROCESSOR="hunyuan"
        GENERATE_CONFIG="examples/diffusion/generate/configs/generate_hunyuan.yaml"
        MODEL_NAME="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"
        INFER_NUM_FRAMES=5
        PREPROCESS_EXTRA_ARGS="--target_frames 13"
        ;;
    *)
        echo "ERROR: Unknown recipe '$RECIPE_NAME'. Add a case to diffusion_finetune_launcher.sh."
        exit 1
        ;;
esac
echo "[config] Recipe=$RECIPE_NAME  Processor=$PROCESSOR  Model=$MODEL_NAME"

# ============================================
# Stage 1: Download dissolve dataset
# ============================================
echo "============================================"
echo "[data] Downloading dissolve dataset..."
echo "============================================"
uv run --extra diffusion python -c "
from huggingface_hub import snapshot_download
snapshot_download('modal-labs/dissolve', repo_type='dataset', local_dir='$DATA_DIR/raw')
print('Dataset downloaded successfully')
"

# ============================================
# Stage 2: Preprocess videos to latents
# ============================================
echo "============================================"
echo "[preprocess] Converting videos to latents..."
echo "============================================"
uv run --extra diffusion python -m tools.diffusion.preprocessing_multiprocess video \
    --video_dir "$DATA_DIR/raw" \
    --output_dir "$DATA_DIR/cache" \
    --processor "$PROCESSOR" \
    --resolution_preset 512p \
    --caption_format sidecar \
    $PREPROCESS_EXTRA_ARGS

# ============================================
# Stage 3: Finetune
# ============================================
echo "============================================"
echo "[finetune] Running finetuning..."
echo "============================================"
CONFIG="--config /opt/Automodel/${CONFIG_PATH} \
    --data.dataloader.cache_dir $DATA_DIR/cache \
    --checkpoint.checkpoint_dir $CKPT_DIR \
    --step_scheduler.max_steps ${MAX_STEPS:-100} \
    --step_scheduler.ckpt_every_steps 100 \
    --step_scheduler.save_checkpoint_every_epoch false \
    --fsdp.dp_size ${NPROC_PER_NODE} \
    --wandb.mode disabled"

CMD="uv run --extra diffusion torchrun --nproc-per-node=${NPROC_PER_NODE} \
              --nnodes=${TEST_NODE_COUNT} \
              --rdzv_backend=c10d \
              --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
              --rdzv_id=${SLURM_JOB_ID}"

eval $CMD examples/diffusion/finetune/finetune.py $CONFIG

# ============================================
# Stage 4: Inference smoke test
# ============================================
echo "============================================"
echo "[inference] Running inference smoke test..."
echo "============================================"
CKPT_STEP_DIR=$(ls -d $CKPT_DIR/epoch_*_step_* | sort -t_ -k4 -n | tail -1)

uv run --extra diffusion python examples/diffusion/generate/generate.py \
    --config "$GENERATE_CONFIG" \
    --model.pretrained_model_name_or_path "$MODEL_NAME" \
    --model.checkpoint "$CKPT_STEP_DIR" \
    --inference.num_inference_steps 5 \
    --inference.pipeline_kwargs.num_frames "$INFER_NUM_FRAMES" \
    --output.output_dir "$INFER_DIR" \
    --vae.enable_slicing true \
    --vae.enable_tiling true

# Verify output
if ls $INFER_DIR/sample_*.mp4 1>/dev/null 2>&1; then
    echo "[inference] SUCCESS: Output video(s) generated"
else
    echo "[inference] FAILURE: No output videos found"
    exit 1
fi

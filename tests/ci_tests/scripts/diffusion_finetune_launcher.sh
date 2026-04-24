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
        MEDIA_TYPE="video"
        PROCESSOR="wan"
        GENERATE_CONFIG="examples/diffusion/generate/configs/generate_wan.yaml"
        MODEL_NAME="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        INFER_NUM_FRAMES=9
        PREPROCESS_EXTRA_ARGS=""
        ;;
    hunyuan_t2v_flow*)
        MEDIA_TYPE="video"
        PROCESSOR="hunyuan"
        GENERATE_CONFIG="examples/diffusion/generate/configs/generate_hunyuan.yaml"
        MODEL_NAME="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v"
        INFER_NUM_FRAMES=5
        PREPROCESS_EXTRA_ARGS="--target_frames 13"
        ;;
    flux_t2i_flow*)
        MEDIA_TYPE="image"
        PROCESSOR="flux"
        GENERATE_CONFIG="examples/diffusion/generate/configs/generate_flux.yaml"
        MODEL_NAME="black-forest-labs/FLUX.1-dev"
        PREPROCESS_EXTRA_ARGS=""
        ;;
    *)
        echo "ERROR: Unknown recipe '$RECIPE_NAME'. Add a case to diffusion_finetune_launcher.sh."
        exit 1
        ;;
esac

# LoRA recipes are named *_lora.yaml — they save PEFT adapter artifacts
# (adapter_model.safetensors + adapter_config.json) that generate.py loads via
# model.lora_weights, not model.checkpoint.
IS_LORA=false
if [[ "$RECIPE_NAME" == *_lora ]]; then
    IS_LORA=true
fi
echo "[config] Recipe=$RECIPE_NAME  MediaType=$MEDIA_TYPE  Processor=$PROCESSOR  Model=$MODEL_NAME  LoRA=$IS_LORA"

# ============================================
# Stage 1: Download dataset
# ============================================
echo "============================================"
echo "[data] Downloading dataset..."
echo "============================================"
if [ "$MEDIA_TYPE" = "image" ]; then
    uv run --extra diffusion python -c "
from datasets import load_dataset
from pathlib import Path
import json

ds = load_dataset('diffusers/tuxemon', split='train')
out_dir = Path('$DATA_DIR/raw')
out_dir.mkdir(parents=True, exist_ok=True)

jsonl_entries = []
for i, row in enumerate(ds):
    fname = f'tuxemon_sample_{i:04d}.png'
    row['image'].save(out_dir / fname)
    jsonl_entries.append({'file_name': fname, 'internvl': row['gpt4_turbo_caption']})

jsonl_path = out_dir / 'tuxemon_internvl.json'
with open(jsonl_path, 'w') as jf:
    for entry in jsonl_entries:
        jf.write(json.dumps(entry) + '\n')

print(f'Extracted {len(ds)} images to {out_dir}')
"
else
    uv run --extra diffusion python -c "
from huggingface_hub import snapshot_download
snapshot_download('modal-labs/dissolve', repo_type='dataset', local_dir='$DATA_DIR/raw')
print('Dataset downloaded successfully')
"
fi

# ============================================
# Stage 2: Preprocess to latents
# ============================================
echo "============================================"
echo "[preprocess] Converting ${MEDIA_TYPE}s to latents..."
echo "============================================"
if [ "$MEDIA_TYPE" = "image" ]; then
    uv run --extra diffusion python -m tools.diffusion.preprocessing_multiprocess image \
        --image_dir "$DATA_DIR/raw" \
        --output_dir "$DATA_DIR/cache" \
        --processor "$PROCESSOR" \
        $PREPROCESS_EXTRA_ARGS
else
    uv run --extra diffusion python -m tools.diffusion.preprocessing_multiprocess video \
        --video_dir "$DATA_DIR/raw" \
        --output_dir "$DATA_DIR/cache" \
        --processor "$PROCESSOR" \
        --resolution_preset 512p \
        --caption_format sidecar \
        $PREPROCESS_EXTRA_ARGS
fi

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

# generate.py loads LoRA adapters from model.lora_weights and consolidated
# full-finetune checkpoints from model.checkpoint — they are distinct code paths.
# PEFT saves adapter_model.safetensors + adapter_config.json under the `model/`
# subdirectory of the step checkpoint; generate.py reads those files from the
# directory passed to --model.lora_weights without descending further.
if [ "$IS_LORA" = "true" ]; then
    CKPT_FLAG="--model.lora_weights"
    CKPT_STEP_DIR="$CKPT_STEP_DIR/model"
else
    CKPT_FLAG="--model.checkpoint"
fi

if [ "$MEDIA_TYPE" = "image" ]; then
    uv run --extra diffusion python examples/diffusion/generate/generate.py \
        --config "$GENERATE_CONFIG" \
        --model.pretrained_model_name_or_path "$MODEL_NAME" \
        $CKPT_FLAG "$CKPT_STEP_DIR" \
        --inference.num_inference_steps 5 \
        --output.output_dir "$INFER_DIR" \
        --vae.enable_slicing true \
        --vae.enable_tiling true

    if ls $INFER_DIR/sample_*.png 1>/dev/null 2>&1; then
        echo "[inference] SUCCESS: Output image(s) generated"
    else
        echo "[inference] FAILURE: No output images found"
        exit 1
    fi
else
    uv run --extra diffusion python examples/diffusion/generate/generate.py \
        --config "$GENERATE_CONFIG" \
        --model.pretrained_model_name_or_path "$MODEL_NAME" \
        $CKPT_FLAG "$CKPT_STEP_DIR" \
        --inference.num_inference_steps 5 \
        --inference.pipeline_kwargs.num_frames "$INFER_NUM_FRAMES" \
        --output.output_dir "$INFER_DIR" \
        --vae.enable_slicing true \
        --vae.enable_tiling true

    if ls $INFER_DIR/sample_*.mp4 1>/dev/null 2>&1; then
        echo "[inference] SUCCESS: Output video(s) generated"
    else
        echo "[inference] FAILURE: No output videos found"
        exit 1
    fi
fi

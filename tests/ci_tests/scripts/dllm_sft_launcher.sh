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
# Optional:
#   MAX_STEPS, LOCAL_BATCH_SIZE, CONFIG_NPROC_PER_NODE

DATA_DIR="$PIPELINE_DIR/$TEST_NAME/data"
CKPT_DIR="$PIPELINE_DIR/$TEST_NAME/checkpoint"
GEN_DIR="$PIPELINE_DIR/$TEST_NAME/generation"
TEST_NODE_COUNT="${TEST_NODE_COUNT:-1}"

cd /opt/Automodel

# dLLM SFT recipes (DiffusionLMSFTRecipe / DiffusionGemmaSFTRecipe) are LLM-SFT
# subclasses: they train a text model as a denoiser rather than a next-token
# predictor.  A single entry point -- examples/dllm_sft/finetune.py -- selects
# the recipe class from the config's `recipe:` field, so every dLLM recipe
# launches the same way.  This is a train -> generate functional smoke test: a
# few SFT steps, then a short generation from the trained checkpoint (analogous
# to the finetune -> inference flow in diffusion_finetune_launcher.sh).

# W&B: dLLM recipe YAMLs ship `wandb.enable: false`, but disable it globally too
# so CI never attempts a login/network call regardless of the config.
export WANDB_MODE=disabled

# Per-config GPUs/node override (recipe ci.nproc_per_node -> CONFIG_NPROC_PER_NODE)
NPROC_PER_NODE=${CONFIG_NPROC_PER_NODE:-$NPROC_PER_NODE}

RECIPE_NAME=$(basename "$CONFIG_PATH" .yaml)

# ============================================
# Map recipe -> generation sampler
# ============================================
# examples/dllm_generate/generate.py ships only two sampler presets:
#   - llada    : pure MDLM, full-forward no-cache (LLaDA family)
#   - nemotron : block-diffusion KV-cache via model.generate (Nemotron-Labs)
# Recipes with no preset (dflash / diffusion_gemma) run TRAIN-ONLY: the generate
# stage is skipped rather than guessed.
SAMPLER=""
case "$RECIPE_NAME" in
    llada_sft|llada2_sft)        SAMPLER="llada" ;;
    nemotron_labs_diffusion_sft) SAMPLER="nemotron" ;;
    *)                           SAMPLER="" ;;
esac

echo "[config] Recipe=$RECIPE_NAME  GPUs/node=$NPROC_PER_NODE  Nodes=$TEST_NODE_COUNT  Sampler=${SAMPLER:-<none, train-only>}  Level=${TEST_LEVEL:-nightly}"

# ============================================
# Stage 1: Dataset prep (config-specific)
# ============================================
# Most recipes stream a public HuggingFace dataset (e.g. allenai/tulu-3-sft-mixture)
# and need no prep.  The DiffusionGemma recipes consume a local OpenAI-messages
# JSONL built from GSM8K via examples/dllm_sft/prep_gsm8k.py -- materialize it
# into the per-test sandbox and point the recipe at it.
DATASET_ARGS=""
case "$RECIPE_NAME" in
    diffusion_gemma_*)
        echo "============================================"
        echo "[data] Preparing GSM8K chat JSONL for $RECIPE_NAME..."
        echo "============================================"
        mkdir -p "$DATA_DIR"
        GSM8K_JSONL="$DATA_DIR/gsm8k_chat_train.jsonl"
        python examples/dllm_sft/prep_gsm8k.py --output "$GSM8K_JSONL"
        DATASET_ARGS="--dataset.path_or_dataset_id $GSM8K_JSONL"
        ;;
esac

# ============================================
# Stage 2: Finetune (smoke)
# ============================================
# Cap optimizer steps (MAX_STEPS) so only a few batches run.  When we will
# generate, persist a CONSOLIDATED HF checkpoint so generate.py can reload it:
# ckpt_every_steps=max_steps + save_checkpoint_every_epoch=false write exactly
# one checkpoint, at the final step (save_consolidated=final emits the HF dir).
# Otherwise disable saving to keep the train-only smoke light.
if [[ -n "$SAMPLER" ]]; then
    CKPT_ARGS="--checkpoint.enabled true \
        --checkpoint.checkpoint_dir $CKPT_DIR \
        --checkpoint.model_save_format safetensors \
        --checkpoint.save_consolidated final \
        --step_scheduler.ckpt_every_steps ${MAX_STEPS:-10} \
        --step_scheduler.save_checkpoint_every_epoch false"
else
    CKPT_ARGS="--checkpoint.enabled false --checkpoint.checkpoint_dir $CKPT_DIR"
fi

# lr_warmup_steps must stay below the capped max_steps (scheduler asserts
# warmup < decay); recipe values (e.g. nemotron's 50) assume a full run.
# group_by_length precomputes token lengths over the whole dataset (~86 min on
# tulu-3's 939k samples) -- pointless for a few-step smoke, and it blows the
# CI wall clock.  The recipe delivers group_by_length=false safely even for
# recipes that never set it (train_ft.py pops the key before the dataloader
# is instantiated).
CONFIG="--config /opt/Automodel/${CONFIG_PATH} \
    --step_scheduler.max_steps ${MAX_STEPS:-10} \
    --lr_scheduler.lr_warmup_steps 1 \
    --dataloader.group_by_length false \
    ${CKPT_ARGS} \
    ${DATASET_ARGS}"

# Optional per-config local batch size override (recipe ci.local_batch_size).
# NOTE: global_batch_size stays at the recipe value; LOCAL_BATCH_SIZE must divide
# global_batch_size / dp_size or StepScheduler raises.
if [[ -n "${LOCAL_BATCH_SIZE:-}" ]]; then
    CONFIG="${CONFIG} --step_scheduler.local_batch_size ${LOCAL_BATCH_SIZE}"
fi

CMD="torchrun --nproc-per-node=${NPROC_PER_NODE} \
              --nnodes=${TEST_NODE_COUNT} \
              --rdzv_backend=c10d \
              --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
              --rdzv_id=${SLURM_JOB_ID}"

echo "============================================"
echo "[finetune] Running dLLM SFT finetune..."
echo "============================================"
eval $CMD examples/dllm_sft/finetune.py $CONFIG

# ============================================
# Stage 3: Generation smoke
# ============================================
# Reload the trained consolidated checkpoint single-process and decode a short
# sequence.  Runs only for recipes with a sampler preset, and only on node 0
# (under multi-node srun the launcher runs per node; generation is single-GPU).
if [[ -n "$SAMPLER" && "${SLURM_NODEID:-0}" == "0" ]]; then
    # Latest step checkpoint -> .../epoch_<e>_step_<s>/model/consolidated (HF format).
    # Select by the trailing step number so an underscore-bearing $TEST_NAME in
    # the path can't skew the ordering.
    CKPT_STEP_DIR=$(ls -d "$CKPT_DIR"/epoch_*_step_* 2>/dev/null \
        | awk -F'_step_' '{print $NF"\t"$0}' | sort -k1,1n | tail -1 | cut -f2- || true)
    if [[ -z "$CKPT_STEP_DIR" ]]; then
        echo "[generate] FAILURE: no checkpoint found under $CKPT_DIR"
        ls -la "$CKPT_DIR" 2>/dev/null || echo "  (checkpoint dir does not exist)"
        exit 1
    fi
    CONSOLIDATED="$CKPT_STEP_DIR/model/consolidated"
    if [[ ! -f "$CONSOLIDATED/config.json" ]]; then
        echo "[generate] FAILURE: no consolidated HF checkpoint at $CONSOLIDATED"
        ls -la "$CKPT_STEP_DIR/model" 2>/dev/null || true
        exit 1
    fi

    echo "============================================"
    echo "[generate] Generating from $CONSOLIDATED (sampler=$SAMPLER)..."
    echo "============================================"
    mkdir -p "$GEN_DIR"
    GEN_LOG="$GEN_DIR/generation.log"

    # Short, model-agnostic generation params.  block_size=32 satisfies the
    # nemotron model.generate `max_new_tokens % block_length == 0` constraint and
    # is fine for llada.  --steps only affects the llada (standalone-sampler)
    # path; the nemotron path routes through model.generate and ignores it.
    # --raw skips the chat template so base-model tokenizers without one work.
    python examples/dllm_generate/generate.py \
        --checkpoint "$CONSOLIDATED" \
        --sampler "$SAMPLER" \
        --prompt "The capital of France is" \
        --steps 32 \
        --max_new_tokens 32 \
        --block_size 32 \
        --raw \
        --seed 42 2>&1 | tee "$GEN_LOG"

    # generate.py prints the per-prompt "[Prompt 0] ..." block only AFTER decoding
    # completes, so this is a post-decode signal (a crash is already caught by
    # set -o pipefail). It does not assert output quality -- an undertrained smoke
    # model may decode little -- only that the train -> reload -> generate pipeline
    # ran end to end, mirroring the diffusion launcher's "output file exists" bar.
    if grep -qF "[Prompt 0]" "$GEN_LOG"; then
        echo "[generate] SUCCESS: generation completed"
    else
        echo "[generate] FAILURE: generation did not produce a decoded result"
        exit 1
    fi
elif [[ -z "$SAMPLER" ]]; then
    echo "[generate] No sampler preset for $RECIPE_NAME; skipping generation (train-only)."
fi

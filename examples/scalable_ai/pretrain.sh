#!/bin/bash
# Pre-train Moonlight-V4-16B-A3B on one 8-GPU node inside the NeMo Automodel container (see pretrain_moonlight_v4_16b.yaml).
# Expects $WORK (default /workspace) with Automodel/ (this repo) and models/Moonlight-V4-16B-A3B/ (config + tokenizer).
# Step 1 tokenises FineWeb-edu into $WORK/data/fineweb_edu_moonshot if that directory is missing (needs network).
# Step 2 runs the recipe, restarting from the latest checkpoint after a failure; after three failures it swaps the
# DeepEP dispatcher for the torch one at micro-batch 1. Weights & Biases picks up WANDB_API_KEY or ~/.netrc; without
# either the run is logged offline (sync later with `wandb sync`).
# Extra arguments are passed to the recipe, e.g. `pretrain.sh --step_scheduler.max_steps 200`. Environment knobs: CFG (yaml),
# DATA (shard dir), CKPT (checkpoint dir), PREP_FILES / TRAIN_TOKENS / SHARD_TOKENS / PREP_WORKERS (data preparation),
# WANDB_RUN_ID / WANDB_WAIT_MIN (credential wait in minutes).
set -o pipefail
WORK=${WORK:-/workspace}
cd $WORK/Automodel || exit 1
export PYTHONPATH=$WORK/Automodel:$WORK/Automodel/examples/scalable_ai:${PYTHONPATH:-}
export HF_HOME=$WORK/hf_home TOKENIZERS_PARALLELISM=false
export TILELANG_CACHE_DIR=$WORK/cache/tilelang TRITON_CACHE_DIR=$WORK/cache/triton TORCHINDUCTOR_CACHE_DIR=$WORK/cache/inductor
export WANDB_DIR=$WORK/logs/wandb WANDB_CACHE_DIR=$WORK/cache/wandb WANDB_CONFIG_DIR=$WORK/cache/wandb_config
mkdir -p $WORK/logs $WORK/hf_home $TILELANG_CACHE_DIR $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR $WANDB_DIR $WANDB_CACHE_DIR $WANDB_CONFIG_DIR
M=$WORK/models/Moonlight-V4-16B-A3B
DATA=${DATA:-$WORK/data/fineweb_edu_moonshot}
CFG=${CFG:-examples/scalable_ai/configs/pretrain_moonlight_v4_16b.yaml}
CKPT=${CKPT:-$WORK/checkpoints/moonlight_v4_16b}

if [ ! -f "$DATA/README.txt" ]; then
  echo "=== preparing data in $DATA ($(date))"
  python examples/scalable_ai/prepare_fineweb.py --tokenizer "$M" --out "$DATA" --num-files ${PREP_FILES:-2} \
      --train-tokens ${TRAIN_TOKENS:-560M} --val-tokens 8M --shard-tokens ${SHARD_TOKENS:-35M} --workers ${PREP_WORKERS:-48} \
      2>&1 | tee $WORK/logs/prepare_fineweb.log || { echo "data preparation failed"; exit 1; }
fi
echo "train shards: $(ls $DATA/fineweb_train_*.bin | wc -l), $(du -sh $DATA | cut -f1) total"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# Weights & Biases credentials: WANDB_API_KEY, or a netrc file (NETRC, default ~/.netrc; the sbatch wrapper mounts the
# login user's home at /home_login). Waits up to WANDB_WAIT_MIN minutes for one to appear, then logs offline.
export WANDB_RUN_ID=${WANDB_RUN_ID:-moonlight-v4-16b-a3b-pretrain} WANDB_RESUME=allow
netrc_key() { awk '/machine/{f=($2=="api.wandb.ai")} f{for(i=1;i<=NF;i++) if($i=="password"){print $(i+1); exit}}' "$1" 2>/dev/null; }
for f in "${NETRC:-}" "$HOME/.netrc" /home_login/.netrc; do [ -n "$f" ] && [ -f "$f" ] && NETRC_FILE=$f && break; done
waited=0
while [ -z "${WANDB_API_KEY:-}" ] && [ -z "$(netrc_key "${NETRC_FILE:-/nonexistent}")" ] && [ $waited -lt ${WANDB_WAIT_MIN:-30} ]; do
  [ $waited -eq 0 ] && echo "=== no wandb credentials yet; waiting up to ${WANDB_WAIT_MIN:-30} min for ~/.netrc on the login node ($(date))"
  sleep 60; waited=$((waited+1))
  for f in "$HOME/.netrc" /home_login/.netrc; do [ -f "$f" ] && NETRC_FILE=$f && break; done
done
if [ -z "${WANDB_API_KEY:-}" ]; then
  key=$(netrc_key "${NETRC_FILE:-/nonexistent}")
  if [ -n "$key" ]; then export WANDB_API_KEY=$key; echo "=== wandb credentials found in $NETRC_FILE"; else echo "WARNING: no wandb credentials; logging offline (wandb sync $WANDB_DIR/wandb/offline-* later)"; export WANDB_MODE=offline; WANDB_ARGS="--wandb.mode offline"; fi
  unset key
fi

for attempt in 1 2 3 4 5 6; do
  extra=""
  [ $attempt -ge 4 ] && extra="--model.backend.dispatcher torch --step_scheduler.local_batch_size 1 --dataloader.batch_size 1 --validation_dataloader.batch_size 1"
  echo "=== training attempt $attempt ($(date)) $extra"
  torchrun --nproc-per-node 8 nemo_automodel/recipes/llm/train_ft.py --config $CFG \
      --model.config.pretrained_model_name_or_path "$M" --checkpoint.checkpoint_dir $CKPT \
      --dataset.file_pattern "$DATA/fineweb_train_*.bin" --validation_dataset.file_pattern "$DATA/fineweb_val_0000.bin" \
      ${WANDB_ARGS:-} "$@" $extra 2>&1 | tee $WORK/logs/pretrain_${WANDB_RUN_ID}_attempt${attempt}.log
  rc=${PIPESTATUS[0]}
  [ $rc -eq 0 ] && { echo "=== training finished ($(date))"; exit 0; }
  echo "=== attempt $attempt failed with rc=$rc ($(date)); will resume from the latest checkpoint"; sleep 30
done
echo "=== giving up after 6 attempts"; exit 1

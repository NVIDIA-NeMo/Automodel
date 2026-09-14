#!/bin/bash
# Export a sharded v4_88k checkpoint as an HF directory whose every file matches the base
# model's layout: same 1,045 tensor names, shapes AND dtypes, and the base's own
# config.json / generation_config.json / tokenizer / preprocessor files.
#
# The stock export (checkpoint.save_consolidated: final, or the consolidate.sh inside the
# checkpoint) gets the weights right but not the rest:
#   - A_log / dt_bias in the 30 GatedDeltaNet layers (60 tensors) are kept in fp32 by the
#     model and are marked "intrinsically fp32" in .hf_metadata, so even CAST_DTYPE=bf16
#     leaves them fp32; the base ships them bf16.
#   - config.json, generation_config.json and the tokenizer files are re-serialized from
#     the runtime config (transformers 5.x): use_cache=false, output_hidden_states=true,
#     the vision model_type renamed, and the sampling defaults (temperature, top_k, top_p,
#     the eos list) dropped from generation_config.json.
#   - preprocessor_config.json, video_preprocessor_config.json, vocab.json, merges.txt
#     are not written at all.
#
# Usage (from the repo root, inside the training container; CPU only, ~1 min):
#   NPROC_PER_NODE=16 bash examples/vlm_finetune/qwen3_5_moe/export_hf_v4_88k.sh \
#       checkpoints/<run>/epoch_0_step_199/model  exports/<name>
# BASE_SNAPSHOT defaults to the Qwen/Qwen3.6-35B-A3B snapshot in $HF_HOME.
set -euo pipefail

CKPT_MODEL_DIR=${1:?usage: export_hf_v4_88k.sh <checkpoint>/model <output_dir>}
OUT_DIR=${2:?usage: export_hf_v4_88k.sh <checkpoint>/model <output_dir>}
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
NUM_THREADS=${NUM_THREADS:-5}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
BASE_SNAPSHOT=${BASE_SNAPSHOT:-$(ls -d "$HF_HOME"/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/*/ | head -1)}
HELPER="$(dirname "$0")/export_hf_v4_88k_helper.py"

[ -d "$CKPT_MODEL_DIR/.hf_metadata" ] || { echo "no .hf_metadata in $CKPT_MODEL_DIR"; exit 1; }
[ -f "$BASE_SNAPSHOT/config.json" ] || { echo "base snapshot not found: $BASE_SNAPSHOT"; exit 1; }
[ -e "$OUT_DIR" ] && { echo "refusing to overwrite existing $OUT_DIR"; exit 1; }

# 1. Re-mark the fp32 tensors as bf16 in the dtype mapping so the cast applies to them too.
#    The original mapping is kept alongside as *.orig.
python "$HELPER" remap-dtypes "$CKPT_MODEL_DIR/.hf_metadata/fqn_to_dtype_mapping.json" "$BASE_SNAPSHOT"

# 2. Consolidate the 16 rank shards into the base's 26-shard layout, casting to bf16.
torchrun --nproc-per-node="$NPROC_PER_NODE" tools/offline_hf_consolidation.py \
  --backend gloo --num-threads "$NUM_THREADS" \
  --model-name "$BASE_SNAPSHOT" \
  --input-dir "$CKPT_MODEL_DIR" \
  --output-dir "$OUT_DIR" \
  --cast-dtype bf16

# 3. Overlay the base's own metadata files, then fix the index's total_size, then verify.
python "$HELPER" overlay-and-verify "$OUT_DIR" "$BASE_SNAPSHOT"
echo "EXPORT_OK $OUT_DIR"

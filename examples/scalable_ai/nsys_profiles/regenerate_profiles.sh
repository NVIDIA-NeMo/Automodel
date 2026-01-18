#!/bin/bash
# Regenerate all nsys profiles for the Automodel guest lecture
# This script unsets HF_TOKEN to avoid embedding credentials in profiles

# Continue running even if a command fails
set +e

# Unset HF_TOKEN to prevent it from being captured in profiles
unset HF_TOKEN
unset HUGGING_FACE_HUB_TOKEN
unset HF_HUB_TOKEN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUTPUT_DIR="$SCRIPT_DIR"

cd "$REPO_ROOT"

echo "=== Regenerating profiles in $OUTPUT_DIR ==="
echo "Working directory: $(pwd)"
echo ""

# End-to-End Benchmark Profiles

echo "=== 1/7: moonlight_hf_small.nsys-rep ==="
nsys profile --force-overwrite true \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --output="$OUTPUT_DIR/moonlight_hf_small.nsys-rep" \
  torchrun \
  --nproc-per-node 2 \
  nemo_automodel/recipes/llm/benchmark.py \
  --config examples/scalable_ai/configs/moonlight_16b_hf.yaml \
  --benchmark.nsys_start 3 \
  --benchmark.nsys_end 5 \
  --step_scheduler.max_steps 6 \
  --benchmark.warmup_steps 1 \
  --step_scheduler.global_batch_size 4 \
  --dataset.seq_len 1024 \
  --model.config.num_hidden_layers 3

echo "=== 2/7: moonlight_te_deepep-false_small.nsys-rep ==="
nsys profile --force-overwrite true \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --output="$OUTPUT_DIR/moonlight_te_deepep-false_small.nsys-rep" \
  torchrun \
  --nproc-per-node 2 \
  nemo_automodel/recipes/llm/benchmark.py \
  --config examples/scalable_ai/configs/moonlight_16b_te_deepep.yaml \
  --benchmark.nsys_start 3 \
  --benchmark.nsys_end 5 \
  --step_scheduler.max_steps 6 \
  --benchmark.warmup_steps 1 \
  --distributed.ep_size 2 \
  --step_scheduler.global_batch_size 16 \
  --model.backend.enable_deepep False \
  --model.config.num_hidden_layers 3

echo "=== 3/7: moonlight_te_deepep_small.nsys-rep ==="
nsys profile --force-overwrite true \
  --trace=cuda,nvtx \
  --cuda-memory-usage=true \
  --output="$OUTPUT_DIR/moonlight_te_deepep_small.nsys-rep" \
  torchrun \
  --nproc-per-node 2 \
  nemo_automodel/recipes/llm/benchmark.py \
  --config examples/scalable_ai/configs/moonlight_16b_te_deepep.yaml \
  --benchmark.nsys_start 3 \
  --benchmark.nsys_end 5 \
  --step_scheduler.max_steps 6 \
  --benchmark.warmup_steps 1 \
  --distributed.ep_size 2 \
  --step_scheduler.global_batch_size 16 \
  --model.config.num_hidden_layers 3

# Layer-Level Profiles

echo "=== 4/7: mla_profile_hf.nsys-rep ==="
nsys profile --force-overwrite true \
  -c cudaProfilerApi \
  -t cuda,nvtx \
  -o "$OUTPUT_DIR/mla_profile_hf" \
  python examples/benchmark/profile_layer.py \
  --model-id moonshotai/Moonlight-16B-A3B \
  --layer mla \
  --use-hf

echo "=== 5/7: mla_profile_te.nsys-rep ==="
nsys profile --force-overwrite true \
  -c cudaProfilerApi \
  -t cuda,nvtx \
  -o "$OUTPUT_DIR/mla_profile_te" \
  python examples/benchmark/profile_layer.py \
  --model-id moonshotai/Moonlight-16B-A3B \
  --layer mla \
  --backend-attn te

echo "=== 6/7: rmsnorm_profile_hf.nsys-rep ==="
nsys profile --force-overwrite true \
  -c cudaProfilerApi \
  -t cuda,nvtx \
  -o "$OUTPUT_DIR/rmsnorm_profile_hf" \
  python examples/benchmark/profile_layer.py \
  --model-id moonshotai/Moonlight-16B-A3B \
  --layer rmsnorm \
  --use-hf

echo "=== 7/7: rmsnorm_profile_te.nsys-rep ==="
nsys profile --force-overwrite true \
  -c cudaProfilerApi \
  -t cuda,nvtx \
  -o "$OUTPUT_DIR/rmsnorm_profile_te" \
  python examples/benchmark/profile_layer.py \
  --model-id moonshotai/Moonlight-16B-A3B \
  --layer rmsnorm \
  --backend-rms-norm te

echo ""
echo "=== All profiles regenerated successfully ==="
ls -lh "$OUTPUT_DIR"/*.nsys-rep

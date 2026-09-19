#!/bin/bash
# Regenerate the nsys profiles for the Automodel guest lecture (2026 September edition, Moonlight-V4-16B-A3B).
# Portable profiles run on any CUDA GPU; set RUN_HOPPER=1 to also produce the TileLang / DeepEP / TE profiles,
# which need Hopper-class GPUs and the tilelang, tile_kernels, deep_ep and transformer_engine packages.
set +e   # keep going if one profile fails

# Do not embed credentials in the profiles
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HF_HUB_TOKEN

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
OUT="$SCRIPT_DIR"
cd "$REPO_ROOT"
echo "=== Regenerating profiles in $OUT (repo root: $REPO_ROOT) ==="

E2E_COMMON=(--benchmark.nsys_start 3 --benchmark.nsys_end 5 --step_scheduler.max_steps 6 --benchmark.warmup_steps 1
            --step_scheduler.global_batch_size 4 --step_scheduler.local_batch_size 1 --dataset.seq_len 1024)

e2e() {  # name config extra-args...
  local name=$1 config=$2; shift 2
  echo "=== e2e: $name ==="
  nsys profile --force-overwrite true --trace=cuda,nvtx --cuda-memory-usage=true --output="$OUT/$name.nsys-rep" \
    torchrun --nproc-per-node 2 nemo_automodel/recipes/llm/benchmark.py --config "$config" "${E2E_COMMON[@]}" "$@"
}
layer() {  # name profile_layer args...
  local name=$1; shift
  echo "=== layer: $name ==="
  nsys profile --force-overwrite true -c cudaProfilerApi -t cuda,nvtx -o "$OUT/$name" \
    python examples/scalable_ai/profile_layer.py "$@"
}

# End-to-end (2 layers = sliding-window only, the only fair comparison with the transformers implementation)
e2e moonlight_v4_hf_small        examples/scalable_ai/configs/moonlight_v4_16b_hf.yaml    --model.config.num_hidden_layers 2
e2e moonlight_v4_torch_small     examples/scalable_ai/configs/moonlight_v4_16b_torch.yaml --model.config.num_hidden_layers 2 --distributed.ep_size 1
e2e moonlight_v4_torch_csa_small examples/scalable_ai/configs/moonlight_v4_16b_torch.yaml --model.config.num_hidden_layers 3 --distributed.ep_size 2

# Layer level
layer attn_swa_hf              --layer attn --compress-ratio 0 --use-hf
layer attn_swa_automodel       --layer attn --compress-ratio 0
layer attn_csa_automodel_eager --layer attn --compress-ratio 4 --backend-attn eager
layer attn_hca_hf              --layer attn --compress-ratio 128 --use-hf
layer attn_hca_automodel       --layer attn --compress-ratio 128
layer moe_hf                   --layer moe --use-hf
layer moe_automodel_torchmm    --layer moe --backend-experts torch_mm
layer rmsnorm_hf               --layer rmsnorm --use-hf
layer rmsnorm_automodel_fp32   --layer rmsnorm --backend-rms-norm torch_fp32
layer hc_hf                    --layer hc --use-hf
layer hc_automodel             --layer hc
layer block_csa_automodel      --layer block --compress-ratio 4

if [[ "${RUN_HOPPER:-0}" == "1" ]]; then
  e2e moonlight_v4_tilelang_deepep_small examples/scalable_ai/configs/moonlight_v4_16b_tilelang_deepep.yaml --model.config.num_hidden_layers 3 --distributed.ep_size 2
  layer attn_csa_automodel_tilelang --layer attn --compress-ratio 4 --backend-attn tilelang
  layer moe_automodel_te            --layer moe --backend-experts te
  layer rmsnorm_automodel_te        --layer rmsnorm --backend-rms-norm te
fi

echo "=== done ==="
ls -lh "$OUT"/*.nsys-rep 2>/dev/null

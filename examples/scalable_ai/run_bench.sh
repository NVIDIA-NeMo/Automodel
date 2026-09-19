#!/bin/bash
# Moonlight-V4-16B-A3B benchmarks on 8 GPUs (branch scalable-ai-2026_sept). Usage: run_bench.sh [torch|tilelang|deepep|hf|hf2layer ...]
# Expects $WORK (default /workspace) to contain Automodel/ (this repo), models/Moonlight-V4-16B-A3B/ (config + tokenizer) and logs/.
set -o pipefail
WORK=${WORK:-/workspace}
cd $WORK/Automodel
export PYTHONPATH=$WORK/Automodel:${PYTHONPATH:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$WORK/hf_home TOKENIZERS_PARALLELISM=false
# Writable caches: inside the container $HOME is not writable; tilelang aborts at import (and Automodel then
# silently falls back to the transformers model class) if it cannot create its cache directory.
export TILELANG_CACHE_DIR=$WORK/cache/tilelang TRITON_CACHE_DIR=$WORK/cache/triton TORCHINDUCTOR_CACHE_DIR=$WORK/cache/inductor
mkdir -p $WORK/logs $WORK/hf_home $TILELANG_CACHE_DIR $TRITON_CACHE_DIR $TORCHINDUCTOR_CACHE_DIR
M=$WORK/models/Moonlight-V4-16B-A3B
STEPS=${STEPS:-12}; WARMUP=${WARMUP:-4}

echo "=== environment ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
python - <<'PY' 2>&1 | grep -v "Warning\|warn(\|DSA\|pynvml"
import importlib, torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__, "| deepseek_v4 native:", importlib.util.find_spec("transformers.models.deepseek_v4") is not None)
for m in ("tilelang", "tile_kernels", "deep_ep", "transformer_engine", "grouped_gemm"):
    try:
        mod = importlib.import_module(m); print(" ", m, getattr(mod, "__version__", "ok"))
    except Exception as e:
        print(" ", m, "MISSING:", type(e).__name__)
import nemo_automodel; print("nemo_automodel from", nemo_automodel.__file__)
from nemo_automodel.components.models.deepseek_v4.optimized_kernels import is_dsv4_kernel_available
print("dsv4 kernels: sinkhorn", is_dsv4_kernel_available("sinkhorn"), "sparse_attn", is_dsv4_kernel_available("sparse_attn"), "indexer", is_dsv4_kernel_available("indexer"))
PY

run() {  # name config [extra overrides...]
  local name=$1 cfg=$2; shift 2
  echo; echo "########## $name ##########"
  timeout 1800 torchrun --nproc-per-node 8 nemo_automodel/recipes/llm/benchmark.py --config "$cfg" \
      --model.config.pretrained_model_name_or_path "$M" --step_scheduler.max_steps "$STEPS" --benchmark.warmup_steps "$WARMUP" \
      --benchmark.json_output_path "$WORK/logs/$name.json" "$@" > "$WORK/logs/$name.log" 2>&1
  local rc=$?
  echo "exit=$rc"
  grep -E "TFLOPs/GPU:|Total parameters|Average iteration time|Average MFU|Total iteration time" "$WORK/logs/$name.log" | grep -v "MTP" | sed 's/.*| INFO | __main__ | //;s/.*| INFO | root | //' | tail -5
  echo "peak memory (rank 0, GB): $(grep -o 'Max Memory Allocated: [0-9.]* GB' "$WORK/logs/$name.log" | awk '{print $4}' | sort -n | tail -1)"
  grep -E "OutOfMemoryError|Error:|Error\b" "$WORK/logs/$name.log" | grep -v "Warning\|DSA\|AttributeError: DSA" | tail -2 | cut -c1-200
  return $rc
}

# try decreasing micro-batch sizes (global batch fixed by the yaml) until a run succeeds
run_lbs() {  # name config [extra overrides...]
  local name=$1 cfg=$2; shift 2
  for lbs in ${LBS_LIST:-2 1}; do
    run "${name}_lbs${lbs}" "$cfg" --step_scheduler.local_batch_size "$lbs" "$@" && return 0
  done
  return 1
}

C=examples/scalable_ai/configs
for which in "${@:-torch tilelang hf}"; do
  case $which in
    torch)    run_lbs automodel_torch          $C/moonlight_v4_16b_torch.yaml ;;
    tilelang) run_lbs automodel_tilelang_deepep $C/moonlight_v4_16b_tilelang_deepep.yaml ;;
    deepep)   run_lbs automodel_eager_deepep   $C/moonlight_v4_16b_torch.yaml --model.backend.dispatcher deepep ;;
    torch_ac) run automodel_torch_lbs4_ac      $C/moonlight_v4_16b_torch.yaml --distributed.activation_checkpointing true ;;
    hf)       run hf_ootb_seq2048           $C/moonlight_v4_16b_hf.yaml
              run hf_ootb_seq1024           $C/moonlight_v4_16b_hf.yaml --dataset.seq_len 1024
              run hf_ootb_seq512            $C/moonlight_v4_16b_hf.yaml --dataset.seq_len 512 ;;
    hf2layer) run hf_ootb_2layers_seq2048   $C/moonlight_v4_16b_hf.yaml --model.config.num_hidden_layers 2
              run automodel_torch_2layers   $C/moonlight_v4_16b_torch.yaml --model.config.num_hidden_layers 2 ;;
  esac
done
echo; echo "=== logs ==="; ls -la $WORK/logs

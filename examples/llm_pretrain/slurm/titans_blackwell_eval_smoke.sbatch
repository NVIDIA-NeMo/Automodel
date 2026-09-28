#!/bin/bash
# Load every completed Titans architecture and generate one token.

set -euo pipefail

AUTOMODEL_CHECKOUT=${AUTOMODEL_CHECKOUT:-$PWD}
TITANS_WORK_ROOT=${TITANS_WORK_ROOT:-$(cd "$AUTOMODEL_CHECKOUT/../.." && pwd)}
TITANS_IMAGE=${TITANS_IMAGE:-nvcr.io#nvidia/nemo-automodel:26.04}
TOKENIZER=${TITANS_TOKENIZER:-NousResearch/Llama-2-7b-hf}
PROMPT=${TITANS_SMOKE_PROMPT:-"The passkey is 314159. Question: What is the passkey? Answer:"}
MODEL_REGISTRY=${TITANS_MODEL_REGISTRY:-"$AUTOMODEL_CHECKOUT/examples/llm_pretrain/titans_eval_registry.json"}
MODEL_NAME=${TITANS_MODEL_NAME:-}

if [[ ${TITANS_EVAL_IN_CONTAINER:-0} != 1 ]]; then
  exec srun \
    --nodes=1 \
    --ntasks=1 \
    --container-image="$TITANS_IMAGE" \
    --container-mounts=/scratch:/scratch \
    --container-workdir="$AUTOMODEL_CHECKOUT" \
    env \
      TITANS_EVAL_IN_CONTAINER=1 \
      AUTOMODEL_CHECKOUT="$AUTOMODEL_CHECKOUT" \
      TITANS_WORK_ROOT="$TITANS_WORK_ROOT" \
      TITANS_TOKENIZER="$TOKENIZER" \
      TITANS_SMOKE_PROMPT="$PROMPT" \
      TITANS_MODEL_REGISTRY="$MODEL_REGISTRY" \
      TITANS_MODEL_NAME="$MODEL_NAME" \
      PYTHONPATH="$AUTOMODEL_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}" \
      HF_HOME="$TITANS_WORK_ROOT/cache/huggingface" \
      bash "$AUTOMODEL_CHECKOUT/examples/llm_pretrain/slurm/titans_blackwell_eval_smoke.sbatch"
fi

cd "$AUTOMODEL_CHECKOUT"
export PYTHONPATH="$AUTOMODEL_CHECKOUT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-"$TITANS_WORK_ROOT/cache/huggingface"}

RESULT_ROOT="$TITANS_WORK_ROOT/results/eval/smoke"
mkdir -p "$RESULT_ROOT"

model_args=()
[[ -n $MODEL_NAME ]] && model_args=(--model "$MODEL_NAME")
mapfile -t model_rows < <(
  python examples/llm_pretrain/titans_eval_registry.py \
    "$MODEL_REGISTRY" --work-root "$TITANS_WORK_ROOT" "${model_args[@]}"
)
(( ${#model_rows[@]} )) || { echo "Model registry has no enabled checkpoints" >&2; exit 1; }
declare -A seen=()

for model_row in "${model_rows[@]}"; do
  IFS=$'\t' read -r architecture checkpoint enable_ttt <<<"$model_row"
  [[ -z ${seen[$architecture]:-} ]] || continue
  seen[$architecture]=1
  output="$RESULT_ROOT/${architecture}.jsonl"

  echo "=== $architecture: $checkpoint ==="
  python examples/llm_pretrain/titans_generate.py \
    --checkpoint "$checkpoint" \
    --tokenizer "$TOKENIZER" \
    --prompt "$PROMPT" \
    --max-new-tokens 1 \
    --enable-ttt-updates \
    --output-jsonl "$output"

  python - "$output" "$architecture" <<'PY'
import json
import math
import sys

path, architecture = sys.argv[1:]
rows = [json.loads(line) for line in open(path) if line.strip()]
if len(rows) != 1:
    raise SystemExit(f"{architecture}: expected one output row, found {len(rows)}")
row = rows[0]
if row["generated_tokens"] != 1:
    raise SystemExit(f"{architecture}: expected one generated token: {row}")
if not isinstance(row["text"], str):
    raise SystemExit(f"{architecture}: generated text is not a string: {row}")
print(json.dumps({"architecture": architecture, **row}, sort_keys=True))
PY
done

echo "All Titans architectures loaded and generated one token."

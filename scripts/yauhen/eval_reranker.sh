#!/bin/bash
set -euo pipefail

cd "/lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/nemotron_4_embed_vl/nemo-retriever-research"
# TODO: change paths
export TRITON_CACHE_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/$SLURM_JOB_USER/triton_cache"
LOGS_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/$SLURM_JOB_USER/eval_logs"

echo "$HF_HOME"
echo "$@"

INSTALL_ATTEMPTS=5
for ((attempt = 1; attempt <= INSTALL_ATTEMPTS; attempt++)); do
    if /opt/venv/bin/python -m pip install beir==2.0.0 mteb[xet]==2.18.16 polars==1.35.2; then
        break
    fi
    if ((attempt == INSTALL_ATTEMPTS)); then
        echo "Error: dependency installation failed after $INSTALL_ATTEMPTS attempts."
        exit 1
    fi
    echo "Dependency installation attempt $attempt failed; retrying in 30 seconds."
    sleep 30
done
/opt/venv/bin/python -m pip uninstall -y mistral_common
# Dynamically detect number of available GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Create logs directory if it doesn't exist
mkdir -p "$LOGS_DIR"

model_log_suffix() {
    local model_path="${1%/}"
    local name
    name="$(basename "$model_path")"

    while [[ "$name" == "consolidated" || "$name" == "model" || "$name" == "LATEST" ]]; do
        model_path="$(dirname "$model_path")"
        name="$(basename "$model_path")"
    done

    name="${name//[^A-Za-z0-9_.-]/_}"
    printf "_%s" "$name"
}

# Parse --model_name_or_path from arguments, pass everything else through
MODEL_SUFFIX=""
PASS_THROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_name_or_path)
            MODEL_SUFFIX="$(model_log_suffix "$2")"
            PASS_THROUGH_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            PASS_THROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done
if ((${#PASS_THROUGH_ARGS[@]} > 0)); then
    set -- "${PASS_THROUGH_ARGS[@]}"
else
    set --
fi

echo "Model suffix: $MODEL_SUFFIX"
printf "Pass-through args:"
printf " %q" "$@"
printf "\n"

LOG_FILE="$LOGS_DIR/reranker_eval_$(date +%Y%m%d_%H%M%S)${MODEL_SUFFIX}.log"

torchrun --nproc_per_node="$NUM_GPUS" --master_port=29520 \
    src/nemo_retriever_research/evaluation/ranker_eval.py \
    "$@" \
    2>&1 | tee "$LOG_FILE"

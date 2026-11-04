#!/bin/bash
set -euo pipefail

cd "/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/ybabakhin/git/nemo-retriever-research"
# TODO: change paths
export TRITON_CACHE_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/$SLURM_JOB_USER/triton_cache"
LOGS_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/$SLURM_JOB_USER/eval_logs"

echo "$HF_HOME"
echo "$@"

INSTALL_ATTEMPTS=5
for ((attempt = 1; attempt <= INSTALL_ATTEMPTS; attempt++)); do
    if /opt/venv/bin/python -m pip install transformers==4.57.0 beir==2.0.0 mteb[xet]==2.18.16 polars==1.35.2 "qwen-vl-utils>=0.0.14"; then
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

# Parse --eval_type and --model_name_or_path from arguments, pass everything else through
EVAL_TYPE="mteb"
MODEL_SUFFIX=""
PASS_THROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --eval_type)
            EVAL_TYPE="$2"
            shift 2
            ;;
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

echo "Eval type: $EVAL_TYPE"
echo "Model suffix: $MODEL_SUFFIX"
printf "Pass-through args:"
printf " %q" "$@"
printf "\n"

MTEB_ARGS=("$@")
if [[ "$EVAL_TYPE" == "mteb" ]]; then
    # mteb_ranker_eval.py forwards unknown args into model construction, so
    # remove submitter metadata and BEIR/multimodal-only arguments.
    MTEB_ARGS=()
    ORIGINAL_ARGS=("$@")
    ARG_INDEX=0
    while ((ARG_INDEX < ${#ORIGINAL_ARGS[@]})); do
        ARG="${ORIGINAL_ARGS[$ARG_INDEX]}"
        case "$ARG" in
            --eval_group_name|--dataset_mapping|--log_wandb|--dataset_dir|--multimodal)
                if ((ARG_INDEX + 1 >= ${#ORIGINAL_ARGS[@]})); then
                    echo "Error: $ARG requires a value"
                    exit 1
                fi
                ARG_INDEX=$((ARG_INDEX + 2))
                ;;
            --eval_group_name=*|--dataset_mapping=*|--log_wandb=*|--dataset_dir=*|--multimodal=*)
                ARG_INDEX=$((ARG_INDEX + 1))
                ;;
            *)
                MTEB_ARGS+=("$ARG")
                ARG_INDEX=$((ARG_INDEX + 1))
                ;;
        esac
    done

    printf "MTEB args:"
    printf " %q" "${MTEB_ARGS[@]}"
    printf "\n"
fi

case "$EVAL_TYPE" in
    mteb)
        # Run one independently sharded MTEB worker per GPU.
        TOTAL_CORES=$(nproc)
        CPUS_PER_PROCESS=$((TOTAL_CORES / NUM_GPUS))

        REMAINDER=$((TOTAL_CORES % NUM_GPUS))
        if [ "$REMAINDER" -ne 0 ]; then
            echo "Warning: Total cores ($TOTAL_CORES) is not divisible by $NUM_GPUS. Some cores will be unused."
            echo "Using $CPUS_PER_PROCESS cores per process, leaving $REMAINDER cores unused."
        fi

        if [ "$CPUS_PER_PROCESS" -lt 1 ]; then
            echo "Error: Not enough cores ($TOTAL_CORES) for $NUM_GPUS processes. Need at least $NUM_GPUS cores."
            exit 1
        fi

        echo "Total cores: $TOTAL_CORES, CPUs per process: $CPUS_PER_PROCESS"

        MTEB_WORKERS=()
        for i in $(seq 0 $((NUM_GPUS - 1))); do
            START_CPU=$((i * CPUS_PER_PROCESS))
            END_CPU=$((START_CPU + CPUS_PER_PROCESS - 1))
            LOG_FILE="$LOGS_DIR/mteb_qwen3_vl_reranker_gpu_${i}_$(date +%Y%m%d_%H%M%S)${MODEL_SUFFIX}.log"

            CUDA_VISIBLE_DEVICES="$i" taskset -c "$START_CPU-$END_CPU" env -u LOCAL_RANK /opt/venv/bin/python \
                src/nemo_retriever_research/evaluation/mteb_ranker_eval.py \
                --model_type qwen3_vl_reranker \
                --part="$i" \
                --total_parts="$NUM_GPUS" \
                "${MTEB_ARGS[@]}" > "$LOG_FILE" 2>&1 &
            MTEB_WORKERS+=("$!")
        done

        for _ in "${MTEB_WORKERS[@]}"; do
            if ! wait -n; then
                echo "Error: an MTEB Qwen reranker worker failed; terminating the remaining workers."
                mapfile -t RUNNING_WORKERS < <(jobs -pr)
                if ((${#RUNNING_WORKERS[@]} > 0)); then
                    kill "${RUNNING_WORKERS[@]}" 2>/dev/null || true
                    wait "${RUNNING_WORKERS[@]}" 2>/dev/null || true
                fi
                exit 1
            fi
        done
        ;;

    beir_reranker_eval)
        LOG_FILE="$LOGS_DIR/reranker_eval_$(date +%Y%m%d_%H%M%S)${MODEL_SUFFIX}.log"

        torchrun --nproc_per_node="$NUM_GPUS" --master_port=29520 \
            src/nemo_retriever_research/evaluation/ranker_eval.py \
            --model_type qwen3_vl_reranker \
            --max_length 10240 \
            --batch_size 8 \
            --export_results_only_when_complete True \
            --log_json True \
            --skip_existing_results True \
            "$@" \
            2>&1 | tee "$LOG_FILE"
        ;;

    *)
        echo "Error: Unknown eval_type '$EVAL_TYPE'. Supported types: mteb, beir_reranker_eval"
        exit 1
        ;;
esac

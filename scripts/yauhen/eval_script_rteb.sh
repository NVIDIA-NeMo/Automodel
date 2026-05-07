#!/bin/bash
set -euo pipefail


cd /lustre/fs12/portfolios/llmservice/users/$SLURM_JOB_USER/git/nemo-retriever-research

echo $HF_HOME
echo "$@"

/opt/venv/bin/python -m pip install "sentence-transformers>=2.0.0,<4.0.0" beir==2.0.0 mteb[xet]==2.10.0 polars==1.35.2
/opt/venv/bin/python -m pip install transformers==5.5.0
export TRITON_CACHE_DIR="/lustre/fsw/portfolios/llmservice/users/$SLURM_JOB_USER/triton_cache"

# Dynamically detect number of available GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Create logs directory if it doesn't exist
mkdir -p logs

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
            MODEL_SUFFIX="_$(basename "$2")"
            PASS_THROUGH_ARGS+=("$1" "$2")
            shift 2
            ;;
        *)
            PASS_THROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${PASS_THROUGH_ARGS[@]}"

echo "Eval type: $EVAL_TYPE"
echo "Model suffix: $MODEL_SUFFIX"
echo "Pass-through args: $@"

case "$EVAL_TYPE" in
    mteb)
        # Dynamically calculate total cores and CPUs per process
        TOTAL_CORES=$(nproc)
        CPUS_PER_PROCESS=$((TOTAL_CORES / NUM_GPUS))

        # Sanity check: warn if TOTAL_CORES is not divisible by NUM_GPUS
        REMAINDER=$((TOTAL_CORES % NUM_GPUS))
        if [ $REMAINDER -ne 0 ]; then
            echo "Warning: Total cores ($TOTAL_CORES) is not divisible by $NUM_GPUS. Some cores will be unused."
            echo "Using $CPUS_PER_PROCESS cores per process, leaving $REMAINDER cores unused."
        fi

        # Ensure we have at least 1 core per process
        if [ $CPUS_PER_PROCESS -lt 1 ]; then
            echo "Error: Not enough cores ($TOTAL_CORES) for $NUM_GPUS processes. Need at least $NUM_GPUS cores."
            exit 1
        fi

        echo "Total cores: $TOTAL_CORES, CPUs per process: $CPUS_PER_PROCESS"

        for i in $(seq 0 $((NUM_GPUS - 1))); do
            # Calculate CPU core range for this process
            START_CPU=$((i * CPUS_PER_PROCESS))
            END_CPU=$((START_CPU + CPUS_PER_PROCESS - 1))

            # Create log file name with timestamp
            LOG_FILE="logs/mteb_gpu_${i}_$(date +%Y%m%d_%H%M%S)${MODEL_SUFFIX}.log"

            # Run process bound to specific CPU cores with logging
            CUDA_VISIBLE_DEVICES=$i taskset -c $START_CPU-$END_CPU python3 src/nemo_retriever_research/evaluation/mteb_text_eval.py \
            --part=$i \
            --total_parts=$NUM_GPUS \
            --max_length 4096 \
            --embedding_dim 16384 \
            --batch_size 4 \
            --model_type HF \
            --precision bf16 \
            --attn_implementation flash_attention_2 \
            --padding_side left \
            --pooling avg \
            --normalize_embeddings True "$@" > "$LOG_FILE" 2>&1 &

        done

        wait
        ;;

    text_embedding_eval)
        LOG_FILE="logs/text_embedding_eval_$(date +%Y%m%d_%H%M%S)${MODEL_SUFFIX}.log"

        torchrun --nproc_per_node=$NUM_GPUS --master_port=29510 \
            src/nemo_retriever_research/evaluation/embedding_eval.py \
            --max_length 4096 \
            --embedding_dim 16384 \
            --batch_size 4 \
            --model_type HF \
            --precision bf16 \
            --attn_implementation flash_attention_2 \
            --padding_side left \
            --pooling avg \
            --normalize_embeddings True \
            --corpus_modalities text \
            "$@" 2>&1 | tee "$LOG_FILE"
        ;;

    *)
        echo "Error: Unknown eval_type '$EVAL_TYPE'. Supported types: mteb, text_embedding_eval"
        exit 1
        ;;
esac

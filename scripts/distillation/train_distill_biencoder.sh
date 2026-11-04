#!/bin/bash
set -euo pipefail

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "N_NODES: $N_NODES"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
printf "Arguments:"
printf " %q" "$@"
printf "\n"

export TRITON_CACHE_DIR="/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/users/$SLURM_JOB_USER/triton_cache"

/opt/venv/bin/python -m pip install -e . --no-deps
/opt/venv/bin/python -m pip uninstall -y mistral_common
pip install --index-url=https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-mlwfo-pypi/simple --upgrade one-logger-utils

if [ "$N_NODES" -gt 1 ]; then
    echo "Running multi-node distillation with torchrun"
    torchrun \
        --nproc-per-node="$GPUS_PER_NODE" \
        --nnodes="$N_NODES" \
        --rdzv-backend=c10d \
        --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
        nemo_automodel/recipes/retrieval/distill_bi_encoder.py \
        "$@"
else
    echo "Running single-node distillation with torchrun"
    torchrun \
        --nproc-per-node="$GPUS_PER_NODE" \
        nemo_automodel/recipes/retrieval/distill_bi_encoder.py \
        "$@"
fi

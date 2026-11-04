#!/bin/bash
#SBATCH -A nemotron_retriever_dev    #account
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -t 04:00:00    #wall time limit, hr:min:sec
#SBATCH -N 8             # node count — change per job (always multi-node)
#SBATCH --gres=gpu:8

#SBATCH --ntasks-per-node=1
# One srun task per node; each runs torchrun with --nnodes=$SLURM_NNODES.
# Request enough CPUs per node for dataloader workers (Slurm often defaults to 1 CPU/task).
#SBATCH --cpus-per-task=240
# If your site prefers whole nodes: use  #SBATCH --exclusive  instead (and drop cpus-per-task if redundant).
# If sbatch rejects --gres, ask your admin for the correct GPU syntax for this partition.
#SBATCH -J ministral3_3b_reranker #job name
#SBATCH --output=/lustre/fsw/portfolios/llmservice/users/ronaya/jobs/%x_%j.out
#SBATCH --error=/lustre/fsw/portfolios/llmservice/users/ronaya/jobs/%x_%j.err
##SBATCH --dependency=singleton

echo "Running on hosts: $(echo $(scontrol show hostname))"

echo "SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-<unset>}"
echo "SLURM_NNODES=${SLURM_NNODES:-<unset>} SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-<unset>}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-<unset>} SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-<unset>}"

if [[ -z "${1:-}" ]]; then
  echo "Usage: sbatch nemotron_vl_multinode_reranker.sh <config_yaml>"
  exit 2
fi
HOST_AUTOMODEL_DIR=/lustre/fsw/portfolios/llmservice/users/ronaya/automodel-internal
CONTAINER_AUTOMODEL_DIR=/opt/Automodel
CONFIG_YAML="$(realpath "$1")"
if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "Config YAML not found: $1" >&2
  exit 2
fi
if [[ "${CONFIG_YAML}" != "${HOST_AUTOMODEL_DIR}/"* ]]; then
  echo "Config YAML must be under ${HOST_AUTOMODEL_DIR}: ${CONFIG_YAML}" >&2
  exit 2
fi
CONTAINER_CONFIG_YAML="${CONTAINER_AUTOMODEL_DIR}/${CONFIG_YAML#${HOST_AUTOMODEL_DIR}/}"
echo "Config YAML: ${CONFIG_YAML} -> ${CONTAINER_CONFIG_YAML}"

CONT=/lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidia-nemo-automodel-26.04.00.sqsh
DS_RAPIDS=/lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/01_datasets
HF_CACHE=/lustre/fsw/portfolios/llmservice/users/ronaya/hf_cache
TRITON_CACHE=/lustre/fsw/portfolios/llmservice/users/ronaya/triton_cache
CHECKPOINT_DIR=/lustre/fsw/portfolios/llmservice/users/ronaya/model_checkpoints
PYXIS_MOUNTS=(
  "${HOST_AUTOMODEL_DIR}:${CONTAINER_AUTOMODEL_DIR}"
  "${HF_CACHE}:${HF_CACHE}"
  "${TRITON_CACHE}:${TRITON_CACHE}"
  "${CHECKPOINT_DIR}:${CHECKPOINT_DIR}"
  "/lustre/fsw/portfolios/llmservice/users/ronaya/Ministral-3-3B-Instruct-2512-BF16:/lustre/fsw/portfolios/llmservice/users/ronaya/Ministral-3-3B-Instruct-2512-BF16"
  # Training jsonl files referenced by dataloader.dataset.data_dir_list
  "${DS_RAPIDS}/03_training_datasets:${DS_RAPIDS}/03_training_datasets"
  "/lustre/fsw/portfolios/llmservice/users/ronaya/synthetic_query_generation:/lustre/fsw/portfolios/llmservice/users/ronaya/synthetic_query_generation"
  # Corpus metadata (merlin_metadata.json etc.) loaded by retrieval_dataset.load_corpus_metadata
  "${DS_RAPIDS}/04_corpus:${DS_RAPIDS}/04_corpus"
)
CONT_MOUNT=$(IFS=,; echo "${PYXIS_MOUNTS[*]}")

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"
export HF_HOME="${HF_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export TRITON_CACHE_DIR="${TRITON_CACHE}"

echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "World size: $((${SLURM_GPUS_PER_NODE:-8} * ${SLURM_NNODES:-1})) GPUs across ${SLURM_NNODES:-<unset>} nodes"

# Credentials: same sources as interactive JupyterLab (huggingface-cli login + wandb login).
# Optional override before sbatch: export HF_TOKEN=... and/or export WANDB_API_KEY=...
if [[ -z "${HF_TOKEN:-}" && -f "${HF_HOME}/token" ]]; then
  export HF_TOKEN="$(tr -d "[:space:]" < "${HF_HOME}/token")"
fi
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi
if [[ -z "${WANDB_API_KEY:-}" && -f "${HOME}/.netrc" ]]; then
  WANDB_API_KEY="$(awk '
    /^machine api\.wandb\.ai$/ { in_block=1; next }
    in_block && /^  password / { print $2; exit }
    in_block && /^machine / { in_block=0 }
  ' "${HOME}/.netrc")"
  if [[ -n "${WANDB_API_KEY}" ]]; then
    export WANDB_API_KEY
  fi
fi

echo "HF_TOKEN length: ${#HF_TOKEN}  WANDB_API_KEY length: ${#WANDB_API_KEY}"

# NCCL / PyTorch: avoid default ~60s collective watchdog timeouts on multi-node
export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

# After an NCCL collective failure: submit with ENABLE_NCCL_DEBUG=1 (verbose). In the Slurm .err,
# compare [rank0] vs other ranks for desync, stuck collectives, or network errors.
if [[ "${ENABLE_NCCL_DEBUG:-0}" == "1" ]]; then
  export NCCL_DEBUG=INFO
  export TORCH_DISTRIBUTED_DEBUG=DETAIL
fi

srun \
        --container-image="${CONT}" \
        --container-mounts="${CONT_MOUNT}" \
        --container-entrypoint \
        --no-container-mount-home \
        --export=ALL \
        bash -c 'set -euo pipefail
            cd /opt/Automodel/
            [[ -f "$1" ]] || { echo "Config YAML not found in container: $1" >&2; exit 2; }
            torchrun \
            --nproc_per_node=${SLURM_GPUS_PER_NODE:-8} \
            --nnodes=${SLURM_NNODES:-1} \
            --rdzv_backend=c10d \
            --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
            examples/retrieval/cross_encoder/finetune.py --config "$1"' _ "${CONTAINER_CONFIG_YAML}"

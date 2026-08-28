#!/bin/bash
#SBATCH -A nemotron_retriever_dev
#SBATCH -p batch_block1,batch_block3,batch_block4
#SBATCH -t 04:00:00
#SBATCH -N 8
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=240
#SBATCH -J automodel_distill
#SBATCH --output=/lustre/fsw/portfolios/datascience/projects/datascience_nemo_retriever/users/jiaruic/slurm_jobs/%x_%j.out
#SBATCH --error=/lustre/fsw/portfolios/datascience/projects/datascience_nemo_retriever/users/jiaruic/slurm_jobs/%x_%j.err

echo "Running on hosts: $(echo $(scontrol show hostname))"
echo "SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-<unset>}"
echo "SLURM_NNODES=${SLURM_NNODES:-<unset>} SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-<unset>}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-<unset>} SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE:-<unset>}"

if [[ $# -lt 1 || -z "${1:-}" ]]; then
  echo "Usage: $0 <config-yaml>" >&2
  exit 2
fi

CONFIG_YAML_INPUT=$1

SUBMIT_TS="${SUBMIT_TS:-}"
if [[ -z "${SUBMIT_TS}" && -n "${SLURM_JOB_ID:-}" ]]; then
  JOB_SUBMIT_TIME="$(scontrol show job "${SLURM_JOB_ID}" | awk -F= '/SubmitTime=/ { split($2, a, " "); print a[1]; exit }')"
  if [[ -n "${JOB_SUBMIT_TIME}" && "${JOB_SUBMIT_TIME}" != "Unknown" ]]; then
    SUBMIT_TS="$(date -d "${JOB_SUBMIT_TIME}" +%Y%m%d_%H%M%S 2>/dev/null || true)"
  fi
fi
if [[ -z "${SUBMIT_TS}" ]]; then
  SUBMIT_TS="$(date +%Y%m%d_%H%M%S)"
fi

CONFIG_YAML="${CONFIG_YAML_INPUT}"
if [[ "${DISABLE_SUBMIT_TS_CONFIG:-0}" != "1" ]]; then
  CONFIG_DIR="$(dirname "${CONFIG_YAML_INPUT}")"
  CONFIG_BASE="$(basename "${CONFIG_YAML_INPUT}" .yaml)"
  CONFIG_YAML="${CONFIG_DIR}/${CONFIG_BASE}_${SUBMIT_TS}.yaml"
  python3 - "${CONFIG_YAML_INPUT}" "${CONFIG_YAML}" "${SUBMIT_TS}" <<'PY'
import re
import sys
from pathlib import Path

src, dst, submit_ts = sys.argv[1:]


def stamp(line: str, sep: str) -> str:
    key, value = line.split(":", 1)
    leading = value[: len(value) - len(value.lstrip())]
    raw = value.strip()
    quote = raw[0] if raw[:1] in {"'", '"'} and raw[-1:] == raw[0] else ""
    inner = raw[1:-1] if quote else raw
    if "__SUBMIT_TS__" in inner:
        inner = inner.replace("__SUBMIT_TS__", submit_ts)
    elif not re.search(r"[_-]\d{8}_\d{6}(?:_v\d+)?$", inner):
        inner = f"{inner}{sep}{submit_ts}"
    rendered = f"{quote}{inner}{quote}" if quote else inner
    return f"{key}:{leading}{rendered}\n"


section = None
out = []
for line in Path(src).read_text().splitlines(keepends=True):
    stripped = line.strip()
    if stripped and not line.startswith((" ", "\t")) and stripped.endswith(":"):
        section = stripped[:-1]
    if section == "checkpoint" and stripped.startswith("checkpoint_dir:"):
        out.append(stamp(line, "_"))
    elif section == "wandb" and stripped.startswith("name:"):
        out.append(stamp(line, "-"))
    else:
        out.append(line)

Path(dst).write_text("".join(out))
PY
fi

echo "Config YAML input: ${CONFIG_YAML_INPUT}"
echo "Submit timestamp: ${SUBMIT_TS}"
echo "Config YAML generated: ${CONFIG_YAML}"

CONT=/lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/03_sqsh/nvidian-nemo-automodel-26.08.rc5.sqsh
AUTOMODEL_INTERNAL=/lustre/fsw/portfolios/datascience/projects/datascience_nemo_retriever/users/jiaruic/code/vlemb/automodel-private-internal
DS_RAPIDS=/lustre/fsw/portfolios/datascience/projects/datascience_rapids_research/retriever_models_research/01_datasets
GMO_AUTO_BLEND=/lustre/fsw/portfolios/datascience/users/gmoreira/nemo_retriever_assets/auto_blend_research
GMO_CHECKPOINTS=/lustre/fsw/portfolios/datascience/users/gmoreira/model_checkpoints
HF_CACHE=/lustre/fsw/portfolios/datascience/projects/datascience_nemo_retriever/users/jiaruic/hf_cache
TRITON_CACHE=/lustre/fsw/portfolios/datascience/projects/datascience_nemo_retriever/users/jiaruic/triton_cache
CHECKPOINT_DIR=/lustre/fsw/portfolios/datascience/projects/datascience_nemo_retriever/users/jiaruic/model_checkpoints
PYXIS_MOUNTS=(
  "${AUTOMODEL_INTERNAL}:/opt/Automodel"
  "${HF_CACHE}:${HF_CACHE}"
  "${TRITON_CACHE}:${TRITON_CACHE}"
  "${DS_RAPIDS}/03_training_datasets:${DS_RAPIDS}/03_training_datasets"
  "${DS_RAPIDS}/04_corpus:${DS_RAPIDS}/04_corpus"
  "${GMO_AUTO_BLEND}:${GMO_AUTO_BLEND}"
  "${GMO_CHECKPOINTS}:${GMO_CHECKPOINTS}"
  "/lustre/fsw/portfolios/datascience/projects/datascience_nemo_retriever/users/jiaruic:/lustre/fsw/portfolios/datascience/projects/datascience_nemo_retriever/users/jiaruic"
  "${CHECKPOINT_DIR}:${CHECKPOINT_DIR}"
)
CONT_MOUNT=$(IFS=,; echo "${PYXIS_MOUNTS[*]}")

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=13742
export HF_HOME="${HF_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE}"
export HF_DATASETS_CACHE="${HF_CACHE}"
export TRANSFORMERS_CACHE="${HF_CACHE}"
export TRITON_CACHE_DIR="${TRITON_CACHE}"

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
echo "MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "World size: $((${SLURM_GPUS_PER_NODE:-8} * ${SLURM_NNODES:-1})) GPUs across ${SLURM_NNODES:-<unset>} nodes"

export NCCL_TIMEOUT=1800
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

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
        bash -c "set -euo pipefail
            cd /opt/Automodel/
            torchrun \
            --nproc_per_node=${SLURM_GPUS_PER_NODE:-8} \
            --nnodes=${SLURM_NNODES:-1} \
            --rdzv_backend=c10d \
            --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
            -m nemo_automodel.cli.app ${CONFIG_YAML}"

#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Launch single-node 8-GPU ASR fine-tuning of Qwen/Qwen2.5-Omni-3B on the
# six-corpus English mix (AMI + earnings22 + voxpopuli_en + gigaspeech +
# spgispeech + librispeech) via make_multi_en_asr_dataset.
#
# The script enforces NPROC_PER_NODE=8 (single-node 8-GPU contract) and uses
# the bundled DTensor venv (transformers 5.x, torch 2.x, soundfile, scipy;
# no torchcodec — audio is decoded via soundfile inside make_multi_en_asr_dataset).
#
# NOTE: gigaspeech and spgispeech are gated on the Hugging Face Hub. Accept
# their dataset terms before launching (gigaspeech also needs trust_remote_code,
# which the builder already sets), or trim the mix via a dataset.sources override.
#
# Optional environment:
#   PY                   Python interpreter (default: bundled DTensorPolicyWorkerV2 venv).
#   CONFIG               YAML config to launch (default: multi_en_sft_3b.yaml next to this script).
#   NPROC_PER_NODE       GPUs per node (must be 8; default: 8).
#   WANDB_PROJECT / WANDB_NAME / WANDB_DIR / WANDB_MODE
#                        Standard WandB env vars. WandB is enabled by default;
#                        set WANDB_MODE=disabled to skip uploading.
#
# Extra arguments are forwarded to ``nemo_automodel.cli.app``, e.g.:
#   examples/audio_finetune/qwen2_5_omni_asr/train_multi_en_qwen2.5_omni_3b.sh \
#       --step_scheduler.max_steps 3 --checkpoint.checkpoint_dir /tmp/multi_en_smoke
#
set -euo pipefail

PY="${PY:-/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/multi_en_sft_3b.yaml}"

# Single-node 8-GPU contract: the YAML's FSDP2 schedule assumes exactly 8 GPUs.
# Running with a different worker count would silently change the parallelization
# plan, so reject the call up front.
if [[ "${NPROC_PER_NODE}" != "8" ]]; then
    echo "NPROC_PER_NODE must be exactly 8 (single-node 8-GPU contract); got '${NPROC_PER_NODE}'." >&2
    exit 1
fi

if [[ ! -x "${PY}" ]]; then
    echo "PY interpreter not found or not executable: ${PY}" >&2
    exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "CONFIG file not found: ${CONFIG}" >&2
    exit 1
fi

exec "${PY}" -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes=1 \
    -m nemo_automodel.cli.app \
    "${CONFIG}" \
    "$@"

#!/usr/bin/env bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Launch single-node 8-GPU ASR fine-tuning of Qwen3-Omni-30B-A3B-Instruct on
# any HF audio dataset (default: yuekai/WenetSpeech_Wu_1k).
#
# Which YAML is launched is controlled by the CONFIG env var:
#   CONFIG=examples/vlm_finetune/qwen3_omni_asr/wenetspeech_wu_sft.yaml  (default)
#   CONFIG=examples/vlm_finetune/qwen3_omni_asr/ami_sft.yaml             (AMI demo)
#
# `automodel`'s CLI only accepts the YAML as a positional argument, so do NOT
# pass --config-file at the script's end — set CONFIG instead.
#
# Required environment:
#   WENETSPEECH_WU_PATH  HF-cached path / HuggingFace dataset id for the gated
#                        yuekai/WenetSpeech_Wu_1k. The Wu YAML resolves
#                        ``${oc.env:WENETSPEECH_WU_PATH}`` from this var, so it
#                        is genuinely required for the Wu recipe. Other YAMLs
#                        (e.g. ami_sft.yaml) hard-code the dataset path and
#                        only need a non-empty stub here:
#                          export WENETSPEECH_WU_PATH=unused
#
# Optional environment:
#   PY                   Python interpreter (default: the bundled
#                        DTensorPolicyWorkerV2 venv that ships torch, nemo_automodel,
#                        soundfile, scipy, transformers; lacks qwen_omni_utils and
#                        torchcodec by design).
#   CONFIG               YAML config to launch (default: wenetspeech_wu_sft.yaml
#                        next to this script). See note above.
#   NPROC_PER_NODE       GPUs per node (default: 8).
#
# Any extra arguments are forwarded verbatim to `nemo_automodel.cli.app`, e.g.:
#   CONFIG=examples/vlm_finetune/qwen3_omni_asr/ami_sft.yaml \
#   WENETSPEECH_WU_PATH=unused \
#   examples/vlm_finetune/qwen3_omni_asr/train.sh \
#       --step_scheduler.max_steps 3 \
#       --checkpoint.checkpoint_dir /tmp/smoke_ckpt
#
set -euo pipefail

PY="${PY:-/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/wenetspeech_wu_sft.yaml}"

# Single-node 8-GPU contract: the YAML's ep_size=8 fsdp2 schedule assumes 8 GPUs.
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
if [[ -z "${WENETSPEECH_WU_PATH:-}" ]]; then
    echo "WENETSPEECH_WU_PATH is unset. Set it to a local HF-cached path or to the gated HF dataset id 'yuekai/WenetSpeech_Wu_1k' (after 'huggingface-cli login')." >&2
    exit 1
fi

exec "${PY}" -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes=1 \
    -m nemo_automodel.cli.app \
    "${CONFIG}" \
    "$@"

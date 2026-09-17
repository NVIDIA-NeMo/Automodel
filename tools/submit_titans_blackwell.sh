#!/bin/bash
# Rank compatible Blackwell clusters and submit a portable Titans run.

set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 SCALE VARIANT (--pilot|--full) [--total-gpus 8|16]" >&2
  echo "       SCALE: 170m | 340m | 760m" >&2
  echo "       VARIANT: baseline or a 170M component ablation" >&2
  exit 2
fi

SCALE=$1
VARIANT=$2
shift 2
TOTAL_GPUS=8
MODE=
MODE_COUNT=0

while (( $# )); do
  case "$1" in
    --pilot) MODE=pilot; MODE_COUNT=$((MODE_COUNT + 1)) ;;
    --full) MODE=full; MODE_COUNT=$((MODE_COUNT + 1)) ;;
    --total-gpus)
      shift
      TOTAL_GPUS=${1:?--total-gpus requires a value}
      ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

case "$SCALE" in 170m|340m|760m) ;; *) echo "unsupported scale: $SCALE" >&2; exit 2 ;; esac
case "$VARIANT" in
  baseline|no_persistent|no_convolution|no_momentum|no_weight_decay|depth3|depth4|linear_memory) ;;
  *) echo "unknown variant: $VARIANT" >&2; exit 2 ;;
esac
if (( MODE_COUNT != 1 )); then
  echo "choose exactly one of --pilot or --full" >&2
  exit 2
fi
if [[ $SCALE != 170m && $VARIANT != baseline ]]; then
  echo "component ablations are currently defined only for 170m" >&2
  exit 2
fi
if [[ $VARIANT == linear_memory ]]; then
  echo "linear_memory remains blocked until its momentum recurrence is vectorized" >&2
  exit 2
fi
if [[ $TOTAL_GPUS != 8 && $TOTAL_GPUS != 16 ]]; then
  echo "--total-gpus must be 8 or 16 to preserve global-batch divisibility" >&2
  exit 2
fi

for command in slurm-cli git python3; do
  command -v "$command" >/dev/null || {
    echo "required command not found: $command" >&2
    exit 1
  }
done
if [[ -n $(git status --short) ]]; then
  echo "AutoModel checkout is dirty; commit and push before submitting." >&2
  exit 1
fi

REMOTE_URL=${TITANS_AUTOMODEL_REMOTE:-https://github.com/NVIDIA-NeMo/Automodel.git}
BRANCH=${TITANS_AUTOMODEL_BRANCH:-ffrujeri/feat/titans-neural-memory}
LOCAL_SHA=$(git rev-parse HEAD)
PUSHED_SHA=$(git ls-remote "$REMOTE_URL" "refs/heads/$BRANCH" | awk '{print $1}')
if [[ $LOCAL_SHA != "$PUSHED_SHA" ]]; then
  echo "local HEAD $LOCAL_SHA is not pushed branch HEAD $PUSHED_SHA" >&2
  exit 1
fi

read -r -a CANDIDATES <<<"${TITANS_BLACKWELL_CLUSTERS:-aws-pdx-slurm-1 nsc-svg-slurm-1 aws-cmh-slurm-1 oci-hsg-cs-001}"
TARGET_ARGS=()
for cluster in "${CANDIDATES[@]}"; do
  TARGET_ARGS+=(--cluster "$cluster")
done

RECOMMENDATION=$(slurm-cli --json job recommend-target \
  "${TARGET_ARGS[@]}" \
  --name "titans-${SCALE}-${VARIANT}-${MODE}" \
  --total-gpus "$TOTAL_GPUS" \
  --gpu-family blackwell \
  --require-shared-root \
  --cpus-per-task 128 \
  --memory 0 \
  --time 4:00:00)

mapfile -t TARGET < <(
  python3 -c '
import json, sys
data = json.load(sys.stdin)
print(data["cluster"])
print(data["account"])
print(data["requested_nodes"])
print(data["requested_gpus_per_node"])
print(data.get("detected_gpu_type") or "unknown")
print(data["work_root"])
' <<<"$RECOMMENDATION"
)
CLUSTER=${TARGET[0]}
ACCOUNT=${TARGET[1]}
NODES=${TARGET[2]}
GPUS_PER_NODE=${TARGET[3]}
GPU_TYPE=${TARGET[4]}
SHARED_ROOT=${TARGET[5]}

if [[ $SHARED_ROOT != /scratch/fsw/portfolios/* ]]; then
  echo "refusing non-FSW work root for $CLUSTER / $ACCOUNT: $SHARED_ROOT" >&2
  exit 1
fi
REMOTE_ROOT=${TITANS_REMOTE_ROOT:-$SHARED_ROOT/titans-automodel}
REMOTE_CHECKOUT=$REMOTE_ROOT/checkouts/$LOCAL_SHA
MOUNT_ROOT=${TITANS_MOUNT_ROOT:-/scratch}

WANDB_STATUS=$(slurm-cli --cluster "$CLUSTER" shell \
  'if grep -qs "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then echo ready; else echo missing; fi')
if [[ $WANDB_STATUS == *missing* ]]; then
  if [[ -z ${WANDB_API_KEY:-} ]]; then
    echo "selected cluster $CLUSTER needs W&B authentication; export WANDB_API_KEY" >&2
    exit 1
  fi
  printf -v WANDB_KEY_QUOTED %q "$WANDB_API_KEY"
  slurm-cli --cluster "$CLUSTER" shell \
    "umask 077; touch \"\$HOME/.netrc\"; printf '\\nmachine api.wandb.ai\\n  login user\\n  password %s\\n' $WANDB_KEY_QUOTED >> \"\$HOME/.netrc\"; chmod 600 \"\$HOME/.netrc\"" \
    >/dev/null
fi

if [[ -n ${HF_TOKEN:-} ]]; then
  printf -v HF_TOKEN_QUOTED %q "$HF_TOKEN"
  slurm-cli --cluster "$CLUSTER" shell \
    "umask 077; mkdir -p '$REMOTE_ROOT/cache/huggingface'; printf '%s' $HF_TOKEN_QUOTED > '$REMOTE_ROOT/cache/huggingface/token'; chmod 600 '$REMOTE_ROOT/cache/huggingface/token'" \
    >/dev/null
fi

SYNC_COMMAND=$(cat <<EOF
set -euo pipefail
mkdir -p '$REMOTE_ROOT/checkouts' '$REMOTE_ROOT/logs'
if [[ -d '$REMOTE_CHECKOUT/.git' ]]; then
  test -z "\$(git -C '$REMOTE_CHECKOUT' status --short)"
  test "\$(git -C '$REMOTE_CHECKOUT' rev-parse HEAD)" = '$LOCAL_SHA'
else
  git clone --no-checkout '$REMOTE_URL' '$REMOTE_CHECKOUT'
  git -C '$REMOTE_CHECKOUT' checkout --detach '$LOCAL_SHA'
fi
EOF
)
slurm-cli --cluster "$CLUSTER" shell "$SYNC_COMMAND" --timeout 300 >/dev/null

if [[ $MODE == pilot ]]; then
  JOB_NAME=titans-${SCALE}-${VARIANT}-blackwell-pilot
  SBATCH_SCRIPT=examples/llm_pretrain/slurm/titans_blackwell_lmm_pilot.sbatch
  SUBMIT_NODES=$NODES
  SUBMIT_TASKS=$NODES
else
  JOB_NAME=titans-${SCALE}-fineweb-prepare
  SBATCH_SCRIPT=examples/llm_pretrain/slurm/titans_blackwell_fineweb_prepare.sbatch
  # Data preparation uses one node; it submits training with the resolved topology.
  SUBMIT_NODES=1
  SUBMIT_TASKS=1
fi
LOG_PATH=$REMOTE_ROOT/logs/${JOB_NAME}_%j.out
JOB_BODY=$(mktemp)
trap 'rm -f "$JOB_BODY"' EXIT
{
  printf 'export AUTOMODEL_CHECKOUT=%q\n' "$REMOTE_CHECKOUT"
  printf 'export TITANS_WORK_ROOT=%q\n' "$REMOTE_ROOT"
  printf 'export TITANS_MOUNT_ROOT=%q\n' "$MOUNT_ROOT"
  printf 'export TITANS_SCALE=%q\n' "$SCALE"
  printf 'export TITANS_EXPERIMENT=%q\n' "$VARIANT"
  printf 'export TITANS_TRAIN_NODES=%q\n' "$NODES"
  printf 'export TITANS_GPUS_PER_NODE=%q\n' "$GPUS_PER_NODE"
  printf 'export TITANS_PILOT_STEPS=%q\n' "${TITANS_PILOT_STEPS:-10}"
  printf 'export TITANS_LOCAL_BATCH_SIZE=%q\n' "${TITANS_LOCAL_BATCH_SIZE:-}"
  printf 'export TITANS_ACTIVATION_CHECKPOINTING=%q\n' "${TITANS_ACTIVATION_CHECKPOINTING:-false}"
  printf 'export TITANS_RUN_SUFFIX=%q\n' "${TITANS_RUN_SUFFIX:-}"
  printf 'export TITANS_TIME_LIMIT=%q\n' "${TITANS_TIME_LIMIT:-4:00:00}"
  awk 'NR == 1 {next} !/^#SBATCH/' "$SBATCH_SCRIPT"
} >"$JOB_BODY"

SUBMIT_RESULT=$(slurm-cli --cluster "$CLUSTER" --json job submit \
  --script "$JOB_BODY" \
  --name "$JOB_NAME" \
  --partition batch \
  --account "$ACCOUNT" \
  --nodes "$SUBMIT_NODES" \
  --ntasks "$SUBMIT_TASKS" \
  --cpus-per-task 128 \
  --gpus "$GPUS_PER_NODE" \
  --memory 0 \
  --time 4:00:00 \
  --workdir "$REMOTE_CHECKOUT" \
  --output "$LOG_PATH")
JOB_ID=$(printf '%s' "$SUBMIT_RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['job_id'])")

echo "Submitted $SCALE/$VARIANT $MODE to $CLUSTER / $ACCOUNT"
echo "Topology: $NODES node(s) x $GPUS_PER_NODE $GPU_TYPE GPU(s) = $TOTAL_GPUS total"
echo "Job:      $JOB_ID"
if [[ $MODE == pilot ]]; then
  echo "W&B:      https://wandb.ai/nvidia/titans-paper-reproduction"
else
  case "$SCALE" in 170m|340m) TOKEN_BUDGET=15B ;; 760m) TOKEN_BUDGET=30B ;; esac
  RUN_ID=titans-${SCALE}-${VARIANT}-${TOKEN_BUDGET}-blackwell-v1
  echo "W&B:      https://wandb.ai/nvidia/titans-paper-reproduction/runs/$RUN_ID"
fi
echo "Status:   slurm-cli --cluster $CLUSTER job get $JOB_ID"
echo "Log:      slurm-cli --cluster $CLUSTER file read ${LOG_PATH//%j/$JOB_ID} --tail 100"

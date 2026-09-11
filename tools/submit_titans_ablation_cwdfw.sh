#!/bin/bash
# Submit an immutable, multi-node Titans ablation through slurm-cli.

set -euo pipefail

if (( $# < 2 )); then
  echo "usage: $0 VARIANT (--pilot|--full) [--nodes 1|2]" >&2
  exit 2
fi

VARIANT=$1
shift
NODES=2
TOTAL_STEPS=
WARMUP_STEPS=
MODE_COUNT=0

while (( $# )); do
  case "$1" in
    --pilot)
      TOTAL_STEPS=10
      WARMUP_STEPS=1
      MODE_COUNT=$((MODE_COUNT + 1))
      ;;
    --full)
      TOTAL_STEPS=28610
      WARMUP_STEPS=286
      MODE_COUNT=$((MODE_COUNT + 1))
      ;;
    --nodes)
      shift
      NODES=${1:?--nodes requires a value}
      ;;
    *)
      echo "usage: $0 VARIANT (--pilot|--full) [--nodes 1|2]" >&2
      exit 2
      ;;
  esac
  shift
done

case "$VARIANT" in
  baseline|no_persistent|no_convolution|no_momentum|no_weight_decay|depth3|depth4|linear_memory) ;;
  *)
    echo "unknown variant: $VARIANT" >&2
    exit 2
    ;;
esac
if (( MODE_COUNT != 1 )); then
  echo "choose exactly one of --pilot or --full" >&2
  exit 2
fi
if [[ $NODES != 1 && $NODES != 2 ]]; then
  echo "--nodes must be 1 or 2; the current 16-shard dataset supports at most 16 ranks" >&2
  exit 2
fi
if [[ $VARIANT == linear_memory ]]; then
  echo "linear_memory is not launchable until its momentum recurrence is vectorized" >&2
  exit 2
fi

CLUSTER=${SLURM_CLUSTER:-cw-dfw}
REMOTE_ROOT=${TITANS_REMOTE_ROOT:-/lustre/fsw/portfolios/coreai/users/ffrujeri/titans-automodel}
REMOTE_URL=${TITANS_AUTOMODEL_REMOTE:-https://github.com/NVIDIA-NeMo/Automodel.git}
BRANCH=${TITANS_AUTOMODEL_BRANCH:-ffrujeri/feat/titans-neural-memory}
SBATCH_SCRIPT=examples/llm_pretrain/slurm/cwdfw_titans_170m_ablation.sbatch

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

LOCAL_SHA=$(git rev-parse HEAD)
PUSHED_SHA=$(git ls-remote "$REMOTE_URL" "refs/heads/$BRANCH" | awk '{print $1}')
if [[ $LOCAL_SHA != "$PUSHED_SHA" ]]; then
  echo "local HEAD $LOCAL_SHA is not pushed branch HEAD $PUSHED_SHA" >&2
  exit 1
fi

REMOTE_CHECKOUT=$REMOTE_ROOT/checkouts/$LOCAL_SHA
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

JOB_NAME=titans-170m-${VARIANT}
if (( TOTAL_STEPS < 28610 )); then
  JOB_NAME=${JOB_NAME}-pilot
  MAX_STEPS_PER_RUN=5
else
  MAX_STEPS_PER_RUN=$((1500 * NODES))
fi
LOG_PATH=$REMOTE_ROOT/logs/${JOB_NAME}_%j.out
JOB_BODY=$(mktemp)
trap 'rm -f "$JOB_BODY"' EXIT
{
  printf 'export AUTOMODEL_CHECKOUT=%q\n' "$REMOTE_CHECKOUT"
  printf 'export TITANS_EXPERIMENT=%q\n' "$VARIANT"
  printf 'export TITANS_TOTAL_STEPS=%q\n' "$TOTAL_STEPS"
  printf 'export TITANS_WARMUP_STEPS=%q\n' "$WARMUP_STEPS"
  printf 'export TITANS_MAX_STEPS_PER_RUN=%q\n' "$MAX_STEPS_PER_RUN"
  awk 'NR == 1 {next} !/^#SBATCH/' "$SBATCH_SCRIPT"
} >"$JOB_BODY"

SUBMIT_RESULT=$(slurm-cli --cluster "$CLUSTER" --json job submit \
  --script "$JOB_BODY" \
  --name "$JOB_NAME" \
  --partition batch \
  --account coreai_dlalgo_compeval \
  --nodes "$NODES" \
  --ntasks "$NODES" \
  --cpus-per-task 128 \
  --gpus 8 \
  --memory 0 \
  --time 4:00:00 \
  --workdir "$REMOTE_CHECKOUT" \
  --output "$LOG_PATH")
JOB_ID=$(printf '%s' "$SUBMIT_RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['data']['job_id'])")

echo "Submitted $VARIANT on $NODES node(s): job $JOB_ID"
echo "Status: slurm-cli --cluster $CLUSTER job get $JOB_ID"
echo "Log:    slurm-cli --cluster $CLUSTER file read ${LOG_PATH//%j/$JOB_ID} --tail 100"

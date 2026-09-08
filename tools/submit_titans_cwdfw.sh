#!/bin/bash
# Submit and optionally monitor the Titans 170M smoke through slurm-cli.

set -euo pipefail

CLUSTER=${SLURM_CLUSTER:-cw-dfw}
REMOTE_ROOT=${TITANS_REMOTE_ROOT:-/lustre/fsw/portfolios/coreai/users/ffrujeri/titans-automodel}
REMOTE_CHECKOUT=$REMOTE_ROOT/automodel
REMOTE_URL=${TITANS_AUTOMODEL_REMOTE:-https://github.com/NVIDIA-NeMo/Automodel.git}
BRANCH=${TITANS_AUTOMODEL_BRANCH:-ffrujeri/feat/titans-neural-memory}
SBATCH_SCRIPT=examples/llm_pretrain/slurm/cwdfw_titans_170m_4k_smoke.sbatch
WAIT=false

if [[ ${1:-} == "--wait" ]]; then
  WAIT=true
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--wait]" >&2
  exit 2
fi

for command in slurm-cli git python3; do
  command -v "$command" >/dev/null || {
    echo "required command not found: $command" >&2
    exit 1
  }
done

if [[ -n $(git status --short) ]]; then
  echo "AutoModel checkout is dirty; commit and push the workflow changes first." >&2
  exit 1
fi

LOCAL_SHA=$(git rev-parse HEAD)
PUSHED_SHA=$(git ls-remote "$REMOTE_URL" "refs/heads/$BRANCH" | awk '{print $1}')
if [[ -z $PUSHED_SHA ]]; then
  echo "remote branch not found: $REMOTE_URL $BRANCH" >&2
  exit 1
fi
if [[ $LOCAL_SHA != "$PUSHED_SHA" ]]; then
  echo "local HEAD $LOCAL_SHA is not the pushed branch HEAD $PUSHED_SHA" >&2
  echo "push $BRANCH before submitting so the cluster runs exactly this checkout." >&2
  exit 1
fi

json_field() {
  local expression=$1
  python3 -c "import json,sys; data=json.load(sys.stdin); print($expression)"
}

SYNC_COMMAND=$(cat <<EOF
set -euo pipefail
mkdir -p '$REMOTE_ROOT/logs'
if [[ -d '$REMOTE_CHECKOUT/.git' ]]; then
  test -z "\$(git -C '$REMOTE_CHECKOUT' status --short)"
  git -C '$REMOTE_CHECKOUT' fetch origin '$BRANCH'
  git -C '$REMOTE_CHECKOUT' checkout '$BRANCH'
  git -C '$REMOTE_CHECKOUT' merge --ff-only 'origin/$BRANCH'
else
  git clone --single-branch --branch '$BRANCH' '$REMOTE_URL' '$REMOTE_CHECKOUT'
fi
test "\$(git -C '$REMOTE_CHECKOUT' rev-parse HEAD)" = '$LOCAL_SHA'
mkdir -p '$REMOTE_CHECKOUT/slurm_jobs'
EOF
)

echo "Syncing $BRANCH@$LOCAL_SHA to $CLUSTER:$REMOTE_CHECKOUT"
SYNC_RESULT=$(slurm-cli --cluster "$CLUSTER" --json shell "$SYNC_COMMAND" --timeout 300)
SYNC_OK=$(printf '%s' "$SYNC_RESULT" | json_field "data.get('success', False)")
if [[ $SYNC_OK != "True" ]]; then
  printf '%s\n' "$SYNC_RESULT" >&2
  exit 1
fi

LOG_PATH=$REMOTE_ROOT/logs/titans-170m-4k-smoke_%j.out
JOB_BODY=$(mktemp)
trap 'rm -f "$JOB_BODY"' EXIT
awk 'NR == 1 {next} !/^#SBATCH/' "$SBATCH_SCRIPT" >"$JOB_BODY"

echo "Submitting the full-shape 4K smoke"
SUBMIT_RESULT=$(slurm-cli --cluster "$CLUSTER" --json job submit \
  --script "$JOB_BODY" \
  --name titans-170m-4k-smoke \
  --partition batch \
  --account coreai_dlalgo_compeval \
  --nodes 1 \
  --ntasks 1 \
  --gpus 8 \
  --memory 0 \
  --time 1:00:00 \
  --workdir "$REMOTE_CHECKOUT" \
  --output "$LOG_PATH")
JOB_ID=$(printf '%s' "$SUBMIT_RESULT" | json_field "data['job_id']")
FINAL_LOG=${LOG_PATH//%j/$JOB_ID}

echo "Submitted job $JOB_ID"
echo "Status: slurm-cli --cluster $CLUSTER job get $JOB_ID"
echo "Log:    slurm-cli --cluster $CLUSTER file read $FINAL_LOG --tail 100"

if [[ $WAIT != true ]]; then
  exit 0
fi

while true; do
  JOB_RESULT=$(slurm-cli --cluster "$CLUSTER" --json job get "$JOB_ID")
  STATE=$(printf '%s' "$JOB_RESULT" | json_field "data.get('state', 'UNKNOWN')")
  echo "job $JOB_ID: $STATE"
  case "$STATE" in
    COMPLETED)
      break
      ;;
    FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL)
      slurm-cli --cluster "$CLUSTER" file read "$FINAL_LOG" --tail 200 || true
      exit 1
      ;;
  esac
  sleep 30
done

slurm-cli --cluster "$CLUSTER" file read "$FINAL_LOG" --tail 200

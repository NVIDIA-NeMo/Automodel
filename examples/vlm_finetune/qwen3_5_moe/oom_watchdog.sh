#!/bin/bash
# Watch every node's log for an OOM or crash and tear the run down the moment one appears.
#
# A failed rank does not end the run: the survivors block in a collective holding all their
# memory until the NCCL timeout (120 min here), so the teardown has to be immediate and
# cannot wait for a human. Each node's log carries all 8 of its local ranks, so all four
# are watched -- an OOM often hits a non-zero rank first.
set -uo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
RUN_NAME=${RUN_NAME:-affine}
RUN_DIR=${RUN_DIR:-$(ls -td "$REPO"/logs/"$RUN_NAME"-* | head -1)}
# Last optimizer step the run logs; the watchdog exits cleanly once it appears.
FINAL_STEP=${FINAL_STEP:-5020}
PATTERN='OutOfMemoryError|CUDA out of memory|DeepEP error|Xid|CUDA error|NCCL.*(unhandled|aborting)'

echo "watchdog: $RUN_DIR"
while true; do
  hit=$(grep -lEm1 "$PATTERN" "$RUN_DIR"/rank*.log 2>/dev/null | head -1)
  if [ -n "$hit" ]; then
    echo "WATCHDOG_TRIGGERED in $hit"
    grep -hEm2 "$PATTERN" "$RUN_DIR"/rank*.log 2>/dev/null | head -4
    RUN_NAME=$RUN_NAME bash "$REPO/examples/vlm_finetune/qwen3_5_moe/teardown.sh"
    echo "WATCHDOG_TEARDOWN_DONE"
    exit 10
  fi
  # Stop watching once the run is over.
  if ! pgrep -f "launch_node.sh 0" > /dev/null 2>&1; then
    grep -qE "step $FINAL_STEP \|" "$RUN_DIR"/rank0.log 2>/dev/null && { echo "WATCHDOG_RUN_COMPLETE"; exit 0; }
    sleep 30
    pgrep -f "launch_node.sh 0" > /dev/null 2>&1 || { echo "WATCHDOG_RANK0_GONE"; exit 11; }
  fi
  sleep 10
done

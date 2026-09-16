#!/bin/bash
# Fan out the N x 8 x H200 run from rank 0's node.
#
# Starts launch_node.sh (NNODES=4) on every node over SSH, rank 0 locally and
# last so the rendezvous master is up before the others connect, and samples nvidia-smi
# on every node for the duration -- the recipe's `mem` field is
# torch.cuda.max_memory_allocated and sits 20-30 GiB below the real footprint, so the smi
# trace is the one to judge headroom by (RUNBOOK.md section 11).
#
#   bash examples/vlm_finetune/qwen3_5_moe/launch_cluster.sh [--key.sub value ...]
#
# Knobs: CONFIG, RUN_NAME, MASTER_ADDR, NNODES, plus anything launch_node.sh takes.
set -euo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
LAUNCHER=examples/vlm_finetune/qwen3_5_moe/launch_node.sh
CONFIG=${CONFIG:-examples/vlm_finetune/qwen3_5_moe/qwen3_6_35b_4node_ep8_base.yaml}
RUN_NAME=${RUN_NAME:-affine}
MASTER_ADDR=${MASTER_ADDR:-10.30.0.2}
NNODES=${NNODES:-4}

# Forward any NCCL tuning vars set in this shell to the per-node launchers.
NCCL_PASSTHROUGH=""
for v in NCCL_DEBUG NCCL_ALGO NCCL_PROTO NCCL_CROSS_NIC NCCL_NVLS_ENABLE \
         NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_COLLNET_ENABLE; do
  [ -n "${!v:-}" ] && NCCL_PASSTHROUGH="$NCCL_PASSTHROUGH $v=${!v}"
done

# node_rank -> private IP. Rank 0 is this box (the NFS server and rendezvous master).
# Spot hosts are re-provisioned: re-check these addresses (and the SSH user below) after
# every re-provisioning, and keep them in sync with teardown.sh and MASTER_ADDR.
NODE_IPS=(10.30.0.2 10.30.0.4 10.30.0.3 10.30.0.7)

STAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$REPO/logs/$RUN_NAME-$STAMP"
RUNDIR="$REPO/logs/$RUN_NAME-$STAMP"
echo "run dir: $RUNDIR"
echo "config:  $CONFIG"

# `ssh -f` backgrounds the client after authentication and the remote redirects close
# every inherited descriptor. Backgrounding inside the remote shell instead (`... & disown`)
# leaves the SSH channel open, so this loop blocks on the first node and the later ranks
# never start.
for rank in $(seq 1 $((NNODES - 1))); do
  ip=${NODE_IPS[$rank]}
  echo "starting rank $rank on $ip"
  ssh -f -n -o BatchMode=yes "chilaingoc@$ip" \
    "cd $REPO && setsid env NNODES=$NNODES MASTER_ADDR=$MASTER_ADDR CONFIG=$CONFIG RUN_NAME=$RUN_NAME \
       ${ENTRY:+ENTRY=$ENTRY} $NCCL_PASSTHROUGH \
       bash $LAUNCHER $rank $* > $RUNDIR/rank$rank.log 2>&1 < /dev/null"
done

# Give the non-zero ranks a head start on `docker run` so they are waiting at the
# rendezvous when rank 0 opens it, rather than the other way round.
sleep 10

# nvidia-smi sampler on every node, including this one.
for rank in $(seq 0 $((NNODES - 1))); do
  ip=${NODE_IPS[$rank]}
  ssh -f -n -o BatchMode=yes "chilaingoc@$ip" \
    "setsid nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
       --format=csv,noheader,nounits -l 5 > $RUNDIR/smi_rank$rank.csv 2>&1 < /dev/null"
done

echo "starting rank 0 locally"
cd "$REPO"
NNODES=$NNODES MASTER_ADDR=$MASTER_ADDR CONFIG=$CONFIG RUN_NAME=$RUN_NAME \
  bash "$LAUNCHER" 0 "$@" 2>&1 | tee "$RUNDIR/rank0.log"

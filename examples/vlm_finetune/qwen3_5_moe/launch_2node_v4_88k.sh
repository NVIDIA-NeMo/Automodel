#!/bin/bash
# Launch one node of the 2 x 8 x H200 v4_88k run (default: qwen3_6_35b_v4_88k_2node_ep8.yaml).
# See MULTINODE_2xH200_v4_88k.md for the node setup this assumes.
#
# Run from the repo checkout on the shared filesystem, once per node, node rank 0 first:
#   node-2:  bash examples/vlm_finetune/qwen3_5_moe/launch_2node_v4_88k.sh 0 [--key.sub value ...]
#   node-3:  bash examples/vlm_finetune/qwen3_5_moe/launch_2node_v4_88k.sh 1 [--key.sub value ...]
#
# Environment knobs: IMAGE, CONFIG, MASTER_ADDR (node-0's private IP), MASTER_PORT,
# TRITON_CACHE_HOST, TORCHRUN_ARGS (extra torchrun flags), NCCL_DEBUG, NVSHMEM_DEBUG,
# TRITON_PRINT_AUTOTUNING=1 (log every autotune benchmark).
# ENTRY swaps the torchrun target for a plain script, e.g. the comms smoke test:
#   ENTRY=examples/vlm_finetune/qwen3_5_moe/comm_check_2node.py bash ...launch_2node_v4_88k.sh 0
set -euo pipefail

NODE_RANK=${1:?usage: launch_2node_v4_88k.sh <node_rank> [overrides...]}
shift
MASTER_ADDR=${MASTER_ADDR:-10.30.0.2}
MASTER_PORT=${MASTER_PORT:-29500}
IMAGE=${IMAGE:-nemo-automodel:26.08.00-deepep564}
CONFIG=${CONFIG:-examples/vlm_finetune/qwen3_5_moe/qwen3_6_35b_v4_88k_2node_ep8.yaml}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)

# HF_TOKEN (and optionally WANDB_API_KEY) come from the repo-root .env, which is not committed.
if [ -f "$REPO/.env" ]; then set -a; . "$REPO/.env"; set +a; fi

if [ -n "${ENTRY:-}" ]; then
  TARGET="$ENTRY"
else
  TARGET="-m nemo_automodel.cli.app $CONFIG"
fi

# Per-rank wrapper: NVSHMEM picks a NIC per PE, and on a3-ultragpu-8g two NICs are equally
# close to each GPU (topo shows PIX for both), so pin GPU N to its own rail NIC gpuNrdma0.
# DeepEP's internode kernels send GPU N -> GPU N on the other node, which keeps traffic on-rail.
#
# Caches. The 8 ranks of a node share one container. The *inductor* cache must be per
# rank: concurrent writes to /tmp/torchinductor_root corrupt kernels another rank then
# loads ("CUDA driver error: invalid argument" from the static Triton launcher inside
# torchao AdamW8bit's compiled step; 2/2 with a shared dir, 0/2 per rank). The *Triton*
# cache is safe to share (3/3 clean on the same probe) and must be: FLA's GatedDeltaNet
# kernels are JIT-specialized and autotuned per batch length, and with length-grouped
# batching a cold cache costs ~30 s stalls on half the steps of the first run. It lives
# on the host RAID so it survives the container and warms across runs.
TRITON_CACHE_HOST=${TRITON_CACHE_HOST:-/mnt/fast/triton_cache}
mkdir -p "$TRITON_CACHE_HOST"
RANK_ENV=$(cat <<'EOF'
HCAS=(rocep145s0 rocep146s0 rocep152s0 rocep153s0 rocep198s0 rocep199s0 rocep205s0 rocep206s0)
export NVSHMEM_HCA_LIST="${HCAS[$LOCAL_RANK]}:1"
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_rank${LOCAL_RANK}
export TRITON_CACHE_DIR=/triton_cache
exec python "$@"
EOF
)

mkdir -p "$REPO/logs"
LOG="$REPO/logs/$(date +%Y%m%d_%H%M%S)_node${NODE_RANK}.log"
echo "node_rank=$NODE_RANK master=$MASTER_ADDR:$MASTER_PORT image=$IMAGE log=$LOG"

docker run --rm --name "v4_88k_node${NODE_RANK}" \
  --gpus all --privileged --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$REPO":/opt/Automodel \
  -v /mnt/fast/hf:/hf \
  -v /usr/local/gib:/usr/local/gib:ro \
  -v "$TRITON_CACHE_HOST":/triton_cache \
  -w /opt/Automodel \
  -e HF_HOME=/hf -e HF_TOKEN -e WANDB_API_KEY \
  -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  -e TORCHDYNAMO_DISABLE -e TRITON_PRINT_AUTOTUNING \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_SOCKET_IFNAME=enp0s19 -e GLOO_SOCKET_IFNAME=enp0s19 \
  -e NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=enp0s19 \
  -e NVSHMEM_IB_GID_INDEX=3 \
  -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" -e NVSHMEM_DEBUG="${NVSHMEM_DEBUG:-WARN}" \
  -e NVSHMEM_DEBUG_SUBSYS -e NVSHMEM_IBGDA_NIC_HANDLER \
  -e RANK_ENV="$RANK_ENV" -e TORCHRUN_ARGS="${TORCHRUN_ARGS:-}" \
  "$IMAGE" \
  bash -c '
    set -e
    # Google gIB NCCL (RoCE on a3-ultragpu-8g): its libnccl + net/tuner plugins.
    source /usr/local/gib/scripts/set_nccl_env.sh
    export LD_LIBRARY_PATH=/usr/local/gib/lib64:${LD_LIBRARY_PATH:-}
    printf "%s\n" "$RANK_ENV" > /tmp/rank_env.sh
    exec torchrun --nnodes 2 --node-rank '"$NODE_RANK"' --nproc-per-node 8 \
      --master-addr '"$MASTER_ADDR"' --master-port '"$MASTER_PORT"' $TORCHRUN_ARGS \
      --no-python /bin/bash /tmp/rank_env.sh '"$TARGET"' "$@"
  ' _ "$@" 2>&1 | tee "$LOG"

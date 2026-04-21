#!/bin/bash
# Usage: bash examples/vlm_finetune/gemma4/run_cp4_test.sh <node_rank>
# Example on master:  bash examples/vlm_finetune/gemma4/run_cp4_test.sh 0
# Example on node1:   bash examples/vlm_finetune/gemma4/run_cp4_test.sh 1

NODE_RANK=${1:-0}

# Always run from repo root regardless of where the script is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../.."

export NCCL_SOCKET_IFNAME=eth0   # 屏蔽 IPv6，只用 eth0；如果网卡名不同改这里
export NCCL_IB_DISABLE=0

torchrun --nnodes=4 --nproc-per-node=8 --node-rank=$NODE_RANK --rdzv-backend=c10d --rdzv-endpoint=10.178.157.101:29500 nemo_automodel/recipes/vlm/finetune.py -c examples/vlm_finetune/gemma4/gemma4_E4B_cp4_mock_image.yaml

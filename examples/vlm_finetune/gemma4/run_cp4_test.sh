#!/bin/bash
# Usage: bash run_cp4_test.sh <node_rank>
# Example on master:  bash run_cp4_test.sh 0
# Example on node1:   bash run_cp4_test.sh 1

NODE_RANK=${1:-0}

torchrun --nnodes=4 --nproc-per-node=8 --node-rank=$NODE_RANK --rdzv-backend=c10d --rdzv-endpoint=10.178.157.101:29500 nemo_automodel/recipes/vlm/finetune.py -c examples/vlm_finetune/gemma4/gemma4_E4B_cp4_mock_image.yaml

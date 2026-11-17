#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Scheduled qualification on FOUR GPUs. Do not add to the two-GPU PR runner.
set -euo pipefail
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
output="${INKLING2_COMPOSITION_OUTPUT_DIR:-$(mktemp -d)}"
mkdir -p "$output"
python -c 'import torch; assert torch.cuda.device_count() >= 4, "PP2+EP2 requires four visible GPUs"'
for case in fp32-pp2-dp2 fp32-pp2-ep2 master-bf16-pp2-ep2 master-bf16-pp2-ep2-interleaved \
    fp32-pp2-dp2-flex fp32-pp2-ep2-flex master-bf16-pp2-ep2-flex master-bf16-pp2-ep2-interleaved-flex; do
    mkdir -p "$output/$case"
    timeout 600 python -m torch.distributed.run --standalone --nproc_per_node=4 \
        tests/functional_tests/models/inkling2/pipeline_composition.py "$case" "$output/$case"
done
for attention in eager flex; do
    for schedule in 1f1b interleaved1f1b; do
        python tests/functional_tests/models/inkling2/pipeline_composition_recipe.py \
            "$schedule" "$output/recipe-$attention-$schedule" "$attention"
    done
done

#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -xeuo pipefail

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export CUDA_VISIBLE_DEVICES="0,1"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

RUN_DIR=$(mktemp -d)
trap 'rm -rf "$RUN_DIR"' EXIT
python tests/functional_tests/parallelism/run_pp_dataloader.py --work-dir "$RUN_DIR" --prepare-only
python -m torch.distributed.run --standalone --nproc-per-node=2 \
    tests/functional_tests/parallelism/run_pp_dataloader.py --work-dir "$RUN_DIR"

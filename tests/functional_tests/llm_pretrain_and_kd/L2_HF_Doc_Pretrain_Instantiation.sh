#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

export PYTHONPATH=${PYTHONPATH:-}:$(pwd)

torchrun --nproc-per-node=8 --nnodes=1 \
  tests/functional_tests/llm_pretrain_and_kd/hf_doc_pretrain_instantiation.py

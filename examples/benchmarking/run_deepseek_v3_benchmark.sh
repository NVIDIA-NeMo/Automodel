#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

# Example script to run DeepSeek V3 benchmarking with different configurations

# Basic run with default configuration
echo "Running basic benchmark with default config..."
torchrun --nproc-per-node=8 examples/benchmarking/deepseek_v3_config.py

# Run with custom number of iterations
echo "Running benchmark with 50 iterations..."
torchrun --nproc-per-node=8 examples/benchmarking/deepseek_v3_config.py \
  --config examples/benchmarking/deepseek_v3_benchmark.yaml \
  --training.iters=50

# Run with different parallelism settings
echo "Running benchmark with PP=8, EP=4..."
torchrun --nproc-per-node=8 examples/benchmarking/deepseek_v3_config.py \
  --config examples/benchmarking/deepseek_v3_benchmark.yaml \
  --distributed.pp=8 \
  --distributed.ep=4 \
  --distributed.dp_shard=1

# Run with profiling enabled
echo "Running benchmark with NSYS profiling..."
torchrun --nproc-per-node=8 examples/benchmarking/deepseek_v3_config.py \
  --config examples/benchmarking/deepseek_v3_benchmark.yaml \
  --profiling.nsys_start=5 \
  --profiling.nsys_end=10

# Run with a different model
echo "Running benchmark with a different model..."
torchrun --nproc-per-node=8 examples/benchmarking/deepseek_v3_config.py \
  --config examples/benchmarking/deepseek_v3_benchmark.yaml \
  --model.pretrained_model_name_or_path="deepseek-ai/deepseek-coder-33b-instruct" \
  --model.num_layers=32

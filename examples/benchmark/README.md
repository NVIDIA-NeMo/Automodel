# Benchmarking V3 - Recipe-Based Benchmarking

This directory contains a new recipe-based approach to benchmarking that subclasses `TrainFinetuneRecipeForNextTokenPrediction` from `nemo_automodel/recipes/llm/train_ft.py`.

## Overview

The V3 benchmarking recipe reuses the `setup()` and `_forward_backward_step()` methods from the parent training recipe while implementing a custom benchmarking loop with timers and profiling support.

## Key Components

### 1. BenchmarkingRecipeForNextTokenPrediction
- **Location**: `benchmark_recipe.py`
- **Purpose**: Main recipe class that extends the training recipe for benchmarking
- **Key Features**:
  - Reuses parent's `setup()` method for model, optimizer, and distributed initialization
  - Reuses parent's `_forward_backward_step()` for forward/backward passes
  - Implements custom `run_benchmark()` method with synthetic data generation
  - Includes timers for performance measurement
  - Supports nsys profiling
  - Calculates Model FLOPs Utilization (MFU)

### 2. MockBenchmarkDataset
- **Location**: `mock_dataset.py`
- **Purpose**: Generates synthetic data for benchmarking (similar to the original benchmarking script)
- **Features**:
  - Creates random token sequences
  - Generates appropriate labels and position_ids
  - Configurable vocab size and sequence length

### 3. Configurations
- **Location**: `configs/`
- **Format**: YAML files following the train_ft.py recipe style
- **Available Configs**:
  - `deepseek_v3_torch.yaml` / `deepseek_v3_te_deepep.yaml`
  - `gptoss_20b_torch.yaml` / `gptoss_20b_te_deepep.yaml`
  - `gptoss_120b_te_deepep.yaml`
  - `kimi_k2_te_deepep.yaml`
  - `moonlight_16b_torch.yaml` / `moonlight_16b_te_deepep.yaml`
  - `qwen3_moe_30b_torch.yaml` / `qwen3_moe_30b_te_deepep.yaml`

## Configuration Structure

Each config includes the following sections:

```yaml
# Training parameters (benchmarking-specific)
training:
  seq_len: 4096
  global_batch_size: 256
  local_batch_size: 4
  steps: 30
  warmup_steps: 10

# Profiling settings
profiling:
  nsys_start: -1  # Set to iteration number to start profiling
  nsys_end: -1    # Set to iteration number to stop profiling

# Model configuration
model:
  _target_: nemo_automodel.components.models.*.model.*ForCausalLM.from_config
  pretrained_model_name_or_path: <model_id>
  is_meta_device: true
  backend:
    # Backend configuration (torch, te, deepep, etc.)

# Distributed training setup
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  tp_size: 1
  cp_size: 1
  pp_size: 1
  ep_size: 8
  # ...

# Pipeline parallelism (if pp_size > 1)
autopipeline:
  _target_: nemo_automodel.components.distributed.pipelining.AutoPipeline
  pp_schedule: interleaved1f1b
  pp_microbatch_size: 4
  # ...

# Dataset and dataloader
dataset:
  _target_: examples.benchmarking_v3.mock_dataset.MockBenchmarkDataset
  vocab_size: 102400
  seq_len: 4096

dataloader:
  _target_: torch.utils.data.DataLoader
  batch_size: 4
  num_workers: 0

# Optimizer
optimizer:
  _target_: torch.optim.Adam
  lr: 1.0e-4
  # ...
```

## Usage

### Running a Benchmark

```bash
# Single GPU
python examples/benchmarking_v3/benchmark_recipe.py --config examples/benchmarking_v3/configs/moonlight_16b_torch.yaml

# Multi-GPU with torchrun
torchrun --nproc_per_node=8 examples/benchmarking_v3/benchmark_recipe.py --config examples/benchmarking_v3/configs/qwen3_moe_30b_te_deepep.yaml

# With nsys profiling
# Edit config to set nsys_start and nsys_end, then run:
nsys profile -o benchmark_profile torchrun --nproc_per_node=8 examples/benchmarking_v3/benchmark_recipe.py --config <config_path>
```

### Customizing Configurations

To create a custom benchmark config:

1. Copy an existing config from `configs/`
2. Modify the relevant sections:
   - `training.seq_len` - Sequence length
   - `training.global_batch_size` - Total batch size across all GPUs
   - `training.local_batch_size` - Batch size per GPU
   - `training.steps` - Number of iterations
   - `distributed.pp_size`, `distributed.ep_size` - Parallelism settings
   - `model.pretrained_model_name_or_path` - Model to benchmark
   - `model.backend` - Backend configuration (torch, te, deepep)

## Differences from Original Benchmarking Script

### Similarities
- Generates synthetic data with random tokens
- Supports timers for performance measurement
- Supports nsys profiling
- Calculates MFU (Model FLOPs Utilization)
- Supports gradient accumulation
- Supports pipeline parallelism

### Key Differences
1. **Structure**: Uses recipe pattern from train_ft.py
2. **Configuration**: Uses YAML configs instead of direct instantiation
3. **Setup**: Reuses parent's setup() method for consistency
4. **Forward/Backward**: Reuses parent's _forward_backward_step() method
5. **Dataset**: Uses a proper Dataset class instead of inline generation
6. **Extensibility**: Easier to extend with additional features from the parent recipe

## Benefits of Recipe-Based Approach

1. **Code Reuse**: Leverages existing setup and forward/backward logic
2. **Consistency**: Ensures benchmarking uses the same code paths as training
3. **Maintainability**: Changes to parent recipe automatically propagate
4. **Configurability**: YAML configs provide better organization
5. **Extensibility**: Easy to add validation, checkpointing, or other features if needed

## Performance Metrics

The benchmarking recipe provides:
- **Iteration time**: Time per training iteration
- **TFLOPs/GPU**: Theoretical FLOPs per GPU
- **MFU**: Model FLOPs Utilization (percentage of peak hardware performance)
- **Memory usage**: Peak GPU memory allocation
- **Loss values**: Training loss per iteration (for verification)

## Future Enhancements

Potential improvements:
- Add communication profiling
- Add memory profiling
- Support for mixed precision benchmarking
- Comparison with baseline metrics
- Automated sweep over different configurations
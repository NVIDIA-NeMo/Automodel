# Nemotron NanoV3 Pipeline Parallelism Support

## Overview

Enables 8-way pipeline parallelism for Nemotron NanoV3 30B (Mamba2 hybrid architecture).

## Usage

### Old Mixtral runs fine
```bash
torchrun --nproc-per-node=8 nemo_automodel/recipes/llm/benchmark.py \
  --config examples/llm_finetune/mistral/mixtral-8x7b-v0-1_benchmark.yaml
```

### Nemotron without PP
```bash
torchrun --nproc-per-node=8 nemo_automodel/recipes/llm/benchmark.py \
  --config examples/llm_finetune/nemotron/nemotron_nano_v3_squad_benchmark.yaml
```

### Nemotron with PP
```bash
torchrun --nproc-per-node=8 nemo_automodel/recipes/llm/benchmark.py \
  --config examples/llm_finetune/nemotron/nemotron_nano_v3_pp_benchmark.yaml
```

## Changes

### Core Fixes
1. **parallelizer.py**: Unpack ModuleList/ModuleDict in layer extraction
2. **functional.py**: Support `backbone.*` model structure (vs `model.*`)
3. **hf_utils.py**: Support `backbone.embeddings` (vs `embed_tokens`)
4. **flops_utils.py**: Calibrated Mamba2 SSM FLOPs (~16× empirical, 2.7× vs matmul training baseline)
5. **train_ft.py**: Pass `trust_remote_code` to AutoConfig

### SSM FLOP Calibration

Mamba2 SSM training operations empirically require **`~16×B×L×D×N` FLOPs**.

To verify or recalibrate:
```bash
python tools/profile_ssm_calibration.py
```

## Architecture Notes

Nemotron uses Mamba2 state-space models instead of attention:
- Structure: `NemotronHForCausalLM` → `backbone.embeddings/layers/norm`
- Layers: 23 Mamba2 + 23 MOE + 6 Attention (52 total)
- PP splits 52 layers across 8 GPUs (~7 layers/stage)

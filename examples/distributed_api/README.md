# Distributed API Examples

This directory contains examples demonstrating how to use `NeMoAutoModelForCausalLM` as a drop-in replacement for Hugging Face's `AutoModelForCausalLM` with built-in distributed training support.

## Overview

NeMo AutoModel extends the Hugging Face Auto API with two new parameters:

- **`device_mesh`**: A pre-built PyTorch `DeviceMesh` for fine-grained control over parallelism topology
- **`distributed`**: A dictionary specifying parallelism sizes (simpler approach)

## Quick Start

### Standard Hugging Face (Single GPU)

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
```

### NeMo AutoModel with Distributed Support

```python
from nemo_automodel import NeMoAutoModelForCausalLM

# Option 1: Using distributed dictionary (simpler)
model = NeMoAutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    distributed={"tp_size": 2, "dp_size": 4},
)

# Option 2: Using explicit device_mesh (more control)
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh(
    "cuda",
    mesh_shape=(1, 1, 1, 1, 2),  # (pp, dp_replicate, dp_shard, cp, tp)
    mesh_dim_names=("pp", "dp_replicate", "dp_shard", "cp", "tp")
)

model = NeMoAutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B",
    device_mesh=mesh,
)
```

## Running the Examples

### Single GPU (No Distributed)

```bash
python distributed_model_loading.py --model-name hf-internal-testing/tiny-random-LlamaForCausalLM
```

### Tensor Parallelism (TP=2)

```bash
torchrun --nproc-per-node=2 distributed_model_loading.py \
    --model-name meta-llama/Llama-3.1-8B \
    --tp-size 2
```

### Data Parallelism with FSDP2 (DP=4)

```bash
torchrun --nproc-per-node=4 distributed_model_loading.py \
    --model-name meta-llama/Llama-3.1-8B \
    --dp-size 4
```

### Combined TP and DP (TP=2, DP=2)

```bash
torchrun --nproc-per-node=4 distributed_model_loading.py \
    --model-name meta-llama/Llama-3.1-8B \
    --tp-size 2 \
    --dp-size 2
```

### Using Explicit Device Mesh

```bash
torchrun --nproc-per-node=4 distributed_model_loading.py \
    --model-name meta-llama/Llama-3.1-8B \
    --tp-size 2 \
    --use-device-mesh
```

## Parallelism Types

| Type | Parameter | Description |
|------|-----------|-------------|
| **Tensor Parallelism (TP)** | `tp_size` | Splits model layers across GPUs |
| **Context Parallelism (CP)** | `cp_size` | Distributes sequence context across GPUs |
| **Pipeline Parallelism (PP)** | `pp_size` | Splits model stages across GPUs |
| **Data Parallelism (DP)** | `dp_size` | Replicates model with FSDP2 sharding |

## Distributed Configuration Options

When using the `distributed` dictionary, you can specify:

```python
distributed = {
    "tp_size": 2,           # Tensor parallelism size
    "cp_size": 1,           # Context parallelism size
    "pp_size": 1,           # Pipeline parallelism size
    "dp_size": 4,           # Data parallelism size (0 = auto)
    "backend": "nccl",      # Distributed backend
    "use_hf_tp_plan": False,  # Use HuggingFace's TP plan
    "sequence_parallel": False,  # Enable sequence parallelism
    "activation_checkpointing": False,  # Enable gradient checkpointing
}
```

## Notes

- When `device_mesh` is provided with `size() > 1`, the model automatically uses distributed loading
- When `distributed` dict is non-empty, the model automatically uses distributed loading
- Liger kernel is automatically disabled when TP or CP > 1
- Context parallelism requires SDPA-compatible attention


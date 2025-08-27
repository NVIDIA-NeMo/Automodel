# Pipeline Parallelism with AutoPipeline

## Introduction

As large language models continue to grow in size, training and fine-tuning them efficiently across multiple GPUs has become increasingly challenging. While data parallelism works well for smaller models, models with billions of parameters require more sophisticated parallelization strategies to overcome memory constraints and communication overhead.

Pipeline parallelism addresses these challenges by splitting a model's layers across different devices and processing them in a pipelined fashion. Each device processes a different stage of the model, enabling training of models that wouldn't fit on a single device while maintaining high GPU utilization through overlapped computation.

AutoPipeline is NeMo AutoModel's high-level pipeline parallelism interface specifically designed for HuggingFace models, making pipeline parallelism as simple as data parallelism. Built on PyTorch's native `torch.distributed.pipelining`, AutoPipeline provides seamless pipeline parallelism support for any HuggingFace decoder-only causal language model with minimal code changes.

For custom models and more granular control, the functional API in `nemo_automodel.components.distributed.pipelining.functional` provides modular, accessible building blocks that can be used with any PyTorch model architecture.

This guide walks you through the complete process of using AutoPipeline for HuggingFace models and the functional API for custom models. You'll learn how to configure pipeline stages, integrate with existing training workflows, optimize performance, and combine pipeline parallelism with other parallelization strategies.

:::{important}
Before proceeding with this guide, please ensure that you have NeMo AutoModel installed on your machine.

**Prerequisites:**

```bash
# Install uv from https://docs.astral.sh/uv/getting-started/installation/
# Initialize the virtual environment using uv
uv venv

# Install the latest stable release from PyPI
uv pip install nemo-automodel

# Or install from source for the latest features
uv pip install git+https://github.com/NVIDIA-NeMo/Automodel.git
```

For a complete guide and additional options please consult the AutoModel [Installation Guide](./installation.md).
:::

## Key Features

AutoPipeline provides enterprise-grade pipeline parallelism with the following features:

- **Universal HuggingFace Support**: Works with any HuggingFace decoder-only causal language model including Llama, Qwen, Mistral, Gemma, and more
- **PyTorch Native Integration**: Built on PyTorch's `torch.distributed.pipelining` for optimal performance
- **Flexible Configuration**: Multiple scheduling strategies, configurable microbatch sizes, and automatic or manual layer splitting
- **Mixed Parallelism Support**: Combine pipeline parallelism with data parallelism, tensor parallelism, and FSDP
- **Modular Functional API**: For custom models, the functional module provides accessible, low-level building blocks
- **Minimal Opinions**: Easy to extend and integrate with existing training workflows

## Quick Start with AutoPipeline (HuggingFace Models)

Here's a minimal example to get started with AutoPipeline using 2 pipeline stages with a HuggingFace model:

```python
import torch
from torch.distributed.device_mesh import init_device_mesh
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from transformers import AutoModelForCausalLM
from transformers.integrations.accelerate import init_empty_weights
from transformers.modeling_utils import no_init_weights
from transformers.utils import ContextManagers

def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Define loss function for pipeline training."""
    return torch.nn.functional.cross_entropy(
        logits.float().view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=-100
    )

if __name__ == "__main__":
    # 1) Initialize device mesh with 2 pipeline stages
    world_mesh = init_device_mesh("cuda", mesh_shape=(2,), mesh_dim_names=("pp",))

    # 2) Load model on meta device to avoid OOM with large models
    init_ctx = ContextManagers([no_init_weights(), init_empty_weights()])
    with init_ctx:
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")

    # 3) Configure and build pipeline
    ap = AutoPipeline(
        world_mesh=world_mesh,
        pp_axis_name="pp",
        pp_schedule="1f1b",
        pp_microbatch_size=1,
        pp_batch_size=8,  # Total batch size across pipeline
        device=torch.cuda.current_device(),
        dtype=torch.bfloat16,
    ).build(model, loss_fn=loss_fn)

    # 4) Access pipeline components
    print(ap.debug_summary())
    print(ap.pretty_print_stages())
```

### Running the Quick Start Example

Save the above code as `pipeline_example.py` and run with:

```bash
# Run with 2 GPUs for 2 pipeline stages
uv run torchrun --nproc_per_node=2 pipeline_example.py
```

For a complete training example:

```bash
# Run fine-tuning with 2-way pipeline parallelism using Llama 3.1 8B
uv run torchrun --nproc_per_node=2 examples/llm/finetune.py \
    --config examples/llm_finetune/llama3_1/llama3_1_8b_hellaswag_pp.yaml
```

## Configuration Options

### Basic Configuration

AutoPipeline provides comprehensive control over pipeline behavior:

```python
ap = AutoPipeline(
    # Device mesh configuration
    world_mesh=world_mesh,           # DeviceMesh with pipeline axis
    pp_axis_name="pp",              # Name of pipeline axis (default: "pp")

    # Schedule configuration
    pp_schedule="1f1b",             # Pipeline schedule ("1f1b", "looped_bfs", etc.)
    pp_microbatch_size=1,           # Microbatch size per stage
    # pp_batch_size is automatically inferred from dataloader.batch_size

    # Stage configuration
    layers_per_stage=None,          # Layers per stage (None for auto)
    module_fqns_per_model_part=None,  # Manual module assignment

    # Model patching
    patch_inner_model=True,         # Patch HF model internals
    patch_causal_lm_model=True,     # Patch causal LM wrapper
).build(model, loss_fn=loss_fn)
```

### Automatic vs Manual Layer Distribution

AutoPipeline offers flexible control over how your model is split across pipeline stages:

#### Automatic Distribution
Let AutoPipeline automatically balance layers across stages:

```python
ap = AutoPipeline(
    world_mesh=world_mesh,
    pp_schedule="1f1b",
    layers_per_stage=8,  # Each stage gets ~8 transformer layers
).build(model, loss_fn=loss_fn)
```

#### Manual Distribution
Specify exactly which modules go to each stage:

```python
from nemo_automodel.components.distributed.pipelining.functional import (
    generate_hf_model_fqn_per_model_part
)

# Generate balanced assignments
module_fqns = generate_hf_model_fqn_per_model_part(
    num_stages=4,
    num_layers=32,
    include_embeddings=True,
    include_lm_head=True,
    include_rotary_emb=True,
    fqn_prefix="model."
)

# Or define custom assignments
custom_module_fqns = [
    # Stage 0: Embeddings + first 8 layers
    ["model.embed_tokens", "model.rotary_emb"] +
    [f"model.layers.{i}" for i in range(8)],

    # Stage 1: Next 8 layers
    ["model.rotary_emb"] + [f"model.layers.{i}" for i in range(8, 16)],

    # Stage 2: Next 8 layers
    ["model.rotary_emb"] + [f"model.layers.{i}" for i in range(16, 24)],

    # Stage 3: Final 8 layers + output
    ["model.rotary_emb"] + [f"model.layers.{i}" for i in range(24, 32)] +
    ["model.norm", "lm_head"]
]

ap = AutoPipeline(
    world_mesh=world_mesh,
    module_fqns_per_model_part=custom_module_fqns,
).build(model, loss_fn=loss_fn)
```

## Understanding Model Splitting

When AutoPipeline splits your model, it intelligently distributes components across pipeline stages. Here's how a typical model gets split:

### Example: 32-Layer Model Across 2 Stages

```python
# Stage 0 (Rank 0): Input processing + first half
stage_0_modules = [
    "model.embed_tokens",     # Token embeddings
    "model.layers.0-15",      # First 16 transformer layers
    "model.rotary_emb"        # Position embeddings (shared)
]

# Stage 1 (Rank 1): Second half + output processing
stage_1_modules = [
    "model.layers.16-31",     # Last 16 transformer layers
    "model.norm",             # Final layer norm
    "lm_head",               # Language modeling head
    "model.rotary_emb"        # Position embeddings (shared)
]
```

### Example: 32-Layer Model Across 4 Stages

```python
# Stage 0 (Rank 0): Input processing
stage_0_modules = [
    "model.embed_tokens",     # Token embeddings
    "model.layers.0-7",       # First 8 transformer layers
    "model.rotary_emb"        # Position embeddings (shared)
]

# Stage 1 (Rank 1): Early layers
stage_1_modules = [
    "model.layers.8-15",      # Next 8 transformer layers
    "model.rotary_emb"
]

# Stage 2 (Rank 2): Middle layers
stage_2_modules = [
    "model.layers.16-23",     # Next 8 transformer layers
    "model.rotary_emb"
]

# Stage 3 (Rank 3): Output processing
stage_3_modules = [
    "model.layers.24-31",     # Final 8 transformer layers
    "model.norm",             # Final layer norm
    "lm_head",               # Language modeling head
    "model.rotary_emb"
]
```

Key observations:
- **Embeddings** only exist on the first stage
- **Language modeling head** only exists on the last stage
- **Rotary embeddings** are shared across all stages (for position encoding)
- **Transformer layers** are evenly distributed

## Using the Functional API for Custom Models

While AutoPipeline is specifically designed as a high-level interface for HuggingFace models, the functional API in `nemo_automodel.components.distributed.pipelining.functional` provides more modular and accessible building blocks that can be used with any PyTorch model, including custom architectures. This separation allows for cleaner code organization where AutoPipeline handles HuggingFace-specific optimizations while the functional module remains model-agnostic.

### Key Functional API Components

The functional API provides several utilities for building custom pipeline parallel systems:

#### 1. **Stage ID Calculation**
```python
from nemo_automodel.components.distributed.pipelining.functional import stage_ids_this_rank

# Calculate which stages run on this rank
# For a "loop" style schedule (default)
stage_ids = stage_ids_this_rank(pp_rank=0, pp_size=4, num_stages=8, style="loop")
# Returns: (0, 4) - rank 0 gets stages 0 and 4

# For a "v" style schedule (for zero-bubble schedules)
stage_ids = stage_ids_this_rank(pp_rank=0, pp_size=4, num_stages=8, style="v")
# Returns: (0, 7) - rank 0 gets stages 0 and 7
```

#### 2. **Module Name Generation**
```python
from nemo_automodel.components.distributed.pipelining.functional import (
    generate_hf_model_fqn_per_model_part
)

# Generate balanced module assignments for any model
module_names = generate_hf_model_fqn_per_model_part(
    num_stages=4,
    num_layers=32,
    include_embeddings=True,
    include_lm_head=True,
    include_rotary_emb=False,  # Set based on your model
    fqn_prefix=""  # Use "model." for nested models
)
```

#### 3. **Virtual Stage Calculation**
```python
from nemo_automodel.components.distributed.pipelining.functional import calculate_virtual_stages

# Calculate virtual stages for interleaved schedules
num_virtual_stages, stages_per_rank = calculate_virtual_stages(
    num_layers=32,
    layers_per_stage=4,  # Each virtual stage has 4 layers
    pp_size=4,
    is_single_stage_schedule=False,
    round_to_pp_multiple="up"  # Round up to nearest multiple of pp_size
)
```

#### 4. **Pipeline Schedule Building**
```python
from nemo_automodel.components.distributed.pipelining.functional import build_pipeline_schedule

# Build a schedule for your stages
schedule = build_pipeline_schedule(
    pipeline_parallel_schedule_csv=None,  # Optional CSV schedule
    pipeline_parallel_schedule="1f1b",
    microbatch_size=1,
    local_batch_size=8,
    stages=stages,  # List of PipelineStage objects
    loss_fn=loss_fn,
    scale_grads=False
)
```

### Example: Pipeline Parallelism for Custom Models

Here's how to use the functional API to implement pipeline parallelism for a custom model:

```python
import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.pipelining import PipelineStage
from nemo_automodel.components.distributed.pipelining.functional import (
    stage_ids_this_rank,
    build_pipeline_schedule,
    calculate_virtual_stages
)

class CustomTransformerBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # Simplified transformer block
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x

class CustomModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            CustomTransformerBlock(hidden_size) for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x)

def split_custom_model_for_pipeline(model, pp_rank, pp_size, num_stages):
    """Split a custom model into pipeline stages."""

    # Determine which stages this rank handles
    stage_indices = stage_ids_this_rank(pp_rank, pp_size, num_stages, style="loop")

    stages = []
    for stage_idx in stage_indices:
        # Create a stage-specific version of the model
        # This is a simplified example - you'd need to implement proper splitting
        stage_model = create_stage_model(model, stage_idx, num_stages)

        # Create PipelineStage
        stage = PipelineStage(
            stage_model,
            stage_idx,
            num_stages,
            device=torch.cuda.current_device(),
            group=None  # Set your process group here
        )
        stages.append(stage)

    return stages

# Usage
def main():
    # Initialize device mesh
    world_mesh = init_device_mesh("cuda", mesh_shape=(4,), mesh_dim_names=("pp",))
    pp_rank = world_mesh["pp"].get_local_rank()
    pp_size = world_mesh["pp"].size()

    # Create model
    model = CustomModel(vocab_size=50000, hidden_size=768, num_layers=24)

    # Calculate virtual stages
    num_virtual_stages, stages_per_rank = calculate_virtual_stages(
        num_layers=24,
        layers_per_stage=3,  # 8 virtual stages total
        pp_size=4,
        is_single_stage_schedule=False
    )

    # Split model into stages
    stages = split_custom_model_for_pipeline(model, pp_rank, pp_size, num_virtual_stages)

    # Define loss function
    def loss_fn(logits, targets):
        return nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

    # Build pipeline schedule
    schedule = build_pipeline_schedule(
        pipeline_parallel_schedule_csv=None,
        pipeline_parallel_schedule="interleaved_1f1b",  # Good for multi-stage
        microbatch_size=1,
        local_batch_size=8,
        stages=stages,
        loss_fn=loss_fn,
        scale_grads=True
    )

    # Training loop
    for batch in dataloader:
        # Use schedule.step() for training
        losses = []
        schedule.step(batch["input_ids"], target=batch["labels"], losses=losses)

        # losses will contain the loss values from the last stage
        if losses:
            print(f"Loss: {sum(losses) / len(losses)}")
```

### Advanced: Custom Model Splitting Logic

For more complex custom models, you can implement your own splitting logic:

```python
from nemo_automodel.components.distributed.pipelining.functional import pipeline_model

def custom_parallelize_fn(
    model, world_mesh, moe_mesh, *,
    pp_enabled, dp_axis_names, **kwargs
):
    """Custom parallelization function for each pipeline stage."""
    # Apply your custom parallelization logic here
    # This is called for each pipeline stage
    if dp_axis_names:
        # Apply data parallelism
        pass
    # Add any other parallelization strategies
    pass

# Use pipeline_model for complete pipeline setup
schedule, model_parts, has_first, has_last, stages = pipeline_model(
    model=your_custom_model,
    world_mesh=world_mesh,
    moe_mesh=None,
    pp_axis_name="pp",
    dp_axis_names=("dp",),
    layers_per_stage=4,
    pipeline_parallel_schedule="1f1b",
    pipeline_parallel_schedule_csv=None,
    microbatch_size=1,
    local_batch_size=8,
    device=torch.cuda.current_device(),
    loss_fn=loss_fn,
    parallelize_fn=custom_parallelize_fn,
    module_fqns_per_model_part=None,  # Provide custom module names
    patch_inner_model=False,  # Disable HF-specific patching
    patch_causal_lm_model=False,  # Disable HF-specific patching
)
```

### Tips for Using Functional API with Custom Models

The functional API is designed to be more accessible and modular than AutoPipeline:

1. **Module Naming**: Ensure your model has consistent module naming that can be mapped to stages
2. **State Management**: Handle model state (embeddings, buffers) carefully across stages
3. **Communication**: First and last stages need special handling for inputs/outputs
4. **Flexibility**: The functional API gives you complete control over how models are split and parallelized
5. **Testing**: Start with a small model and verify correct splitting before scaling up

The functional module's modular design makes it easier to integrate pipeline parallelism into existing custom model training workflows without the HuggingFace-specific assumptions that AutoPipeline makes.

## Mixed Parallelism

AutoPipeline can be combined with other parallelization strategies for optimal performance:

```python
def parallelize_fn(
    model, world_mesh, moe_mesh, *,
    pp_enabled, dp_axis_names,
    cp_axis_name=None, tp_axis_name=None, ep_axis_name=None
):
    """Apply additional parallelization to each pipeline stage."""
    # Example: Apply FSDP to each stage
    if dp_axis_names:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        # Wrap model with FSDP (simplified example)
        # In practice, you'd configure FSDP parameters
        pass

    # Example: Apply tensor parallelism
    if tp_axis_name:
        # Apply tensor parallelism to attention/MLP layers
        pass

# Build pipeline with custom parallelization
ap = AutoPipeline(world_mesh=world_mesh).build(
    model,
    loss_fn=loss_fn,
    parallelize_fn=parallelize_fn
)
```

## Monitoring and Debugging

AutoPipeline provides comprehensive tools for understanding your pipeline configuration:

### Pipeline Information
```python
# Get pipeline info
info = ap.info
print(f"Pipeline enabled: {info.enabled}")
print(f"Has first stage: {info.has_first_stage}")
print(f"Has last stage: {info.has_last_stage}")

# Access model parts
model_parts = ap.parts  # List of pipeline stages
stage_modules = ap.list_stage_modules()  # Module names per stage
```

### Analysis
```python
# Parameter distribution
stage_param_counts = ap.get_stage_param_counts()
total_params = ap.get_total_param_count()
trainable_params = ap.get_total_param_count(trainable_only=True)

for i, params in enumerate(stage_param_counts):
    percentage = (params / total_params) * 100
    print(f"Stage {i}: {params:,} parameters ({percentage:.1f}%)")

# Debug summary
print(ap.debug_summary())
print(ap.pretty_print_stages(max_modules_per_stage=10))

# Visualize schedule
ap.visualize_current_schedule("pipeline_schedule.png")
```

### Gradient Management
```python
# Scale gradients for mixed parallelism
ap.scale_grads_by_divisor(divisor=8)

# Clip gradients across pipeline stages
grad_norm = ap.clip_grad_norm(max_norm=1.0, norm_type=2.0)
```

## Adding Pipeline Parallelism to Existing Configurations

You can easily add pipeline parallelism to any existing training configuration through command-line overrides or YAML modifications.

### Command-Line Override Method

Add pipeline parallelism to an existing config using command-line arguments:

```bash
uv run torchrun --nproc_per_node=2 examples/llm/finetune.py \
    --config examples/llm/llama_3_2_1b_squad.yaml \
    --distributed._target_ nemo_automodel.components.distributed.fsdp2.FSDP2Manager \
    --distributed.pp_size 2 \
    --autopipeline._target_ nemo_automodel.components.distributed.pipelining.AutoPipeline \
    --autopipeline.pp_schedule 1f1b \
    --autopipeline.pp_microbatch_size 1 \
    --autopipeline.round_virtual_stages_to_pp_multiple up \
    --autopipeline.scale_grads_in_schedule false
```

Key parameters to override:
- `--distributed.pp_size`: Number of pipeline stages (must match nproc_per_node)
- `--autopipeline._target_`: Specify AutoPipeline class
- `pp_batch_size` is automatically inferred from `--dataloader.batch_size`
- `--autopipeline.pp_schedule`: Pipeline schedule (1f1b, interleaved_1f1b, etc.)

### YAML Configuration Method

Add these sections to your existing YAML config:

```yaml
# Modify existing distributed section
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: 1
  tp_size: 1
  cp_size: 1
  pp_size: 4  # Enable 4-way pipeline parallelism
  sequence_parallel: false

# Add new autopipeline section
autopipeline:
  _target_: nemo_automodel.components.distributed.pipelining.AutoPipeline
  pp_schedule: 1f1b
  pp_microbatch_size: 1
  # pp_batch_size is automatically inferred from dataloader.batch_size
  round_virtual_stages_to_pp_multiple: up
  scale_grads_in_schedule: false
  layers_per_stage: null  # Auto-compute, or specify number
```

### Mixed Parallelism Examples

#### Pipeline + Data Parallelism (4 GPUs total)
```bash
uv run torchrun --nproc_per_node=4 examples/llm/finetune.py \
    --config your_config.yaml \
    --distributed.pp_size 2 \
    --distributed.dp_size 2 \
    --dataloader.batch_size 16
```

#### Pipeline + Tensor Parallelism (4 GPUs total)
```bash
uv run torchrun --nproc_per_node=4 examples/llm/finetune.py \
    --config your_config.yaml \
    --distributed.pp_size 2 \
    --distributed.tp_size 2 \
    --dataloader.batch_size 8
```

#### Full Hybrid: PP + DP + TP (8 GPUs total)
```bash
uv run torchrun --nproc_per_node=8 examples/llm/finetune.py \
    --config your_config.yaml \
    --distributed.pp_size 2 \
    --distributed.dp_size 2 \
    --distributed.tp_size 2 \
    --dataloader.batch_size 32
```

## Integration with Training Recipes

AutoPipeline seamlessly integrates with NeMo AutoModel's recipe system. Here's a complete example YAML configuration:

```yaml
# config.yaml
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: 1
  tp_size: 1
  cp_size: 1
  pp_size: 2          # 2-way pipeline parallelism
  sequence_parallel: false

autopipeline:
  _target_: nemo_automodel.components.distributed.pipelining.AutoPipeline
  pp_schedule: 1f1b
  pp_microbatch_size: 1
  # pp_batch_size is automatically inferred from dataloader.batch_size
  layers_per_stage: null  # Auto-compute layer distribution
  round_virtual_stages_to_pp_multiple: up
  scale_grads_in_schedule: false

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.SQuAD
  path_or_dataset: squad
  split: train

dataloader:
  batch_size: 8
  shuffle: true
```

Run training with:
```bash
# Run with 2 GPUs for 2-way pipeline parallelism
uv run torchrun --nproc_per_node=2 examples/llm/finetune.py --config config.yaml
```

## Troubleshooting

### Common Issues

**Model doesn't fit in memory:**
- Increase number of pipeline stages
- Reduce microbatch size
- Enable gradient checkpointing

**Pipeline bubbles reducing efficiency:**
- Increase batch size to have more microbatches
- Try different schedules (e.g., `interleaved_1f1b`)
- Adjust virtual stages configuration

**Uneven stage distribution:**
- Use manual module assignment for fine control
- Adjust `layers_per_stage` parameter
- Check parameter counts with `get_stage_param_counts()`

## Conclusion

AutoPipeline and the functional API together provide a complete pipeline parallelism solution for both HuggingFace and custom models. AutoPipeline offers a high-level, optimized interface specifically for HuggingFace models, while the functional module provides modular, accessible building blocks for custom architectures.

Key takeaways:
- Pipeline parallelism enables training of models too large for a single GPU
- AutoPipeline provides a simple API for HuggingFace models with powerful customization options
- The functional API offers modular components for implementing pipeline parallelism with any PyTorch model
- Both can be combined with other parallelization strategies for optimal performance
- Use built-in monitoring tools to understand and optimize your pipeline
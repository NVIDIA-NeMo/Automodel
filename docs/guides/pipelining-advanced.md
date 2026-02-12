# Advanced Pipeline Reference

This page covers pipeline internals, the functional API for custom models, model patching details, and debugging tools. For the quick-start guide, see [Pipeline Parallelism with AutoPipeline](pipelining.md).

## Model Patching

AutoPipeline splits a model by deep-copying it per stage and pruning away modules that don't belong to that stage. Many Hugging Face models assume the full module tree is present and return `ModelOutput` objects; after pruning, their original `forward()` often breaks.

Two flags switch AutoPipeline to lightweight, pipeline-friendly `forward()` implementations that return plain tensors (see `nemo_automodel.components.distributed.pipelining.hf_utils.patch_hf_model_for_pp`):

```python
ap = AutoPipeline(
    ...,
    patch_inner_model=True,        # Patch the decoder module
    patch_causal_lm_model=True,    # Patch the CausalLM wrapper
).build(model, loss_fn=loss_fn)
```

### `patch_inner_model`

Patches the *decoder module* (`model.model` for `...ForCausalLM`, otherwise the module itself):

- **Stage 0** (has `embed_tokens`): takes token IDs, produces hidden states
- **Middle stages** (no `embed_tokens`): takes hidden states from previous stage via `inputs_embeds`
- Handles sliced layer containers and returns a **tensor** of hidden states

For compilation/performance, this patched forward prefers a precomputed `causal_mask_mapping` dict.

### `patch_causal_lm_model`

Patches the `...ForCausalLM` wrapper forward:

- Returns **hidden states** when `lm_head` is absent on that stage
- Returns **logits** when `lm_head` is present (typically only the last stage)
- Supports `logits_to_keep` to compute logits for only the last `k` tokens

Only applies when the module is a `...ForCausalLM`-style wrapper (has a `.model` attribute).

### When to Change These

| Scenario | `patch_inner_model` | `patch_causal_lm_model` |
|---|---|---|
| Standard HF `AutoModelForCausalLM` (default) | `True` | `True` |
| Custom model with pipeline-friendly forward | `False` | `False` |
| Custom inner model, but wrapper needs simplification | `False` | `True` |

If you disable `patch_causal_lm_model`, your last stage outputs hidden states instead of logits -- make sure your `loss_fn` applies the LM head explicitly.

## Understand Model Splitting

When AutoPipeline splits a model, it distributes components across pipeline stages.

### 32-Layer Model Across 2 Stages

| Stage | Rank | Modules |
|---|---|---|
| 0 | 0 | `embed_tokens`, `layers.0-15`, `rotary_emb` |
| 1 | 1 | `layers.16-31`, `norm`, `lm_head`, `rotary_emb` |

### 32-Layer Model Across 4 Stages

| Stage | Rank | Modules |
|---|---|---|
| 0 | 0 | `embed_tokens`, `layers.0-7`, `rotary_emb` |
| 1 | 1 | `layers.8-15`, `rotary_emb` |
| 2 | 2 | `layers.16-23`, `rotary_emb` |
| 3 | 3 | `layers.24-31`, `norm`, `lm_head`, `rotary_emb` |

Key observations:
- **Embeddings** only on the first stage
- **LM head** only on the last stage
- **Rotary embeddings** shared across all stages
- **Transformer layers** evenly distributed

## Functional API for Custom Models

The functional API in `nemo_automodel.components.distributed.pipelining.functional` provides modular building blocks for any PyTorch model, without Hugging Face-specific assumptions.

### Stage ID Calculation

```python
from nemo_automodel.components.distributed.pipelining.functional import stage_ids_this_rank

# "loop" style (default)
stage_ids = stage_ids_this_rank(pp_rank=0, pp_size=4, num_stages=8, style="loop")
# Returns: (0, 4) - rank 0 gets stages 0 and 4

# "v" style (for zero-bubble schedules)
stage_ids = stage_ids_this_rank(pp_rank=0, pp_size=4, num_stages=8, style="v")
# Returns: (0, 7) - rank 0 gets stages 0 and 7
```

### Module Name Generation

```python
from nemo_automodel.components.distributed.pipelining.functional import (
    generate_hf_model_fqn_per_model_part
)

module_names = generate_hf_model_fqn_per_model_part(
    num_stages=4, num_layers=32,
    include_embeddings=True, include_lm_head=True,
    include_rotary_emb=False,
    fqn_prefix=""
)
```

### Virtual Stage Calculation

```python
from nemo_automodel.components.distributed.pipelining.functional import calculate_virtual_stages

num_virtual_stages, stages_per_rank = calculate_virtual_stages(
    num_layers=32,
    layers_per_stage=4,
    pp_size=4,
    is_single_stage_schedule=False,
    round_to_pp_multiple="up"
)
```

### Build Pipeline Schedule

```python
from nemo_automodel.components.distributed.pipelining.functional import build_pipeline_schedule

schedule = build_pipeline_schedule(
    pipeline_parallel_schedule_csv=None,
    pipeline_parallel_schedule="1f1b",
    microbatch_size=1,
    local_batch_size=8,
    stages=stages,
    loss_fn=loss_fn,
    scale_grads=False
)
```

### Example: Custom Model Pipeline

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

def main():
    world_mesh = init_device_mesh("cuda", mesh_shape=(4,), mesh_dim_names=("pp",))
    pp_rank = world_mesh["pp"].get_local_rank()
    pp_size = world_mesh["pp"].size()

    model = CustomModel(vocab_size=50000, hidden_size=768, num_layers=24)

    num_virtual_stages, stages_per_rank = calculate_virtual_stages(
        num_layers=24, layers_per_stage=3,
        pp_size=4, is_single_stage_schedule=False
    )

    # Split model into stages (implement your own splitting logic)
    stage_indices = stage_ids_this_rank(pp_rank, pp_size, num_virtual_stages, style="loop")
    stages = []
    for stage_idx in stage_indices:
        stage_model = create_stage_model(model, stage_idx, num_virtual_stages)
        stage = PipelineStage(
            stage_model, stage_idx, num_virtual_stages,
            device=torch.cuda.current_device(), group=None
        )
        stages.append(stage)

    def loss_fn(logits, targets):
        return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

    schedule = build_pipeline_schedule(
        pipeline_parallel_schedule_csv=None,
        pipeline_parallel_schedule="interleaved_1f1b",
        microbatch_size=1, local_batch_size=8,
        stages=stages, loss_fn=loss_fn, scale_grads=True
    )

    # Training loop
    for batch in dataloader:
        losses = []
        schedule.step(batch["input_ids"], target=batch["labels"], losses=losses)
        if losses:
            print(f"Loss: {sum(losses) / len(losses)}")
```

### Advanced: Custom Parallelization Function

```python
from nemo_automodel.components.distributed.pipelining.functional import pipeline_model

def custom_parallelize_fn(
    model, world_mesh, moe_mesh, *,
    pp_enabled, dp_axis_names, **kwargs
):
    """Apply additional parallelization to each pipeline stage."""
    if dp_axis_names:
        # Apply data parallelism (FSDP, etc.)
        pass

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
    module_fqns_per_model_part=None,
    patch_inner_model=False,
    patch_causal_lm_model=False,
)
```

### Tips for Custom Models

1. **Module Naming**: Ensure consistent module naming that maps to stages
2. **State Management**: Handle model state (embeddings, buffers) carefully across stages
3. **Communication**: First and last stages need special handling for inputs/outputs
4. **Testing**: Start with a small model and verify correct splitting before scaling up

## Mixed Parallelism (Code)

AutoPipeline can be combined with other parallelization strategies via a `parallelize_fn`:

```python
def parallelize_fn(
    model, world_mesh, moe_mesh, *,
    pp_enabled, dp_axis_names,
    cp_axis_name=None, tp_axis_name=None, ep_axis_name=None
):
    """Apply additional parallelization to each pipeline stage."""
    if dp_axis_names:
        # Apply FSDP to each stage
        pass
    if tp_axis_name:
        # Apply tensor parallelism to attention/MLP layers
        pass

ap = AutoPipeline(world_mesh=world_mesh).build(
    model, loss_fn=loss_fn, parallelize_fn=parallelize_fn
)
```

## Monitor and Debug

### Pipeline Information

```python
info = ap.info
print(f"Pipeline enabled: {info.enabled}")
print(f"Has first stage: {info.has_first_stage}")
print(f"Has last stage: {info.has_last_stage}")

model_parts = ap.parts               # List of pipeline stages
stage_modules = ap.list_stage_modules()  # Module names per stage
```

### Parameter Analysis

```python
stage_param_counts = ap.get_stage_param_counts()
total_params = ap.get_total_param_count()
trainable_params = ap.get_total_param_count(trainable_only=True)

for i, params in enumerate(stage_param_counts):
    percentage = (params / total_params) * 100
    print(f"Stage {i}: {params:,} parameters ({percentage:.1f}%)")
```

### Debug Output

```python
print(ap.debug_summary())
print(ap.pretty_print_stages(max_modules_per_stage=10))
ap.visualize_current_schedule("pipeline_schedule.png")
```

### Gradient Management

```python
ap.scale_grads_by_divisor(divisor=8)
grad_norm = ap.clip_grad_norm(max_norm=1.0, norm_type=2.0)
```

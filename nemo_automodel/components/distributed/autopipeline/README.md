## Autopipeline

High-level, fork-friendly pipeline-parallel training interface built on PyTorch `torch.distributed.pipelining`.

- Works with HuggingFace decoder-only models (e.g., Llama, Qwen, Mistral)
- Minimal opinions; easy to extend
- Public API: `AutoPipeline`, `AutoPipelineConfig`, `PipelineInfo`

Source layout in this package:
- `core.py`: orchestration (`AutoPipeline`, `AutoPipelineConfig`, `PipelineInfo`)
- `functional.py`: stage splitting, schedule builder, helpers
- `hf_utils.py`: HF forward patchers for pipeline compatibility
- `training_utils.py`: training helpers (step, grad utils)

### Quickstart

```python
import torch
from torch.distributed.device_mesh import init_device_mesh
from nemo_automodel.components.distributed.autopipeline import AutoPipeline, AutoPipelineConfig
from transformers import AutoModelForCausalLM

# 1) Device mesh with a pipeline axis 'pp'
world_mesh = init_device_mesh("cuda", mesh_shape=(4,), mesh_dim_names=("pp",))

# 2) Load the model (initialize and materialize weights outside AutoPipeline)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 3) Define a loss used by the pipeline schedule (runs on the last stage)
def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        logits.float().view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
    )

# 4) Configure and build AutoPipeline
cfg = AutoPipelineConfig(
    world_mesh=world_mesh,
    pp_axis_name="pp",
    pp_schedule="1f1b",
    pp_microbatch_size=1,
    pp_batch_size=4,
    layers_per_stage=None,
)
ap = AutoPipeline(cfg).build(model, loss_fn=loss_fn)

# 5) Access pipeline components
pipeline_info = ap.info
model_parts = ap.parts
stage_modules = ap.list_stage_modules()

# 6) Debug and monitoring
print(ap.debug_summary())
print(ap.pretty_print_stages())
ap.visualize_current_schedule("schedule.png")
```

### Configuration (selected)

- `world_mesh`: `DeviceMesh` with a pipeline axis (default name `pp`)
- `pp_axis_name`: name of the pipeline axis in `world_mesh` (default `"pp"`)
- `pp_schedule`: schedule name understood by PyTorch (e.g., `"1f1b"`)
- `pp_microbatch_size`: microbatch size per stage
- `pp_batch_size`: per-rank batch size; must be divisible by microbatch size
- `layers_per_stage`: int for virtual stage granularity; `None` auto-computes
- `module_fqns_per_model_part`: explicit module names per stage (overrides auto)
- `patch_inner_model`, `patch_causal_lm_model`: enable HF patchers (default True)

See `core.AutoPipelineConfig` for the full list.

### Custom parallelization (optional)

You can wrap each stage in-place (e.g., FSDP/TP/EP) by providing `parallelize_fn` to `build()`.

```python
from nemo_automodel.components.distributed.autopipeline.core import ParallelizeFnProtocol

def my_parallelize_fn(
    model, world_mesh, moe_mesh, *, pp_enabled, dp_axis_names, cp_axis_name=None,
    tp_axis_name=None, ep_axis_name=None, ep_shard_axis_names=None,
) -> None:
    # in-place wrapping only; do not return a new module
    pass

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
ap = AutoPipeline(cfg).build(model, loss_fn=loss_fn, parallelize_fn=my_parallelize_fn)
```

### Controlling stage splits

- Use `layers_per_stage` for simple virtual stage control, or
- Provide exact modules per stage via `module_fqns_per_model_part`.

```python
from nemo_automodel.components.distributed.autopipeline.functional import generate_hf_model_fqn_per_model_part

module_fqns = generate_hf_model_fqn_per_model_part(num_stages=8, num_layers=32)
cfg = AutoPipelineConfig(
    world_mesh=world_mesh,
    pp_axis_name="pp",
    pp_schedule="1f1b",
    pp_microbatch_size=1,
    pp_batch_size=8,
    module_fqns_per_model_part=module_fqns,
)
```

### Introspection and helpers

```python
# Access pipeline information
info = ap.info                                # PipelineInfo
model_parts = ap.parts                        # list[nn.Module] 
names = ap.list_stage_modules()               # list[list[str]]
print(ap.pretty_print_stages())               # human-readable summary
print(ap.debug_summary())                     # pipeline statistics

# Parameter analysis
stage_param_counts = ap.get_stage_param_counts()
total_params = ap.get_total_param_count()
trainable_params = ap.get_total_param_count(trainable_only=True)

# Gradient utilities
ap.scale_grads_by_divisor(divisor=8)
grad_norm = ap.clip_grad_norm(max_norm=1.0)

# Schedule visualization
ap.visualize_current_schedule("schedule.png")
ap.log_debug_summary()                        # Log summary with stages
```

### HuggingFace specifics

HF models are patched automatically for pipeline compatibility (embeddings, masks, rotary, KV cache, optional `lm_head`). If needed, see `hf_utils.patch_hf_model_for_pp`.
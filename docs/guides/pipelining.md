# Pipeline Parallelism with AutoPipeline

Pipeline parallelism splits a model's layers across devices and processes them in a pipelined fashion, enabling training of models that wouldn't fit on a single GPU.

**AutoPipeline** is NeMo AutoModel's high-level interface for pipeline parallelism. Built on PyTorch's native `torch.distributed.pipelining`, it works with any Hugging Face decoder-only causal LM with minimal code changes.

- **Universal Hugging Face Support**: Llama, Qwen, Mistral, Gemma, and more
- **Flexible Configuration**: Multiple schedules, configurable microbatch sizes, automatic or manual layer splitting
- **Mixed Parallelism**: Combine with FSDP, TP, CP, and DP
- **Functional API**: For custom (non-HF) models, see the [Advanced Pipeline Reference](pipelining-advanced.md)

**Prerequisites:**

```bash
# Install uv from https://docs.astral.sh/uv/getting-started/installation/
uv venv
uv pip install nemo-automodel
```
:::{important}
Before proceeding with this guide, please ensure that you have NeMo AutoModel installed on your machine.
For a complete guide and additional options please consult the AutoModel [Installation Guide](./installation.md).
:::

## Quick Start

Here's a minimal example with 2 pipeline stages on a Hugging Face model:

```python
import torch
from torch.distributed.device_mesh import init_device_mesh
from nemo_automodel.components.distributed.pipelining import AutoPipeline
from transformers import AutoModelForCausalLM
from transformers.integrations.accelerate import init_empty_weights
from transformers.modeling_utils import no_init_weights
from transformers.utils import ContextManagers

def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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
        pp_batch_size=8,
        device=torch.cuda.current_device(),
        dtype=torch.bfloat16,
    ).build(model, loss_fn=loss_fn)

    # 4) Inspect
    print(ap.debug_summary())
    print(ap.pretty_print_stages())
```

Run with:

```bash
uv run torchrun --nproc-per-node=2 pipeline_example.py
```

For a complete training example:

```bash
uv run torchrun --nproc-per-node=2 examples/llm_finetune/finetune.py \
    --config examples/llm_finetune/llama3_1/llama3_1_8b_hellaswag_pp.yaml
```

## Configuration

### Basic Parameters

```python
ap = AutoPipeline(
    world_mesh=world_mesh,           # DeviceMesh with pipeline axis
    pp_axis_name="pp",              # Name of pipeline axis (default: "pp")
    pp_schedule="1f1b",             # Schedule: "1f1b", "looped_bfs", "interleaved_1f1b"
    pp_microbatch_size=1,           # Microbatch size per stage
    # pp_batch_size is automatically inferred from dataloader.batch_size
    layers_per_stage=None,          # Layers per stage (None = auto)
).build(model, loss_fn=loss_fn)
```

### Automatic vs. Manual Layer Distribution

**Automatic** -- let AutoPipeline balance layers:

```python
ap = AutoPipeline(
    world_mesh=world_mesh,
    pp_schedule="1f1b",
    layers_per_stage=8,  # Each stage gets ~8 transformer layers
).build(model, loss_fn=loss_fn)
```

**Manual** -- specify exactly which modules go to each stage:

```python
from nemo_automodel.components.distributed.pipelining.functional import (
    generate_hf_model_fqn_per_model_part
)

module_fqns = generate_hf_model_fqn_per_model_part(
    num_stages=4, num_layers=32,
    include_embeddings=True, include_lm_head=True,
    include_rotary_emb=True, fqn_prefix="model."
)

ap = AutoPipeline(
    world_mesh=world_mesh,
    module_fqns_per_model_part=module_fqns,
).build(model, loss_fn=loss_fn)
```

## Add Pipeline Parallelism to Existing Configs

### Command-Line Override

```bash
uv run torchrun --nproc-per-node=2 examples/llm_finetune/finetune.py \
    --config examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml \
    --distributed._target_ nemo_automodel.components.distributed.fsdp2.FSDP2Manager \
    --distributed.pp_size 2 \
    --autopipeline._target_ nemo_automodel.components.distributed.pipelining.AutoPipeline \
    --autopipeline.pp_schedule 1f1b \
    --autopipeline.pp_microbatch_size 1 \
    --autopipeline.round_virtual_stages_to_pp_multiple up \
    --autopipeline.scale_grads_in_schedule false
```

Key parameters:
- `--distributed.pp_size` -- number of pipeline stages (must match `--nproc-per-node`)
- `--autopipeline._target_` -- specify AutoPipeline class
- `pp_batch_size` is automatically inferred from `--dataloader.batch_size`
- `--autopipeline.pp_schedule` -- pipeline schedule (1f1b, interleaved_1f1b, etc.)

### YAML Configuration

```yaml
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: 1
  tp_size: 1
  cp_size: 1
  pp_size: 4                  # 4-way pipeline parallelism
  sequence_parallel: false

autopipeline:
  _target_: nemo_automodel.components.distributed.pipelining.AutoPipeline
  pp_schedule: 1f1b
  pp_microbatch_size: 1
  # pp_batch_size is automatically inferred from dataloader.batch_size
  round_virtual_stages_to_pp_multiple: up
  scale_grads_in_schedule: false
  layers_per_stage: null      # Auto-compute, or specify number
```

### Mixed Parallelism Examples

```bash
# Pipeline + Data Parallelism (4 GPUs: 2 PP x 2 DP)
uv run torchrun --nproc-per-node=4 examples/llm_finetune/finetune.py \
    --config your_config.yaml \
    --distributed.pp_size 2 --distributed.dp_size 2

# Pipeline + Tensor Parallelism (4 GPUs: 2 PP x 2 TP)
uv run torchrun --nproc-per-node=4 examples/llm_finetune/finetune.py \
    --config your_config.yaml \
    --distributed.pp_size 2 --distributed.tp_size 2

# Full Hybrid: PP + DP + TP (8 GPUs: 2 PP x 2 DP x 2 TP)
uv run torchrun --nproc-per-node=8 examples/llm_finetune/finetune.py \
    --config your_config.yaml \
    --distributed.pp_size 2 --distributed.dp_size 2 --distributed.tp_size 2
```

## Complete Recipe Example

```yaml
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: 1
  tp_size: 1
  cp_size: 1
  pp_size: 2
  sequence_parallel: false

autopipeline:
  _target_: nemo_automodel.components.distributed.pipelining.AutoPipeline
  pp_schedule: 1f1b
  pp_microbatch_size: 1
  layers_per_stage: null
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

```bash
uv run torchrun --nproc-per-node=2 examples/llm_finetune/finetune.py --config config.yaml
```

## Troubleshooting

| Problem | Fix |
|---|---|
| Model doesn't fit in memory | Increase pipeline stages, reduce microbatch size, enable gradient checkpointing |
| Pipeline bubbles reducing efficiency | Increase batch size for more microbatches, try `interleaved_1f1b`, adjust virtual stages |
| Uneven stage distribution | Use manual module assignment, adjust `layers_per_stage`, check with `get_stage_param_counts()` |

## Next Steps

- [Advanced Pipeline Reference](pipelining-advanced.md) -- model patching, functional API for custom models, model splitting internals, monitoring/debug API
- [Gradient Checkpointing](gradient-checkpointing.md) -- reduce per-stage memory
- [FP8 Training](fp8-training.md) -- combine with quantized training

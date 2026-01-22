# NeMo 2 → NeMo Automodel migration guide (LLM fine-tuning example)

## Overview

This guide shows how to migrate an LLM fine-tuning workflow from **NeMo 2 (NeMo-Run + NeMo LLM recipes)** to **NeMo Automodel**. It focuses on:

- Translating configuration knobs (batch size, parallelism, LoRA/PEFT, precision/FP8).
- Converting from a “Python recipe + external YAML” style to an Automodel “YAML-driven recipe” style.
- Keeping the training loop “end-to-end runnable” through `torchrun`.

## Baseline: NeMo 2 / NeMo-Run (what you are migrating from)

The NeMo 2 example uses:

- A Python recipe (`recipe.py`) built with **NeMo-Run** (`import nemo_run as run`) and `nemo.collections.llm`.
- A `torchrun` launcher script (`run.sh`) that calls `recipe.py` and points it at an external YAML file (`config-fp16.yaml` or `config-fp8.yaml`).

### How to run the NeMo 2 example

From the directory that contains `nemo2/`:

```bash
cd nemo2

# BF16 mixed precision (FP8 knobs are commented out in the YAML)
bash run.sh fp16

# BF16 mixed precision + FP8 enabled in YAML
bash run.sh fp8
```

### Launch scripts (side-by-side)

<table style="border-collapse:collapse; width:100%;">
  <thead>
    <tr>
      <th style="border:1px solid #d0d7de; padding:8px; text-align:left; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        NeMo 2 (<code>nemo2/run.sh</code>)
      </th>
      <th style="border:1px solid #d0d7de; padding:8px; text-align:left; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        NeMo Automodel (<code>automodel/run.sh</code>)
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <pre style="white-space:pre-wrap; word-break:break-word; overflow-wrap:anywhere;"><code class="language-sh">#!/bin/bash

set -ex

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    recipe.py \
    --factory 'migration_recipe()' \
    --yes \
    --yaml config-${1}.yaml \
    -v
</code></pre>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <pre style="white-space:pre-wrap; word-break:break-word; overflow-wrap:anywhere;"><code class="language-sh"># I'm assuming there's Automodel folder in the same directory as this script, if not
# you can git clone git@github.com:NVIDIA-NeMo/Automodel.git
# If you have Automodel installed, you do not need to change the PYTHONPATH,
# instead you only need to provide the absolute path to the finetune.py script
export PYTHONPATH=$(realpath Automodel/)
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRANSFORMERS_OFFLINE=1

torchrun \
    --nproc-per-node=4 \
    --nnodes=1 \
    Automodel/examples/llm_finetune/finetune.py \
    --config config-fp16.yaml
</code></pre>
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <em>(no direct equivalent)</em>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <strong>Optional smaller-model launcher</strong> (<code>automodel/run_llama3_2_1b.sh</code>)
        <pre style="white-space:pre-wrap; word-break:break-word; overflow-wrap:anywhere;"><code class="language-sh">export PYTHONPATH=$(realpath Automodel/)
export CUDA_VISIBLE_DEVICES=0,1
export TRANSFORMERS_OFFLINE=1

torchrun \
    --nproc-per-node=2 \
    --nnodes=1 \
    Automodel/examples/llm_finetune/finetune.py \
    --config config-fp16.yaml \
    --model.pretrained_model_name_or_path meta-llama/Llama-3.2-1B \
    --distributed.tp_size 1
</code></pre>
      </td>
    </tr>
  </tbody>
</table>

### NeMo 2 recipe (`nemo2/recipe.py`)

```python
import nemo_run as run

from nemo.collections import llm
from megatron.core.dist_checkpointing.validation import StrictHandling


def migration_recipe() -> run.Partial:
    recipe = llm.llama31_70b.finetune_recipe(
        dir="/app/output",
        name="cust-llama33-70b",
        num_nodes=1,
        num_gpus_per_node=4,
        peft_scheme="lora",
    )

    recipe.data = run.Config(
        llm.FineTuningDataModule,
        seq_length=4096,
        dataset_kwargs=run.Config(dict),
    )
    recipe.trainer.strategy.ckpt_load_strictness = StrictHandling.LOG_ALL
    recipe.log.extra_loggers = []
    recipe.log.tensorboard = None
    return recipe


def run_finetuning():
    run.cli.main(llm.finetune, default_factory=migration_recipe)


if __name__ == "__main__":
    run_finetuning()
```

This recipe defines a `migration_recipe()` that:

- Starts from `llm.llama31_70b.finetune_recipe(...)`
  - `dir="/app/output"`
  - `name="cust-llama33-70b"`
  - `num_nodes=1`
  - `num_gpus_per_node=4`
  - `peft_scheme="lora"`
- Overrides data with `llm.FineTuningDataModule`:
  - `seq_length=4096`
  - `dataset_kwargs=run.Config(dict)` (filled by YAML)
- Tweaks trainer/logging:
  - `recipe.trainer.strategy.ckpt_load_strictness = StrictHandling.LOG_ALL`
  - `recipe.log.extra_loggers = []`
  - `recipe.log.tensorboard = None`

### What the NeMo 2 YAML files configure

The two YAMLs in `nemo2/` configure:

- **Dataset**
  - `data.dataset_root: /mount/models/data/`
  - `data.seq_length: 4096`
  - `data.global_batch_size: 8`
  - `data.micro_batch_size: 1`
  - `data.dataset_kwargs` (instruction-style prompt + completion)
    - `prompt_template: '{prompt} {completion}'`
    - `label_key: completion`
    - `truncation_field: prompt`
- **Trainer / distribution**
  - `trainer.strategy.tensor_model_parallel_size: 4`
  - `trainer.strategy.pipeline_model_parallel_size: 1`
  - `trainer.strategy.context_parallel_size: 1`
  - `trainer.strategy.ckpt_async_save: false`
- **Precision**
  - `trainer.plugins.precision: bf16-mixed`
  - FP8:
    - In `config-fp16.yaml`: FP8-related keys are present but commented out.
    - In `config-fp8.yaml`: FP8 keys are enabled.
- **Checkpointing**
  - `log.ckpt.save_last: link`
  - `log.ckpt.save_top_k: 1`
  - `log.ckpt.train_time_interval: null`
- **Optimization**
  - Warmup: `optim.lr_scheduler.warmup_steps: 50`
  - LR: `optim.config.lr: 0.0001`
- **Restore / init weights**
  - `resume.restore_config.path: /mount/models/llama-3_3-70b-instruct_v0.0.1`
- **LoRA (PEFT) hyperparameters**
  - `peft.dim: 8`
  - `peft.alpha: 16`
  - `peft.dropout: 0.1`

## Target: NeMo Automodel (what you are migrating to)

The NeMo Automodel example uses:

- A **single YAML config** that drives an end-to-end recipe.
- The **LLM finetuning recipe** script `examples/llm_finetune/finetune.py`.
- An Automodel component graph (via `_target_` keys), including:
  - model loader
  - dataset
  - loss
  - distributed manager
  - checkpointing
  - optimizer / scheduler

### How to run the Automodel example

There are two launcher scripts in `automodel/`:

- `run.sh`: 4 GPUs, uses `config-fp16.yaml`
- `run_llama3_2_1b.sh`: 2 GPUs, overrides model and TP settings

From the directory that contains `automodel/`:

```bash
cd automodel
bash run.sh
```

To run the FP8 config, point at `config-fp8.yaml`:

```bash
cd automodel

export PYTHONPATH="$(realpath Automodel/)"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRANSFORMERS_OFFLINE=1

torchrun \
  --nproc-per-node=4 \
  --nnodes=1 \
  Automodel/examples/llm_finetune/finetune.py \
  --config config-fp8.yaml
```

::::{note}
The scripts in `automodel/` assume there is an `Automodel/` directory next to the script (see the comment at the top of `automodel/run.sh`). If you are already in the Automodel repository root (i.e., you do not have a nested `Automodel/` folder), adapt the paths accordingly (for example, use `examples/llm_finetune/finetune.py` and set `PYTHONPATH` to the repository root).
::::

### What the Automodel YAML files configure

The two YAMLs in `automodel/` configure:

- **Step scheduler**
  - Checkpoint cadence: `ckpt_every_steps: 10`
  - Validation cadence: `val_every_steps: 10`
  - Training horizon: `num_epochs: 3`, `max_steps: 30`
  - Batch sizes: `global_batch_size: 8`, `local_batch_size: 1`
- **Distributed env**
  - `backend: nccl`
  - `timeout_minutes: 1`
- **RNG**
  - `seed: 1111`
  - `ranked: true`
- **Model**
  - `_target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained`
  - `pretrained_model_name_or_path: meta-llama/Llama-3.3-70B-Instruct`
  - `torch_dtype: bf16`
- **Checkpointing**
  - `checkpoint_dir: checkpoints/`
  - `model_save_format: safetensors`
  - `save_consolidated: false`
- **LoRA (PEFT)**
  - `_target_: nemo_automodel.components._peft.lora.PeftConfig`
  - `match_all_linear: True`
  - `dim: 8`, `alpha: 16`, `dropout: 0.1`
  - `target_modules` differs across the two example YAMLs:
    - BF16 config: `['gate_proj', 'up_proj']`
    - FP8 config: `['*_proj']`
- **Distributed training manager**
  - `_target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager`
  - `tp_size: 4`, `cp_size: 1`
  - `dp_size: none`, `dp_replicate_size: 1`
  - `sequence_parallel: false`
- **Loss**
  - `_target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy`
- **FP8 (only in `config-fp8.yaml`)**
  - `enabled: true`
  - `recipe_name: tensorwise`
  - `enable_fsdp_float8_all_gather: true`
  - `precompute_float8_dynamic_scale_for_fsdp: true`
  - `force_recompute_fp8_weight_in_bwd: true`
  - `filter_fqns: ["lm_head"]`
  - `emulate: false`
- **Dataset + dataloader**
  - Dataset class: `ColumnMappedTextInstructionDataset`
  - Data paths in the example: `../data/training.jsonl` and `../data/validation.jsonl`
  - Column mapping: `context ← prompt`, `answer ← completion`

## Config files (side-by-side)

### BF16 / BF16-mixed

<table style="border-collapse:collapse; width:100%;">
  <thead>
    <tr>
      <th style="border:1px solid #d0d7de; padding:8px; text-align:left; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        NeMo 2 (<code>nemo2/config-fp16.yaml</code>)
      </th>
      <th style="border:1px solid #d0d7de; padding:8px; text-align:left; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        NeMo Automodel (<code>automodel/config-fp16.yaml</code>)
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <pre style="white-space:pre-wrap; word-break:break-word; overflow-wrap:anywhere;"><code class="language-yaml">data:
  dataset_root: /mount/models/data/
  seq_length: 4096
  global_batch_size: 8
  micro_batch_size: 1
  dataset_kwargs:
    prompt_template: '{prompt} {completion}'
    label_key: completion
    truncation_field: prompt
trainer:
  accelerator: gpu
  max_epochs: 3
  max_steps: 30
  limit_val_batches: 1.0
  log_every_n_steps: 10
  val_check_interval: 10
  strategy:
    tensor_model_parallel_size: 4
    pipeline_model_parallel_size: 1
    context_parallel_size: 1
    ckpt_async_save: false
  plugins:
    precision: bf16-mixed
    # fp8: hybrid
    # fp8_margin: 0
    # fp8_amax_history_len: 1024
    # fp8_amax_compute_algo: max
    # fp8_params: true
log:
  ckpt:
    save_last: link
    save_top_k: 1
    train_time_interval: null
optim:
  lr_scheduler:
    warmup_steps: 50
  config:
    lr: 0.0001
resume:
  restore_config:
    path: /mount/models/llama-3_3-70b-instruct_v0.0.1
peft:
  dim: 8
  alpha: 16
  dropout: 0.1
</code></pre>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <pre style="white-space:pre-wrap; word-break:break-word; overflow-wrap:anywhere;"><code class="language-yaml">step_scheduler:
  ckpt_every_steps: 10
  val_every_steps: 10  # will run every x number of gradient steps
  num_epochs: 3
  max_steps: 30

  global_batch_size: 8
  local_batch_size: 1

dist_env:
  backend: nccl
  timeout_minutes: 1

rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 1111
  ranked: true

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.3-70B-Instruct
  torch_dtype: bf16

checkpoint:
  enabled: true
  checkpoint_dir: checkpoints/
  model_save_format: safetensors # torch_save or safetensors
  save_consolidated: false # saves the model in a consolidated safetensors format. Requires model_save_format to be safetensors.

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules: ['gate_proj', 'up_proj']
  match_all_linear: True
  dim: 8
  alpha: 16
  dropout: 0.1

distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none
  dp_replicate_size: 1 # dp_shard_size = dp_size / dp_replicate_size and dp_shard_size &lt; dp_size. For DDP usecase, use DDPManager
  tp_size: 4
  cp_size: 1
  sequence_parallel: false

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: ../data/training.jsonl
  column_mapping:
    context: prompt
    answer: completion

packed_sequence:
  # Set packed_sequence_size > 0 to run with packed sequences
  packed_sequence_size: 0
  split_across_pack: False

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  shuffle: false

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: ../data/validation.jsonl
  column_mapping:
    context: prompt
    answer: completion

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater

lr_scheduler:
  lr_decay_style: cosine
  # lr_warmup_steps: 50
  min_lr: 0.0

optimizer:
  _target_: torch.optim.Adam
  betas: [0.9, 0.999]
  eps: 1e-8
  lr: 1e-5
  weight_decay: 0

# Uncomment and configure for W&B logging
# wandb:
#   project: &lt;your_wandb_project&gt;
#   entity: &lt;your_wandb_entity&gt;
#   name: &lt;your_wandb_exp_name&gt;
#   save_dir: &lt;your_wandb_save_dir&gt;
</code></pre>
      </td>
    </tr>
  </tbody>
</table>

### FP8

<table style="border-collapse:collapse; width:100%;">
  <thead>
    <tr>
      <th style="border:1px solid #d0d7de; padding:8px; text-align:left; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        NeMo 2 (<code>nemo2/config-fp8.yaml</code>)
      </th>
      <th style="border:1px solid #d0d7de; padding:8px; text-align:left; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        NeMo Automodel (<code>automodel/config-fp8.yaml</code>)
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <pre style="white-space:pre-wrap; word-break:break-word; overflow-wrap:anywhere;"><code class="language-yaml">data:
  dataset_root: /mount/models/data/
  seq_length: 4096
  global_batch_size: 8
  micro_batch_size: 1
  dataset_kwargs:
    prompt_template: '{prompt} {completion}'
    label_key: completion
    truncation_field: prompt
trainer:
  accelerator: gpu
  max_epochs: 3
  max_steps: 30
  limit_val_batches: 1.0
  log_every_n_steps: 10
  val_check_interval: 10
  strategy:
    tensor_model_parallel_size: 4
    pipeline_model_parallel_size: 1
    context_parallel_size: 1
    ckpt_async_save: false
  plugins:
    precision: bf16-mixed
    fp8: hybrid
    fp8_margin: 0
    fp8_amax_history_len: 1024
    fp8_amax_compute_algo: max
    fp8_params: true
log:
  ckpt:
    save_last: link
    save_top_k: 1
    train_time_interval: null
optim:
  lr_scheduler:
    warmup_steps: 50
  config:
    lr: 0.0001
resume:
  restore_config:
    path: /mount/models/llama-3_3-70b-instruct_v0.0.1
peft:
  dim: 8
  alpha: 16
  dropout: 0.1
</code></pre>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <pre style="white-space:pre-wrap; word-break:break-word; overflow-wrap:anywhere;"><code class="language-yaml">step_scheduler:
  ckpt_every_steps: 10
  val_every_steps: 10  # will run every x number of gradient steps
  num_epochs: 3
  max_steps: 30

  global_batch_size: 8
  local_batch_size: 1

dist_env:
  backend: nccl
  timeout_minutes: 1

rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 1111
  ranked: true

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.3-70B-Instruct
  torch_dtype: bf16

checkpoint:
  enabled: true
  checkpoint_dir: checkpoints/
  model_save_format: safetensors # torch_save or safetensors
  save_consolidated: false # saves the model in a consolidated safetensors format. Requires model_save_format to be safetensors.

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules: ['*_proj']
  match_all_linear: True
  dim: 8
  alpha: 16
  dropout: 0.1

distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none
  dp_replicate_size: 1 # dp_shard_size = dp_size / dp_replicate_size and dp_shard_size &lt; dp_size. For DDP usecase, use DDPManager
  tp_size: 4
  cp_size: 1
  sequence_parallel: false

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

fp8:
  enabled: true
  recipe_name: tensorwise  # Options: tensorwise, rowwise, rowwise_with_gw_hp
  enable_fsdp_float8_all_gather: true
  precompute_float8_dynamic_scale_for_fsdp: true
  force_recompute_fp8_weight_in_bwd: true
  filter_fqns: ["lm_head"]
  emulate: false  # Set to true for testing on older GPUs without native FP8 support


dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: ../data/training.jsonl
  column_mapping:
    context: prompt
    answer: completion

packed_sequence:
  # Set packed_sequence_size > 0 to run with packed sequences
  packed_sequence_size: 0
  split_across_pack: False

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  shuffle: false

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: ../data/validation.jsonl
  column_mapping:
    context: prompt
    answer: completion

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater


lr_scheduler:
  lr_decay_style: cosine
  # lr_warmup_steps: 50
  min_lr: 0.0

optimizer:
  _target_: torch.optim.Adam
  betas: [0.9, 0.999]
  eps: 1e-8
  lr: 1e-5
  weight_decay: 0

# Uncomment and configure for W&B logging
# wandb:
#   project: &lt;your_wandb_project&gt;
#   entity: &lt;your_wandb_entity&gt;
#   name: &lt;your_wandb_exp_name&gt;
#   save_dir: &lt;your_wandb_save_dir&gt;
</code></pre>
      </td>
    </tr>
  </tbody>
</table>

## Key knob mapping (side-by-side)

<table style="border-collapse:collapse; width:100%;">
  <thead>
    <tr>
      <th style="border:1px solid #d0d7de; padding:8px; text-align:left; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Topic
      </th>
      <th style="border:1px solid #d0d7de; padding:8px; text-align:left; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        NeMo 2
      </th>
      <th style="border:1px solid #d0d7de; padding:8px; text-align:left; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        NeMo Automodel
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Model initialization / restore
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>resume.restore_config.path: /mount/models/llama-3_3-70b-instruct_v0.0.1</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>model.pretrained_model_name_or_path: meta-llama/Llama-3.3-70B-Instruct</code>
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Batch sizes
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>data.global_batch_size: 8</code>, <code>data.micro_batch_size: 1</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>step_scheduler.global_batch_size: 8</code>, <code>step_scheduler.local_batch_size: 1</code>
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Parallelism
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>trainer.strategy.tensor_model_parallel_size: 4</code>, <code>pipeline_model_parallel_size: 1</code>, <code>context_parallel_size: 1</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>distributed.tp_size: 4</code>, <code>distributed.cp_size: 1</code>
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        LoRA (PEFT)
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Enabled via <code>peft_scheme="lora"</code> (in <code>recipe.py</code>); parameters under <code>peft.*</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Configured via <code>peft._target_: ...PeftConfig</code>; parameters under <code>peft.*</code>
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        FP8
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Under <code>trainer.plugins</code> (example keys include <code>fp8</code>, <code>fp8_params</code>, and amax settings)
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Under the top-level <code>fp8:</code> block (example keys include <code>enabled</code>, <code>recipe_name</code>, and FSDP float8 settings)
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Dataset wiring
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>data.dataset_root</code> + <code>data.dataset_kwargs</code> (<code>prompt_template</code>, <code>label_key</code>, <code>truncation_field</code>)
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>dataset.path_or_dataset_id</code> + <code>dataset.column_mapping</code> (<code>context: prompt</code>, <code>answer: completion</code>)
      </td>
    </tr>
  </tbody>
</table>

## Example dataset schema (illustrative)

The Automodel configs use `ColumnMappedTextInstructionDataset` with:

- **context** ← `prompt`
- **answer** ← `completion`

Your JSONL records can include extra keys (like `taskname`), but the example mapping only consumes `prompt` and `completion`.

### `training.jsonl` (example)

Below is a **made-up** (illustrative) sample in the same JSONL shape:

```json
{"taskname": "email", "prompt": "Email Alex to confirm the meeting time tomorrow at 3pm.", "completion": "Hi Alex, just confirming our meeting tomorrow at 3pm. See you then."}
{"taskname": "email", "prompt": "Write to the team to remind them the deadline is Friday.", "completion": "Hi team, friendly reminder that the deadline is this Friday. Please share any blockers ASAP."}
{"taskname": "email", "prompt": "Message Priya to ask if she can review my draft today.", "completion": "Hi Priya, could you review my draft sometime today? Any quick feedback would be appreciated."}
```

### `validation.jsonl` (example)

Below is a **made-up** (illustrative) sample in the same JSONL shape:

```json
{"taskname": "email", "prompt": "Email Sam to thank them for the help last week.", "completion": "Hi Sam, thanks again for your help last week. I really appreciate it."}
{"taskname": "email", "prompt": "Write to Morgan to reschedule the call to next Tuesday.", "completion": "Hi Morgan, can we reschedule our call to next Tuesday? Let me know what time works best."}
```

## Appendices (verbatim source files)

### Appendix A: `nemo2/recipe.py`

```python
import nemo_run as run

from nemo.collections import llm
from megatron.core.dist_checkpointing.validation import StrictHandling


def migration_recipe() -> run.Partial:
    recipe = llm.llama31_70b.finetune_recipe(
        dir="/app/output",
        name="cust-llama33-70b",
        num_nodes=1,
        num_gpus_per_node=4,
        peft_scheme="lora",
    )

    recipe.data = run.Config(
        llm.FineTuningDataModule,
        seq_length=4096,
        dataset_kwargs=run.Config(dict),
    )
    recipe.trainer.strategy.ckpt_load_strictness = StrictHandling.LOG_ALL
    recipe.log.extra_loggers = []
    recipe.log.tensorboard = None
    return recipe


def run_finetuning():
    run.cli.main(llm.finetune, default_factory=migration_recipe)


if __name__ == "__main__":
    run_finetuning()
```

### Appendix B: `nemo2/run.sh`

```bash
#!/bin/bash

set -ex

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    recipe.py \
    --factory 'migration_recipe()' \
    --yes \
    --yaml config-${1}.yaml \
    -v
```

### Appendix C: `nemo2/config-fp16.yaml`

```yaml
data:
  dataset_root: /mount/models/data/
  seq_length: 4096
  global_batch_size: 8
  micro_batch_size: 1
  dataset_kwargs:
    prompt_template: '{prompt} {completion}'
    label_key: completion
    truncation_field: prompt
trainer:
  accelerator: gpu
  max_epochs: 3
  max_steps: 30
  limit_val_batches: 1.0
  log_every_n_steps: 10
  val_check_interval: 10
  strategy:
    tensor_model_parallel_size: 4
    pipeline_model_parallel_size: 1
    context_parallel_size: 1
    ckpt_async_save: false
  plugins:
    precision: bf16-mixed
    # fp8: hybrid
    # fp8_margin: 0
    # fp8_amax_history_len: 1024
    # fp8_amax_compute_algo: max
    # fp8_params: true
log:
  ckpt:
    save_last: link
    save_top_k: 1
    train_time_interval: null
optim:
  lr_scheduler:
    warmup_steps: 50
  config:
    lr: 0.0001
resume:
  restore_config:
    path: /mount/models/llama-3_3-70b-instruct_v0.0.1
peft:
  dim: 8
  alpha: 16
  dropout: 0.1
```

### Appendix D: `nemo2/config-fp8.yaml`

```yaml
data:
  dataset_root: /mount/models/data/
  seq_length: 4096
  global_batch_size: 8
  micro_batch_size: 1
  dataset_kwargs:
    prompt_template: '{prompt} {completion}'
    label_key: completion
    truncation_field: prompt
trainer:
  accelerator: gpu
  max_epochs: 3
  max_steps: 30
  limit_val_batches: 1.0
  log_every_n_steps: 10
  val_check_interval: 10
  strategy:
    tensor_model_parallel_size: 4
    pipeline_model_parallel_size: 1
    context_parallel_size: 1
    ckpt_async_save: false
  plugins:
    precision: bf16-mixed
    fp8: hybrid
    fp8_margin: 0
    fp8_amax_history_len: 1024
    fp8_amax_compute_algo: max
    fp8_params: true
log:
  ckpt:
    save_last: link
    save_top_k: 1
    train_time_interval: null
optim:
  lr_scheduler:
    warmup_steps: 50
  config:
    lr: 0.0001
resume:
  restore_config:
    path: /mount/models/llama-3_3-70b-instruct_v0.0.1
peft:
  dim: 8
  alpha: 16
  dropout: 0.1
```

### Appendix E: `automodel/config-fp16.yaml`

```yaml
step_scheduler:
  ckpt_every_steps: 10
  val_every_steps: 10  # will run every x number of gradient steps
  num_epochs: 3
  max_steps: 30

  global_batch_size: 8
  local_batch_size: 1

dist_env:
  backend: nccl
  timeout_minutes: 1

rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 1111
  ranked: true

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.3-70B-Instruct
  torch_dtype: bf16

checkpoint:
  enabled: true
  checkpoint_dir: checkpoints/
  model_save_format: safetensors # torch_save or safetensors
  save_consolidated: false # saves the model in a consolidated safetensors format. Requires model_save_format to be safetensors.

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules: ['gate_proj', 'up_proj']
  match_all_linear: True
  dim: 8
  alpha: 16
  dropout: 0.1

distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none
  dp_replicate_size: 1 # dp_shard_size = dp_size / dp_replicate_size and dp_shard_size < dp_size. For DDP usecase, use DDPManager
  tp_size: 4
  cp_size: 1
  sequence_parallel: false

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: ../data/training.jsonl
  column_mapping:
    context: prompt
    answer: completion

packed_sequence:
  # Set packed_sequence_size > 0 to run with packed sequences
  packed_sequence_size: 0
  split_across_pack: False

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  shuffle: false

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: ../data/validation.jsonl
  column_mapping:
    context: prompt
    answer: completion

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater

lr_scheduler:
  lr_decay_style: cosine
  # lr_warmup_steps: 50
  min_lr: 0.0

optimizer:
  _target_: torch.optim.Adam
  betas: [0.9, 0.999]
  eps: 1e-8
  lr: 1e-5
  weight_decay: 0

# Uncomment and configure for W&B logging
# wandb:
#   project: <your_wandb_project>
#   entity: <your_wandb_entity>
#   name: <your_wandb_exp_name>
#   save_dir: <your_wandb_save_dir>
```

### Appendix F: `automodel/config-fp8.yaml`

```yaml
step_scheduler:
  ckpt_every_steps: 10
  val_every_steps: 10  # will run every x number of gradient steps
  num_epochs: 3
  max_steps: 30

  global_batch_size: 8
  local_batch_size: 1

dist_env:
  backend: nccl
  timeout_minutes: 1

rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 1111
  ranked: true

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.3-70B-Instruct
  torch_dtype: bf16

checkpoint:
  enabled: true
  checkpoint_dir: checkpoints/
  model_save_format: safetensors # torch_save or safetensors
  save_consolidated: false # saves the model in a consolidated safetensors format. Requires model_save_format to be safetensors.

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules: ['*_proj']
  match_all_linear: True
  dim: 8
  alpha: 16
  dropout: 0.1

distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none
  dp_replicate_size: 1 # dp_shard_size = dp_size / dp_replicate_size and dp_shard_size < dp_size. For DDP usecase, use DDPManager
  tp_size: 4
  cp_size: 1
  sequence_parallel: false

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

fp8:
  enabled: true
  recipe_name: tensorwise  # Options: tensorwise, rowwise, rowwise_with_gw_hp
  enable_fsdp_float8_all_gather: true
  precompute_float8_dynamic_scale_for_fsdp: true
  force_recompute_fp8_weight_in_bwd: true
  filter_fqns: ["lm_head"]
  emulate: false  # Set to true for testing on older GPUs without native FP8 support


dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: ../data/training.jsonl
  column_mapping:
    context: prompt
    answer: completion

packed_sequence:
  # Set packed_sequence_size > 0 to run with packed sequences
  packed_sequence_size: 0
  split_across_pack: False

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  shuffle: false

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: ../data/validation.jsonl
  column_mapping:
    context: prompt
    answer: completion

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater


lr_scheduler:
  lr_decay_style: cosine
  # lr_warmup_steps: 50
  min_lr: 0.0

optimizer:
  _target_: torch.optim.Adam
  betas: [0.9, 0.999]
  eps: 1e-8
  lr: 1e-5
  weight_decay: 0

# Uncomment and configure for W&B logging
# wandb:
#   project: <your_wandb_project>
#   entity: <your_wandb_entity>
#   name: <your_wandb_exp_name>
#   save_dir: <your_wandb_save_dir>
```

### Appendix G: `automodel/run.sh`

```bash
# I'm assuming there's Automodel folder in the same directory as this script, if not
# you can git clone git@github.com:NVIDIA-NeMo/Automodel.git
# If you have Automodel installed, you do not need to change the PYTHONPATH,
# instead you only need to provide the absolute path to the finetune.py script
export PYTHONPATH=$(realpath Automodel/)
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRANSFORMERS_OFFLINE=1

torchrun \
    --nproc-per-node=4 \
    --nnodes=1 \
    Automodel/examples/llm_finetune/finetune.py \
    --config config-fp16.yaml
```

### Appendix H: `automodel/run_llama3_2_1b.sh`

```bash
export PYTHONPATH=$(realpath Automodel/)
export CUDA_VISIBLE_DEVICES=0,1
export TRANSFORMERS_OFFLINE=1

torchrun \
    --nproc-per-node=2 \
    --nnodes=1 \
    Automodel/examples/llm_finetune/finetune.py \
    --config config-fp16.yaml \
    --model.pretrained_model_name_or_path meta-llama/Llama-3.2-1B \
    --distributed.tp_size 1
```


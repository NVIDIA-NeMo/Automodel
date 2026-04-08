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
        <pre style="white-space:pre-wrap; word-break:break-word; overflow-wrap:anywhere;"><code class="language-sh">&#35; I'm assuming there's Automodel folder in the same directory as this script, if not
&#35; you can git clone git@github.com:NVIDIA-NeMo/Automodel.git
&#35; If you have Automodel installed, you do not need to change the PYTHONPATH,
&#35; instead you only need to provide the absolute path to the finetune.py script
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
&#35; Uncomment and configure for W&B logging
&#35; wandb:
&#35;   project: &lt;your_wandb_project&gt;
&#35;   entity: &lt;your_wandb_entity&gt;
&#35;   name: &lt;your_wandb_exp_name&gt;
&#35;   save_dir: &lt;your_wandb_save_dir&gt;
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
&#35; Uncomment and configure for W&B logging
&#35; wandb:
&#35;   project: &lt;your_wandb_project&gt;
&#35;   entity: &lt;your_wandb_entity&gt;
&#35;   name: &lt;your_wandb_exp_name&gt;
&#35;   save_dir: &lt;your_wandb_save_dir&gt;
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
        Explanation
      </th>
      <th style="border:1px solid #d0d7de; padding:8px; text-align:left; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        What the NeMo 2 YAML files configure
      </th>
      <th style="border:1px solid #d0d7de; padding:8px; text-align:left; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        What the Automodel YAML files configure
      </th>
      <th style="border:1px solid #d0d7de; padding:8px; text-align:left; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Notes
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Model initialization / restore
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>resume.restore_config.path</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>model.pretrained_model_name_or_path</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Same concept, different key name. Both can usually be a local path or an HF model id.
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Training schedule (epochs/steps + val/ckpt cadence)
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>trainer.max_epochs</code><br/>
        <code>trainer.max_steps</code><br/>
        <code>trainer.val_check_interval</code><br/>
        <code>trainer.log_every_n_steps</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>step_scheduler.num_epochs</code><br/>
        <code>step_scheduler.max_steps</code><br/>
        <code>step_scheduler.val_every_steps</code><br/>
        <code>step_scheduler.ckpt_every_steps</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        The Automodel example expresses scheduling in a single <code>step_scheduler</code> block. There is no 1:1 YAML key for <code>log_every_n_steps</code> in the Automodel example config.
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Batch sizes
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>data.global_batch_size</code><br/>
        <code>data.micro_batch_size</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>step_scheduler.global_batch_size</code><br/>
        <code>step_scheduler.local_batch_size</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>micro_batch_size</code> and <code>local_batch_size</code> both represent per-rank batch size; naming differs.
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Dataset source (train/val paths)
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>data.dataset_root</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>dataset.path_or_dataset_id</code><br/>
        <code>validation_dataset.path_or_dataset_id</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Automodel supports local files or an HF dataset id. NeMo 2 example uses a dataset root directory plus dataset kwargs handled by the NeMo 2 data module.
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Dataset column wiring (prompt/answer fields)
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>data.dataset_kwargs.prompt_template</code><br/>
        <code>data.dataset_kwargs.label_key</code><br/>
        <code>data.dataset_kwargs.truncation_field</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>dataset.column_mapping</code><br/>
        <code>validation_dataset.column_mapping</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        There is no direct <code>prompt_template</code> equivalent in the example Automodel YAML; formatting is handled by the dataset + collate function (and optionally <code>use_hf_chat_template</code>).
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Sequence length, padding, truncation
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>data.seq_length</code><br/>
        <code>data.dataset_kwargs.truncation_field</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>dataset.seq_length</code> (optional)<br/>
        <code>dataset.padding</code><br/>
        <code>dataset.truncation</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Automodel’s dataset class exposes <code>seq_length</code>/<code>padding</code>/<code>truncation</code>, but the example YAML does not set them (defaults are <code>do_not_pad</code>/<code>do_not_truncate</code>).
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Parallelism (tensor/context/pipeline)
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>trainer.strategy.tensor_model_parallel_size</code><br/>
        <code>trainer.strategy.context_parallel_size</code><br/>
        <code>trainer.strategy.pipeline_model_parallel_size</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>distributed.tp_size</code><br/>
        <code>distributed.cp_size</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        The example Automodel config does not include a direct pipeline-parallel knob (no 1:1 mapping shown).
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Data-parallel / sharding strategy
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <em>(implicit in NeMo 2 strategy + world size)</em>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>distributed._target_</code> (e.g. <code>...FSDP2Manager</code>)<br/>
        <code>distributed.dp_size</code><br/>
        <code>distributed.dp_replicate_size</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        These are effectively “new” YAML knobs in the Automodel example; NeMo 2 expresses more of this via the chosen Lightning strategy and the launcher’s world size.
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Precision (BF16 / mixed precision)
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>trainer.plugins.precision</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>model.torch_dtype</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Keying differs: NeMo 2 sets precision via the trainer plugin; Automodel example sets the model dtype directly.
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        FP8 enablement + recipe
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>trainer.plugins.fp8</code><br/>
        <code>trainer.plugins.fp8_params</code><br/>
        <code>trainer.plugins.fp8_margin</code><br/>
        <code>trainer.plugins.fp8_amax_history_len</code><br/>
        <code>trainer.plugins.fp8_amax_compute_algo</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>fp8.enabled</code><br/>
        <code>fp8.recipe_name</code><br/>
        <code>fp8.enable_fsdp_float8_all_gather</code><br/>
        <code>fp8.precompute_float8_dynamic_scale_for_fsdp</code><br/>
        <code>fp8.force_recompute_fp8_weight_in_bwd</code><br/>
        <code>fp8.filter_fqns</code><br/>
        <code>fp8.emulate</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Automodel exposes extra FP8 knobs in the example config (e.g. <code>filter_fqns</code>, <code>emulate</code>) that don’t have a direct NeMo 2 YAML equivalent.
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Checkpointing (directory + format)
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>log.ckpt.save_last</code><br/>
        <code>log.ckpt.save_top_k</code><br/>
        <code>log.ckpt.train_time_interval</code><br/>
        <code>trainer.strategy.ckpt_async_save</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>checkpoint.enabled</code><br/>
        <code>checkpoint.checkpoint_dir</code><br/>
        <code>checkpoint.model_save_format</code><br/>
        <code>checkpoint.save_consolidated</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>model_save_format</code>/<code>save_consolidated</code> are “new” (or at least much more explicit) in the Automodel YAML. Retention policies like <code>save_top_k</code> are not shown in the Automodel example.
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Optimizer (LR + hyperparameters)
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>optim.config.lr</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>optimizer._target_</code><br/>
        <code>optimizer.lr</code><br/>
        <code>optimizer.betas</code><br/>
        <code>optimizer.eps</code><br/>
        <code>optimizer.weight_decay</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Automodel example makes the optimizer fully explicit in YAML; NeMo 2 example sets fewer optimizer knobs in YAML (others come from recipe defaults).
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        LR scheduler / warmup
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>optim.lr_scheduler.warmup_steps</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>lr_scheduler.lr_warmup_steps</code><br/>
        <code>lr_scheduler.lr_decay_style</code><br/>
        <code>lr_scheduler.min_lr</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        In the example Automodel YAML, <code>lr_warmup_steps</code> is present but commented out—enable it if you want the same behavior as NeMo 2’s <code>warmup_steps</code>.
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        LoRA (PEFT) hyperparameters + targeting
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>peft.dim</code><br/>
        <code>peft.alpha</code><br/>
        <code>peft.dropout</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>peft._target_</code><br/>
        <code>peft.target_modules</code><br/>
        <code>peft.match_all_linear</code><br/>
        <code>peft.dim</code><br/>
        <code>peft.alpha</code><br/>
        <code>peft.dropout</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        NeMo 2 also needs <code>peft_scheme="lora"</code> in <code>recipe.py</code> (not a YAML key). Automodel enables/configures PEFT entirely in YAML (and exposes targeting knobs like <code>target_modules</code>).
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Distributed backend + RNG (new in Automodel YAML)
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <em>(not configured in the example NeMo 2 YAML)</em>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>dist_env.backend</code><br/>
        <code>dist_env.timeout_minutes</code><br/>
        <code>rng.seed</code><br/>
        <code>rng.ranked</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        These knobs are explicit in the Automodel example YAML (previously often set via defaults, CLI flags, or launcher/environment).
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Packed sequences (new in Automodel example)
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <em>(no equivalent in the example NeMo 2 YAML)</em>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>packed_sequence.packed_sequence_size</code><br/>
        <code>packed_sequence.split_across_pack</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Set <code>packed_sequence_size</code> to a positive value to enable packed sequences.
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Loss function (explicit in Automodel YAML)
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <em>(implicit via recipe defaults)</em>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>loss_fn._target_</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Automodel makes the loss component explicit/configurable in YAML in this example.
      </td>
    </tr>
    <tr>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Validation limiting / subsetting
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>trainer.limit_val_batches</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        <code>validation_dataset.limit_dataset_samples</code>
      </td>
      <td style="border:1px solid #d0d7de; padding:8px; vertical-align:top; word-break:break-word; overflow-wrap:anywhere; white-space:normal;">
        Not a perfect 1:1 mapping: NeMo 2 can limit “batches”; Automodel can limit “samples” at the dataset level (in this dataset implementation).
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


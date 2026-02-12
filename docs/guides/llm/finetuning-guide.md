# Fine-Tuning Hyperparameter Guide

## Introduction

The standard NeMo AutoModel [fine-tuning guide](finetune.md) walks through an end-to-end example using the SQuAD benchmark dataset. While that example demonstrates the mechanics of fine-tuning, getting good results on your own data requires careful hyperparameter tuning. Learning rate, LoRA rank, batch size, number of epochs -- the right values depend on your dataset size, task complexity, and available compute.

There is no formula that gives you the perfect hyperparameters up front. Fine-tuning is inherently iterative -- you run an experiment, read the loss curves, adjust, and repeat. The goal of this guide is to make that loop faster by giving you good starting points, clear signals to watch for, and concrete knobs to turn when something is off.

This guide covers recommended parameter ranges by dataset scale, how to interpret training signals and decide what to change, common failure modes and how to fix them, and step-by-step strategies to go from an initial run to a production-quality fine-tuned model.

## Before You Start

Before diving into parameter tuning, ensure the following:

1. **Data quality**: Your dataset is clean, consistently formatted, and representative of your target task. Garbage in, garbage out applies strongly to fine-tuning.
2. **Evaluation set**: You have a held-out validation set (ideally 5-10% of your data) from the same distribution. Never tune parameters without a validation signal.
3. **Baseline**: Run the pretrained model (without fine-tuning) on your evaluation set to establish a baseline. This tells you how much improvement to expect.

## Quick-Start: Recommended Parameter Ranges

The tables below summarize recommended starting points for fine-tuning. These are intended as starting ranges -- the optimal values depend on your dataset size, task complexity, and model architecture.

### SFT (Full Fine-Tuning)

| Parameter | Small Dataset (<5K) | Medium Dataset (5K-50K) | Large Dataset (>50K) |
|-----------|-------------------|------------------------|---------------------|
| `optimizer.lr` | 1e-6 to 5e-6 | 5e-6 to 2e-5 | 1e-5 to 3e-5 |
| `optimizer.weight_decay` | 0.01 to 0.1 | 0.01 | 0 to 0.01 |
| `step_scheduler.num_epochs` | 1-3 | 2-5 | 1-3 |
| `step_scheduler.global_batch_size` | 16-32 | 32-64 | 64-128 |
| `step_scheduler.local_batch_size` | 2-4 | 4-8 | 8-16 |
| `clip_grad_norm.max_norm` | 1.0 | 1.0 | 1.0 |
| `lr_scheduler.lr_decay_style` | `cosine` | `cosine` | `cosine` |
| `lr_scheduler.min_lr` | 1e-7 | 1e-6 | 1e-6 |

### PEFT (LoRA)

| Parameter | Small Dataset (<5K) | Medium Dataset (5K-50K) | Large Dataset (>50K) |
|-----------|-------------------|------------------------|---------------------|
| `optimizer.lr` | 1e-5 to 5e-5 | 1e-5 to 1e-4 | 5e-5 to 2e-4 |
| `peft.dim` (rank) | 4-8 | 8-16 | 16-64 |
| `peft.alpha` | 2x to 4x rank | 2x to 4x rank | 2x rank |
| `peft.dropout` | 0.05-0.1 | 0.05 | 0.0-0.05 |
| `step_scheduler.num_epochs` | 3-5 | 2-4 | 1-3 |
| `step_scheduler.global_batch_size` | 16-32 | 32-64 | 64-128 |

### QLoRA (Quantized LoRA)

| Parameter | Recommended Range |
|-----------|------------------|
| `optimizer.lr` | 1e-5 to 2e-4 |
| `optimizer._target_` | `torch.optim.AdamW` |
| `optimizer.weight_decay` | 0.01 |
| `peft.dim` | 16-64 |
| `peft.alpha` | 32 |
| `peft.dropout` | 0.05-0.1 |
| `quantization.bnb_4bit_quant_type` | `nf4` |
| `quantization.bnb_4bit_compute_dtype` | `bfloat16` |
| `quantization.bnb_4bit_use_double_quant` | `True` |

## Reading the Loss Curves: When to Adjust What

The tables above are starting points. In practice, you will need to adjust based on what you observe during training. This section tells you what to watch for and which knob to turn.

### Learning Rate

The learning rate is the single most impactful hyperparameter. Get this wrong and nothing else matters.

| What you see | What it means | What to do |
|---|---|---|
| Loss barely decreases over the first 50-100 steps | LR is too low -- the model is barely updating | Increase `optimizer.lr` by 5-10x |
| Loss decreases very slowly but steadily | LR is on the low side -- safe but inefficient | Try 2-3x higher; you can converge in fewer steps |
| Loss decreases quickly in early steps, then plateaus at a good value | LR is in the right range | Keep it; fine-tune the schedule (`min_lr`, warmup) |
| Loss decreases quickly but then **spikes** or oscillates | LR is too high -- the optimizer is overshooting | Reduce by 2-5x, or add/increase `lr_warmup_steps` |
| Loss goes to NaN or inf within the first few steps | LR is far too high, or there's a data issue | Reduce by 10x; also check data and gradient clipping |
| Training loss keeps falling but validation loss starts climbing | Overfitting, not an LR issue per se | See the overfitting section below; reduce epochs first |

:::{tip}
**Rule of thumb for PEFT vs SFT:** LoRA fine-tuning tolerates ~10x higher learning rates than full SFT because only a small fraction of parameters are being updated. If you switch from SFT to PEFT (or vice versa), adjust the LR accordingly.
:::

### Batch Size

Batch size affects both training stability and convergence speed. In NeMo AutoModel, `global_batch_size` is the effective batch size per optimizer step, while `local_batch_size` is the per-GPU micro-batch size. The ratio determines how many gradient accumulation steps occur.

| What you see | What it means | What to do |
|---|---|---|
| Loss curve is very noisy (jumps around a lot) | Effective batch size may be too small -- gradients are noisy | Increase `global_batch_size` (e.g., 16 -> 32 -> 64) |
| Loss is smooth but converges very slowly | Batch size may be too large -- fewer updates per epoch | Decrease `global_batch_size` and/or increase LR proportionally |
| OOM error | `local_batch_size` is too large for your GPU memory | Reduce `local_batch_size`; keep `global_batch_size` the same (this increases gradient accumulation steps automatically) |
| Training is very slow (high step time) with low GPU utilization | `local_batch_size` might be too small to saturate the GPU | Increase `local_batch_size` if memory allows |

:::{tip}
**Scaling rule:** If you double `global_batch_size`, you can often increase `optimizer.lr` by ~1.4x (square-root scaling) to maintain similar convergence behavior. This is a rough guideline, not a strict rule.
:::

### Number of Epochs

More epochs means the model sees the data more times. This is useful when you have little data, but risky because of overfitting.

| What you see | What it means | What to do |
|---|---|---|
| Validation loss is still decreasing at the end of training | Model hasn't finished learning -- you can train longer | Increase `num_epochs` by 1-2 |
| Validation loss flattens while training loss keeps dropping | Classic overfitting onset | Stop here; use the checkpoint from the flat point |
| Validation loss **increases** while training loss drops | Overfitting is already happening | Reduce `num_epochs`, increase regularization (`weight_decay`, `peft.dropout`), or add more data |
| Both losses plateau very early (within the first epoch) | LR is likely too low, or the model has already learned what it can from this data | Increase LR first; if that doesn't help, the data may be too small or too easy |

:::{tip}
**Small datasets (<5K samples):** 1-3 epochs is usually enough. Going beyond 3 epochs almost always leads to overfitting. If results aren't good after 3 epochs, the problem is likely elsewhere (data quality, LR, or model choice).

**Large datasets (>50K samples):** 1 epoch may be all you need, especially for full SFT. The model sees enough variety in a single pass.
:::

### Weight Decay

Weight decay is a regularizer that penalizes large weights. It helps prevent overfitting but can hurt performance if set too high.

| What you see | What it means | What to do |
|---|---|---|
| Overfitting (val loss rises) and you've already reduced epochs | Model needs more regularization | Add or increase `optimizer.weight_decay` (try 0.01 -> 0.05 -> 0.1) |
| Training loss converges to a higher value than expected | Weight decay may be too aggressive | Reduce `optimizer.weight_decay` (try 0.01 or 0) |
| Model is underfitting (both train and val loss are high) | Weight decay is not the issue, but don't make it worse | Set `weight_decay: 0` and focus on LR and data |

### LoRA Rank (`peft.dim`)

LoRA rank controls the capacity of the adapters. Higher rank = more trainable parameters = more capacity to learn, but also more risk of overfitting.

| What you see | What it means | What to do |
|---|---|---|
| Training loss plateaus well above what you'd expect | Adapter capacity is too low -- the model can't represent the task | Increase `peft.dim` (e.g., 8 -> 16 -> 32) |
| Overfitting on a small dataset even with low LR and few epochs | Rank may be too high for the data size | Decrease `peft.dim` (e.g., 16 -> 8 -> 4) |
| Good results but you want to reduce checkpoint size / inference cost | Rank is higher than needed | Try halving `peft.dim` and see if quality holds |

### Gradient Clipping

Gradient clipping prevents exploding gradients. In most cases, `max_norm: 1.0` is a safe default and you don't need to change it.

| What you see | What it means | What to do |
|---|---|---|
| Loss spikes or NaN, especially with high LR | Gradients are exploding | Ensure `clip_grad_norm.max_norm: 1.0` is set; if already set, reduce LR |
| Training is stable | Clipping is working as intended | Leave it at 1.0 |

### Summary: The Tuning Loop

In practice, a typical tuning session looks like this:

1. **Start with a conservative config** from the recommended ranges above.
2. **Run for a few hundred steps** and watch the training + validation loss.
3. **Adjust the learning rate first** -- it has the biggest impact.
4. **Then adjust batch size** if the loss curve is too noisy or training is too slow.
5. **Then adjust epochs** based on where validation loss starts to rise.
6. **Then fine-tune regularization** (weight decay, LoRA dropout, LoRA rank) if overfitting persists.
7. **Repeat** until validation loss is good and generated outputs look right.

This is not a one-shot process. Expect 3-5 iterations to find a good configuration for a new dataset.

## Example Datasets by Scale

To help you get started, here are practical, widely-used Hugging Face datasets that represent real fine-tuning scenarios at different scales. All instruction-style datasets can be loaded via `ColumnMappedTextInstructionDataset`, which maps arbitrary column names to the standard `context`, `question`, and `answer` fields expected by NeMo AutoModel.

### Dataset Overview

| Scale | Dataset | Task | Size | HF ID |
|-------|---------|--------|------|-------|
| **Small** | Medical Meadow MedQA | Medical QA | ~10K | `medalpaca/medical_meadow_medqa` |
| **Small** | Databricks Dolly | General instruction | ~15K | `databricks/databricks-dolly-15k` |
| **Medium** | CodeAlpaca | Code generation | ~20K | `sahil2801/CodeAlpaca-20k` |
| **Medium** | Stanford Alpaca | General instruction | ~52K | `tatsu-lab/alpaca` |
| **Large** | xLAM Function Calling | Tool/function calling | ~60K | `Salesforce/xlam-function-calling-60k` |
| **Large** | OpenAssistant oasst1 | Multi-turn chat | ~85K | `OpenAssistant/oasst1` |
| **Large** | SlimOrca | Instruction following | ~518K | `Open-Orca/SlimOrca` |

### Small: Medical Meadow MedQA (~10K samples)

[Medical Meadow MedQA](https://huggingface.co/datasets/medalpaca/medical_meadow_medqa) contains medical exam-style question-answer pairs. This is a realistic example of a small, specialized dataset for healthcare applications -- the kind of data you would encounter when fine-tuning for a clinical or biomedical use case.

```yaml
# Small: Medical QA (~10K)
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: medalpaca/medical_meadow_medqa
  split: train
  column_mapping:
    question: input
    answer: output

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: medalpaca/medical_meadow_medqa
  split: train
  limit_dataset_samples: 500
  column_mapping:
    question: input
    answer: output
```

### Small: Databricks Dolly (~15K samples)

[Databricks Dolly](https://huggingface.co/datasets/databricks/databricks-dolly-15k) is a human-generated instruction-following dataset covering QA, summarization, brainstorming, and classification. Its small size, human authorship, and diverse task coverage make it a practical starting point for general-purpose fine-tuning.

```yaml
# Small: Dolly instruction-following (~15K)
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: databricks/databricks-dolly-15k
  split: train
  column_mapping:
    context: context
    question: instruction
    answer: response

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: databricks/databricks-dolly-15k
  split: train
  limit_dataset_samples: 500
  column_mapping:
    context: context
    question: instruction
    answer: response
```

### Medium: CodeAlpaca (~20K samples)

[CodeAlpaca](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) contains 20K instruction-following examples for code generation. This is representative of medium-scale fine-tuning for developer tools and code assistants.

```yaml
# Medium: Code generation (~20K)
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: sahil2801/CodeAlpaca-20k
  split: train
  column_mapping:
    question: instruction
    context: input
    answer: output

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: sahil2801/CodeAlpaca-20k
  split: train
  limit_dataset_samples: 500
  column_mapping:
    question: instruction
    context: input
    answer: output
```

### Medium: Stanford Alpaca (~52K samples)

[Stanford Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) is one of the most widely-used instruction-tuning datasets. Its 52K instruction-input-output triples make it a solid baseline for general-purpose fine-tuning at medium scale.

```yaml
# Medium: Alpaca instruction-following (~52K)
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: tatsu-lab/alpaca
  split: train
  column_mapping:
    question: instruction
    context: input
    answer: output

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: tatsu-lab/alpaca
  split: train
  limit_dataset_samples: 1000
  column_mapping:
    question: instruction
    context: input
    answer: output
```

### Large: xLAM Function Calling (~60K samples)

[xLAM](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) from Salesforce is a specialized dataset for training models to generate structured function/tool calls -- a high-demand production use case for AI agents. NeMo AutoModel includes a dedicated loader for this dataset.

```yaml
# Large: Tool/function calling (~60K)
dataset:
  _target_: nemo_automodel.components.datasets.llm.xlam.make_xlam_dataset
  dataset_name: Salesforce/xlam-function-calling-60k
  split: train

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.xlam.make_xlam_dataset
  dataset_name: Salesforce/xlam-function-calling-60k
  split: train[:256]
  limit_dataset_samples: 256
```

### Large: SlimOrca (~518K samples)

[SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) is a large-scale, curated instruction-following dataset with ~518K conversations in OpenAI chat format. Commonly used for building general-purpose assistants, this represents the scale where full SFT becomes most effective.

```yaml
# Large: SlimOrca instruction-following (~518K)
# SlimOrca uses OpenAI chat format (messages list).
# Use the ChatDataset loader for this format.
dataset:
  _target_: nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset
  path_or_dataset_id: Open-Orca/SlimOrca
  split: train

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset
  path_or_dataset_id: Open-Orca/SlimOrca
  split: train
  limit: 500
```

### Custom Dataset: Your Own JSONL Files

If you have your own data in JSONL format, `ColumnMappedTextInstructionDataset` is the most common path for proprietary data. Just map your column names:

```yaml
# Custom JSONL file with columns: "instruction", "input", "response"
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: /path/to/your/train.jsonl
  column_mapping:
    context: instruction
    question: input
    answer: response

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: /path/to/your/val.jsonl
  column_mapping:
    context: instruction
    question: input
    answer: response
```

:::{tip}
**Choosing the right dataset loader:**
- **`ColumnMappedTextInstructionDataset`** -- Use for any dataset with instruction/context/answer columns (covers most cases). See the [ColumnMappedTextInstructionDataset guide](column-mapped-text-instruction-dataset.md).
- **`ChatDataset`** -- Use for datasets in OpenAI chat format with a `messages` list (e.g., SlimOrca, ShareGPT-style data).
- **`make_xlam_dataset`** -- Dedicated loader for the xLAM function-calling format.
- **`ColumnMappedTextInstructionIterableDataset`** -- Streaming variant for very large datasets or Delta Lake tables. See the [IterableDataset guide](column-mapped-text-instruction-iterable-dataset.md).

For full details on building custom datasets, see [Integrate Your Own Text Dataset](dataset.md).
:::

---

## Learning Rate Experiments with Evaluation

Finding the right learning rate is the single most impactful tuning decision. This section provides complete, runnable configs at different LR values so you can compare them side by side using validation loss.

### Experiment Design

We recommend running 3-5 short experiments with different learning rates on a small subset of your data, validating frequently, and comparing validation loss curves. Use `max_steps` to keep runs short.

The strategy:
1. Run each config below (or use CLI overrides to sweep LRs).
2. Compare validation loss at the same step across runs.
3. Pick the LR with the lowest validation loss, then run a full training with that LR.

### Experiment 1: Conservative LR (1e-6) -- SFT Baseline

Best for: verifying the pipeline works; small datasets where overfitting is a risk.

```yaml
# experiment_lr_1e6.yaml -- Conservative LR for SFT
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

step_scheduler:
  global_batch_size: 32
  local_batch_size: 4
  max_steps: 200
  val_every_steps: 10
  ckpt_every_steps: 50

optimizer:
  _target_: torch.optim.Adam
  lr: 1.0e-6
  betas: [0.9, 0.999]
  eps: 1e-8
  weight_decay: 0

lr_scheduler:
  lr_decay_style: cosine
  min_lr: 1.0e-7

clip_grad_norm:
  max_norm: 1.0

# Using Databricks Dolly as the example dataset (~15K instruction-following)
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: databricks/databricks-dolly-15k
  split: train
  column_mapping:
    context: context
    question: instruction
    answer: response

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: databricks/databricks-dolly-15k
  split: train
  limit_dataset_samples: 500
  column_mapping:
    context: context
    question: instruction
    answer: response

distributed:
  dp_size: none
  tp_size: 1
  cp_size: 1

distributed_config:
  _target_: nemo_automodel.components.distributed.config.FSDP2Config

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  shuffle: false

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater

rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 42
  ranked: true

dist_env:
  backend: nccl
  timeout_minutes: 1
```

### Experiment 2: Standard LR (1e-5) -- Recommended Default

Best for: most use cases; the default in NeMo AutoModel recipes.

```bash
# Just override LR from experiment 1 via CLI
torchrun --nproc-per-node=1 examples/llm_finetune/finetune.py \
  --config experiment_lr_1e6.yaml \
  --optimizer.lr 1.0e-5 \
  --lr_scheduler.min_lr 1.0e-6
```

### Experiment 3: Aggressive LR (5e-5) -- Fast Convergence

Best for: larger datasets where you want faster convergence; PEFT where higher LRs are typical.

```bash
torchrun --nproc-per-node=1 examples/llm_finetune/finetune.py \
  --config experiment_lr_1e6.yaml \
  --optimizer.lr 5.0e-5 \
  --lr_scheduler.min_lr 5.0e-6
```

### Experiment 4: PEFT with LR Sweep

For PEFT, learning rates are typically 5-10x higher than SFT. Run three quick experiments:

```bash
# PEFT LR = 5e-5 (conservative for PEFT)
torchrun --nproc-per-node=1 examples/llm_finetune/finetune.py \
  --config experiment_lr_1e6.yaml \
  --optimizer.lr 5.0e-5 \
  --peft._target_ nemo_automodel.components._peft.lora.PeftConfig \
  --peft.match_all_linear True \
  --peft.dim 8 \
  --peft.alpha 32 \
  --peft.use_triton True

# PEFT LR = 1e-4 (standard for PEFT)
torchrun --nproc-per-node=1 examples/llm_finetune/finetune.py \
  --config experiment_lr_1e6.yaml \
  --optimizer.lr 1.0e-4 \
  --peft._target_ nemo_automodel.components._peft.lora.PeftConfig \
  --peft.match_all_linear True \
  --peft.dim 8 \
  --peft.alpha 32 \
  --peft.use_triton True

# PEFT LR = 2e-4 (aggressive for PEFT)
torchrun --nproc-per-node=1 examples/llm_finetune/finetune.py \
  --config experiment_lr_1e6.yaml \
  --optimizer.lr 2.0e-4 \
  --peft._target_ nemo_automodel.components._peft.lora.PeftConfig \
  --peft.match_all_linear True \
  --peft.dim 8 \
  --peft.alpha 32 \
  --peft.use_triton True
```

### Reading the Validation Results

Each run logs validation loss at the steps specified by `val_every_steps`. Look for lines like:
```
INFO:root:[val] step 20 | epoch 0 | loss 0.2469
```

Compare across experiments:

| Step | LR=1e-6 Val Loss | LR=1e-5 Val Loss | LR=5e-5 Val Loss |
|------|-------------------|-------------------|-------------------|
| 10   | 1.55              | 1.42              | 1.30              |
| 20   | 1.48              | 1.15              | 0.95              |
| 50   | 1.35              | 0.78              | 0.82 (rising!)    |

In this example, `lr=1e-5` gives the best final validation loss. `lr=5e-5` converges faster initially but starts overfitting by step 50.

### Logging Experiments to W&B or MLflow

For easier comparison, enable experiment tracking. Each run will appear as a separate entry you can overlay:

```yaml
# Add to your config to track experiments in Weights & Biases
wandb:
  project: finetune-lr-sweep
  entity: your-wandb-entity
  name: lr_1e-5_dolly_sft   # Change per experiment
  save_dir: ./wandb_logs

# Or use MLflow
mlflow:
  experiment_name: finetune-lr-sweep
  run_name: lr_1e-5_dolly_sft   # Change per experiment
  tracking_uri: null              # Uses default local tracking
  tags:
    task: finetune
    learning_rate: "1e-5"
```

### Post-Training Evaluation with Hugging Face Generate

After training, evaluate your best checkpoint by generating answers on held-out examples:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # Only needed for PEFT checkpoints

# --- For SFT checkpoint ---
ckpt_path = "checkpoints/epoch_0_step_50/model/consolidated"
tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
model = AutoModelForCausalLM.from_pretrained(ckpt_path)

# --- For PEFT checkpoint ---
# base_model_name = "meta-llama/Llama-3.2-1B"
# tokenizer = AutoTokenizer.from_pretrained(base_model_name)
# model = AutoModelForCausalLM.from_pretrained(base_model_name)
# model = PeftModel.from_pretrained(model, "checkpoints/epoch_0_step_50/model/")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Evaluate on task-specific prompts
test_prompts = [
    "Context: The patient presented with elevated troponin levels. Question: What condition does this suggest? Answer: ",
    "Context: Section 12(b) of the Securities Exchange Act requires... Question: What must be filed? Answer: ",
]

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Prompt: {prompt[:60]}...")
    print(f"Response: {response}\n")
```

### Post-Training Evaluation with LM Evaluation Harness

For standardized benchmarking, use the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) on your fine-tuned checkpoint:

```bash
# Evaluate SFT checkpoint on standard benchmarks
python3 -m lm_eval --model hf \
    --model_args pretrained=checkpoints/epoch_0_step_50/model/consolidated \
    --tasks hellaswag,arc_easy,winogrande \
    --device cuda:0 \
    --batch_size 8

# Evaluate PEFT checkpoint (pass base model + adapter)
python3 -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-1B,peft=checkpoints/epoch_0_step_50/model/ \
    --tasks hellaswag,arc_easy,winogrande \
    --device cuda:0 \
    --batch_size 8
```

:::{tip}
Run `lm_eval` on your base model **before** fine-tuning to establish a baseline. Then compare with the fine-tuned checkpoint to measure improvement -- and also to check for regression on general capabilities (catastrophic forgetting).
:::

---

## Additional Hyperparameters to Consider

Beyond the core parameters in the tables above, the following settings can have a meaningful impact on fine-tuning quality and efficiency.

### LR Warmup

NeMo AutoModel supports learning rate warmup via the `lr_scheduler` config. During warmup, the learning rate linearly increases from near-zero to the target LR, which stabilizes early training and prevents loss spikes.

By default, warmup is set to **10% of total training steps** (capped at 1000 steps). You can override this explicitly:

```yaml
lr_scheduler:
  lr_decay_style: cosine
  min_lr: 1.0e-6
  lr_warmup_steps: 100    # Explicit warmup steps
```

**Recommendations:**
- For **short runs** (<500 steps): use 5-10 warmup steps.
- For **medium runs** (500-5000 steps): use 50-200 warmup steps or ~5-10% of total steps.
- For **long runs** (>5000 steps): the default 10% / max 1000 is usually fine.
- If you see **loss spikes in the first few steps**, increase warmup.

### Sequence Length and Packing

Sequence length directly affects GPU memory usage and training throughput. Longer sequences require more memory per sample.

**Sequence packing** concatenates shorter samples into a single sequence (separated by EOS tokens), which reduces wasted computation on padding tokens. This can dramatically improve GPU utilization when your dataset has variable-length samples.

```yaml
packed_sequence:
  packed_sequence_size: 2048      # Pack samples into 2048-token sequences
  split_across_pack: False        # Don't split individual samples across packs
```

**Recommendations:**
- Enable packing when your dataset has **highly variable** sequence lengths (common with instruction datasets).
- Set `packed_sequence_size` to roughly the **95th percentile** of your sample lengths.
- Packing can improve training throughput by **1.5-3x** on datasets with many short samples.
- See [Packed Sequence documentation](dataset.md#packed-sequence-support-in-nemo-automodel) for details.

### DoRA (Weight-Decomposed Low-Rank Adaptation)

NeMo AutoModel supports [DoRA](https://arxiv.org/abs/2402.09353), which decomposes weight updates into magnitude and direction components. DoRA can improve fine-tuning quality over standard LoRA at the cost of slightly more compute.

```yaml
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: True
  dim: 16
  alpha: 32
  use_dora: True      # Enable DoRA
```

**When to use DoRA:**
- When standard LoRA results are not meeting quality expectations.
- When you have enough compute budget (DoRA is ~10-20% slower than LoRA).
- Currently supported for `nn.Linear` layers only.

### LoRA Alpha Rule of Thumb

The `alpha` parameter controls the scaling of LoRA updates (effective scaling = `alpha / dim`). A common and effective starting point:

- **`alpha` = `dim`** (scaling factor = 1.0): Conservative, stable. Good default.
- **`alpha` = 2 x `dim`** (scaling factor = 2.0): More aggressive adaptation. Try this if `alpha = dim` underfits.
- **`alpha` > 4 x `dim`**: Rarely needed. Can cause instability.

### Reproducibility

For reproducible experiments, set the RNG seed explicitly:

```yaml
rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 42
  ranked: true    # Different seed per rank for data parallelism
```

When comparing hyperparameter experiments, keep the same seed across runs so differences in results are attributable to the parameter changes, not randomness.

---

## Step-by-Step Tuning Strategy

### Step 1: Start Conservative

Begin with a low learning rate and few epochs. The goal of the first run is to confirm the pipeline works and to see a decreasing loss curve.

```yaml
# conservative_finetune.yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

step_scheduler:
  global_batch_size: 32
  local_batch_size: 4
  val_every_steps: 10
  ckpt_every_steps: 50
  num_epochs: 1

optimizer:
  _target_: torch.optim.Adam
  lr: 1.0e-6
  betas: [0.9, 0.999]
  eps: 1e-8
  weight_decay: 0

lr_scheduler:
  lr_decay_style: cosine
  min_lr: 1.0e-7

clip_grad_norm:
  max_norm: 1.0

dataset:
  _target_: /path/to/your/custom_dataset.py:build_my_dataset
  split: train

validation_dataset:
  _target_: /path/to/your/custom_dataset.py:build_my_dataset
  split: validation
  limit_dataset_samples: 100

# Use FSDP2 for distribution
distributed:
  dp_size: none
  tp_size: 1
  cp_size: 1

distributed_config:
  _target_: nemo_automodel.components.distributed.config.FSDP2Config

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  shuffle: false

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater

rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 42
  ranked: true

dist_env:
  backend: nccl
  timeout_minutes: 1
```

Run it:
```bash
torchrun --nproc-per-node=1 examples/llm_finetune/finetune.py --config conservative_finetune.yaml
```

**What to look for:**
- Training loss should decrease steadily.
- Validation loss should decrease or stay flat (not increase).
- If loss does not decrease at all, increase the learning rate by 5-10x.

### Step 2: Increase Learning Rate Gradually

Once you confirm the pipeline works, increase the learning rate. A common approach:

1. Try `lr: 1.0e-5` -- this is the default in most NeMo AutoModel recipes.
2. If loss decreases well but you want faster convergence, try `lr: 5.0e-5`.
3. If loss spikes or becomes unstable, reduce back to the previous stable value.

Override the learning rate from the CLI without editing the YAML:
```bash
torchrun --nproc-per-node=1 examples/llm_finetune/finetune.py \
  --config conservative_finetune.yaml \
  --optimizer.lr 1.0e-5
```

### Step 3: Monitor for Overfitting

Overfitting is the most common failure mode when fine-tuning, especially on small datasets.

**Symptoms:**
- Training loss keeps decreasing, but validation loss starts increasing after a certain number of steps.
- Generated outputs are repetitive, copy training examples verbatim, or degrade in general quality.

**Mitigations:**

| Strategy | YAML Parameter | Recommended Value |
|----------|---------------|-------------------|
| Fewer epochs | `step_scheduler.num_epochs` | 1-2 for small datasets |
| Early stopping | `step_scheduler.max_steps` | Stop at the step where val loss is minimal |
| Weight decay | `optimizer.weight_decay` | 0.01-0.1 |
| LoRA dropout | `peft.dropout` | 0.05-0.1 |
| Smaller LoRA rank | `peft.dim` | 4-8 for small datasets |
| Lower learning rate | `optimizer.lr` | Reduce by 2-5x |
| More data | `dataset` | Augment or collect more training data |

### Step 4: Switch to PEFT if Resources Are Limited

If full fine-tuning is too expensive or you have limited data, switch to LoRA. PEFT is particularly effective because:

- It updates <1% of parameters, reducing overfitting risk.
- It requires significantly less GPU memory.
- The original model capabilities are better preserved.

```yaml
# Add this to your config for PEFT
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: True  # Apply LoRA to all linear layers
  dim: 8                  # Start small, increase if underfitting
  alpha: 32               # Typically 2-4x the rank
  dropout: 0.05           # Regularization
  use_triton: True        # Optimized LoRA kernel

# Higher LR is typical for PEFT
optimizer:
  _target_: torch.optim.Adam
  lr: 5.0e-5
  betas: [0.9, 0.999]
  eps: 1e-8
  weight_decay: 0
```

:::{tip}
**Choosing `target_modules` vs `match_all_linear`:**
- `match_all_linear: True` applies LoRA to all linear layers. This is a good default.
- For more control, use `target_modules: "*.proj"` to match specific layers (e.g., attention projections only). This reduces the number of trainable parameters further and can be useful if you want to preserve certain model capabilities.
- Use `exclude_modules` to skip specific layers (e.g., the output head).
:::

### Step 5: Use Checkpointing and Pick the Best

Always enable checkpointing and validate frequently so you can pick the checkpoint with the lowest validation loss rather than using the final one.

```yaml
checkpoint:
  enabled: true
  checkpoint_dir: checkpoints/
  model_save_format: safetensors
  save_consolidated: True

step_scheduler:
  val_every_steps: 10     # Validate frequently
  ckpt_every_steps: 10    # Save frequently (adjust based on disk space)
```

After training, evaluate each saved checkpoint on your validation set and select the best one.

## Scenarios by Dataset Scale

### Scenario 1: Small Dataset (<1K Samples)

Use PEFT with aggressive regularization. Example: Medical Meadow MedQA, a real-world medical QA dataset.

```yaml
# small_dataset_peft.yaml -- PEFT on a small medical dataset
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: True
  dim: 4                   # Very low rank to prevent overfitting
  alpha: 16
  dropout: 0.1             # Higher dropout for regularization
  use_triton: True

optimizer:
  _target_: torch.optim.AdamW
  lr: 2.0e-5
  betas: [0.9, 0.999]
  eps: 1e-8
  weight_decay: 0.05       # Moderate weight decay

step_scheduler:
  global_batch_size: 16
  local_batch_size: 4
  num_epochs: 3            # Few epochs
  val_every_steps: 5       # Frequent validation
  ckpt_every_steps: 20

# Medical Meadow MedQA (~10K medical QA samples)
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: medalpaca/medical_meadow_medqa
  split: train
  column_mapping:
    question: input
    answer: output

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: medalpaca/medical_meadow_medqa
  split: train
  limit_dataset_samples: 500
  column_mapping:
    question: input
    answer: output

checkpoint:
  enabled: true
  checkpoint_dir: checkpoints/small_dataset/
  model_save_format: safetensors
  save_consolidated: false

distributed:
  dp_size: none
  tp_size: 1
  cp_size: 1

distributed_config:
  _target_: nemo_automodel.components.distributed.config.FSDP2Config

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  shuffle: false

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater

rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 42
  ranked: true

dist_env:
  backend: nccl
  timeout_minutes: 1
```

Run it:
```bash
torchrun --nproc-per-node=1 examples/llm_finetune/finetune.py --config small_dataset_peft.yaml
```

### Scenario 2: Medium Dataset (5K-50K Samples)

Standard PEFT with moderate parameters. Example: Stanford Alpaca (~52K instruction-following samples).

```yaml
# medium_dataset_peft.yaml -- PEFT on Alpaca
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
  torch_dtype: bf16

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: True
  dim: 16
  alpha: 32
  dropout: 0.05
  use_triton: True

optimizer:
  _target_: torch.optim.Adam
  lr: 1.0e-5
  betas: [0.9, 0.999]
  eps: 1e-8
  weight_decay: 0

step_scheduler:
  global_batch_size: 64
  local_batch_size: 8
  num_epochs: 2
  val_every_steps: 10
  ckpt_every_steps: 50

dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: tatsu-lab/alpaca
  split: train
  column_mapping:
    question: instruction
    context: input
    answer: output

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: tatsu-lab/alpaca
  split: train
  limit_dataset_samples: 1000
  column_mapping:
    question: instruction
    context: input
    answer: output

checkpoint:
  enabled: true
  checkpoint_dir: checkpoints/medium_dataset/
  model_save_format: safetensors
  save_consolidated: false

distributed:
  dp_size: none
  tp_size: 1
  cp_size: 1

distributed_config:
  _target_: nemo_automodel.components.distributed.config.FSDP2Config

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  shuffle: false

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater

rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 42
  ranked: true

dist_env:
  backend: nccl
  timeout_minutes: 1
```

Run it (multi-GPU):
```bash
torchrun --nproc-per-node=8 examples/llm_finetune/finetune.py --config medium_dataset_peft.yaml
```

### Scenario 3: Large Dataset (>50K Samples)

Full SFT is viable with large datasets. Example: xLAM function calling (~60K) or Natural Instructions (~800K+).

```yaml
# large_dataset_sft.yaml -- Full SFT on xLAM function calling
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

optimizer:
  _target_: torch.optim.Adam
  lr: 2.0e-5
  betas: [0.9, 0.999]
  eps: 1e-8
  weight_decay: 0

lr_scheduler:
  lr_decay_style: cosine
  min_lr: 1.0e-6

step_scheduler:
  global_batch_size: 128
  local_batch_size: 8
  num_epochs: 2
  val_every_steps: 50
  ckpt_every_steps: 200

clip_grad_norm:
  max_norm: 1.0

dataset:
  _target_: nemo_automodel.components.datasets.llm.xlam.make_xlam_dataset
  dataset_name: Salesforce/xlam-function-calling-60k
  split: train

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.xlam.make_xlam_dataset
  dataset_name: Salesforce/xlam-function-calling-60k
  split: train[:256]
  limit_dataset_samples: 256

checkpoint:
  enabled: true
  checkpoint_dir: checkpoints/large_dataset/
  model_save_format: safetensors
  save_consolidated: True

distributed:
  dp_size: none
  tp_size: 1
  cp_size: 1

distributed_config:
  _target_: nemo_automodel.components.distributed.config.FSDP2Config

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  shuffle: false

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater

rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 42
  ranked: true

dist_env:
  backend: nccl
  timeout_minutes: 1
```

Run it (multi-GPU):
```bash
torchrun --nproc-per-node=8 examples/llm_finetune/finetune.py --config large_dataset_sft.yaml
```

:::{tip}
For the Natural Instructions dataset (~800K+), swap the dataset section:
```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: Muennighoff/natural-instructions
  split: train
  column_mapping:
    context: definition
    question: inputs
    answer: targets
```
:::

### Scenario 4: Memory-Constrained (Single Consumer GPU)

Use QLoRA for maximum memory efficiency. Example: Llama-3.1-8B with 4-bit quantization on Databricks Dolly.

```yaml
# qlora_constrained.yaml -- QLoRA on a single GPU
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.1-8B
  force_hf: true

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  match_all_linear: true
  dim: 16
  alpha: 32
  dropout: 0.1

quantization:
  load_in_4bit: True
  load_in_8bit: False
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_use_double_quant: True
  bnb_4bit_quant_type: nf4
  bnb_4bit_quant_storage: bfloat16

optimizer:
  _target_: torch.optim.AdamW
  lr: 1.0e-5
  betas: [0.9, 0.999]
  eps: 1e-8
  weight_decay: 0.01

step_scheduler:
  global_batch_size: 32
  local_batch_size: 4
  max_steps: 500
  val_every_steps: 50
  ckpt_every_steps: 100

dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: databricks/databricks-dolly-15k
  split: train
  column_mapping:
    context: context
    question: instruction
    answer: response

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: databricks/databricks-dolly-15k
  split: train
  limit_dataset_samples: 500
  column_mapping:
    context: context
    question: instruction
    answer: response

distributed:
  dp_size: none
  tp_size: 1
  cp_size: 1

distributed_config:
  _target_: nemo_automodel.components.distributed.config.FSDP2Config
  sequence_parallel: false

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

packed_sequence:
  packed_sequence_size: 0

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  shuffle: false

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater

rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 42
  ranked: true

dist_env:
  backend: nccl
  timeout_minutes: 1
```

Run it (single GPU):
```bash
torchrun --nproc-per-node=1 examples/llm_finetune/finetune.py --config qlora_constrained.yaml
```

## Using Your Own Dataset

NeMo AutoModel supports custom datasets via YAML configuration. You have two options:

### Option A: Python Dotted Path

If your dataset class is installable:
```yaml
dataset:
  _target_: my_package.datasets.build_legal_dataset
  split: train
```

### Option B: File Path

Reference a local Python file directly:
```yaml
dataset:
  _target_: /path/to/my_dataset.py:build_legal_dataset
  split: train
```

:::{note}
For file path imports, you may need to set `NEMO_ENABLE_USER_MODULES=1` in your environment. See the [configuration guide](../configuration.md) for details.
:::

For full details on building custom datasets, see [Integrate Your Own Text Dataset](dataset.md).

## Troubleshooting Common Issues

### Loss Not Decreasing

| Possible Cause | Solution |
|----------------|----------|
| Learning rate too low | Increase `optimizer.lr` by 5-10x |
| Data formatting issue | Verify your dataset preprocessing; check that loss mask is correct |
| Model too small for the task | Try a larger base model |
| Too few trainable parameters (PEFT) | Increase `peft.dim` or use `match_all_linear: True` |

### Loss Spikes or NaN

| Possible Cause | Solution |
|----------------|----------|
| Learning rate too high | Reduce `optimizer.lr` by 2-5x |
| Missing gradient clipping | Add `clip_grad_norm.max_norm: 1.0` |
| Data quality issue | Check for corrupted or extremely long samples |
| Numerical instability | Ensure `bfloat16` is used (default in FSDP2Config) |

### Validation Loss Increases (Overfitting)

| Possible Cause | Solution |
|----------------|----------|
| Too many epochs | Reduce `step_scheduler.num_epochs` |
| Learning rate too high | Reduce `optimizer.lr` |
| LoRA rank too high | Reduce `peft.dim` |
| Too little data | Augment dataset or use more aggressive regularization |

### Out of Memory (OOM)

| Possible Cause | Solution |
|----------------|----------|
| Batch size too large | Reduce `step_scheduler.local_batch_size`; increase `global_batch_size` relative to `local_batch_size` to use gradient accumulation |
| Model too large for GPU | Use `model.is_meta_device: true` with FSDP2, or switch to QLoRA |
| Sequence length too long | Reduce sequence length or enable `packed_sequence` |
| Activation memory | Enable `distributed_config.activation_checkpointing: true` |

### Generated Output is Repetitive or Low Quality

| Possible Cause | Solution |
|----------------|----------|
| Overfitting | Use an earlier checkpoint, reduce epochs, add regularization |
| Catastrophic forgetting (SFT) | Switch to PEFT to preserve base model capabilities |
| Poor data quality | Review and clean training data |
| Wrong task framing | Ensure your prompt format matches what the model expects |

## Parameter Reference

### `step_scheduler` Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `global_batch_size` | int | Total samples per optimizer step across all GPUs |
| `local_batch_size` | int | Samples per micro-batch per GPU |
| `num_epochs` | int | Total training epochs |
| `max_steps` | int | Maximum training steps (overrides `num_epochs` if set) |
| `val_every_steps` | int | Steps between validation runs |
| `ckpt_every_steps` | int | Steps between checkpoint saves |

:::{note}
Gradient accumulation steps are computed automatically:
`grad_acc_steps = global_batch_size / (local_batch_size * dp_size)`
:::

### `peft` Parameters (LoRA)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_modules` | list | `[]` | Glob pattern for modules to apply LoRA (e.g., `"*.proj"`) |
| `match_all_linear` | bool | `False` | Apply LoRA to all `nn.Linear` layers |
| `exclude_modules` | list | `[]` | Modules to exclude from LoRA |
| `dim` | int | `8` | LoRA rank (low-rank dimension) |
| `alpha` | int | `32` | LoRA scaling factor |
| `dropout` | float | `0.0` | LoRA dropout rate |
| `dropout_position` | str | `"post"` | Where to apply dropout: `"pre"` or `"post"` |
| `lora_A_init` | str | `"xavier"` | Initialization method for LoRA A matrix |
| `use_dora` | bool | `False` | Enable DoRA (Weight-Decomposed Low-Rank Adaptation) |
| `use_triton` | bool | `False` | Use optimized Triton kernel for LoRA |

### `optimizer` Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `_target_` | str | Optimizer class (e.g., `torch.optim.Adam`, `torch.optim.AdamW`) |
| `lr` | float | Learning rate |
| `betas` | list | Adam beta parameters |
| `eps` | float | Adam epsilon for numerical stability |
| `weight_decay` | float | L2 regularization strength |

### `lr_scheduler` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lr_decay_style` | str | `cosine` | Decay style: `cosine`, `linear`, `constant`, `WSD`, `inverse-square-root` |
| `min_lr` | float | -- | Minimum learning rate at end of decay |
| `lr_warmup_steps` | int | 10% of total steps | Number of linear warmup steps (auto-computed if not set) |

### `distributed` Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `dp_size` | int/none | Data parallel size (auto-inferred if `none`) |
| `tp_size` | int | Tensor parallel size |
| `cp_size` | int | Context parallel size |

## Summary of Key Recommendations

1. **Start conservative**: Low learning rate, few epochs, frequent validation.
2. **Prefer PEFT for small datasets**: LoRA with rank 4-16 prevents overfitting and preserves base capabilities.
3. **Monitor validation loss**: Always use a held-out set. The best model is the one with the lowest validation loss, not the lowest training loss.
4. **Use gradient clipping**: Set `clip_grad_norm.max_norm: 1.0` to prevent training instability.
5. **Checkpoint frequently**: Save every N steps and pick the best checkpoint post-training.
6. **Iterate**: Fine-tuning is empirical. Plan for 3-5 experimental runs to find optimal parameters.

## Further Reading

- [Supervised Fine-Tuning (SFT) and PEFT Guide](finetune.md)
- [Integrate Your Own Text Dataset](dataset.md)
- [Dataset Overview](../dataset-overview.md)
- [Configuration and Environment Variables](../configuration.md)
- [Checkpointing Guide](../checkpointing.md)
- [Gradient Checkpointing](../gradient-checkpointing.md)

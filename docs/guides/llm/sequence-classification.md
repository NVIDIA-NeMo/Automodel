# Sequence Classification (SFT/PEFT) with NeMo Automodel

## Introduction

Sequence classification tasks (e.g., sentiment analysis, topic classification, GLUE tasks) map input text to a discrete label. NeMo Automodel provides a lightweight recipe specialized for this setting that integrates with popular pretrained model formats and dataset sources. Integration with Hugging Face is supported.

This guide shows how to train a sequence classification model using the `TrainFinetuneRecipeForSequenceClassification` recipe, including optional Parameter-Efficient Fine-Tuning (LoRA).

## Quickstart

Use the example config for GLUE MRPC with RoBERTa-large + LoRA:

```bash
python3 examples/llm_seq_cls/seq_cls.py --config examples/llm_seq_cls/glue/mrpc_roberta_lora.yaml
```

- Loads `roberta-large` with `num_labels: 2`
- Builds GLUE MRPC datasets (train/validation)
- Optionally, enables LoRA via the `peft` block
- Trains and validates per `step_scheduler`

## What is the Sequence Classification Recipe?

`TrainFinetuneRecipeForSequenceClassification` is a config-driven trainer that orchestrates:
- Model and optimizer construction
- Dataset/Dataloader setup
- Training and validation loops
- Checkpointing and logging

It follows the same design as the SFT recipe in the fine-tune guide, but uses a standard cross-entropy classification loss and a simplified batching pipeline.

## Minimal Config Anatomy

```yaml
# GLUE MRPC with RoBERTa-large + LoRA
step_scheduler:
  global_batch_size: 32
  local_batch_size: 32
  ckpt_every_steps: 200
  val_every_steps: 100
  num_epochs: 2
  max_steps: 10

dist_env:
  backend: nccl
  timeout_minutes: 1

model:
  _target_: nemo_automodel.NeMoAutoModelForSequenceClassification.from_pretrained
  pretrained_model_name_or_path: roberta-large
  num_labels: 2

checkpoint:
  enabled: true
  checkpoint_dir: checkpoints/
  model_save_format: safetensors
  save_consolidated: true

distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none
  dp_replicate_size: 1
  tp_size: 1
  cp_size: 1
  sequence_parallel: false

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules:
  - "*.query"
  - "*.value"
  dim: 8
  alpha: 16
  dropout: 0.1

dataset:
  _target_: nemo_automodel.components.datasets.llm.seq_cls.GLUE_MRPC
  split: train

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.seq_cls.GLUE_MRPC
  split: validation

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater

optimizer:
  _target_: torch.optim.AdamW
  betas: [0.9, 0.999]
  eps: 1e-8
  lr: 3.0e-4
  weight_decay: 0


```

## Dataset Notes

- For single-sentence datasets (e.g., `yelp_review_full`, `imdb`), use `YelpReviewFull` or `IMDB` from `nemo_automodel.components.datasets.llm.seq_cls`.
- For GLUE MRPC (sentence-pair classification), use `GLUE_MRPC`, which tokenizes `(sentence1, sentence2)` with padding/truncation.

## LoRA (PEFT) Settings

- `target_modules`: glob to select linear layers (e.g., `"*.proj"`).
- `dim` (rank), `alpha`, `dropout`: tune per model/compute budget. Values `dim=8, alpha=16, dropout=0.1` are a good starting point for RoBERTa.
- The recipe automatically applies the adapters; no additional code changes are required.

## Running with torchrun

```bash
torchrun --nproc-per-node=2 examples/llm_seq_cls/seq_cls.py --config examples/llm_seq_cls/glue/mrpc_roberta_lora.yaml
```
You can adjust the number of GPUs as necessary via the `--nproc-per-node` knob.


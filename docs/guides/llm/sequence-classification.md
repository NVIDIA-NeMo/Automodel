# Sequence Classification (SFT/PEFT) with NeMo Automodel

## Introduction

Sequence classification tasks (e.g., sentiment analysis, topic classification, GLUE tasks) map input text to a discrete label. NeMo Automodel provides a lightweight recipe specialized for this setting that integrates with popular pretrained model formats and dataset sources. Integration with Hugging Face is supported.

This guide shows how to train a sequence classification model using the `TrainFinetuneRecipeForSequenceClassification` recipe, including optional Parameter-Efficient Fine-Tuning (LoRA).

## Quickstart

Use the example config for GLUE MRPC with RoBERTa-large + LoRA:

```bash
python -m examples.llm_seq_cls.seq_cls --config examples/llm_seq_cls/glue/mrpc_roberta_lora.yaml
```

- Loads `roberta-large` with `num_labels: 2`
- Builds GLUE MRPC datasets (train/validation)
- Enables LoRA via the `peft` block
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
model:
  _target_: nemo_automodel.NeMoAutoModelForSequenceClassification.from_pretrained
  pretrained_model_name_or_path: roberta-large
  num_labels: 2

peft:  # optional (enable LoRA)
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules: "*.proj"
  dim: 8
  alpha: 16
  dropout: 0.1

# GLUE MRPC dataset
dataset:
  _target_: nemo_automodel.components.datasets.llm.seq_cls.GLUE_MRPC
  split: train

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.seq_cls_collater

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.seq_cls.GLUE_MRPC
  split: validation

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.seq_cls_collater

step_scheduler:
  global_batch_size: 64
  local_batch_size: 8
  num_epochs: 3

optimizer:
  _target_: torch.optim.Adam
  lr: 3.0e-4
```

## Dataset Notes

- For single-sentence datasets (e.g., `yelp_review_full`, `imdb`), use `YelpReviewFull` or `IMDB` from `nemo_automodel.components.datasets.llm.seq_cls`.
- For GLUE MRPC (sentence-pair classification), use `GLUE_MRPC`, which tokenizes `(sentence1, sentence2)` with padding/truncation.
- The `seq_cls_collater` expects fixed-length tokenized inputs and outputs tensors: `input_ids`, `attention_mask`, `labels`.

## LoRA (PEFT) Settings

- `target_modules`: glob to select linear layers (e.g., `"*.proj"`).
- `dim` (rank), `alpha`, `dropout`: tune per model/compute budget. Values `dim=8, alpha=16, dropout=0.1` are a good starting point for RoBERTa.
- The recipe automatically applies the adapters; no additional code changes are required.

## Additional Examples

- `examples/llm_seq_cls/yelp/yelp_bert.yaml`: 5-way Yelp classification with BERT
- `examples/llm_seq_cls/imdb/imdb_qwen.yaml`: IMDB sentiment with Qwen
- `examples/llm_seq_cls/glue/mrpc_roberta_lora.yaml`: MRPC + RoBERTa-large + LoRA

## Running with torchrun

```bash
torchrun --nproc-per-node=2 -m examples.llm_seq_cls.seq_cls --config examples/llm_seq_cls/glue/mrpc_roberta_lora.yaml
```

## Inference and Adapters

Checkpoints are compatible with common tooling, including Hugging Face. For PEFT, the recipe saves adapter weights alongside a minimal adapter config. You can load them with standard APIs (e.g., Hugging Face + PEFT), similar to the examples in the fine-tune guide.

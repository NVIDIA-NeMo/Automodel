# Sequence Classification (SFT/PEFT) with NeMo AutoModel

## Introduction

You have text that needs to be categorized — sentiment, topic, paraphrase detection, entailment — and a pretrained language model that knows language but nothing about your labels. Sequence classification fine-tuning bridges that gap: you train the model on your labeled examples so it predicts the correct class for new inputs, without the cost of training from scratch. The result is a classifier you can evaluate and deploy directly. This guide walks you through that process with NeMo AutoModel — from installation through training and inference — using [RoBERTa-large](https://huggingface.co/roberta-large) and the [GLUE MRPC](https://huggingface.co/datasets/nyu-mll/glue) paraphrase detection dataset as a running example.

NeMo AutoModel supports two fine-tuning modes:

- **Supervised Fine-Tuning (SFT)** updates every parameter in the model — both the pretrained backbone and the classifier head. SFT gives the model the most freedom to adapt to your data, but requires enough GPU memory to hold the full optimizer state.
- **Parameter-Efficient Fine-Tuning (PEFT)** using [LoRA](https://arxiv.org/abs/2106.09685) freezes the pretrained backbone and injects small low-rank adapters into selected layers (e.g., attention projections). Only the adapters and the classifier head are trained. Because the vast majority of parameters stay frozen, PEFT uses significantly less GPU memory and produces smaller checkpoints — making it practical when the base model is large or GPU budget is limited, at the cost of slightly less representational flexibility than SFT.

### Workflow Overview

```text
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────────┐
│ 1. Install   │--->│ 2. Configure │--->│  3. Train    │--->│ 4. Evaluate  │--->│ 5. Inference   │
│              │    │              │    │              │    │              │    │                │
│ pip install  │    │ YAML recipe  │    │ python3 or   │    │ Val loss +   │    │ HF Transformers│
│ or Docker    │    │ Choose SFT   │    │ torchrun     │    │ accuracy in  │    │ load checkpoint│
│              │    │ or PEFT      │    │              │    │ training log │    │                │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └────────────────┘
```


| Step             | Section                                                           | SFT                                        | PEFT                        |
| ---------------- | ----------------------------------------------------------------- | ------------------------------------------ | --------------------------- |
| **1. Install**   | [Install NeMo AutoModel](#install-nemo-automodel)                 | Same                                       | Same                        |
| **2. Configure** | [Configure Your Training Recipe](#configure-your-training-recipe) | YAML without `peft:` section               | YAML with `peft:` section   |
| **3. Train**     | [Train the Model](#train-the-model)                               | Same command for both modes                | Same command for both modes |
| **4. Evaluate**  | [Evaluate](#evaluate)                                             | Validation loss + accuracy in training log | Same                        |
| **5. Inference** | [Run Inference](#run-inference)                                   | Load consolidated checkpoint directly      | Load base model + adapter   |


:::{note}
**Scope.** This guide covers training and inference for sequence classification. It does not cover QLoRA, large-model meta-device loading, checkpoint resumption, publishing to Hugging Face Hub, or vLLM deployment. For those topics, see the [SFT/PEFT fine-tune guide](finetune.md). The `automodel` CLI does not currently support sequence classification; use the recipe script directly as shown below.
:::

## Install NeMo AutoModel

```bash
pip3 install nemo-automodel
```

Alternatively, for ease-of-use, if you run into dependency or driver issues, use the pre-built Docker container:

```bash
docker pull nvcr.io/nvidia/nemo-automodel:26.02.00
docker run --gpus all -it --rm --shm-size=8g nvcr.io/nvidia/nemo-automodel:26.02.00
```

:::{important}
**Docker users:** Checkpoints are lost when the container exits unless you bind-mount the checkpoint directory to the host. See [Install with NeMo Docker Container](../installation.md#install-with-nemo-docker-container) and [Saving Checkpoints When Using Docker](../checkpointing.md#saving-checkpoints-when-using-docker).
:::

For the full set of installation methods, see the [installation guide](../installation.md).

## Configure Your Training Recipe

Training is configured through a [YAML](https://en.wikipedia.org/wiki/YAML) file with three required sections — **model**, **dataset**, and **step_scheduler** — plus an optional **peft** section. The sections below walk through each one. For the complete copy-pastable config, see [Full Recipe YAML](#full-recipe-yaml).

Both SFT and PEFT are driven by a **recipe** — a self-contained Python class (`[TrainFinetuneRecipeForSequenceClassification](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/llm/train_seq_cls.py)`) that wires together model loading, dataset preparation, training, checkpointing, and logging.

### Model

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForSequenceClassification.from_pretrained
  pretrained_model_name_or_path: roberta-large
  num_labels: 2
```

This guide uses **RoBERTa-large** as a running example. Replace `pretrained_model_name_or_path` with any Hugging Face model ID compatible with `AutoModelForSequenceClassification` (encoder models like BERT, DeBERTa; decoder models like GPT-2). Set `num_labels` to the number of classes in your task.

:::{details} Accessing gated models
Some Hugging Face models are **gated**. If the model page shows a "Request access" button:

1. Log in with your Hugging Face account and accept the license.
2. Ensure the token you use (from `huggingface-cli login` or `HF_TOKEN`) belongs to the approved account.

Pulling a gated model without an authorized token triggers a 403 error.
:::

### Dataset

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.seq_cls.GLUE_MRPC
  split: train

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.seq_cls.GLUE_MRPC
  split: validation
```

This guide uses **GLUE MRPC** as a running example. The dataset downloads automatically from Hugging Face on first run. MRPC is a sentence-pair classification task: given two sentences, predict whether they are paraphrases (label 1) or not (label 0). To use your own dataset, see [Integrate Your Own Text Dataset](dataset.md).

:::{details} About GLUE MRPC
The Microsoft Research Paraphrase Corpus (MRPC) is a binary classification task from the [GLUE benchmark](https://gluebenchmark.com/). Each example contains two sentences and a label:

- **Label 0**: Not a paraphrase.
- **Label 1**: Paraphrase.

The training set has 3,668 examples; the validation set has 408 examples. Here is what the data looks like:

**Paraphrase (label 1):**

```json
{
  "sentence1": "Revenue in the first quarter of the year dropped 15 percent from the same period a year earlier .",
  "sentence2": "With the scandal hanging over Stewart 's company , revenue the first quarter dropped 15 percent from the same period a year earlier .",
  "label": 1
}
```

**Not a paraphrase (label 0):**

```json
{
  "sentence1": "The Nasdaq had a weekly gain of 17.27 , or 1.2 percent , closing at 1,520.15 on Friday .",
  "sentence2": "The tech-laced Nasdaq Composite .IXIC advanced 30.46 points , or 2.04 percent , to 1,520.15 .",
  "label": 0
}
```

The `GLUE_MRPC` class tokenizes both sentences together with truncation and returns `input_ids`, `attention_mask`, and `labels`.
:::

### PEFT (Optional)

```yaml
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules:
  - "*.query"       # glob pattern matching self-attention query layers
  - "*.value"       # glob pattern matching self-attention value layers
  dim: 8            # low-rank dimension of the adapters
  alpha: 16         # scaling factor (effective scale = alpha / dim)
  dropout: 0.1      # dropout applied to the LoRA pathway
```

Including a `peft:` section enables LoRA fine-tuning. Remove it entirely to run SFT instead.

When PEFT is enabled, the recipe freezes the entire pretrained backbone and then does two things: (1) injects LoRA adapters into the layers matched by `target_modules`, and (2) unfreezes the classifier head. So the trainable parameters are the LoRA adapters plus the classifier head — everything else stays frozen. The classifier head is fully trained in both SFT and PEFT modes.

:::{details} Choosing target_modules for other architectures
The `target_modules` patterns use wildcards against fully-qualified module names. For RoBERTa, `"*.query"` and `"*.value"` target the self-attention projections. For other architectures, list the model's modules to find the right patterns:

```python
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained("roberta-large")
print([name for name, _ in model.named_modules()])
```

:::

### Training Schedule

```yaml
step_scheduler:
  num_epochs: 3       # Train over the dataset three times.
```

When only `num_epochs` is specified, the recipe uses the following defaults: batch size of 1 per GPU (`global_batch_size` and `local_batch_size` both default to 1), AdamW optimizer with lr=2e-5, FSDP2 distribution, checkpoints saved once per epoch to `checkpoints/`, and validation run at each checkpoint step. To set a larger batch size or more frequent validation, see the [Full Configuration Reference](#full-configuration-reference).

### Full Recipe YAML

:::{details} seq_cls_config.yaml (click to expand)
Save as `seq_cls_config.yaml`. This config runs PEFT (LoRA). To run SFT instead, remove the `peft:` section.

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForSequenceClassification.from_pretrained
  pretrained_model_name_or_path: roberta-large
  num_labels: 2

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

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.seq_cls.GLUE_MRPC
  split: validation

step_scheduler:
  num_epochs: 3
```

:::

## Train the Model

```bash
torchrun --nproc-per-node=1 examples/llm_seq_cls/seq_cls.py --config seq_cls_config.yaml
```

:::{note}
For multi-GPU training, increase `--nproc-per-node` (e.g., `--nproc-per-node=2` for two GPUs). See the [torchrun docs](https://docs.pytorch.org/docs/stable/elastic/run.html) for additional options.
:::

### Sample Output

```text
INFO:root:step 99  | epoch 0 | loss 0.4312 | accuracy 0.8125 | grad_norm 1.2344 | lr 2.00e-05 | mem 4.52 GiB | tps 9245.30 (9245.30/gpu)
INFO:root:step 199 | epoch 1 | loss 0.1987 | accuracy 0.9375 | grad_norm 0.5625 | lr 2.00e-05 | mem 4.52 GiB | tps 9312.14 (9312.14/gpu)
Saving checkpoint to checkpoints/epoch_1_step_200
INFO:root:[val] step 199 | epoch 1 | loss 0.2469 | accuracy 0.8750 | lr 2.00e-05
```

Each log line reports the current loss, accuracy, gradient norm, peak GPU memory, and tokens per second. Exact step numbers depend on your dataset size, batch configuration, and number of GPUs. Small fluctuations between steps are normal — look at the overall downward trend in loss and upward trend in accuracy rather than individual values.

### Checkpoint Contents

Checkpoints are saved in native Hugging Face format, so no conversion is required — they work directly with Transformers, PEFT, and other tools in the HF ecosystem. **SFT checkpoints** contain the full model weights at `model/consolidated/` and can be loaded directly. **PEFT checkpoints** contain only the adapter weights — at inference time you must load the original base model and apply the adapter on top.

:::{details} Checkpoint directory structure
**SFT checkpoint:**

```bash
$ tree checkpoints/epoch_0_step_200/
checkpoints/epoch_0_step_200/
├── config.yaml
├── dataloader.pt
├── model
│   ├── consolidated
│   │   ├── config.json
│   │   ├── model-00001-of-00001.safetensors
│   │   ├── model.safetensors.index.json
│   │   ├── tokenizer.json
│   │   └── ...
│   └── shard-*.safetensors
├── optim
│   ├── __0_0.distcp
│   └── __1_0.distcp
├── rng.pt
└── step_scheduler.pt
```

**PEFT checkpoint:**

```bash
$ tree checkpoints/epoch_0_step_200/
checkpoints/epoch_0_step_200/
├── config.yaml
├── dataloader.pt
├── model
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── automodel_peft_config.json
├── optim
│   ├── __0_0.distcp
│   └── __1_0.distcp
├── rng.pt
└── step_scheduler.pt
```

The `model/` directory is what you use for inference. The other files (`dataloader.pt`, `optim/`, `rng.pt`, `step_scheduler.pt`) exist for resuming training and are not needed for inference.
:::

## Evaluate

### During Training: Validation Loss and Accuracy

The recipe automatically computes validation loss and accuracy at the interval set by `val_every_steps`. Look for `[val]` lines in the training log:

```text
INFO:root:[val] step 199 | epoch 1 | loss 0.2469 | accuracy 0.8750 | lr 2.00e-05
```

A decreasing validation loss and increasing accuracy across checkpoints indicate the model is learning. If validation loss plateaus or increases while training loss continues to drop, the model may be overfitting — consider stopping earlier or reducing the learning rate.

Metrics are also logged to JSONL files for programmatic analysis:

- `<checkpoint_dir>/training.jsonl` — one JSON object per training step.
- `<checkpoint_dir>/validation.jsonl` — one JSON object per validation run.

### After Training: Computing F1 and Precision/Recall

The in-training `val_accuracy` gives a quick signal, but the standard metric for MRPC is F1 (the harmonic mean of precision and recall). To compute it, run the trained model over the validation set and compare predictions to ground-truth labels:

```python
import json
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import classification_report
from datasets import load_dataset

# Load model (use PeftModel.from_pretrained for PEFT — see Run Inference below)
ckpt_path = "checkpoints/epoch_0_step_200/model/consolidated"
tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load validation data
val_data = load_dataset("glue", "mrpc", split="validation")

preds, labels = [], []
for example in val_data:
    inputs = tokenizer(example["sentence1"], example["sentence2"],
                       truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    preds.append(logits.argmax(dim=-1).item())
    labels.append(example["label"])

print(classification_report(labels, preds, target_names=["not paraphrase", "paraphrase"]))
```

```text
               precision    recall  f1-score   support

not paraphrase       0.82      0.78      0.80       140
    paraphrase       0.88      0.91      0.89       268

      accuracy                           0.86       408
   macro avg         0.85      0.84      0.85       408
weighted avg         0.86      0.86      0.86       408
```

:::{tip}
The numbers above are illustrative. Your results will vary depending on the model, number of epochs, learning rate, and whether you use SFT or PEFT.
:::

## Run Inference

Because SFT checkpoints are self-contained while PEFT checkpoints store only adapter weights (see [Checkpoint Contents](#checkpoint-contents)), the loading procedure differs between the two modes.

### SFT Inference

The SFT checkpoint at `model/consolidated/` is a complete Hugging Face model and can be loaded directly:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ckpt_path = "checkpoints/epoch_0_step_200/model/consolidated"
tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

inputs = tokenizer("He said the foodservice revenue was up .",
                    "The food service revenue was up .",
                    return_tensors="pt").to(device)
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(dim=-1).item()
# For MRPC: 0 = not paraphrase, 1 = paraphrase
print(predicted_class)  # 1
```

### PEFT Inference

PEFT adapters must be loaded on top of the base model:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

base_model_name = "roberta-large"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)

adapter_path = "checkpoints/epoch_0_step_200/model/"
model = PeftModel.from_pretrained(model, adapter_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

inputs = tokenizer("He said the foodservice revenue was up .",
                    "The food service revenue was up .",
                    return_tensors="pt").to(device)
outputs = model(**inputs)
predicted_class = outputs.logits.argmax(dim=-1).item()
# For MRPC: 0 = not paraphrase, 1 = paraphrase
print(predicted_class)  # 1
```

## Full Configuration Reference

This section documents all available config fields for the sequence classification recipe. For the quick-start config, see [Configure Your Training Recipe](#configure-your-training-recipe).

### Switching Between SFT and PEFT

The `peft:` section controls which mode runs:


| Mode                     | What to do in the YAML               |
| ------------------------ | ------------------------------------ |
| **PEFT (LoRA)**          | Include the `peft:` section.         |
| **SFT (full-parameter)** | Remove the `peft:` section entirely. |


All other config sections remain the same for both modes. In both modes, the classifier head is fully trained.

### Full Configuration

All config blocks use a Hydra-style `_target_` field to specify which Python class or factory method to instantiate. For example, `_target_: torch.optim.AdamW` means "construct a `torch.optim.AdamW` instance with the remaining fields as keyword arguments."

:::{details} Full Config
:open:

```yaml
# ── Model ──
model:
  _target_: nemo_automodel.NeMoAutoModelForSequenceClassification.from_pretrained
  pretrained_model_name_or_path: roberta-large
  num_labels: 2  # number of output classes

# ── PEFT (remove this section entirely for SFT) ──
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules:       # glob patterns matching module FQNs
  - "*.query"
  - "*.value"
  dim: 8                # low-rank dimension
  alpha: 16             # scaling factor (effective scale = alpha / dim)
  dropout: 0.1          # LoRA dropout

# ── Dataset ──
dataset:
  _target_: nemo_automodel.components.datasets.llm.seq_cls.GLUE_MRPC
  split: train

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.seq_cls.GLUE_MRPC
  split: validation

# ── Training schedule ──
step_scheduler:
  global_batch_size: 32   # effective batch size per optimizer step (all GPUs)
  local_batch_size: 32    # micro-batch size per GPU per forward pass
  ckpt_every_steps: 200   # save checkpoint every N optimizer steps
  val_every_steps: 100    # run validation every N optimizer steps
  num_epochs: 3
  max_steps: null         # null = derived from num_epochs; if set, takes precedence

# ── Distributed ──
dist_env:
  backend: nccl
  timeout_minutes: 1

distributed:
  strategy: fsdp2         # fsdp2 (recommended), megatron_fsdp, or ddp
  dp_size: null           # null = auto-detected from available GPUs
  tp_size: 1              # tensor-parallel size (1 = disabled)
  cp_size: 1              # context-parallel size (1 = disabled)
  sequence_parallel: false

# ── Dataloaders ──
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater

# ── Checkpointing ──
checkpoint:
  enabled: true
  checkpoint_dir: checkpoints/
  model_save_format: safetensors      # "safetensors" (recommended) or "torch"
  save_consolidated: true             # single HF-compatible bundle
  skip_task_head_prefixes_for_base_model: ["classifier."]
    # Skip classifier head from base model (randomly initialized instead).
    # Applies to both SFT and PEFT.

# ── Optimizer ──
optimizer:
  _target_: torch.optim.AdamW
  betas: [0.9, 0.999]
  eps: 1e-8
  lr: 2.0e-5
  weight_decay: 0.01

# ── Logging (optional) ──
# wandb:
#   project: <your_wandb_project>
#   entity: <your_wandb_entity>
#   name: <your_wandb_exp_name>
```

:::

### Config Field Reference


| Section          | Required?   | What to change                                                                                                                                                                                                    |
| ---------------- | ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`          | Yes         | Set `pretrained_model_name_or_path` to your Hugging Face model ID. Set `num_labels` to your number of classes.                                                                                                    |
| `peft`           | PEFT only   | Remove entirely for SFT. Adjust `dim` and `alpha` to tune adapter capacity. Adjust `target_modules` for your model architecture.                                                                                  |
| `dataset`        | Yes         | Change `_target_` and `split` for your data. See [Integrate Your Own Text Dataset](dataset.md).                                                                                                                   |
| `step_scheduler` | Yes         | `global_batch_size` and `local_batch_size` control batch sizes. `max_steps` overrides `num_epochs` if set. Gradient accumulation is derived: `grad_acc_steps = global_batch_size / (local_batch_size * dp_size)`. |
| `distributed`    | Yes         | `dp_size: null` means auto-detect from world size. Adjust `tp_size` for tensor parallelism across GPUs.                                                                                                           |
| `checkpoint`     | Recommended | Set `checkpoint_dir` to a persistent path, especially in Docker.                                                                                                                                                  |
| `optimizer`      | Optional    | Defaults are reasonable. Any `torch.optim` class can be substituted via `_target_`.                                                                                                                               |
| `wandb`          | Optional    | Uncomment and configure to enable Weights & Biases logging.                                                                                                                                                       |


### Metrics Reference

**Training metrics** (logged per optimizer step, accumulated across all micro-batches and all-reduced across data-parallel ranks):


| Metric        | Description                                                                                                           |
| ------------- | --------------------------------------------------------------------------------------------------------------------- |
| `loss`        | Cross-entropy classification loss (mean over micro-batches).                                                          |
| `accuracy`    | Fraction of correct predictions across all micro-batches in the optimizer step.                                       |
| `grad_norm`   | Gradient norm after clipping (default max_norm=1.0).                                                                  |
| `lr`          | Current learning rate.                                                                                                |
| `mem`         | Peak GPU memory allocated (GiB) since the previous training step. Reset after each log.                               |
| `tps`         | Tokens processed per second (throughput).                                                                             |
| `tps_per_gpu` | Throughput per GPU.                                                                                                   |
| `mfu`         | Model FLOPs utilization as a percentage of device peak TFLOPs (e.g., 989 TFLOPs for H100). Omitted if not computable. |


**Validation metrics** (logged per validation run):


| Metric         | Description                                                     |
| -------------- | --------------------------------------------------------------- |
| `val_loss`     | Mean of per-batch cross-entropy losses over the validation set. |
| `val_accuracy` | Fraction of correct predictions over the entire validation set. |


## Further Reading

- [SFT/PEFT Fine-Tune Guide](finetune.md) — the LLM fine-tuning guide for next-token prediction, covering QLoRA, large-model meta-device loading, checkpoint resumption, publishing to Hugging Face Hub, and vLLM deployment.
- [Integrate Your Own Text Dataset](dataset.md) — how to define custom datasets for use with NeMo AutoModel recipes.
- [Installation Guide](../installation.md) — full installation and Docker instructions.
- [GLUE Benchmark](https://gluebenchmark.com/) — details on the GLUE tasks including MRPC.


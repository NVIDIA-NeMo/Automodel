# Supervised Fine-Tuning (SFT) and Parameter-Efficient Fine-Tuning (PEFT) with NeMo AutoModel

## Introduction

Fine-tuning teaches a pretrained language model to follow instructions, answer questions, or perform tasks specific to your data. You start with a general-purpose model, train it on your own examples, and end up with a model you can deploy. This guide walks you through that process end-to-end with NeMo AutoModel — from installation through training, evaluation, and deployment — using [Meta LLaMA 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B) and the [SQuAD v1.1](https://huggingface.co/datasets/rajpurkar/squad) dataset as a running example.

NeMo AutoModel supports two fine-tuning modes:

- **Supervised Fine-Tuning (SFT)** updates all model parameters. Use SFT when you need maximum accuracy and have sufficient compute.
- **Parameter-Efficient Fine-Tuning (PEFT)** via [LoRA](https://arxiv.org/abs/2106.09685) freezes the base model and trains small low-rank adapters. PEFT reduces trainable parameters to less than 1% of the original model, lowering memory and storage costs.

NeMo AutoModel integrates directly with Hugging Face Transformers, so no checkpoint conversion is required.

### Workflow Overview

```text
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 1. Install   │───▶│ 2. Configure │───▶│  3. Train    │───▶│ 4. Inference │───▶│ 5. Evaluate  │───▶│ 6. Publish   │───▶│  7. Deploy   │
│              │    │              │    │              │    │              │    │              │    │  (optional)  │    │  (optional)  │
│ pip install  │    │ Write YAML   │    │ automodel CLI│    │ HF generate  │    │ Val loss +   │    │ HF Hub       │    │ vLLM serving │
│ or Docker    │    │ Choose SFT   │    │ or torchrun  │    │ API          │    │ lm-eval-     │    │ upload       │    │              │
│              │    │ or PEFT      │    │              │    │              │    │ harness      │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

| Step | Section | SFT | PEFT |
|------|---------|-----|------|
| **1. Install** | [Install NeMo AutoModel](#install-nemo-automodel) | Same | Same |
| **2. Configure** | [Write the Config](#write-the-config) | YAML without `peft:` section | YAML with `peft:` section |
| **3. Train** | [Run Training](#run-training) | Same command for both modes | Same command for both modes |
| **4. Inference** | [Run Inference](#run-inference) | Load consolidated checkpoint directly | Load base model + adapter |
| **5. Evaluate** | [Evaluate the Fine-Tuned Model](#evaluate-the-fine-tuned-model) | Validation loss during training; lm-eval-harness post-training | Same |
| **6. Publish** | [Publish to HF Hub](#publish-to-the-hugging-face-hub) | Upload `model/consolidated/` | Upload `model/` (adapter only) |
| **7. Deploy** | [Deploy with vLLM](#deploy-with-vllm) | `vllm.LLM(model=...)` | `vLLMHFExporter` with `--lora-model` |

### Model and Dataset

This guide uses **Meta LLaMA 3.2 1B** (`meta-llama/Llama-3.2-1B`) and the **SQuAD v1.1** dataset (`rajpurkar/squad`) as a running example. Both are placeholders — replace the model with any supported [Hugging Face model ID](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/model-coverage/llm.md), and swap the dataset by changing the `dataset` / `validation_dataset` sections in the YAML (see [Integrate Your Own Text Dataset](dataset.md) and [Dataset Overview](../dataset-overview.md)).

:::{details} About LLaMA 3.2 1B
LLaMA is a family of decoder-only transformer models developed by Meta. The 1B variant is a compact model suitable for research and edge deployment, featuring RoPE positional embeddings, grouped-query attention (GQA), and SwiGLU activations.
:::

:::{details} About SQuAD v1.1
The Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset where each example consists of a Wikipedia passage, a question, and a span answer. SQuAD v1.1 guarantees all questions are answerable from the context, making it suitable for straightforward fine-tuning.

Example:
```json
{
    "context": "Architecturally, the school has a Catholic character. ...",
    "question": "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?",
    "answers": { "text": ["Saint Bernadette Soubirous"], "answer_start": [515] }
}
```
:::

## Install NeMo AutoModel

```bash
pip3 install nemo-automodel
```

Alternatively, if you run into dependency or driver issues, use the pre-built Docker container:

```bash
docker pull nvcr.io/nvidia/nemo-automodel:25.11.00
docker run --gpus all -it --rm --shm-size=8g nvcr.io/nvidia/nemo-automodel:25.11.00
```

:::{important}
**Docker users:** Checkpoints are lost when the container exits unless you bind-mount the checkpoint directory to the host. See [Install with NeMo Docker Container](../installation.md#install-with-nemo-docker-container) and [Saving Checkpoints When Using Docker](../checkpointing.md#saving-checkpoints-when-using-docker).
:::

For the full set of installation methods, see the [installation guide](../installation.md).

### Access Gated Models

Some Hugging Face models are **gated**. If the model page shows a "Request access" button:

1. Log in with your Hugging Face account and accept the license.
2. Ensure the token you pass (via `huggingface-cli login` or `HF_TOKEN`) belongs to the approved account.

:::{note}
Pulling a gated model without an authorized token triggers a 403 error.
:::

## Write the Config

Both SFT and PEFT are driven by a **recipe** — a self-contained module that wires together model loading, dataset preparation, training, checkpointing, and logging (see the `TrainFinetuneRecipeForNextTokenPrediction` [source](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/llm/train_ft.py)). Recipes are configured entirely through YAML; the only difference between SFT and PEFT is whether a `peft:` section is present (see [Switching Between SFT and PEFT](#switching-between-sft-and-peft)). For a quick standalone example, see the [finetune.py recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/finetune.py).

Save the following as `finetune_config.yaml`. This config launches a PEFT (LoRA) fine-tuning run. To run SFT instead, remove the `peft:` section — see [Switching Between SFT and PEFT](#switching-between-sft-and-peft).

Each `_target_` field specifies the Python class or function that NeMo AutoModel instantiates for that config section.

```yaml
# ── Model ──
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

# ── PEFT (remove this section entirely for SFT) ──
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules: "*.proj"  # glob pattern matching linear layer FQNs
  dim: 8                    # low-rank dimension of the adapters
  alpha: 32                 # scaling factor for learned weights

# ── Dataset ──
dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: train

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: validation

# ── Training schedule ──
step_scheduler:
  num_epochs: 1
```

All other settings (distributed strategy, optimizer, checkpointing, logging) use sensible defaults. See the [Full Configuration Reference](#full-configuration-reference) to customize them.

## Run Training

You can run the recipe via the AutoModel CLI or directly with torchrun.

### AutoModel CLI

```bash
automodel finetune llm -c finetune_config.yaml
```

where `finetune` is the command and `llm` is the model domain.

### torchrun

```bash
torchrun --nproc-per-node=8 examples/llm_finetune/finetune.py --config finetune_config.yaml
```

See the recipe [source](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/llm/train_ft.py) and [torchrun docs](https://docs.pytorch.org/docs/stable/elastic/run.html) for details.

### Sample Output

```text
$ automodel finetune llm -c finetune_config.yaml
INFO:root:Domain:  llm
INFO:root:Command: finetune
INFO:root:Config:  /mnt/4tb/auto/Automodel/finetune_config.yaml
INFO:root:Running job using source from: /mnt/4tb/auto/Automodel
INFO:root:Launching job locally on 2 devices
cfg-path: /mnt/4tb/auto/Automodel/finetune_config.yaml
INFO:root:step 4 | epoch 0 | loss 1.5514 | grad_norm 102.0000 | mem: 11.66 GiB | tps 6924.50
INFO:root:step 8 | epoch 0 | loss 0.7913 | grad_norm 46.2500 | mem: 14.58 GiB | tps 9328.79
Saving checkpoint to checkpoints/epoch_0_step_10
INFO:root:step 12 | epoch 0 | loss 0.4358 | grad_norm 23.8750 | mem: 15.48 GiB | tps 9068.99
INFO:root:step 16 | epoch 0 | loss 0.2057 | grad_norm 12.9375 | mem: 16.47 GiB | tps 9148.28
INFO:root:step 20 | epoch 0 | loss 0.2557 | grad_norm 13.4375 | mem: 12.35 GiB | tps 9196.97
Saving checkpoint to checkpoints/epoch_0_step_20
INFO:root:[val] step 20 | epoch 0 | loss 0.2469
```

Each log line reports the current loss, gradient norm, peak GPU memory, and tokens per second (TPS). Small fluctuations between steps (e.g., 0.2057 to 0.2557 above) are normal — look at the overall downward trend rather than individual values.

### Checkpoint Contents

SFT and PEFT produce different checkpoint layouts. **SFT checkpoints** contain the full model weights at `model/consolidated/` and can be loaded directly. **PEFT checkpoints** contain only the adapter weights (~MBs instead of GBs) — at inference time you must load the original base model and apply the adapter on top. This distinction affects every downstream step (inference, publishing, deployment).

:::{details} Checkpoint directory structure
**SFT checkpoint:**
```bash
$ tree checkpoints/epoch_0_step_10/
checkpoints/epoch_0_step_10/
├── config.yaml
├── dataloader.pt
├── model
│   ├── consolidated
│   │   ├── config.json
│   │   ├── model-00001-of-00001.safetensors
│   │   ├── model.safetensors.index.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   ├── tokenizer_config.json
│   │   └── generation_config.json
│   ├── shard-00001-model-00001-of-00001.safetensors
│   └── shard-00002-model-00001-of-00001.safetensors
├── optim
│   ├── __0_0.distcp
│   └── __1_0.distcp
├── rng.pt
└── step_scheduler.pt

4 directories, 11 files
```

**PEFT checkpoint:**
```bash
$ tree checkpoints/epoch_0_step_10/
checkpoints/epoch_0_step_10/
├── dataloader.pt
├── config.yaml
├── model
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── automodel_peft_config.json
├── optim
│   ├── __0_0.distcp
│   └── __1_0.distcp
├── rng.pt
└── step_scheduler.pt

2 directories, 8 files
```
:::

## Run Inference

Inference uses the Hugging Face `generate` API. Because SFT checkpoints are self-contained while PEFT checkpoints store only adapter weights (see [Checkpoint Contents](#checkpoint-contents)), the loading procedure differs between the two modes.

### PEFT Inference

PEFT adapters must be loaded on top of the base model:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
model = AutoModelForCausalLM.from_pretrained(base_model_name)

adapter_path = "checkpoints/epoch_0_step_10/model/"
model = PeftModel.from_pretrained(model, adapter_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

prompt = (
    "Context: Architecturally, the school has a Catholic character. "
    "Atop the Main Building's gold dome is a golden statue of the Virgin Mary. "
    "Immediately in front of the Main Building and facing it, is a copper statue of Christ "
    "with arms upraised with the legend 'Venite Ad Me Omnes'.\n\n"
    "Question: What is atop the Main Building?\n\n"
    "Answer:"
)
inputs = tokenizer(prompt, return_tensors="pt").to(device)
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

### SFT Inference

The SFT checkpoint at `model/consolidated/` is a complete Hugging Face model and can be loaded directly:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ckpt_path = "checkpoints/epoch_0_step_10/model/consolidated"
tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
model = AutoModelForCausalLM.from_pretrained(ckpt_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

prompt = (
    "Context: Architecturally, the school has a Catholic character. "
    "Atop the Main Building's gold dome is a golden statue of the Virgin Mary. "
    "Immediately in front of the Main Building and facing it, is a copper statue of Christ "
    "with arms upraised with the legend 'Venite Ad Me Omnes'.\n\n"
    "Question: What is atop the Main Building?\n\n"
    "Answer:"
)
inputs = tokenizer(prompt, return_tensors="pt").to(device)
output = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Evaluate the Fine-Tuned Model

### During Training: Validation Loss

The recipe automatically computes validation loss at the interval set by `val_every_steps`. Look for `[val]` lines in the training log:

```text
INFO:root:[val] step 20 | epoch 0 | loss 0.2469
```

A decreasing validation loss across checkpoints indicates the model is learning. If validation loss plateaus or increases while training loss continues to drop, the model may be overfitting — consider stopping earlier or reducing the learning rate.

### After Training: lm-eval-harness

For task-specific benchmarks (e.g., MMLU, GSM8k, HellaSwag accuracy), use [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) with the fine-tuned checkpoint:

```bash
pip install lm-eval

# SFT checkpoint (using vLLM backend for faster evaluation)
lm_eval --model vllm \
  --model_args pretrained=checkpoints/epoch_0_step_20/model/consolidated/ \
  --tasks hellaswag \
  --batch_size auto

# PEFT adapter (using Hugging Face backend with built-in PEFT support)
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-1B,peft=checkpoints/epoch_0_step_20/model/ \
  --tasks hellaswag \
  --batch_size auto
```

:::{tip}
The SFT example uses the `vllm` backend for faster evaluation (requires `pip install vllm`; see [Deploy with vLLM](#deploy-with-vllm) for setup details). The PEFT example uses the `hf` backend with lm-eval's built-in PEFT support to load the adapter on top of the base model.
:::

:::{tip}
Run lm-eval-harness on the base model *before* fine-tuning to establish a baseline, then compare against the fine-tuned checkpoint.
:::

## Publish to the Hugging Face Hub

Fine-tuned checkpoints and PEFT adapters are stored in Hugging Face-native format and can be uploaded directly to the Hub.

1. Install the Hugging Face Hub library (if not already installed):

```bash
pip3 install huggingface_hub
```

2. Log in to Hugging Face:

```bash
huggingface-cli login
```

3. Upload:

**SFT checkpoint:**
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="checkpoints/epoch_0_step_10/model/consolidated",
    repo_id="your-username/llama3.2_1b-finetuned-squad",
    repo_type="model",
)
```

**PEFT adapter:**
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="checkpoints/epoch_0_step_10/model",
    repo_id="your-username/llama3.2_1b-lora-squad",
    repo_type="model",
)
```

Once uploaded, load the checkpoint or adapter directly from the Hub:

**SFT:**
```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("your-username/llama3.2_1b-finetuned-squad")
```

**PEFT:**
```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
model = PeftModel.from_pretrained(model, "your-username/llama3.2_1b-lora-squad")
```

## Deploy with vLLM

[vLLM](https://github.com/vllm-project/vllm) is an efficient inference engine for production deployment of LLMs.

:::{note}
Make sure vLLM is installed (`pip install vllm`, or use an environment that includes it).
:::

### SFT Checkpoint with vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(model="checkpoints/epoch_0_step_10/model/consolidated/", model_impl="transformers")
params = SamplingParams(max_tokens=20)
outputs = llm.generate("Toronto is a city in Canada.", sampling_params=params)
print(f"Generated text: {outputs[0].outputs[0].text}")
```
```text
>>> Generated text:  It is the capital of Ontario. Toronto is a global hub for cultural tourism. The City of Toronto
```

### PEFT Adapter with vLLM

PEFT adapter serving uses the `vLLMHFExporter` class, which is provided by the `nemo` package — a separate dependency from `nemo-automodel`.

:::{important}
Install both packages before proceeding:
```bash
pip install nemo vllm
```
:::

```python
from nemo.export.vllm_hf_exporter import vLLMHFExporter

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, help="Local path of the base model")
    parser.add_argument('--lora-model', required=True, type=str, help="Local path of the LoRA adapter")
    args = parser.parse_args()

    lora_model_name = "lora_model"

    exporter = vLLMHFExporter()
    exporter.export(model=args.model, enable_lora=True)
    exporter.add_lora_models(lora_model_name=lora_model_name, lora_model=args.lora_model)

    print("vLLM Output: ", exporter.forward(input_texts=["How are you doing?"], lora_model_name=lora_model_name))
```

## Full Configuration Reference

This section documents all available config fields for the fine-tuning recipe. For the quick-start config, see [Write the Config](#write-the-config).

### Switching Between SFT and PEFT

The `peft:` section controls which mode runs:

| Mode | What to do in the YAML |
|------|----------------------|
| **PEFT (LoRA)** | Include the `peft:` section as shown below. |
| **SFT (full-parameter)** | Remove the `peft:` section entirely. |

All other config sections remain the same for both modes.

### Full Config

:::{details} Full Config
:open:
```yaml
# ── Model ──
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

# ── PEFT (remove this section entirely for SFT) ──
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules: "*.proj"  # glob pattern matching linear layer FQNs
  dim: 8                    # low-rank dimension of the adapters
  alpha: 32                 # scaling factor for learned weights
  use_triton: True          # use optimized Triton-based LoRA kernel (requires triton)

# ── Dataset ──
dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: train

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: validation
  limit_dataset_samples: 64

# ── Training schedule ──
step_scheduler:
  grad_acc_steps: 4         # micro-batches accumulated before each optimizer step
  ckpt_every_steps: 10      # save checkpoint every N gradient steps
  val_every_steps: 10       # run validation every N gradient steps
  num_epochs: 1

# ── Distributed ──
dist_env:
  backend: nccl
  timeout_minutes: 1

distributed:
  strategy: fsdp2
  dp_size: null   # auto-detected from world size, tp_size, and cp_size
  tp_size: 1
  cp_size: 1
  sequence_parallel: false

# ── RNG ──
rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 1111
  ranked: true

# ── Loss ──
loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

# ── Dataloaders ──
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  batch_size: 8
  shuffle: false

validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  batch_size: 8

# ── Checkpointing ──
checkpoint:
  enabled: true
  checkpoint_dir: checkpoints/
  model_save_format: safetensors
  save_consolidated: True  # single HF-compatible bundle (requires safetensors format)

# ── Optimizer ──
optimizer:
  _target_: torch.optim.Adam
  betas: [0.9, 0.999]
  eps: 1e-8
  lr: 1.0e-5
  weight_decay: 0

# ── Logging (optional) ──
# wandb:
#   project: <your_wandb_project>
#   entity: <your_wandb_entity>
#   name: <your_wandb_exp_name>
#   save_dir: <your_wandb_save_dir>
```
:::

### Config Field Reference

| Section | Required? | What to change |
|---------|-----------|----------------|
| `model` | Yes | Set `pretrained_model_name_or_path` to your Hugging Face model ID. |
| `peft` | PEFT only | Remove entirely for SFT. Adjust `dim` and `alpha` to tune adapter capacity. `use_triton: True` enables an optimized LoRA kernel (requires the `triton` package). For reduced memory usage, see [QLoRA](#qlora-quantized-low-rank-adaptation) below. |
| `dataset` | Yes | Change `_target_`, `dataset_name`, and `split` for your data. |
| `step_scheduler` | Yes | `grad_acc_steps` sets how many micro-batches accumulate per gradient step. `ckpt_every_steps` and `val_every_steps` are counted in gradient steps. |
| `distributed` | Yes | `dp_size: null` means auto-detect from world size. Adjust `tp_size` for tensor parallelism across GPUs. |
| `checkpoint` | Recommended | Set `checkpoint_dir` to a persistent path, especially in Docker. |
| `optimizer` | Optional | Defaults are reasonable. Any `torch.optim` class can be substituted via `_target_`. |
| `wandb` | Optional | Uncomment and configure to enable Weights & Biases logging. |

## Advanced Topics

### QLoRA (Quantized Low-Rank Adaptation)

If GPU memory is a constraint, [QLoRA](https://arxiv.org/abs/2305.14314) combines LoRA with 4-bit NormalFloat (NF4) quantization to reduce memory usage by up to 75% compared to full-precision fine-tuning, while maintaining comparable quality to standard LoRA.

To enable QLoRA, add a `quantization:` section alongside the `peft:` section in your config. Note two differences from the standard PEFT config above: `target_modules` uses the broader `"*_proj"` pattern to apply LoRA to all projection layers (wider coverage compensates for precision loss from 4-bit weights), and `dim` is increased from 8 to 16 for additional adapter capacity.

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules: "*_proj"  # broader glob than "*.proj" to cover all projection layers
  dim: 16                   # LoRA rank (higher than default to offset quantization)
  alpha: 32                # scaling factor
  dropout: 0.1             # LoRA dropout rate

quantization:
  load_in_4bit: True                   # enable 4-bit quantization
  load_in_8bit: False                  # use 4-bit, not 8-bit
  bnb_4bit_compute_dtype: bfloat16     # compute dtype
  bnb_4bit_use_double_quant: True      # double quantization for extra savings
  bnb_4bit_quant_type: nf4             # NormalFloat quantization type
  bnb_4bit_quant_storage: bfloat16     # storage dtype for quantized weights
```

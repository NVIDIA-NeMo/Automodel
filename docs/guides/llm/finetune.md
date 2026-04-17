# Supervised Fine-Tuning (SFT) and Parameter-Efficient Fine-Tuning (PEFT)

## Introduction

Pretrained language models are general-purpose: they know a lot about language but nothing about your particular domain, terminology, or task. Fine-tuning bridges that gap — you fine-tune the model on your own examples so it produces answers that are accurate and relevant for your use case, without the cost of training a model from scratch. The result is a model optimized for your data that you can evaluate, publish, and deploy. This guide walks you through that process end-to-end with NeMo AutoModel — from installation through training, evaluation, and deployment — using [Meta LLaMA 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B) and the [SQuAD v1.1](https://huggingface.co/datasets/rajpurkar/squad) dataset as a running example.

NeMo AutoModel supports two fine-tuning modes:

- **Supervised Fine-Tuning (SFT)** updates all model parameters. Use SFT when you need maximum accuracy and have sufficient compute.
- **Parameter-Efficient Fine-Tuning (PEFT)** using [LoRA](https://arxiv.org/abs/2106.09685) freezes the base model and trains small low-rank adapters. PEFT reduces trainable parameters to less than 1% of the original model, lowering memory and storage costs.

### Workflow Overview

```text
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 1. Install   │--->│ 2. Configure │--->│  3. Train    │--->│ 4. Inference │--->│ 5. Evaluate  │--->│ 6. Publish   │--->│  7. Deploy   │
│              │    │              │    │              │    │              │    │              │    │  (optional)  │    │  (optional)  │
│ pip install  │    │ YAML config  │    │ automodel CLI│    │ HF generate  │    │ Val loss +   │    │ HF Hub       │    │ vLLM serving │
│ or Docker    │    │ Choose SFT   │    │ or torchrun  │    │ API          │    │ lm-eval-     │    │ upload       │    │              │
│              │    │ or PEFT      │    │              │    │              │    │ harness      │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

| Step | Section | SFT | PEFT |
|------|---------|-----|------|
| **1. Install** | [Install NeMo AutoModel](#install-nemo-automodel) | Same | Same |
| **2. Configure** | [Configure Your Training Recipe](#configure-your-training-recipe) | YAML without `peft:` section | YAML with `peft:` section |
| **3. Train** | [Fine-Tune the Model](#fine-tune-the-model) | Same command for both modes | Same command for both modes |
| **4. Inference** | [Run Inference](#run-inference) | Load consolidated checkpoint directly | Load base model + adapter |
| **5. Evaluate** | [Evaluate the Fine-Tuned Model](#evaluate-the-fine-tuned-model) | Validation loss during training; lm-eval-harness post-training | Same |
| **6. Publish** | [Publish to HF Hub](#publish-to-the-hugging-face-hub) | Upload `model/consolidated/` | Upload `model/` (adapter only) |
| **7. Deploy** | [Deploy with vLLM](#deploy-with-vllm) | `vllm.LLM(model=...)` | `vLLMHFExporter` with `--lora-model` |

## Install NeMo AutoModel

```bash
pip3 install nemo-automodel
```

Alternatively, if you run into dependency or driver issues, use the pre-built Docker container:

```bash
docker pull nvcr.io/nvidia/nemo-automodel:26.02.00
docker run --gpus all -it --rm --shm-size=8g -v $(pwd)/checkpoints:/tmp/checkpoints/ nvcr.io/nvidia/nemo-automodel:26.02.00
```

:::{important}
Docker containers are ephemeral — files written inside the container are lost when it stops. The `-v` flag in the `docker run` command above bind-mounts a local `checkpoints/` directory into the container so that saved checkpoints persist across runs. For more details, see [Saving Checkpoints When Using Docker](../checkpointing.md#saving-checkpoints-when-using-docker).
:::

For the full set of installation methods, see the [installation guide](../installation.md).

## Configure Your Training Recipe


Training is configured through a [YAML](https://en.wikipedia.org/wiki/YAML) config file with three required sections — **model**, **dataset**, and **step_scheduler** — plus an optional **peft** section. The sections below walk through each one. For the complete copy-pastable file, see [Full Config YAML](#full-config-yaml).

Under the hood, both SFT and PEFT are executed by a **recipe**: a self-contained Python class that wires together model loading, dataset preparation, training, checkpointing, and logging. The fine-tuning recipe is [`TrainFinetuneRecipeForNextTokenPrediction`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/llm/train_ft.py). The config file tells the recipe *what* to build; the recipe decides *how* to build it.

:::{dropdown} How the Config System Works

NeMo AutoModel configs use a convention borrowed from [Hydra](https://hydra.cc/): the special **`_target_`** key tells the framework *which* Python class or function to call, and **every other key** in the same YAML block is passed as a keyword argument to that call. For example:

```yaml
optimizer:
  _target_: torch.optim.Adam
  lr: 1.0e-5
  weight_decay: 0
```

is equivalent to writing this Python code:

```python
from torch.optim import Adam

optimizer = Adam(lr=1.0e-5, weight_decay=0)
```

The `_target_` value is a **dotted Python import path**: the same string you would use in an `import` statement. The framework resolves it at runtime by importing the module and looking up the attribute. This means you can point `_target_` at any class constructor or factory function, and the remaining keys become its arguments.

:::{tip}
To discover which parameters a section accepts, look up the Python signature of its `_target_`. For instance, `torch.optim.Adam` accepts `lr`, `betas`, `eps`, and `weight_decay` — those are the keys you can set in the YAML.
:::

**From YAML to running code.** Here is the path a config takes through the framework:

```text
finetune_config.yaml
        │
        ▼
  ┌──────────────┐     load_yaml_config() parses the file into
  │  ConfigNode  │◄─── a tree of ConfigNode objects, one per
  └──────┬───────┘     YAML section.
         │
         ▼
  ┌──────────────┐     The recipe's setup() method reads
  │   Recipe     │◄─── each section from the ConfigNode tree
  │   setup()    │     and passes it to the matching builder.
  └──────┬───────┘
         │
    ┌────┴─────────────────────────────────┐
    ▼            ▼            ▼            ▼
build_model  build_optimizer build_dataloader build_loss_fn ...
    │            │            │            │
    ▼            ▼            ▼            ▼
cfg.model     cfg.optimizer cfg.dataset   cfg.loss_fn
 .instantiate() .instantiate() .instantiate() .instantiate()
    │            │            │            │
    ▼            ▼            ▼            ▼
 Resolves      Resolves     Resolves     Resolves
 _target_,     _target_,    _target_,    _target_,
 calls it      calls it     calls it     calls it
 with kwargs   with kwargs  with kwargs  with kwargs
```

Each builder function calls **`.instantiate()`** on its config section. `.instantiate()` does two things:

1. **Resolves `_target_`** — imports the Python path and obtains the callable (class or function).
2. **Calls it** — passes every other key in the section as a keyword argument.

Nested `_target_` blocks (like `collate_fn` inside `dataloader`) are recursively instantiated the same way.

**The `recipe` key.** Every config file includes a top-level `recipe` key that tells the CLI *which recipe class* to run. You can write it as a **short name** or as a **fully-qualified Python path** — both resolve to the same class:

```yaml
# Short name (the CLI looks up the class automatically)
recipe: TrainFinetuneRecipeForNextTokenPrediction

# Fully-qualified path (used as-is)
recipe: nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction
```

The short name form is a convenience — the CLI scans all recipe modules under `nemo_automodel.recipes` and matches the bare class name. If you invoke the recipe script directly with `torchrun` instead of the `automodel` CLI, the `recipe` key is not required because the script itself *is* the recipe.

**Not every section uses `_target_`.** Some sections like `step_scheduler`, `distributed`, and `checkpoint` are plain key-value groups consumed directly by the recipe — they control training schedule, parallelism strategy, and checkpoint behavior without instantiating a Python object.
:::

### Model

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
```

| Key | Role |
|-----|------|
| `_target_` | Points to [`NeMoAutoModelForCausalLM.from_pretrained`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/_transformers/auto_model.py) — a factory method that downloads (or loads from cache) a pretrained Hugging Face model and wraps it with NeMo distributed-training support. |
| `pretrained_model_name_or_path` | A keyword argument to `from_pretrained`. Any argument that [`from_pretrained`](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained) accepts can be added here (e.g. `cache_dir`, `torch_dtype`). |

This guide uses **Meta Llama 3.2 1B** as a running example. Replace `pretrained_model_name_or_path` with any supported [Hugging Face model ID](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/model-coverage/llm.md).

:::{dropdown} About Llama 3.2 1B
Llama is a family of decoder-only transformer models developed by Meta. The 1B variant is a compact model suitable for research and edge deployment, featuring RoPE positional embeddings, grouped-query attention (GQA), and SwiGLU activations.
:::

:::{dropdown} Accessing gated models
Some Hugging Face models are **gated**. If the model page shows a "Request access" button:

1. Log in with your Hugging Face account and accept the license.
2. Ensure the token you use (from `huggingface-cli login` or `HF_TOKEN`) belongs to the approved account.

Pulling a gated model without an authorized token triggers a 403 error.
:::

### Dataset

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad  # HF-Hub ID used to pull the dataset
  split: train

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: validation
```

| Key | Role |
|-----|------|
| `_target_` | Points to [`make_squad_dataset`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/datasets/llm/squad.py) — a factory function that downloads the SQuAD dataset, tokenizes it, and returns a `torch.utils.data.Dataset`. To use a different dataset, change `_target_` to a different factory function (see [Integrate Your Own Text Dataset](dataset.md)). |
| `dataset_name`, `split` | Keyword arguments passed to `make_squad_dataset`. Each dataset factory defines its own parameters — check the function signature to see what's available. |

This guide uses **SQuAD v1.1** as a running example. Swap the dataset by changing `_target_` and the dataset arguments — see [Integrate Your Own Text Dataset](dataset.md) and [Dataset Overview](../dataset-overview.md).

:::{dropdown} About SQuAD v1.1
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

### PEFT (Optional)

```yaml
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules: "*.proj"  # glob pattern matching linear layer FQNs
  dim: 8                    # low-rank dimension of the adapters
  alpha: 32                 # scaling factor for learned weights
```

| Key | Role |
|-----|------|
| `_target_` | Points to [`PeftConfig`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/_peft/lora.py) — a dataclass that describes which layers to adapt and how. Unlike the model and dataset sections, this instantiation produces a *config object*, not the adapter itself. The recipe passes the resulting `PeftConfig` into `build_model`, which applies LoRA adapters to the model. |
| `target_modules` | A glob pattern matched against fully-qualified layer names (e.g. `"*.proj"` matches every layer whose name ends in `proj`). |
| `dim` | The low-rank dimension *r* — controls adapter capacity. Larger values learn more but use more memory. |
| `alpha` | Scaling factor applied to the adapter output (`alpha / dim`). Higher values give adapters more influence during training. |

Including a `peft:` section enables LoRA fine-tuning. Remove it entirely to run SFT instead — see [Switching Between SFT and PEFT](#switching-between-sft-and-peft).

#### QLoRA (Quantized Low-Rank Adaptation)

If GPU memory is a constraint, [QLoRA](https://arxiv.org/abs/2305.14314) combines LoRA with 4-bit NormalFloat (NF4) quantization to reduce memory usage by up to 75% compared to full-parameter SFT in 16-bit precision, while maintaining comparable quality to standard LoRA.

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

### Training Schedule

```yaml
step_scheduler:
  num_epochs: 1     # Will train over the dataset once.
```

Unlike the sections above, `step_scheduler` has **no `_target_`** — it is not instantiated into a Python object. Instead, the recipe reads its keys directly to control the training loop (how many epochs to run, when to checkpoint, when to validate). This is typical of sections that configure *behavior* rather than *components*.

All other settings (distributed strategy, optimizer, checkpointing, logging) use sensible defaults. See the [Full Configuration Reference](#full-configuration-reference) to customize them.

### Full Config YAML

:::{dropdown} finetune_config.yaml (click to expand)
Save as `finetune_config.yaml`. This config runs PEFT (LoRA). To run SFT instead, remove the `peft:` section. For production-ready examples, see the hosted configs: [Llama 3.2 1B SFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml) and [Llama 3.2 1B PEFT](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_finetune/llama3_2/llama3_2_1b_squad_peft.yaml).

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules: "*.proj"
  dim: 8
  alpha: 32

dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: train

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: validation

step_scheduler:
  num_epochs: 1
```
:::

## Fine-Tune the Model

You can run the recipe using the AutoModel CLI or directly with `torchrun` (advanced).

```bash
automodel --nproc-per-node=8 finetune_config.yaml
```

The `--nproc-per-node=8` flag specifies the number of GPUs per node. Adjust to your case (for a single GPU, omit the `--nproc-per-node` option).

### Invoke the Recipe Script Directly (advanced)

Alternatively, you can invoke the recipe [script](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/llm/train_ft.py) directly using [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html), as shown below.

``` bash
torchrun --nproc-per-node=8 nemo_automodel/recipes/llm/train_ft.py -c finetune_config.yaml
```

### Sample Output
Running the recipe using either the `automodel` app or by directly invoking the recipe script should produce
the following log:
```
$ automodel finetune_config.yaml
INFO:nemo_automodel.cli.app:Config: finetune_config.yaml
INFO:nemo_automodel.cli.app:Recipe: nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction
INFO:nemo_automodel.cli.app:Launching job interactively (local)
cfg-path: finetune_config.yaml
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

Checkpoints are saved in native Hugging Face format, so no conversion is required — they work directly with Transformers, PEFT, vLLM, lm-eval-harness, and other tools in the Hugging Face ecosystem. SFT and PEFT produce different checkpoint layouts. **SFT checkpoints** contain the full model weights at `model/consolidated/` — a single, self-contained Hugging Face model directory created by gathering distributed shards into one location — and can be loaded directly. **PEFT checkpoints** contain only the adapter weights (~MBs instead of GBs) — at inference time you must load the original base model and apply the adapter on top. This distinction affects every downstream step (inference, publishing, deployment).

:::{dropdown} Checkpoint directory structure
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

This section documents all available config fields for the fine-tuning recipe. For the quick-start config, see [Configure Your Training Recipe](#configure-your-training-recipe).

### Switching Between SFT and PEFT

The `peft:` section controls which mode runs:

| Mode | What to do in the YAML |
|------|----------------------|
| **PEFT (LoRA)** | Include the `peft:` section as shown below. |
| **SFT (full-parameter)** | Remove/comment the `peft:` section entirely. |

All other config sections remain the same for both modes.

### Full Configuration

:::{dropdown} Full Config
:open:
```yaml
# Recipe
# Selects which recipe class runs the training loop.
# Use a short name (auto-discovered) or a fully-qualified Python path:
#   recipe: nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction
recipe: TrainFinetuneRecipeForNextTokenPrediction

# Training Schedule
# Controls epoch count, batch sizes, and how often to checkpoint / validate.
# No _target_ — these are plain values read directly by the recipe.
step_scheduler:
  grad_acc_steps: 4       # number of micro-batches accumulated before each optimizer
                          # step. Effective batch = grad_acc_steps × batch_size.
  ckpt_every_steps: 10    # save a checkpoint every N gradient steps
  val_every_steps: 10     # run the validation loop every N gradient steps
  num_epochs: 1           # how many full passes over the training dataset

# Process Group
# Initializes the PyTorch distributed process group.
# No _target_ — consumed directly by the recipe.
# You normally would not need to tune this.
dist_env:
  backend: nccl           # communication backend: "nccl" (GPU, recommended) or "gloo" (CPU)
  timeout_minutes: 1      # timeout for collective operations; increase for large models
                          # that take longer to initialize

# Distributed Strategy
# Determines how model weights, data, and compute are split across GPUs.
# No _target_ — consumed directly by the recipe.
# See "Distributed Training: TP, PP, CP, and EP" in Advanced Topics for details.
distributed:
  strategy: fsdp2         # parallelism strategy: "fsdp2" (recommended), "megatron_fsdp",
                          # or "ddp". FSDP2 shards parameters and optimizer states across
                          # the data-parallel group.
  dp_size: null           # data-parallel group size. null = auto-detect from
                          # world_size ÷ (tp_size × cp_size × pp_size).
  tp_size: 1              # tensor-parallel size: splits weight matrices across GPUs.
                          # Set to 2, 4, or 8 if the model doesn't fit on one GPU.
                          # Should divide evenly into the number of attention heads.
  cp_size: 1              # context-parallel size: splits the input sequence across GPUs.
                          # Increase for very long contexts (e.g. 32k+ tokens).
  sequence_parallel: false # when true, extends TP to also shard activations along
                          # the sequence dimension for additional memory savings

# Random Number Generator
# _target_ → StatefulRNG: a checkpointable RNG that ensures identical sequences
# across training restarts. Seed and ranked are kwargs to StatefulRNG().
rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 1111              # global random seed for reproducibility
  ranked: true            # when true, each GPU rank gets a unique RNG stream derived
                          # from the seed, so data shuffling differs per GPU

# Model
# _target_ → NeMoAutoModelForCausalLM.from_pretrained: downloads (or loads from
# cache) a pretrained HuggingFace model and wraps it with NeMo distributed-training
# support. Any from_pretrained kwarg is accepted (cache_dir, torch_dtype, etc.).
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

# PEFT (remove / comment this entire section for full-parameter SFT)
# _target_ → PeftConfig: a dataclass describing which layers get LoRA adapters.
# The recipe passes this config into build_model(), which attaches adapters
# to the matching layers.
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules: "*.proj" # glob pattern matched against fully-qualified layer names;
                           # "*.proj" matches every layer ending in "proj"
  dim: 8                   # low-rank dimension r — controls adapter capacity.
                           # Larger values are more expressive but use more memory.
  alpha: 32                # LoRA scaling factor: adapter output is scaled by alpha/dim.
                           # Higher values give adapters more influence during training.
  use_triton: True         # use an optimized Triton kernel for LoRA forward/backward
                           # (requires the triton package)

# Checkpointing
# No _target_ — plain key-value group consumed by the recipe.
checkpoint:
  enabled: true            # set to false to skip saving checkpoints entirely
  checkpoint_dir: checkpoints/  # output directory. Docker users: bind-mount this path
                                # (e.g. -v $(pwd)/checkpoints:/workspace/checkpoints)
                                # to persist checkpoints across container restarts.
  model_save_format: safetensors  # "safetensors" (recommended, faster and safer) or
                                  # "torch_save" (legacy pickle-based format)
  save_consolidated: True  # when true, writes a single HuggingFace-compatible checkpoint
                           # to model/consolidated/ that can be loaded directly by
                           # Transformers, vLLM, etc. Requires safetensors format.

# Training Dataset
# _target_ → make_squad_dataset: a factory function that downloads the SQuAD
# dataset, tokenizes it, and returns a torch Dataset. To use a different dataset,
# change _target_ to another factory function (see the dataset guide).
dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad  # HuggingFace Hub dataset ID
  split: train                   # which split to use (train, validation, test)

# Validation Dataset
validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  dataset_name: rajpurkar/squad
  split: validation
  limit_dataset_samples: 64  # cap validation set to 64 samples for faster eval loops;
                             # remove this line to use the full validation set

# Training Dataloader
# _target_ → StatefulDataLoader: a checkpointable DataLoader from torchdata that
# saves and restores iteration state across training restarts, so resumed runs
# don't re-process already-seen batches.
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
                               # function that pads and batches individual samples
                               # into tensors; can be swapped for custom collation
  batch_size: 8                # samples per micro-batch per GPU
  shuffle: true                # whether to shuffle the dataset each epoch

# Validation Dataloader
validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn: nemo_automodel.components.datasets.utils.default_collater
  batch_size: 8

# Loss Function
# _target_ → MaskedCrossEntropy: standard cross-entropy loss that automatically
# ignores padding tokens so they don't affect the gradient.
# Other available loss functions (swap _target_ to use):
#   - nemo_automodel.components.loss.chunked_ce.ChunkedCrossEntropy
#       Computes CE in chunks along the sequence dimension to reduce peak memory.
#       Useful for very long sequences. Accepts chunk_len (default 32).
#   - nemo_automodel.components.loss.linear_ce.FusedLinearCrossEntropy
#       Fuses the final linear projection (lm_head) with the CE computation,
#       avoiding the full logit tensor. Significant **memory savings** for large vocabs.
#   - nemo_automodel.components.loss.te_parallel_ce.TEParallelCrossEntropy
#       TE-based parallel CE with a Triton kernel. Designed for tensor-parallel
#       setups where logits are sharded across TP ranks.
loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

# Optimizer
# _target_ → torch.optim.Adam: any torch.optim class can be used here (e.g.
# AdamW, SGD). All remaining keys become kwargs to the constructor.
optimizer:
  _target_: torch.optim.Adam
  lr: 1.0e-5               # learning rate — the most important hyperparameter to tune
  betas: [0.9, 0.999]      # Adam momentum coefficients (β₁ for mean, β₂ for variance)
  eps: 1e-8                 # small constant added to the denominator for numerical stability
  weight_decay: 0           # L2 regularization strength (0 = no regularization)

# Logging (optional)
# Uncomment to enable Weights & Biases experiment tracking.
# wandb:
#   project: <your_wandb_project>    # W&B project name
#   entity: <your_wandb_entity>      # W&B team or username
#   name: <your_wandb_exp_name>      # display name for this run
#   save_dir: <your_wandb_save_dir>  # local directory for W&B artifacts
```
:::

### Config Field Reference

| Section | Required? | What to change |
|---------|-----------|----------------|
| `model` | Yes | Set `pretrained_model_name_or_path` to your Hugging Face model ID. Source: [`auto_model.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/_transformers/auto_model.py). |
| `peft` | PEFT only | Remove entirely for SFT. Adjust `dim` and `alpha` to tune adapter capacity. `use_triton: True` enables an optimized LoRA kernel (requires the `triton` package). For reduced memory usage, see [QLoRA](#qlora-quantized-low-rank-adaptation). Source: [`lora.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/_peft/lora.py). |
| `dataset` | Yes | Change `_target_`, `dataset_name`, and `split` for your data. Source: [`squad.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/datasets/llm/squad.py). |
| `dataloader` | Optional | Adjust `batch_size` and `shuffle`. Uses [`StatefulDataLoader`](https://meta-pytorch.org/data/main/torchdata.stateful_dataloader.html) for checkpointable iteration. Collation: [`utils.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/datasets/utils.py). |
| `loss_fn` | Optional | Default is [`MaskedCrossEntropy`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/loss/masked_ce.py). Alternatives: [`ChunkedCrossEntropy`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/loss/chunked_ce.py) (long sequences), [`FusedLinearCrossEntropy`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/loss/linear_ce.py) (large vocabs), [`TEParallelCrossEntropy`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/loss/te_parallel_ce.py) (tensor-parallel). |
| `rng` | Optional | Controls reproducibility. Source: [`rng.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/training/rng.py). |
| `step_scheduler` | Yes | `grad_acc_steps` sets how many micro-batches accumulate per gradient step. `ckpt_every_steps` and `val_every_steps` are counted in gradient steps. |
| `distributed` | Yes | `dp_size: null` means auto-detect from world size. Adjust `tp_size` for tensor parallelism across GPUs. |
| `checkpoint` | Recommended | Set `checkpoint_dir` to a persistent path, especially in Docker. |
| `optimizer` | Optional | Defaults are reasonable. Any `torch.optim` class can be substituted via `_target_`. |
| `wandb` | Optional | Uncomment and configure to enable Weights & Biases logging. |

For the fine-tuning recipe itself, see [`train_ft.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/llm/train_ft.py). For more example configs, browse [`examples/llm_finetune/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_finetune).

## Distributed Training: TP, PP, CP, and EP

The `distributed:` section controls how the model and data are split across GPUs. NeMo AutoModel supports four parallelism dimensions, each of which slices the workload differently:

| Dimension | Key | What it shards | When to use |
|-----------|-----|---------------|-------------|
| **Data Parallel (DP)** | `dp_size` | Replicates the model on each group of GPUs; each replica trains on a different data batch. | Default. Scales batch size linearly with GPU count. |
| **Tensor Parallel (TP)** | `tp_size` | Splits individual weight matrices (attention, MLP) across GPUs within a node. | Model is too large to fit on a single GPU, or you want to reduce per-GPU memory at the cost of extra communication. |
| **Pipeline Parallel (PP)** | `pp_size` | Assigns different *layers* (stages) to different GPUs and pipelines micro-batches through them. | Very deep models that don't fit even with TP, or multi-node training where TP's all-reduce is too expensive across nodes. |
| **Context Parallel (CP)** | `cp_size` | Splits the input *sequence* across GPUs so each GPU processes a portion of the context. | Very long sequences that exceed single-GPU memory. |
| **Expert Parallel (EP)** | `ep_size` | Distributes MoE experts across GPUs so each GPU holds a subset of experts. | Mixture-of-Experts models only. |

These dimensions compose with each other. The relationship between them and total GPU count is:

```text
world_size = pp_size × dp_size × cp_size × tp_size
```

When `dp_size` is set to `null` (the default), it is inferred automatically:

```text
dp_size = world_size ÷ (tp_size × cp_size × pp_size)
```

EP does not appear in this formula — experts are distributed across the DP×CP rank groups, with the constraint that `(dp_size × cp_size)` must be divisible by `ep_size`.

#### Data Parallel (default)

Data parallelism is the default. With `strategy: fsdp2`, FSDP2 shards both model parameters and optimizer states across the DP group, so memory usage shrinks as you add GPUs:

```yaml
distributed:
  strategy: fsdp2
  dp_size: null   # auto-detected from world_size ÷ (tp × cp × pp)
  tp_size: 1
  cp_size: 1
```

#### Tensor Parallelism

TP splits weight matrices across GPUs within a single node. Set `tp_size` to the number of GPUs you want to shard over (typically 2, 4, or 8 — should divide evenly into the number of attention heads):

```yaml
distributed:
  strategy: fsdp2
  dp_size: null
  tp_size: 4
  cp_size: 1
  sequence_parallel: false   # set to true for additional memory savings
```

`sequence_parallel: true` extends TP to also shard activation memory along the sequence dimension, further reducing per-GPU memory at the cost of additional communication.

#### Pipeline Parallelism

PP assigns groups of layers to different GPUs and streams micro-batches through the stages. It requires an additional nested `pipeline:` section:

```yaml
distributed:
  strategy: fsdp2
  dp_size: null
  tp_size: 4
  pp_size: 4
  cp_size: 1
  activation_checkpointing: true

  pipeline:
    pp_schedule: interleaved1f1b  # pipeline schedule (1f1b or interleaved1f1b)
    pp_microbatch_size: 1         # micro-batch size per pipeline step
    layers_per_stage: 4           # how many layers each stage handles
    scale_grads_in_schedule: false
```

| Key | Role |
|-----|------|
| `pp_schedule` | The micro-batch schedule. `1f1b` is simpler; `interleaved1f1b` overlaps compute and communication for better throughput. |
| `pp_microbatch_size` | Number of samples per micro-batch fed into the pipeline. Must satisfy: `local_batch_size ÷ pp_microbatch_size ≥ pp_size`. |
| `layers_per_stage` | How many transformer layers each pipeline stage contains. If omitted, the framework splits layers evenly across `pp_size` stages. |

:::{note}
PP requires the model to define a `_pp_plan` that tells the framework how to split layers into stages. All built-in models include this plan; custom models must add one.
:::

#### Context Parallelism

CP splits the sequence across GPUs — useful for very long contexts that exceed single-GPU memory. Set `cp_size` to the desired split factor:

```yaml
distributed:
  strategy: fsdp2
  dp_size: null
  tp_size: 1
  cp_size: 2
```

:::{important}
When `cp_size > 1`, fused RoPE is automatically disabled. Some models also require the Transformer Engine (TE) attention backend for CP with packed sequences — the framework will raise an error with instructions if this applies.
:::

#### Expert Parallelism (MoE models)

EP distributes MoE experts across GPUs. Set `ep_size` to the number of GPUs that share the full set of experts:

```yaml
distributed:
  strategy: fsdp2
  tp_size: 1
  cp_size: 1
  pp_size: 1
  ep_size: 8
  activation_checkpointing: true
```

EP only applies to Mixture-of-Experts models (e.g. Qwen3-MoE, Mixtral, DeepSeek-V3). For dense models, leave `ep_size` at `1` or omit it.

#### Combining Multiple Dimensions

You can combine TP, PP, CP, and EP in a single config. For example, a large MoE model on a multi-node cluster might use:

```yaml
distributed:
  strategy: fsdp2
  dp_size: null
  tp_size: 1
  cp_size: 2
  pp_size: 1
  ep_size: 4
  activation_checkpointing: true
```

When choosing a combination, keep these rules in mind:

- **`world_size` must divide evenly** into `pp_size × tp_size × cp_size` (the remainder becomes `dp_size`).
- **`(dp_size × cp_size) % ep_size == 0`** — EP shares the DP×CP groups.
- **TP within a node, PP across nodes** is the typical layout — TP requires fast NVLink bandwidth, while PP tolerates higher latency.
- **Start simple.** Use DP-only first. Add TP if the model doesn't fit on one GPU. Add PP for very large models. Add CP for long sequences. Add EP only for MoE architectures.

## Next Steps

- [Integrate Your Own Text Dataset](dataset.md) — swap the SQuAD example for your own data.
- [Recipes and End-to-End Examples](../overview.md) — browse the full set of recipes available in NeMo AutoModel. See also the [`examples/llm_finetune/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_finetune) directory for ready-to-run configs.
- [Dataset Overview](../dataset-overview.md) — see all supported dataset types across LLM, VLM, and retrieval tasks.
- [Knowledge Distillation](knowledge-distillation.md) — distill a fine-tuned model into a smaller one.

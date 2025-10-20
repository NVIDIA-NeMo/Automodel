# NanoGPT-style Pre-Training with NeMo AutoModel

This guide walks you through **data preparation** and **model training** for a [NanoGPT-like](https://github.com/KellerJordan/modded-nanogpt) run using the new `NanogptDataset` and pre-training recipe.

In particular, it will show you how to:
1. [Install NeMo AutoModel from git](#1-environment-setup).
2. [Pre-process and tokenize the FineWeb dataset](#2-pre-process-the-fineweb-dataset).
3. [Define your own model architecture](#3-define-your-own-model-architecture).
4. [Setup the YAML configuration](#4-inspect-and-adjust-the-yaml-configuration).
5. [Launch training](#5-launch-training).
6. [Monitor the training](#6-monitoring-and-evaluation).

---

## 1. Environment setup

In this guide we will use an interactive environment, to install NeMo AutoModel from git. You can always install NeMo AutoModel from pypi or use our bi-monthly docker container.

```bash
# clone / install AutoModel (editable for local hacks)
cd /path/to/workspace/ # specify to your path as needed.
git clone git@github.com:NVIDIA-NeMo/AutoModel.git
cd AutoModel/
pip install -e .[all]    # installs NeMo AutoModel + optional extras
```

:::note
For this guide we will use a single machine equipped with 8xH100 NVIDIA GPUs.
:::

:::tip
You can run this guide with a single GPU by changing the config.
:::

---

## 2. Pre-process the FineWeb dataset

### Quick intro to the FineWeb dataset
The 🍷 [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) dataset consists of more than 18.5T tokens (originally 15T tokens) of cleaned and deduplicated english web data from [CommonCrawl](https://commoncrawl.org/). The data processing pipeline is optimized for LLM performance and ran on the 🏭 datatrove library, our large scale data processing library.

Briefly, FineWeb is built by extracting main text from CommonCrawl WARC HTML, keeping English pages via fastText language scoring, applying multiple quality filters (e.g., Gopher repetition/quality checks, C4-style rules, and custom heuristics for list-like or repeated/poorly formatted lines), and then MinHash-deduplicating each crawl independently (5-gram shingling with 14×8 hash functions). Basic PII normalization is applied (e.g., anonymizing emails and public IPs). The result is released per-crawl (and convenient sampled subsets), ready for high-throughput streaming.

### Pre-processing and Tokenization

For the purposes of this guide, we provide a data preprocessing tool at [`nanogpt_data_processor.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/tools/nanogpt_data_processor.py) that streams datasets from the Hugging Face Hub, tokenizes with GPT-2 BPE (using the [`tiktoken`](https://github.com/openai/tiktoken) library), and writes the output in **memory-mapped binary shards** to files. During training, we use the [`NanogptDataset`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/datasets/llm/nanogpt_dataset.py) class that can stream efficiently at training time.


```bash
# Step into repo root
cd /path/to/workspace/AutoModel/

# Generate 500 million tokens using the 10B raw split
python tools/nanogpt_data_processor.py \
  --dataset HuggingFaceFW/fineweb \
  --set-name sample-10BT \
  --max-tokens 500M      # stop after 500 million tokens; specify as needed, reduce for smaller runs.

# Shards are stored in:  tools/fineweb_max_tokens_500M/
#    dataset.bin (single binary file with all tokens)
```

**How the preprocessor works:** The script streams data iteratively from the Hugging Face Hub (avoiding loading the entire dataset into memory), uses a multiprocessing pipeline with separate reader and writer processes, and parallelizes tokenization across multiple CPU cores using `ProcessPoolExecutor`. This design enables efficient processing of very large datasets while maintaining low memory overhead. By default, uses the `gpt2` tokenizer, but can support other tokenizers via `--tokenizer` option.

Consider the following options:
1. Drop the `--max-tokens` flag to stream the **entire** split (tens of billions of tokens).
2. Adjust `--chunk-size` for processing batch size.
3. Use `--num-workers` to control parallelization.
4. Specify `--output-dir` to change the output location.

---

## 3. Define your own model architecture

You can plug in your own model by pointing the YAML `model._target_` to any callable or class that returns an `nn.Module` (or a compatible Hugging Face model). The config loader resolves `_target_` in three ways:

- Import path to a Python object (e.g., `my_pkg.models.build_model`)
- Local Python file with object name (e.g., `/abs/path/to/my_model.py:build_model`)
- A library callable such as Hugging Face `transformers.AutoModelForCausalLM.from_config`

Below are examples for each pattern.

### 4.1 Import path to your code (function or class)

```yaml
# examples/llm_pretrain/nanogpt_pretrain.yaml
model:
  _target_: my_project.models.my_gpt.build_model
  vocab_size: 50258
  n_positions: 2048
  n_embd: 768
  n_layer: 12
  n_head: 12
```

Your Python implements either a class constructor or a builder function that returns the model:

```python
# my_project/models/my_gpt.py
import torch.nn as nn

class MyGPT(nn.Module):
    def __init__(self, vocab_size: int, n_positions: int, n_embd: int, n_layer: int, n_head: int):
        super().__init__()
        # initialize your modules here

    def forward(self, input_ids, attention_mask=None):
        # return logits
        ...

def build_model(vocab_size: int, n_positions: int, n_embd: int, n_layer: int, n_head: int) -> nn.Module:
    return MyGPT(vocab_size, n_positions, n_embd, n_layer, n_head)
```

### 4.2 Local Python file path

Point `_target_` to a concrete file on disk and the object inside it using the `path.py:object_name` form. Absolute paths are recommended.

```yaml
model:
  _target_: /absolute/path/to/my_model.py:build_model
  vocab_size: 50258
  n_positions: 2048
  n_embd: 768
  n_layer: 12
  n_head: 12
```

This loads `/absolute/path/to/my_model.py`, then calls its `build_model(...)` with the remaining YAML keys as keyword arguments.

### 4.3 Hugging Face models via `from_config`

You can instantiate any Hugging Face causal LM with a config-first flow by targeting a `from_config` callable and providing a nested `config` node. The nested node is itself resolved via `_target_`, so you can compose HF configs directly in YAML.

```yaml
model:
  _target_: transformers.AutoModelForCausalLM.from_config
  # Nested object: built first, then passed to from_config(config=...)
  config:
    _target_: transformers.AutoConfig.from_pretrained
    pretrained_model_name_or_path: gpt2   # or "Qwen/Qwen2-1.5B", etc.
    n_layer: 12
    n_head: 12
    n_positions: 2048
    vocab_size: 50258
```

Alternatively, target a specific architecture:

```yaml
model:
  _target_: transformers.GPT2LMHeadModel.from_config
  config:
    _target_: transformers.GPT2Config
    n_layer: 12
    n_head: 12
    n_positions: 2048
    vocab_size: 50258
```

Notes:
- The `model._target_` may reference an import path or a local Python file using the `path.py:object` form.
- Any nested mapping that includes `_target_` (e.g., `config:`) is instantiated first and its result is passed upward. This is how the Hugging Face `from_config` pattern works.
- You can keep using the same training recipe (optimizer, data, distributed settings); only the `model:` block changes.

---

## 4. Inspect and adjust the YAML configuration

`examples/llm_pretrain/nanogpt_pretrain.yaml` is a complete configuration that:
* Defines a GPT-2 model via the `build_gpt2_model` shorthand (easy to scale up).
* Points `file_pattern` at preprocessed binary data files (configure based on your preprocessing output).
* Uses the new `NanogptDataset` with `seq_len=1024`.
* Sets a vanilla `AdamW` optimizer with learning rate `2e-4`.
* Includes FSDP2 distributed training configuration.

Key configuration sections:

```yaml
# Model configuration (two options available)
model:
  _target_: nemo_AutoModel.components.models.gpt2.build_gpt2_model
  vocab_size: 50258
  n_positions: 2048
  n_embd: 768
  n_layer: 12
  n_head: 12

# Dataset configuration
dataset:
  _target_: nemo_AutoModel.components.datasets.llm.nanogpt_dataset.NanogptDataset
  file_pattern: "tools/fineweb_max_tokens_500M/dataset.bin"
  seq_len: 1024
  shuffle_files: true

# Distributed training
distributed:
  _target_: nemo_AutoModel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none
  tp_size: 1
  cp_size: 1
```

**About `_target_` configuration**: The `_target_` field specifies import paths to classes and functions within the nemo_AutoModel repository (or any Python module). For example, `nemo_AutoModel.components.models.gpt2.build_gpt2_model` imports and calls the GPT-2 model builder function. You can also specify paths to your own Python files (e.g., `my_custom_models.MyTransformer`) to use custom `nn.Module` implementations, allowing full flexibility in model architecture while leveraging the training infrastructure.

Update the `file_pattern` to match your data location. For example, if using `tools/nanogpt_data_processor.py` with the default settings: `"tools/fineweb_max_tokens_500M/dataset.bin"`

Scale **width/depth**, `batch_size`, or `seq_len` as needed - the recipe is model-agnostic.

---

## 5. Launch training

```bash
# Single-GPU run (good for local testing)
python examples/llm_pretrain/pretrain.py \
  --config examples/llm_pretrain/nanogpt_pretrain.yaml

# Multi-GPU (e.g. 8x H100)
torchrun --standalone --nproc-per-node 8 \
  examples/llm_pretrain/pretrain.py \
  --config examples/llm_pretrain/nanogpt_pretrain.yaml

# Using the AutoModel CLI:
# single-GPU
AutoModel pretrain llm -c examples/llm_pretrain/nanogpt_pretrain.yaml

# multi-GPU (AutoModel CLI + torchrun on 8 GPUs)
AutoModel --nproc-per-node 8 \
  $(which AutoModel) pretrain llm \
  -c examples/llm_pretrain/nanogpt_pretrain.yaml
```
:::tip
Adjust the `distributed` section in the YAML config to change between DDP, FSDP2, etc.
:::

The `TrainFinetuneRecipeForNextTokenPrediction` class handles:
* Distributed (FSDP2 / TP / CP) wrapping if requested in the YAML.
* Gradient accumulation, LR scheduling, checkpointing, optional W&B logging.
* Validation loops if you supply `validation_dataset`.

Checkpoints are written under `checkpoints/` by default as `safetensors` or `torch_save` (YAML-configurable).

---

## 6. Monitoring and evaluation

* **TPS** (tokens per second), **gradient norm** and **loss** statistics print every optimization step.
* Enable `wandb` in the YAML for dashboards (`wandb.project`, `wandb.entity`, etc.).
* Periodic checkpoints can be loaded via `TrainFinetuneRecipeForNextTokenPrediction.load_checkpoint()`.

Example W&B configuration:
```yaml
wandb:
  project: "nanogpt-pretraining"
  entity: "your-wandb-entity"
  name: "nanogpt-500M-tokens"
```

---

## 7. Further work

1. **Scaling up** - swap the GPT-2 config for `LlamaForCausalLM`, `Qwen2`, or any HF-compatible causal model; increase `n_layer`, `n_embd`, etc.
2. **Mixed precision** - FSDP2 + `bfloat16` (`dtype: bfloat16` in distributed config) for memory savings.
3. **Sequence packing** - set `packed_sequence.packed_sequence_size` > 0 to pack variable-length contexts and boost utilization.
4. **Custom datasets** - implement your own `IterableDataset` or convert existing corpora to the `.bin` format using `tools/nanogpt_data_processor.py` as a template.
5. **BOS alignment** - set `align_to_bos: true` in the dataset config to ensure sequences start with BOS tokens (requires `bos_token` parameter).

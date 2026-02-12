# LLM Pre-Training with NeMo AutoModel

This guide covers **FineWeb** data preparation, **defining** a [NanoGPT‑style](https://github.com/KellerJordan/modded-nanogpt) model, and **launching and monitoring** a NeMo AutoModel pre‑training run.

---

## Set Up Your Environment

In this guide, we will use an interactive environment to install NeMo AutoModel from Git. You can also install NeMo AutoModel from PyPI or use our bi-monthly Docker container (see the [Installation Guide](../installation.md)).

```bash
# clone / install AutoModel (editable for local hacks)
cd /path/to/workspace/ # specify to your path as needed.
git clone https://github.com/NVIDIA-NeMo/Automodel.git
cd Automodel/
pip install -e ".[all]"    # installs NeMo AutoModel + optional extras
```

:::note
For this guide we will use a single machine equipped with 8xH100 NVIDIA GPUs.
:::

:::tip
To run this guide on a single GPU, use the single-GPU command in the **Launch Training** section below and scale down the YAML (for example, reduce `step_scheduler.global_batch_size` / `local_batch_size`, and shrink the model via `model.n_layer` / `model.n_embd` / `model.n_head`). For more launch patterns, see [Run on Your Local Workstation](../../launcher/local-workstation.md).
:::

---

## Pre-Process the FineWeb Dataset

::::{warning}
**File Size Limitation**: The `nanogpt_data_processor.py` script has a **4GB file size limit** (~2^32 bytes) due to 32-bit position tracking in the BOS index. This translates to:
- **~2 billion tokens** when using uint16 (vocabularies < 65,536 tokens, e.g., GPT-2)
- **~1 billion tokens** when using uint32 (larger vocabularies)

Always use the `--max-tokens` flag to stay within these limits (e.g., `--max-tokens 2B` or `--max-tokens 1.5B`).

For larger datasets, please see [pretraining.md](pretraining.md) which supports sharded preprocessing without these constraints.
::::

### Quick Intro to the FineWeb Dataset
[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) is 18.5 T tokens of cleaned, deduplicated English web data from CommonCrawl. For this guide we use the **`sample-10BT` subset** and extract ~500 M tokens that fit within the preprocessing tool's limits.

:::tip
To train on more than 2B tokens from FineWeb, see [pretraining.md](pretraining.md) which uses Megatron Core's sharded dataset format without file size constraints.
:::

### Pre-Processing and Tokenization

For the purposes of this guide, we provide a data preprocessing tool at [`nanogpt_data_processor.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/tools/nanogpt_data_processor.py) that streams datasets from the Hugging Face Hub, tokenizes using Hugging Face's `transformers.AutoTokenizer` (default: GPT-2), and writes the output in **memory-mapped binary shards** to files. During training, we use the [`NanogptDataset`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/datasets/llm/nanogpt_dataset.py) class that can stream efficiently at training time.


```bash
# Step into repo root
cd /path/to/workspace/Automodel/

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
1. Adjust `--max-tokens` to control how many tokens to process (must stay within the 4GB file size limit mentioned above).
2. Adjust `--chunk-size` for processing batch size.
3. Use `--num-workers` to control parallelization.
4. Specify `--output-dir` to change the output location.

---

## Understand the NeMo AutoModel Training Workflow

NeMo AutoModel follows a simple but powerful flow for training:

1. A Python recipe script (for example, [`examples/llm_pretrain/pretrain.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_pretrain/pretrain.py)) serves as the entry point that wires up all training components based on a YAML configuration file. Any configuration option can be overridden via CLI arguments (e.g., `--model.name abc`).
2. The YAML file describes each component of the training job (such as `model`, `dataset`, `optimizer`, `distributed`, `checkpoint`, and optional `wandb`).
3. Each component is constructed from its `_target_`, which points to a Python callable (function or class constructor) to instantiate. The remaining keys in that YAML block become keyword arguments for that callable.

How `_target_` is resolved:
- Import path to a Python object (for example, `my_pkg.models.build_model`).
- Local Python file path plus object name (for example, `/abs/path/to/my_model.py:build_model`).
- Library callables such as Hugging Face `transformers.AutoModelForCausalLM.from_config`.

Nested objects can also specify their own `_target_` (common when building Hugging Face `config` objects first and passing them into a `from_config` method). Any YAML key can be overridden at launch time from the CLI, making it easy to tweak hyperparameters without editing files.

With this context, let's define a model via `_target_`, then point the dataset at your preprocessed shards, and finally review the full YAML.

## Define Your Own Model Architecture

NeMo AutoModel relies on a YAML-driven configuration to build every training component. In particular, the `model._target_` must reference a callable that returns an `nn.Module` (or a compatible Hugging Face model). You can point `_target_` at:

- An import path to a Python object.
- A local Python file plus the object name using `path.py:object_name`.
- A library callable such as `transformers.AutoModelForCausalLM.from_config`.

Below are examples for each pattern.

### NanoGPT Source and File-Path `_target_`

We provide a minimal, pure-PyTorch GPT-2 implementation at [`nemo_automodel/components/models/gpt2.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/models/gpt2.py). It defines:
- `CausalSelfAttention` -- multi-head self-attention with causal mask (uses `F.scaled_dot_product_attention`)
- `MLP` -- feed-forward with GELU activation
- `TransformerBlock` -- LN -> Attn -> Add -> LN -> MLP -> Add
- `GPT2LMHeadModel` -- full model with tied embeddings, stacked blocks, and GPT-2-style init
- `build_gpt2_model(...)` -- factory function that returns a configured `nn.Module`

The model is intentionally lean (no KV-cache or generation helpers) -- designed for forward/backward passes and next-token prediction.

To use it from a file path, point `_target_` to the file and object name (`path.py:object`):

```yaml
model:
  _target_: /abs/path/to/repo/nemo_automodel/components/models/gpt2.py:build_gpt2_model
  vocab_size: 50258
  n_positions: 2048
  n_embd: 768
  n_layer: 12
  n_head: 12
```

This loads the file on disk and calls `build_gpt2_model(...)` with the remaining keys as keyword arguments.

### Import Path to a Callable (Function or Class)

Instead of a file path, you can reference the callable via its import path:

```yaml
# examples/llm_pretrain/nanogpt_pretrain.yaml
model:
  _target_: nemo_automodel.components.models.gpt2.build_gpt2_model
  vocab_size: 50258
  n_positions: 2048
  n_embd: 768
  n_layer: 12
  n_head: 12
```

### Hugging Face Models via `from_config` Function

You can instantiate any Hugging Face causal LM with a config-first flow by targeting a `from_config` callable and providing a nested `config` node. The nested node is itself resolved via `_target_`, so you can compose Hugging Face configs directly in YAML.

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

## Inspect and Adjust the YAML Configuration

[`examples/llm_pretrain/nanogpt_pretrain.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_pretrain/nanogpt_pretrain.yaml) is a complete configuration that:
* Defines a GPT-2 model via the `build_gpt2_model` shorthand (easy to scale up).
* Points `file_pattern` at preprocessed binary data files (configure based on your preprocessing output).
* Uses the new `NanogptDataset` with `seq_len=1024`.
* Sets a vanilla `AdamW` optimizer with learning rate `2e-4`.
* Includes FSDP2 distributed training configuration.

Key configuration sections:

```yaml
# Model configuration (two options available)
model:
  _target_: nemo_automodel.components.models.gpt2.build_gpt2_model
  vocab_size: 50258
  n_positions: 2048
  n_embd: 768
  n_layer: 12
  n_head: 12

# Dataset configuration
dataset:
  _target_: nemo_automodel.components.datasets.llm.nanogpt_dataset.NanogptDataset
  file_pattern: "tools/fineweb_max_tokens_500M/dataset.bin"
  seq_len: 1024
  shuffle_files: true

# Distributed training
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: null
  tp_size: 1
  cp_size: 1
```

**About `_target_` configuration**: The `_target_` field specifies import paths to classes and functions within the nemo_automodel package (or any Python module). For example, `nemo_automodel.components.models.gpt2.build_gpt2_model` imports and calls the GPT-2 model builder function. You can also specify paths to your own Python files (e.g., `my_custom_models.MyTransformer`) to use custom `nn.Module` implementations, allowing full flexibility in model architecture while leveraging the training infrastructure.

Update the `file_pattern` to match your data location. For example, if using `tools/nanogpt_data_processor.py` with the default settings: `"tools/fineweb_max_tokens_500M/dataset.bin"`

Scale **width/depth**, `batch_size`, or `seq_len` as needed - the recipe is model-agnostic.

---

## Launch Training

```bash
# Single-GPU run (good for local testing)
python examples/llm_pretrain/pretrain.py \
  --config examples/llm_pretrain/nanogpt_pretrain.yaml

# Multi-GPU (e.g., 8x H100)
torchrun --standalone --nproc-per-node 8 \
  examples/llm_pretrain/pretrain.py \
  --config examples/llm_pretrain/nanogpt_pretrain.yaml

# Using the automodel CLI:
# single-GPU
automodel pretrain llm -c examples/llm_pretrain/nanogpt_pretrain.yaml

# multi-GPU (automodel CLI + torchrun on 8 GPUs)
automodel --nproc-per-node 8 pretrain llm \
  -c examples/llm_pretrain/nanogpt_pretrain.yaml
```
:::{tip}
Adjust the `distributed` section in the YAML config to change between DDP, FSDP2, etc.
:::

The `TrainFinetuneRecipeForNextTokenPrediction` class handles:
* Distributed (FSDP2 / TP / CP) wrapping if requested in the YAML.
* Gradient accumulation, LR scheduling, checkpointing, optional W&B logging.
* Validation loops if you supply `validation_dataset`.

Checkpoints are written under `checkpoints/` by default as `safetensors` or `torch_save` (YAML-configurable).

---

## Monitor and Evaluate Training

* **TPS** (tokens per second), **gradient norm**, and **loss** statistics print every optimization step.
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

## Explore Further Work

1. **Scaling up**: Swap the GPT-2 config for `LlamaForCausalLM`, `Qwen2`, or any Hugging Face-compatible causal model; increase `n_layer`, `n_embd`, etc.
2. **Mixed precision** - FSDP2 + `bfloat16` (`dtype: bfloat16` in distributed config) for memory savings.
3. **Sequence packing** - set `packed_sequence.packed_sequence_size` > 0 to pack variable-length contexts and boost utilization.
4. **Custom datasets** - implement your own `IterableDataset` or convert existing corpora to the `.bin` format using `tools/nanogpt_data_processor.py` as a template.
5. **BOS alignment** - set `align_to_bos: true` in the dataset config to ensure sequences start with BOS tokens (requires `bos_token` parameter).

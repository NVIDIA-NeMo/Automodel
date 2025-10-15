# NanoGPT-style Pre-Training with NeMo Automodel

This guide walks you through **data preparation** and **model training** for a [NanoGPT-like](https://github.com/KellerJordan/modded-nanogpt) run using the new `NanogptDataset` and pre-training recipe.

---

## 1. Environment setup

In this guide we will use an interactive environment, to install NeMo Automodel from git. You can always install NeMo Automodel from pypi or use our bi-monthly docker container.

```bash
# clone / install Automodel (editable for local hacks)
cd /path/to/workspace/ # specify to your path as needed.
git clone git@github.com:NVIDIA-NeMo/Automodel.git
cd Automodel/
pip install -e .[all]    # installs NeMo Automodel + optional extras
```

:::note
For this guide we will use a single machine equipped with 8xH100 NVIDIA GPUs.
:::

:::tip
You can run this guide with a single GPU by changing the config.
:::

---

## 2. Pre-process the FineWeb dataset

We provide a robust data preprocessing tool at `tools/nanogpt_data_processor.py` that streams datasets from the Hugging Face Hub, tokenizes with GPT-2 BPE (`tiktoken`), and writes **memory-mapped binary shards** that `NanogptDataset` can stream efficiently at training time.

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
1. Drop the `--max-tokens` flag to stream the **entire** split (tens of billions of tokens).
2. Adjust `--chunk-size` for processing batch size.
3. Use `--num-workers` to control parallelization.
4. Specify `--output-dir` to change the output location.

---

## 3. Inspect and adjust the YAML configuration

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
  dp_size: none
  tp_size: 1
  cp_size: 1
```

**About `_target_` configuration**: The `_target_` field specifies import paths to classes and functions within the nemo_automodel repository (or any Python module). For example, `nemo_automodel.components.models.gpt2.build_gpt2_model` imports and calls the GPT-2 model builder function. You can also specify paths to your own Python files (e.g., `my_custom_models.MyTransformer`) to use custom `nn.Module` implementations, allowing full flexibility in model architecture while leveraging the training infrastructure.

Update the `file_pattern` to match your data location. For example, if using `tools/nanogpt_data_processor.py` with the default settings: `"tools/fineweb_max_tokens_500M/dataset.bin"`

Scale **width/depth**, `batch_size`, or `seq_len` as needed - the recipe is model-agnostic.

---

## 4. Launch training

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
automodel pretrain llm -c examples/llm_pretrain/nanogpt_pretrain.yaml

# multi-GPU (AutoModel CLI + torchrun on 8 GPUs)
automodel --nproc-per-node 8 \
  $(which automodel) pretrain llm \
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

## 5. Monitoring and evaluation

* **Throughput** and **loss** statistics print every optimization step.
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

## 6. Further work

1. **Scaling up** - swap the GPT-2 config for `LlamaForCausalLM`, `Qwen2`, or any HF-compatible causal model; increase `n_layer`, `n_embd`, etc.
2. **Mixed precision** - FSDP2 + `bfloat16` (`dtype: bfloat16` in distributed config) for memory savings.
3. **Sequence packing** - set `packed_sequence.packed_sequence_size` > 0 to pack variable-length contexts and boost utilization.
4. **Custom datasets** - implement your own `IterableDataset` or convert existing corpora to the `.bin` format using `tools/nanogpt_data_processor.py` as a template.
5. **BOS alignment** - set `align_to_bos: true` in the dataset config to ensure sequences start with BOS tokens (requires `bos_token` parameter).

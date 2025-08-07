# NanoGPT-style Pre-Training with NeMo Automodel

This short guide walks you through **data preparation** and **model training** for a NanoGPT-like run using the new `BinTokenDataset` and pre-training recipe.

---

## 1. Environment setup

```bash
# clone / install Automodel (editable for local hacks)
cd /path/to/Automodel
pip install -e .[all]    # installs NeMo Automodel + optional extras
```

:::note
For this guide we will use a single machine equiped with 8xH100 NVIDIA GPUs.
:::

:::tip
You can run this guide with a single GPU, by changing the config.
:::

---

## 2. Pre-process the FineWeb dataset

We provide a thin wrapper around NanoGPT’s original `fineweb.py`.  The script streams the dataset from the Hugging Face Hub, tokenises with GPT-2 BPE (`tiktoken`) and writes **memory-mapped binary shards** that `BinTokenDataset` can stream at training time.

```bash
# Step into repo root
cd /path/to/Automodel

# Generate 500 million tokens using the 10B raw split
python tools/data_preprocessing/fineweb.py \
  --split sample-10BT \
  --max-tokens 500M      # stop after 500 million tokens (≈ 2 GPU-hours)

# Shards are stored in:  tools/data_preprocessing/fineweb_10BT_max_500M/
#    fineweb_train_000001.bin  fineweb_val_000000.bin  etc.
```

Consider the following options:
1. Drop the `--max-tokens` flag to stream the **entire** split (tens of billions of tokens).
2. Adjust `--shard_size` for smaller or larger `.bin` files.

---

## 3. Inspect and adjust the YAML configuration

`examples/llm/nanogpt_pretrain.yaml` is a **minimal** configuration that:
* Defines a small GPT-2 12-layer model via `transformers.GPT2Config` (easy to scale up).
* Points `file_pattern` at `tools/data_preprocessing/fineweb_10BT_max_500M/fineweb_train_*.bin` - update this path if you used a different output directory.
* Uses the new `BinTokenDataset` with `seq_len=1024`.
* Sets a vanilla `AdamW` optimiser with a learning rate of `2e-4`.

Scale **width/depth**, `batch_size`, or `seq_len` as needed - the recipe is model-agnostic.

---

## 4. Launch training

```bash
# Single-GPU run (good for local testing)
python examples/llm/pretrain.py \
  --config examples/llm/nanogpt_pretrain.yaml

# Multi-GPU (e.g. 8x A100)
torchrun --standalone --nproc-per-node 8 \
  examples/llm/pretrain.py \
  --config examples/llm/nanogpt_pretrain.yaml

# Or using the **AutoModel CLI** (wraps the same logic under the hood):

```bash
# single-GPU
automodel pretrain llm -c examples/llm/nanogpt_pretrain.yaml

# multi-GPU (AutoModel CLI + torchrun on 8 GPUs)
torchrun --standalone --nproc-per-node 8 \
  $(which automodel) pretrain llm \
  -c examples/llm/nanogpt_pretrain.yaml
```

The `PretrainRecipeForNextTokenPrediction` class handles:
* Distributed (FSDP / TP / CP) wrapping if requested in the YAML.
* Gradient accumulation, LR scheduling, checkpointing, optional W&B logging.
* Validation loops if you supply `validation_dataset`.

Checkpoints are written under `checkpoints/` by default as `safetensors` or `torch_save` (YAML-configurable).

---

## 5. Monitoring and evaluation

* **Throughput** and **loss** statistics print every optimisation step.
* Enable `wandb` in the YAML for nice dashboards (`wandb.project`, `entity`, etc.).
* Periodic checkpoints can be loaded via `FinetuneRecipeForNextTokenPrediction.load_checkpoint()` (same interface for pre-train).

---

## 6. Further work

1. **Scaling up** - swap `transformers.GPT2Config` for a `LlamaForCausalLM`, `Qwen2` or any HF-compatible causal model; increase `n_layer`, `n_embd`, etc.
2. **Mixed precision** - FSDP 2 + `bfloat16` (`dtype: bfloat16` in distributed config) gives great memory savings.
3. **Sequence packing** - set `packed_sequence.packed_sequence_size` > 0 to pack variable-length contexts and boost utilisation.
4. **Custom datasets** - implement your own `IterableDataset` or convert existing corpora to the `.bin` format using the preprocessing script as a template.

Happy training. 
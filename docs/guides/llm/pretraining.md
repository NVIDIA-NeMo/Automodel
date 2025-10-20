# LLM Pre-Training with NeMo AutoModel

This guide covers **FineWeb** data preparation, **defining** a [NanoGPT‑style](https://github.com/KellerJordan/modded-nanogpt) model, and **launching and monitoring** a NeMo AutoModel pre‑training run.

In particular, it will show you how to:
1. [Install NeMo AutoModel from git](#1-environment-setup).
2. [Pre-process and tokenize the FineWeb dataset](#2-pre-process-the-fineweb-dataset).
3. [Introduction to the NeMo AutoModel training workflow](#3-introduction-to-the-nemo-automodel-training-workflow).
4. [Define your own model architecture](#4-define-your-own-model-architecture).
5. [Inspect and adjust the YAML configuration](#5-inspect-and-adjust-the-yaml-configuration).
6. [Launch training](#6-launch-training).
7. [Monitoring and evaluation](#7-monitoring-and-evaluation).

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

## 3. Introduction to the NeMo AutoModel training workflow

NeMo AutoModel follows a simple but powerful flow for training:

1. A Python recipe script (for example, [`examples/llm_pretrain/pretrain.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/llm_pretrain/pretrain.py)) is the entry point. It contains the trainign recipe. For configuration, it reads a YAML file and options can be overriden via CLI arguments (e.g., `--model.name abc`). The recipe wires up all training components.
2. The YAML file describes each component of the training job (such as `model`, `dataset`, `optimizer`, `distributed`, `checkpoint`, and optional `wandb`).
3. Each component is constructed from its `_target_`, which points to a Python callable (function or class constructor) to instantiate. The remaining keys in that YAML block become keyword arguments for that callable.

How `_target_` is resolved:
- Import path to a Python object (for example, `my_pkg.models.build_model`).
- Local Python file path plus object name (for example, `/abs/path/to/my_model.py:build_model`).
- Library callables such as Hugging Face `transformers.AutoModelForCausalLM.from_config`.

Nested objects can also specify their own `_target_` (common when building Hugging Face `config` objects first and passing them into a `from_config` method). Any YAML key can be overridden at launch time from the CLI, making it easy to tweak hyperparameters without editing files.

With this context, let’s define a model via `_target_`, then point the dataset at your preprocessed shards, and finally review the full YAML.

## 4. Define your own model architecture

NeMo AutoModel relies on a YAML-driven configuration to build every training component. In particular, the `model._target_` must reference a callable that returns an `nn.Module` (or a compatible Hugging Face model). You can point `_target_` at:

- An import path to a Python object.
- A local Python file plus the object name using `path.py:object_name`.
- A library callable such as `transformers.AutoModelForCausalLM.from_config`.

Below are examples for each pattern.

### 4.1 NanoGPT source and file-path `_target_`

Below is the minimal GPT‑2 [implementation](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/models/gpt2.py) used for this NanoGPT‑style pretraining flow.
It is a pure‑PyTorch model with tied embeddings and standard transformer blocks:

```
"""
Self-contained GPT-2 (Causal LM) implementation.

This module defines a pure-PyTorch model and defines the necessary
building blocks (attention, MLP, transformer block, and language-model head).
The public *build_gpt2_model* helper returns an ``nn.Module``.
"""
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# The attention layer
class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask."""

    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.0):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = attn_dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C)
        bsz, seq_len, _ = x.shape

        # Project to QKV and reshape: (B, T, 3*C) → (B, n_head, T, head_dim)
        qkv = self.qkv_proj(x).view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # (B, n_head, T, head_dim)

        # Use torch's optimized SDPA when available (PyTorch ≥2.0)
        if hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_dropout, is_causal=True
            )  # (B, n_head, T, head_dim)
        else:
            # Fallback implementation with an explicit causal mask
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal_mask, float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)
            attn_output = attn_weights @ v  # (B, n_head, T, head_dim)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        return self.out_proj(attn_output)

# The MLP
class MLP(nn.Module):
    """GPT-2 feed-forward network (GEGLU → Linear)."""

    def __init__(self, embed_dim: int, expansion_factor: int = 4):
        super().__init__()
        hidden_dim = expansion_factor * embed_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C)
        return self.fc2(self.act(self.fc1(x)))

# Transformers
class TransformerBlock(nn.Module):
    """A single transformer block (LN → Attn → Add → LN → MLP → Add)."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# The GPT2 model definition
class GPT2LMHeadModel(nn.Module):
    """Minimal GPT-2 Causal-LM with tied input/output embeddings."""

    def __init__(
        self,
        *,
        vocab_size: int,
        n_positions: int,
        n_embd: int,
        n_layer: int,
        n_head: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_positions, n_embd)
        self.drop = nn.Dropout(dropout)

        self.h = nn.ModuleList([TransformerBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

        # Language model head (weights tied to token embedding matrix)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # weight tying

        # Initialize parameters following GPT-2 scheme
        self._init_weights()

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:  # (B, T) → (B, T, V)
        batch_size, seq_len = input_ids.shape

        if seq_len > self.wpe.num_embeddings:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum context size {self.wpe.num_embeddings}.")

        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.wte(input_ids) + self.wpe(pos_ids)
        x = self.drop(x)

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def _init_weights(self):
        """Parameter initialization following GPT-2 conventions."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                # GPT-2 uses normal(0, 0.02)
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

# Helper entrypoint
def build_gpt2_model(
    *,
    vocab_size: int = 50257,
    n_positions: int = 2048,
    n_ctx: int | None = None,
    n_embd: int = 768,
    n_layer: int = 12,
    n_head: int = 12,
    bos_token_id: int = 50256,  # kept for API backward-compat (unused)
    eos_token_id: int = 50256,  # kept for API backward-compat (unused)
    attn_implementation: str = "flash_attention_2",  # retained but ignored
    **extra_cfg: Any,  # ignored to preserve call-sites that used to pass config tweaks
) -> nn.Module:
    """Instantiate and return a *pure-PyTorch* GPT-2 language model.

    The function intentionally keeps the same signature as the original
    wrapper so existing YAML/CLI configurations continue to work.
    Extra keyword arguments are quietly ignored.
    """

    # Map legacy *n_ctx* to *n_positions* if provided.
    if n_ctx is not None and n_ctx != n_positions:
        n_positions = n_ctx

    # Issue a gentle warning if the user passes unused extra kwargs.
    if extra_cfg:
        invalid = ", ".join(extra_cfg.keys())
        print(
            f"[build_gpt2_model] Warning: Ignoring unsupported keyword arguments: {invalid}.",
            flush=True,
        )

    return GPT2LMHeadModel(
        vocab_size=vocab_size,
        n_positions=n_positions,
        n_embd=n_embd,
        n_layer=n_layer,
        n_head=n_head,
    )
```

In short, `build_gpt2_model(...)` constructs a compact GPT‑2 with configurable depth/width/heads and returns an `nn.Module` that outputs logits over the vocabulary. It’s intentionally lean (no KV‑cache or generation helpers) but perfectly suited for forward/backward passes and next‑token prediction.

To use this exact implementation directly from a file path, point `_target_` to the file and object name (`path.py:object`). Absolute paths are recommended:

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

### 4.2 Import path to a callable (function or class)

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

## 5. Inspect and adjust the YAML configuration

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

## 6. Launch training

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

## 7. Monitoring and evaluation

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

## 8. Further work

1. **Scaling up** - swap the GPT-2 config for `LlamaForCausalLM`, `Qwen2`, or any HF-compatible causal model; increase `n_layer`, `n_embd`, etc.
2. **Mixed precision** - FSDP2 + `bfloat16` (`dtype: bfloat16` in distributed config) for memory savings.
3. **Sequence packing** - set `packed_sequence.packed_sequence_size` > 0 to pack variable-length contexts and boost utilization.
4. **Custom datasets** - implement your own `IterableDataset` or convert existing corpora to the `.bin` format using `tools/nanogpt_data_processor.py` as a template.
5. **BOS alignment** - set `align_to_bos: true` in the dataset config to ensure sequences start with BOS tokens (requires `bos_token` parameter).

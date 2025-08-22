# Continued Pretraining with NeMo Automodel

## Introduction

As large language models (LLMs) are deployed across new domains, their base pretraining may not fully cover domain-specific vocabulary, style, or distributions. Continued pretraining extends a pretrained model by training it further on additional unlabeled text, typically with the same next-token prediction objective as initial pretraining.

Continued pretraining can yield substantial gains in perplexity and downstream task performance when the new corpus differs meaningfully from the model’s original pretraining data.

NeMo Automodel provides an end-to-end recipe to run continued pretraining using Hugging Face-native models and datasets stored in JSONL chunks.

## How Continued Pretraining Differs from SFT

- **Objective**: Continued pretraining uses the unsupervised next-token prediction objective on unlabeled text. SFT (Supervised Fine-Tuning) uses labeled prompts/targets (e.g., instruction-response pairs) with a next-token prediction objective on target tokens. The key difference in the objective is the type of tokens it's applied to.
- **Data**: Continued pretraining consumes raw text in large volumes (no annotations). SFT requires structured, labeled examples.
- **Behavioral impact**: Continued pretraining primarily improves language modeling quality and domain coverage. SFT shapes instruction-following behavior and task compliance.
- **Configuration**: Both use similar training infrastructure (optimizer, schedulers, distributed strategies). The key difference is the dataset component: continued pretraining uses a large streaming dataset, while SFT often uses curated labeled datasets (e.g., SQuAD in the SFT guide).

## Model and Dataset Context

In this guide, we continue pretraining Meta’s `Llama 3.2 1B` model on a JSONL-based corpus such as FineWeb-Edu.

### Lingua Data Loading
To construct the dataset, we use the data loading functionality introduced in the [Lingua repository](https://github.com/facebookresearch/lingua). The JSONLDataset class constructs an [IterableDataset](https://docs.pytorch.org/docs/stable/data.html#iterable-style-datasets) to continuously read samples from storage instead of reading them all at once. In doing so, we can avoid large memory overheads.

Continued pretraining in this guide expects pre-shuffled JSONL chunks on disk. Each line contains a JSON object with either a `text` or `content` field, for example:

``` json
{"text": "An educational paragraph about topic X ..."}
```

Data are organized as multiple chunk files per source to enable distributed reading:

``` text
<root_dir>/
  <source_name>/
    <source_name>.chunk.00.jsonl
    <source_name>.chunk.01.jsonl
    ...
    <source_name>.val.jsonl
```

::::{note}
Training files follow the pattern `*.chunk.*.jsonl`; validation expects a single `*.val.jsonl` per source. These patterns are enforced by the data loader.
::::

:::{figure} ./dataloader.png
:name: jsonl-dataloader
:alt: Example of JSONL chunked data loading
:align: center

[Source](https://github.com/facebookresearch/lingua/blob/main/dataloader.png) Example of how dataset blending, packing, and shuffling is done in Lingua-style datasets.
:::


### 📚 About the FineWeb-Edu Dataset

[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) is a dataset consisting of 1.3T tokens of educational web pages filtered from the larger [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) dataset. The educational web pages were filtered from the main dataset using a finetuned [Bert](https://huggingface.co/docs/transformers/en/model_doc/bert)-like classifier. Further reading on the filtering process can be found [here](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1).

Here’s a glimpse of what the data looks like:
```json
{
    "id": "<urn:uuid:673b1bf6-2c30-40ae-992b-c387d00a836a>",
    "dump": "CC-MAIN-2013-20",
    "text": "No. 24; Updated March 2011
    Click here to download and print a PDF version of this document.
    Parents are usually the first to recognize that their child has a problem with emotions or behavior. Still, the decision to seek professional help can be difficult and painful for a parent. The first step is to gently try to talk to the child. An honest open talk about feelings can often help. Parents may choose to consult with the child's physicians, teachers, members of the clergy, or other adults who know the child well. These steps may resolve the problems for the child and family.
    Following are a few signs which may indicate that a child and adolescent psychiatric evaluation will be useful ...",
    "url": "http://aacap.org/page.ww?name=When+to+Seek+Help+for+Your+Child&section=Facts+for+Families",
    "date": null,
    "file_path": "s3://commoncrawl/crawl-data/CC-MAIN-2013-20/segments/1368696381249/warc/CC-MAIN-20130516092621-00000-ip-10-60-113-184.ec2.internal.warc.gz",
    "language": "en",
    "language_score": 0.927742,
    "token_count": 755,
    "score": 3.375,
    "int_score": 3,
}
```

#### Downloading the FineWeb-Edu Dataset

For the purposes of this guide, we will be using the FineWeb-Edu 10BT subset which is a subset randomly sampled from FineWeb-Edu of around 10B tokens. In order to prepare the dataset, follow the following commands:

```bash
git clone https://github.com/facebookresearch/lingua.git
cd lingua
pip install -r requirements.txt
python setup/download_prepare_hf_data.py fineweb_edu_10bt <MEMORY> --data_dir <DATA_DIR> --seed 42 --nchunks <NCHUNKS>
```
`<MEMORY>` can be replaced with how much system memory `terashuf` (the tool used to shuffle samples) will be allocated, `<DATA_DIR>` is the root directory where the data will be stored, and `<NCHUNKS>` is the number of shards of the dataset to create. It is recommended that `<NCHUNKS>` either be 1 or the number of GPUs you will be using to run pretraining. An example command you can run is:
```bash
python setup/download_prepare_hf_data.py fineweb_edu_10bt 16 --data_dir ./fineweb_edu --seed 42 --nchunks 1
```

The expected directory structure is like this:
```bash
$ tree fineweb_edu/
fineweb_edu/
├── fineweb_edu_10bt
│   ├── datatrove
│   │   ├── completions
│   │   │   ├── 00000
│   │   │   ├── 00001
│   │   │   ├── 00002
│   │   │   ├── 00003
│   │   │   ├── 00004
│   │   │   ├── 00005
│   │   │   │   ...
│   │   │   └── 00063
│   │   ├── executor.json
│   │   ├── logs
│   │   │   ├── task_00000.log
│   │   │   ├── task_00001.log
│   │   │   ├── task_00002.log
│   │   │   ├── task_00003.log
│   │   │   ├── task_00004.log
│   │   │   ├── task_00005.log
│   │   │   │   ...
│   │   │   └── task_00063.log
│   │   ├── stats
│   │   │   ├── 00000.json
│   │   │   ├── 00001.json
│   │   │   ├── 00002.json
│   │   │   ├── 00003.json
│   │   │   ├── 00004.json
│   │   │   ├── 00005.json
│   │   │   │   ...
│   │   │   └── 00063.json
│   │   └── stats.json
│   ├── fineweb_edu_10bt.chunk.00000.jsonl
│   │   ...
│   ├── fineweb_edu_10bt.chunk.00013.jsonl
│   ├── sample
│   │   └── 10BT
│   │       ├── 000_00000.parquet
│   │       │   ...
│   │       └── 013_00000.parquet
│   └── terashuf
│       ├── LICENSE
│       ├── Makefile
│       ├── README.md
│       ├── terashuf
│       └── terashuf.cc
└── fineweb_edu_10bt_shuffled
    ├── fineweb_edu_10bt.chunk.00.jsonl
    └── fineweb_edu_10bt.val.jsonl
```

### 🔍 About Llama 3.2 1B
**LLaMA** is a family of decoder-only transformer models developed by Meta. The **Llama 3.2 1B** variant is a compact, lightweight model ideal for research and edge deployment. Despite its size, it maintains architectural features consistent with its larger siblings:

- **Decoder-only architecture**: Follows a GPT-style, autoregressive design—optimized for generation tasks.

- **Rotary positional embeddings (RoPE)**: Efficient and extendable positional encoding technique.

- **Grouped-query attention (GQA)**: Enhances scalability by decoupling key/value heads from query heads.

- **SwiGLU activation**: A variant of the GLU activation, offering improved convergence and expressiveness.

- **Multi-layer residual connections**: Enhances training stability and depth scaling.

::::{tip}
`meta-llama/Llama-3.2-1B` is a placeholder. You can use any Hugging Face model ID you have access to (e.g., `Qwen/Qwen2.5-1.5B`).
::::

:::{important}
Some Hugging Face model repositories are **gated**, you must explicitly request permission before you can download their files. If the model page shows a "Request access" or "Agree and access" button:

1.  Log in with your Hugging Face account.
2.  Click the button and accept the license terms.
3.  Wait for approval (usually instant; occasionally manual).
4.  Ensure the token you pass to your script (via `huggingface-cli login` or the `HF_TOKEN` environment variable) belongs to the account that was approved.

 Trying to pull a gated model without an authorized token will trigger a 403 "permission denied" error.
 :::

## Use a Recipe to Continue Pretraining

This example demonstrates how to perform continued pretraining on a large language model using NVIDIA's NeMo Automodel library. We use the LLM [finetune recipe](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/llm/finetune.py), specifically `FinetuneRecipeForNextTokenPrediction`, which orchestrates the pretraining process end-to-end: model loading, dataset preparation, optimizer setup, distributed training, checkpointing, and logging.

### What is a Recipe?

A recipe in NeMo Automodel is a **self-contained orchestration module** that wires together all
components needed to perform a specific task (e.g., continued pretraining).
Think of it as the equivalent of a Trainer class, but highly modular, stateful, and reproducible.

The `FinetuneRecipeForNextTokenPrediction` class is one such recipe. It inherits from `BaseRecipe` and implements:

- `setup()`: builds all training components from the config

- `run_train_validation_loop()`: executes training + validation steps

- Misc: Checkpoint handling, logging, and RNG setup.

### Recipe Config (example)

Below is the configuration from `examples/llm/llama_3_2_1b_fineweb_edu.yaml`:

```yaml
# The model section is responsible for configuring the model we want to finetune.
# Since we want to use the Llama 3 1B model, we pass `meta-llama/Llama-3.2-1B` to the
# `pretrained_model_name_or_path` option.
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
  is_meta_device: false

# As mentioned earlier, we are using the FineWeb-Edu dataset. NeMo Automodel provides the JSONLDataset
# class which prepares the dataset by loading, packing, and shuffling. We use the "train" split for
# training.
dataset:
  _target_: nemo_automodel.components.datasets.llm.jsonl_dataset.JSONLDataset
  root_dir: fineweb_edu
  sources:
    "fineweb_edu_10bt_shuffled": 100.0
  batch_size: 8
  packed_sequence_size: 1024
  split: train
  infinite: false

# Similarly, for validation we use the "validation" split
validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.jsonl_dataset.JSONLDataset
  root_dir: fineweb_edu
  sources:
    "fineweb_edu_10bt_shuffled": 100.0
  batch_size: 8
  packed_sequence_size: 1024
  split: validation
  infinite: false

step_scheduler:
  grad_acc_steps: 4
  ckpt_every_steps: 1000 # checkpoints state every 1000 steps
  val_every_steps: 100 # validates every 100 steps
  num_epochs: 1

dist_env:
  backend: nccl
  timeout_minutes: 1

rng:
  _target_: nemo_automodel.components.training.rng.StatefulRNG
  seed: 1111
  ranked: true

checkpoint:
  enabled: true
  checkpoint_dir: checkpoints/
  model_save_format: safetensors
  save_consolidated: True # saves the model in a consolidated safetensors format. Requires model_save_format to be safetensors.

# For distributed processing, we will FSDP2.
distributed:
  _target_: nemo_automodel.components.distributed.fsdp2.FSDP2Manager
  dp_size: none
  tp_size: 1
  cp_size: 1
  sequence_parallel: false

loss_fn:
  _target_: nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy

# We will use the standard Adam optimizer, but you can specify any optimizer you want, by changing
# the import path using the _target_ option.
optimizer:
  _target_: torch.optim.Adam
  betas: [0.9, 0.999]
  eps: 1e-8
  lr: 1.0e-5
  weight_decay: 0

# If you want to log your experiment on wandb, uncomment and configure the following section
# wandb:
#   project: <your_wandb_project>
#   entity: <your_wandb_entity>
#   name: <your_wandb_exp_name>
#   save_dir: <your_wandb_save_dir>
```

::::{important}
Update `root_dir` and `sources` to match your on-disk dataset. The `sources` map names to sampling weights when mixing multiple sources.
::::

## Loading Large Models
The common model loading pipeline when doing distributed training is that each GPU will load the full model onto it and then hold the shard it needs. However, this is an issue when we want to train models that are larger than the memory of a single GPU. For example, a 70B parameter model takes up 140GB for the model parameters assuming BF16 data type (2 bytes per parameter). Most popular GPUs have a limit of 80GB, which means we cannot directly load the full model onto the GPU.

In these scenarios, you can pass `is_meta_device: true` in the model config. The model will then be instantiated using [PyTorch's Meta device](https://docs.pytorch.org/docs/stable/meta.html) which loads no data, but stores all other parameter metadata necessary for sharding the model. Once the model is sharded, the model weights will be populated by only loading the weights required by the respective model shard.

## High-Level Overview: JSONL Dataset and Data Pipeline

This section explains how `JSONLDataset` and the lingua-based core utilities stream data efficiently and reproducibly.

### `JSONLDataset`

Defined in `nemo_automodel/components/datasets/llm/jsonl_dataset.py`, `JSONLDataset` is a `torch.utils.data.IterableDataset` that:

- Stores constructor arguments (e.g., `root_dir`, `sources`, `batch_size`, `packed_sequence_size`, `seed`, `split`, `add_bos/eos`, `prefetch_size`, `n_views`, `infinite`).
- Builds a persistent iterator state via `init_dataloader_state_from_args(...)` in `lingua_assets/core.py`.
- Constructs a dataloader with `build_dataloader_from_args(...)`, optionally spawning an async prefetch process.
- On iteration, converts the `(batch, state)` stream into a dict with `input_ids`, `labels`, and `loss_mask` tensors suitable for next-token training.
- Persists and restores streaming state across checkpoints via `state_dict()` and `load_state_dict()`.

### Lingua Data Pipeline

The pipeline decomposes into composable iterators and states:

1. **Chunk assignment**: `find_and_sanitize_chunks(...)` and `distribute_data_to_rank(...)` map chunk files to ranks. Validation expects a single `*.val.jsonl` file; training uses `*.chunk.*.jsonl`.
2. **Source sampling**: `init_choice_state(...)` creates a `MultiChoiceState` with RNG state and per-source iterators; `choose_source(...)` samples sources by weight and tracks exhausted sources in single-epoch mode.
3. **Reading lines**: `loop_on_jsonl(...)` wraps `read_jsonl(...)`, which reads lines in an interleaved way using `(block_size, offset)` so multiple ranks can share a file without overlap.
4. **Tokenization**: `tokenize(...)` encodes each JSON line’s `text`/`content` to token IDs, optionally adding BOS/EOS.
5. **Sequence packing**: `pack_tokens(...)` aggregates tokens into windows of shape `(seq_len, n_views)`, using a sliding window so the second column is a 1-token shift of the first (i.e., labels). State logic ensures seamless continuation across chunk boundaries.
6. **Prefetch + shuffle**: `batch_and_shuffle_prefetched_sequences(...)` maintains a prefetch buffer of size `prefetch_size * batch_size`, periodically shuffles to decorrelate sequences, and yields fixed `(batch_size, seq_len, n_views)` arrays with deterministic RNG state.
7. **Async mode (optional)**: `async_iterator(...)` spawns a producer process (`feed_buffer`) feeding a queue; the main process consumes via `consume_buffer` with timeouts and clean shutdown.

Taken together, `build_dataloader_from_args(tokenizer, load_async, prefetch_size, state)` returns a context-managed iterator over `(batch, state)` pairs; `JSONLDataset.__iter__` turns each into tensors.

## Run the Continued Pretraining Recipe

Assuming you saved or will use the provided config at `examples/llm/llama_3_2_1b_fineweb_edu.yaml`:

### Automodel CLI

When NeMo Automodel is installed on your system, it includes the `automodel` CLI program that you
can use to run jobs, locally or on distributed environments.

``` bash
uv run automodel finetune llm -c examples/llm/llama_3_2_1b_fineweb_edu.yaml
```

Where `finetune` is the recipe name and `llm` is the model domain.

### Invoke the Recipe Script Directly

``` bash
uv run torchrun --nproc-per-node=8 examples/llm/finetune.py --config examples/llm/llama_3_2_1b_fineweb_edu.yaml
```

### Sample Output

You should see step-wise logs with loss, memory, and tokens/sec. Checkpoints will be saved under `checkpoints/` as configured.

```
$ uv run automodel finetune llm -c examples/llm/llama_3_2_1b_fineweb_edu.yaml
INFO:root:Domain:  llm
INFO:root:Command: finetune
INFO:root:Config:  /opt/Automodel/examples/llm/llama_3_2_1b_fineweb_edu.yaml
INFO:root:Running job using source from: /opt/Automodel
INFO:root:Launching job locally on 2 devices
cfg-path: /opt/Automodel/examples/llm/llama_3_2_1b_fineweb_edu.yaml
cfg-path: /opt/Automodel/examples/llm/llama_3_2_1b_fineweb_edu.yaml
> initializing torch distributed with 2 workers.
2025-08-22 02:40:26 | INFO | nemo_automodel.components.loggers.log_utils | Setting logging level to 20
2025-08-22 02:40:26 | INFO | root | Experiment_details:
2025-08-22 02:40:26 | INFO | root | Timestamp: '2025-08-22T02:40:26'
2025-08-22 02:40:26 | INFO | root | User: root
2025-08-22 02:40:26 | INFO | root | Host: 2306d67e22e4
2025-08-22 02:40:26 | INFO | root | World size: 2
2025-08-22 02:40:26 | INFO | root | Backend: nccl
2025-08-22 02:40:26 | INFO | root | Recipe: FinetuneRecipeForNextTokenPrediction
2025-08-22 02:40:26 | INFO | root | Model name: meta-llama/Llama-3.2-1B
2025-08-22 02:40:26 | INFO | root | Recipe config:
2025-08-22 02:40:26 | INFO | root |   step_scheduler:
2025-08-22 02:40:26 | INFO | root |     grad_acc_steps: 4
2025-08-22 02:40:26 | INFO | root |     ckpt_every_steps: 2
2025-08-22 02:40:26 | INFO | root |     val_every_steps: 100
2025-08-22 02:40:26 | INFO | root |     num_epochs: 1
2025-08-22 02:40:26 | INFO | root |   dist_env:
2025-08-22 02:40:26 | INFO | root |     backend: nccl
2025-08-22 02:40:26 | INFO | root |     timeout_minutes: 1
2025-08-22 02:40:26 | INFO | root |   rng:
2025-08-22 02:40:26 | INFO | root |     _target_: <class 'nemo_automodel.components.training.rng.StatefulRNG'>
2025-08-22 02:40:26 | INFO | root |     seed: 1111
2025-08-22 02:40:26 | INFO | root |     ranked: True
2025-08-22 02:40:26 | INFO | root |   model:
2025-08-22 02:40:26 | INFO | root |     _target_: <bound method _BaseNeMoAutoModelClass.from_pretrained of <class 'nemo_automodel.components._transformers.auto_model.NeMoAutoModelForCausalLM'>>
2025-08-22 02:40:26 | INFO | root |     pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
2025-08-22 02:40:26 | INFO | root |   checkpoint:
2025-08-22 02:40:26 | INFO | root |     enabled: True
2025-08-22 02:40:26 | INFO | root |     checkpoint_dir: checkpoints/
2025-08-22 02:40:26 | INFO | root |     model_save_format: torch_save
2025-08-22 02:40:26 | INFO | root |     save_consolidated: False
2025-08-22 02:40:26 | INFO | root |   distributed:
2025-08-22 02:40:26 | INFO | root |     _target_: <class 'nemo_automodel.components.distributed.fsdp2.FSDP2Manager'>
2025-08-22 02:40:26 | INFO | root |     dp_size: None
2025-08-22 02:40:26 | INFO | root |     tp_size: 1
2025-08-22 02:40:26 | INFO | root |     cp_size: 1
2025-08-22 02:40:26 | INFO | root |     sequence_parallel: False
2025-08-22 02:40:26 | INFO | root |   loss_fn:
2025-08-22 02:40:26 | INFO | root |     _target_: <class 'nemo_automodel.components.loss.masked_ce.MaskedCrossEntropy'>
2025-08-22 02:40:26 | INFO | root |   dataset:
2025-08-22 02:40:26 | INFO | root |     _target_: <class 'nemo_automodel.components.datasets.llm.jsonl_dataset.JSONLDataset'>
2025-08-22 02:40:26 | INFO | root |     root_dir: fineweb_edu
2025-08-22 02:40:26 | INFO | root |     sources:
2025-08-22 02:40:26 | INFO | root |       fineweb_edu_10bt_shuffled: 100.0
2025-08-22 02:40:26 | INFO | root |     batch_size: 8
2025-08-22 02:40:26 | INFO | root |     packed_sequence_size: 1024
2025-08-22 02:40:26 | INFO | root |     split: train
2025-08-22 02:40:26 | INFO | root |     infinite: False
2025-08-22 02:40:26 | INFO | root |   validation_dataset:
2025-08-22 02:40:26 | INFO | root |     _target_: <class 'nemo_automodel.components.datasets.llm.jsonl_dataset.JSONLDataset'>
2025-08-22 02:40:26 | INFO | root |     root_dir: fineweb_edu
2025-08-22 02:40:26 | INFO | root |     sources:
2025-08-22 02:40:26 | INFO | root |       fineweb_edu_10bt_shuffled: 100.0
2025-08-22 02:40:26 | INFO | root |     batch_size: 8
2025-08-22 02:40:26 | INFO | root |     packed_sequence_size: 1024
2025-08-22 02:40:26 | INFO | root |     split: validation
2025-08-22 02:40:26 | INFO | root |     infinite: False
2025-08-22 02:40:26 | INFO | root |   optimizer:
2025-08-22 02:40:26 | INFO | root |     _target_: <class 'torch.optim.adam.Adam'>
2025-08-22 02:40:26 | INFO | root |     betas: [0.9, 0.999]
2025-08-22 02:40:26 | INFO | root |     eps: 1e-08
2025-08-22 02:40:26 | INFO | root |     lr: 1e-05
2025-08-22 02:40:26 | INFO | root |     weight_decay: 0
2025-08-22 02:40:26 | INFO | root | Library versions:
2025-08-22 02:40:26 | INFO | root | - nemo_automodel: 0.2.0rc0 (/opt/Automodel/nemo_automodel/__init__.py)
2025-08-22 02:40:26 | INFO | root | - transformers: 4.53.0 (/opt/venv/lib/python3.12/site-packages/transformers/__init__.py)
2025-08-22 02:40:26 | INFO | root | - torch: 2.8.0+cu128 CUDA 12.8
2025-08-22 02:40:26 | INFO | root | Patched model with SDPA method= [<SDPBackend.CUDNN_ATTENTION: 3>, <SDPBackend.FLASH_ATTENTION: 1>, <SDPBackend.EFFICIENT_ATTENTION: 2>, <SDPBackend.MATH: 0>]
2025-08-22 02:40:26 | INFO | root | Freezing embeddings
2025-08-22 02:40:28 | INFO | root | Model summary:
2025-08-22 02:40:28 | INFO | root | --------------------------------
2025-08-22 02:40:28 | INFO | root | Trainable parameters: 973,146,112
2025-08-22 02:40:28 | INFO | root | Total parameters: 1,235,814,400
2025-08-22 02:40:28 | INFO | root | Trainable parameters percentage: 78.75%
2025-08-22 02:40:28 | INFO | root | Param L2 norm: 718.4798
2025-08-22 02:40:28 | INFO | root | --------------------------------
2025-08-22 02:40:28 | INFO | root | Using model config to instantiate tokenizer
2025-08-22 02:40:29 | INFO | root | Using model config to instantiate tokenizer
2025-08-22 02:40:29 | INFO | root | Model:
2025-08-22 02:40:29 | INFO | root | FSDPLlamaForCausalLM(
2025-08-22 02:40:29 | INFO | root |   (model): LlamaModel(
2025-08-22 02:40:29 | INFO | root |     (embed_tokens): Embedding(128256, 2048)
2025-08-22 02:40:29 | INFO | root |     (layers): ModuleList(
2025-08-22 02:40:29 | INFO | root |       (0-15): 16 x FSDPLlamaDecoderLayer(
2025-08-22 02:40:29 | INFO | root |         (self_attn): LlamaAttention(
2025-08-22 02:40:29 | INFO | root |           (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
2025-08-22 02:40:29 | INFO | root |           (k_proj): Linear(in_features=2048, out_features=512, bias=False)
2025-08-22 02:40:29 | INFO | root |           (v_proj): Linear(in_features=2048, out_features=512, bias=False)
2025-08-22 02:40:29 | INFO | root |           (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
2025-08-22 02:40:29 | INFO | root |         )
2025-08-22 02:40:29 | INFO | root |         (mlp): LlamaMLP(
2025-08-22 02:40:29 | INFO | root |           (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
2025-08-22 02:40:29 | INFO | root |           (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
2025-08-22 02:40:29 | INFO | root |           (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
2025-08-22 02:40:29 | INFO | root |           (act_fn): SiLU()
2025-08-22 02:40:29 | INFO | root |         )
2025-08-22 02:40:29 | INFO | root |         (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
2025-08-22 02:40:29 | INFO | root |         (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
2025-08-22 02:40:29 | INFO | root |       )
2025-08-22 02:40:29 | INFO | root |     )
2025-08-22 02:40:29 | INFO | root |     (norm): LlamaRMSNorm((2048,), eps=1e-05)
2025-08-22 02:40:29 | INFO | root |     (rotary_emb): LlamaRotaryEmbedding()
2025-08-22 02:40:29 | INFO | root |   )
2025-08-22 02:40:29 | INFO | root |   (lm_head): Linear(in_features=2048, out_features=128256, bias=False)
2025-08-22 02:40:29 | INFO | root | )
2025-08-22 02:40:29 | INFO | root | Optimizer:
2025-08-22 02:40:29 | INFO | root | Adam (
2025-08-22 02:40:29 | INFO | root | Parameter Group 0
2025-08-22 02:40:29 | INFO | root |     amsgrad: False
2025-08-22 02:40:29 | INFO | root |     betas: [0.9, 0.999]
2025-08-22 02:40:29 | INFO | root |     capturable: False
2025-08-22 02:40:29 | INFO | root |     decoupled_weight_decay: False
2025-08-22 02:40:29 | INFO | root |     differentiable: False
2025-08-22 02:40:29 | INFO | root |     eps: 1e-08
2025-08-22 02:40:29 | INFO | root |     foreach: None
2025-08-22 02:40:29 | INFO | root |     fused: None
2025-08-22 02:40:29 | INFO | root |     lr: 1e-05
2025-08-22 02:40:29 | INFO | root |     maximize: False
2025-08-22 02:40:29 | INFO | root |     weight_decay: 0
2025-08-22 02:40:29 | INFO | root | )
2025-08-22 02:40:29 | INFO | root | LR scheduler: <unavailable>
2025-08-22 02:40:29 | INFO | root | Step scheduler:
2025-08-22 02:40:29 | INFO | root | - Gradient accumulation steps: 4
2025-08-22 02:40:29 | INFO | root | - Checkpoint every steps: 2
2025-08-22 02:40:29 | INFO | root | - Current Epoch: 0
2025-08-22 02:40:29 | INFO | root | - Number of epochs: 1
2025-08-22 02:40:29 | INFO | root | - Validation every steps: 100
2025-08-22 02:40:29 | INFO | root | - Max train steps: 9223372036854775807
2025-08-22 02:40:34 | INFO | root | step 1 | epoch 0 | loss 2.4827 | grad_norm 1.8828 | lr 1.00e-05 | mem 30.58 GiB | tps 14611.54 | num_label_tokens 65536
2025-08-22 02:40:37 | INFO | root | step 2 | epoch 0 | loss 2.5554 | grad_norm 1.9922 | lr 1.00e-05 | mem 32.39 GiB | tps 20321.85 | num_label_tokens 65536
Saving checkpoint to checkpoints/epoch_0_step_2
```
For each training batch, the fine-tuning recipe logs the current loss, along with current peak memory usage and tokens per second (TPS).

In addition, the model checkpoint is saved under the `checkpoints/` directory, with the following contents:
```bash
$ tree checkpoints/epoch_0_step_2/
checkpoints/epoch_0_step_2/
├── config.yaml
├── dataloader
│   ├── dataloader_dp_rank_0.pt
│   └── dataloader_dp_rank_1.pt
├── model
│   ├── consolidated
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── model-00001-of-00001.safetensors
│   │   ├── model.safetensors.index.json
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── shard-00001-model-00001-of-00001.safetensors
│   └── shard-00002-model-00001-of-00001.safetensors
├── optim
│   ├── __0_0.distcp
│   └── __1_0.distcp
├── rng.pt
└── step_scheduler.pt

5 directories, 16 files
```
## Run Inference with a Continued-Pretraining Checkpoint

If you saved in Hugging Face-native format (e.g., safetensors consolidated), you can load with Transformers directly; otherwise, use the saved format you configured. Example for a consolidated Transformers layout:

``` python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load finetuned checkpoint
ckpt_path = "checkpoints/epoch_0_step_2/model/consolidated"
tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
model = AutoModelForCausalLM.from_pretrained(ckpt_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Generate text
inputs = tokenizer("Toronto is a city in Canada.", return_tensors="pt").to(device)
output = model.generate(**inputs, max_length=64)

# Decode and print the output
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Export to vLLM

[vLLM](https://github.com/vllm-project/vllm) is an efficient inference
engine designed to optimize the deployment of large language models
(LLMs) for production use. By utilizing advanced techniques like
parallel processing and optimized memory management, vLLM accelerates
inference while maintaining model accuracy.

The following script demonstrates how to use a fine-tuned checkpoint
in vLLM, allowing seamless deployment and efficient inference:

:::{note}
Make sure vLLM is installed (pip install vllm, or use the environment that includes it).
:::


``` python
from vllm import LLM, SamplingParams

llm = LLM(model="checkpoints/epoch_0_step_2/model/consolidated/", model_impl="transformers")
params = SamplingParams(max_tokens=20)
outputs = llm.generate("Toronto is a city in Canada.", sampling_params=params)
print(f"Generated text: {outputs[0].outputs[0].text}")
```

## Practical Tips

- **Chunk-to-world-size**: Ensure the number of training chunks per source divides the world size, this can be done by either setting `--nchunks 1` or `--nchunks <num GPUs>` while generating the dataset. Validation expects exactly one `*.val.jsonl` per source.
- **Mixing sources**: Add multiple entries under `sources` with weights (e.g., `{source_a: 70.0, source_b: 30.0}`) to blend corpora.
- **Throughput vs correlation**: Increase `prefetch_size` to smooth sequence diversity; use async loading for I/O-heavy workloads.
- **Stopping criteria**: Set `infinite` to `false` and control tokens/steps via scheduler and epochs; for large-scale streaming, `infinite: true` and rely on step-based schedules.



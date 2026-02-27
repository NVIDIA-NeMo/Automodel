# Embedding Model Fine-Tuning with NeMo AutoModel

## Introduction

Embedding models map text into dense vector representations used for semantic search, retrieval-augmented generation (RAG), and other information retrieval tasks. Fine-tuning an embedding model on domain-specific data improves retrieval quality by teaching the model to better distinguish relevant documents from irrelevant ones within your target domain.

NeMo AutoModel supports **biencoder (dual-encoder) embedding model fine-tuning** using contrastive learning. Each training example consists of a query paired with one positive document and one or more hard negative documents. The model learns to produce embeddings where queries are close to their relevant documents and far from irrelevant ones.

### Architecture Overview

The biencoder architecture uses two encoders — one for queries and one for passages — built on a **bidirectional** variant of the backbone model. Unlike standard causal LLMs that only attend to preceding tokens, the bidirectional backbone removes the causal mask so every token can attend to every other token, producing richer representations for embedding tasks.

Key features:

- **Shared or separate encoders**: Share weights between query and passage encoders (default), or use independent encoders for asymmetric tasks.
- **Flexible pooling**: Average (`avg`), weighted average (`weighted_avg`), CLS token (`cls`), last token (`last`), or ColBERT-style (`colbert`).
- **L2 normalization**: Optionally normalize embeddings to the unit sphere for cosine similarity.
- **LoRA/PEFT support**: Fine-tune with parameter-efficient adapters.
- **FSDP2 distributed training**: Scale across multiple GPUs seamlessly.

### Supported Backbones

| Backbone Family | Bidirectional Class | Notes |
|-----------------|-------------------|-------|
| Llama (3.x, 3.2, etc.) | `LlamaBidirectionalModel` | Includes Llama-based NeMo Retriever models |

## Quickstart

Fine-tune a Llama 3.2 1B embedding model with a single command:

```bash
torchrun --nproc-per-node=8 examples/biencoder/finetune.py \
    --config examples/biencoder/llama3_2_1b_biencoder.yaml
```

Or using the AutoModel CLI:

```bash
automodel finetune biencoder -c examples/biencoder/llama3_2_1b_biencoder.yaml
```

## Recipe Configuration

The biencoder recipe is configured through a YAML file. Below is the full reference config with explanations.

### Model

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelBiencoder.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
  share_encoder: true
  pooling: avg
  l2_normalize: true
  use_liger_kernel: true
  use_sdpa_patching: true
  torch_dtype: bfloat16
```

| Parameter | Description |
|-----------|-------------|
| `pretrained_model_name_or_path` | Hugging Face model ID or local checkpoint path |
| `share_encoder` | If `true`, query and passage encoders share weights (recommended for most cases) |
| `pooling` | Pooling strategy: `avg`, `weighted_avg`, `cls`, `last`, `colbert` |
| `l2_normalize` | Normalize output embeddings to unit length for cosine similarity |

:::{tip}
You can use any Llama-family model as the backbone, including `nvidia/llama-3.2-nv-embedqa-1b-v2` or `meta-llama/Llama-3.2-1B`. For other backbone families, check the [supported backbones](#supported-backbones) table.
:::

### Contrastive Training Parameters

```yaml
seed: 42
train_n_passages: 5        # 1 positive + 4 hard negatives per query
eval_negative_size: 4       # negatives per query during validation
temperature: 0.02           # temperature scaling for similarity scores
```

- `train_n_passages`: Total passages per query (first is always the positive). More hard negatives generally improve retrieval quality but increase memory usage.
- `temperature`: Lower values sharpen the similarity distribution. Typical range: 0.01–0.1.

### Tokenizer

```yaml
tokenizer:
  _target_: nemo_automodel.NeMoAutoTokenizer.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
```

### Dataset and Collator

The dataset factory and collator handle loading and tokenizing query-passage pairs.

```yaml
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  dataset:
    _target_: nemo_automodel.components.datasets.llm.make_retrieval_dataset
    data_dir_list:
      - /path/to/train_data.json
    data_type: train
    train_n_passages: 5
    seed: 42
    do_shuffle: true
  collate_fn:
    _target_: nemo_automodel.components.datasets.llm.RetrievalBiencoderCollator
    q_max_len: 512
    p_max_len: 512
    query_prefix: "query:"
    passage_prefix: "passage:"
    pad_to_multiple_of: 8
  shuffle: true
  num_workers: 0
```

| Parameter | Description |
|-----------|-------------|
| `data_dir_list` | List of training data files (JSON or JSONL) |
| `train_n_passages` | Must match the top-level `train_n_passages` |
| `q_max_len` / `p_max_len` | Maximum token lengths for queries and passages |
| `query_prefix` / `passage_prefix` | Text prefixed to queries and passages before tokenization |

See [Biencoder Retrieval Dataset](../llm/retrieval-dataset.md) for supported data formats (corpus-ID JSON and inline-text JSONL).

### Validation (Optional)

```yaml
validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  dataset:
    _target_: nemo_automodel.components.datasets.llm.make_retrieval_dataset
    data_dir_list:
      - /path/to/validation_data.json
    data_type: eval
    train_n_passages: 5
    eval_negative_size: 4
    seed: 42
    do_shuffle: false
  collate_fn:
    _target_: nemo_automodel.components.datasets.llm.RetrievalBiencoderCollator
    q_max_len: 512
    p_max_len: 512
    query_prefix: "query:"
    passage_prefix: "passage:"
    pad_to_multiple_of: 8
  batch_size: 2
  shuffle: false
  num_workers: 0
```

The validation loop reports three metrics:

- **val_loss**: Cross-entropy loss on the validation set
- **val_acc1**: Accuracy@1 — fraction of queries where the top-ranked document is the positive
- **val_mrr**: Mean Reciprocal Rank — average of `1/rank` for the positive document

### Optimizer and Scheduler

```yaml
optimizer:
  _target_: transformer_engine.pytorch.optimizers.fused_adam.FusedAdam
  lr: 5.0e-6
  weight_decay: 0.01
  adam_w_mode: true
  bias_correction: true
  master_weights: true

lr_scheduler:
  lr_warmup_steps: 100

step_scheduler:
  global_batch_size: 128
  local_batch_size: 4
  ckpt_every_steps: 500
  val_every_steps: 500
  num_epochs: 1
```

### Distributed Training

```yaml
distributed:
  strategy: fsdp2
  dp_size: none
  tp_size: 1
  cp_size: 1
  sequence_parallel: false
```

The biencoder recipe supports FSDP2 for data parallelism. Tensor parallelism is available for larger backbones.

:::{note}
Pipeline parallelism is not supported for biencoder training.
:::

### Checkpointing

```yaml
checkpoint:
  enabled: true
  checkpoint_dir: ./output/llama3_2_1b_biencoder/checkpoints
  model_save_format: safetensors
  save_consolidated: true
```

Consolidated checkpoints are saved in the Hugging Face `safetensors` format, making them directly loadable for inference or further fine-tuning.

### LoRA/PEFT (Optional)

To use parameter-efficient fine-tuning, add a `peft` block:

```yaml
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
  dim: 16
  alpha: 32
  use_triton: true
```

## Data Preparation

### Supported Formats

NeMo AutoModel supports two input formats for retrieval training data:

#### Inline-Text JSONL (Recommended for Getting Started)

Each line is a standalone example with query and document text inline:

```json
{"query": "Explain transformers", "pos_doc": "Transformers are a type of neural network...", "neg_doc": ["RNNs are...", "CNNs are..."]}
{"query": "What is Python?", "pos_doc": ["A programming language."], "neg_doc": "A snake."}
```

#### Corpus-ID JSON (NeMo Retriever / Merlin Style)

Documents live in a separate corpus directory. Training examples reference documents by ID:

```json
{
  "corpus": [{"path": "/path/to/wiki_corpus"}],
  "data": [
    {
      "question": "Explain transformers",
      "corpus_id": "wiki_corpus",
      "pos_doc": [{"id": "d_123"}],
      "neg_doc": [{"id": "d_456"}, "d_789"]
    }
  ]
}
```

The corpus directory must contain a `merlin_metadata.json` file. See the [Biencoder Retrieval Dataset](../llm/retrieval-dataset.md) guide for full format details.

### Hard Negative Mining

High-quality hard negatives are critical for training effective embedding models. NeMo AutoModel includes a built-in hard negative mining tool that uses an existing embedding model to find challenging negatives from your corpus.

```bash
torchrun --nproc_per_node=8 examples/biencoder/mine_hard_negatives.py \
    --config examples/biencoder/mining_config.yaml \
    --mining.model_name_or_path /path/to/biencoder/checkpoint \
    --mining.train_qa_file_path /path/to/input.json \
    --mining.train_file_output_path /path/to/output_with_negatives.json
```

Key mining parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hard_negatives_to_mine` | 20 | Number of hard negatives to mine per query |
| `hard_neg_margin` | 0.95 | Margin threshold (percentile) for negative selection |
| `mining_batch_size` | 128 | Batch size for similarity computation |
| `query_max_length` / `passage_max_length` | 512 | Max token lengths |

The mining pipeline:
1. Encodes all corpus documents with the passage encoder
2. Encodes all queries with the query encoder
3. Computes similarity scores between queries and all documents
4. Selects documents that are similar but not the positive as hard negatives

## Training Workflow

A typical embedding model fine-tuning workflow:

1. **Prepare training data** in inline JSONL or corpus-ID JSON format
2. **(Optional) Mine hard negatives** using the mining tool
3. **Configure the recipe** by editing the YAML config
4. **Launch training**:

```bash
torchrun --nproc-per-node=8 examples/biencoder/finetune.py \
    --config examples/biencoder/llama3_2_1b_biencoder.yaml
```

### Sample Training Output

```
INFO:root:step 10 | epoch 0 | loss 2.1543 | grad_norm 1.2345 | lr 5.00e-06 | mem 24.50 GiB | time 3.21s
INFO:root:step 20 | epoch 0 | loss 1.3210 | grad_norm 0.8765 | lr 5.00e-06 | mem 24.50 GiB | time 3.18s
INFO:root:step 30 | epoch 0 | val_loss 1.1230 | val_acc1 0.7820 | val_mrr 0.8450
```

## Using the Fine-Tuned Model

The consolidated checkpoint is saved in Hugging Face format and can be loaded directly for inference:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("/path/to/checkpoint/model/consolidated")

queries = ["What is deep learning?"]
documents = [
    "Deep learning is a subset of machine learning...",
    "The weather today is sunny.",
]

q_embeddings = model.encode(queries, prompt="query: ")
d_embeddings = model.encode(documents, prompt="passage: ")

similarities = q_embeddings @ d_embeddings.T
print(similarities)
```

## Full Example Config

Below is the complete reference configuration:

```yaml
seed: 42

train_n_passages: 5
eval_negative_size: 4
temperature: 0.02

step_scheduler:
  global_batch_size: 128
  local_batch_size: 4
  ckpt_every_steps: 500
  val_every_steps: 500
  num_epochs: 1

dist_env:
  backend: nccl
  timeout_minutes: 1

model:
  _target_: nemo_automodel.NeMoAutoModelBiencoder.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
  share_encoder: true
  pooling: avg
  l2_normalize: true
  use_liger_kernel: true
  use_sdpa_patching: true
  torch_dtype: bfloat16

tokenizer:
  _target_: nemo_automodel.NeMoAutoTokenizer.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  dataset:
    _target_: nemo_automodel.components.datasets.llm.make_retrieval_dataset
    data_dir_list:
      - /path/to/train_data.json
    data_type: train
    train_n_passages: 5
    seed: 42
    do_shuffle: true
  collate_fn:
    _target_: nemo_automodel.components.datasets.llm.RetrievalBiencoderCollator
    q_max_len: 512
    p_max_len: 512
    query_prefix: "query:"
    passage_prefix: "passage:"
    pad_to_multiple_of: 8
  shuffle: true
  num_workers: 0

optimizer:
  _target_: transformer_engine.pytorch.optimizers.fused_adam.FusedAdam
  lr: 5.0e-6
  weight_decay: 0.01
  adam_w_mode: true
  bias_correction: true
  master_weights: true

lr_scheduler:
  lr_warmup_steps: 100

checkpoint:
  enabled: true
  checkpoint_dir: ./output/llama3_2_1b_biencoder/checkpoints
  model_save_format: safetensors
  save_consolidated: true

distributed:
  strategy: fsdp2
  dp_size: none
  tp_size: 1
  cp_size: 1
  sequence_parallel: false
```

- Example configs: [`examples/biencoder/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/biencoder)
- Recipe source: [`nemo_automodel/recipes/biencoder/train_biencoder.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/biencoder/train_biencoder.py)
- Dataset guide: [Biencoder Retrieval Dataset](../llm/retrieval-dataset.md)

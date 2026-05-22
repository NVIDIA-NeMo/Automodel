# Retrieval Fine-Tuning (Bi-Encoder and Cross-Encoder)

## Introduction

Retrieval models optimize a model for search, retrieval-augmented generation (RAG), semantic similarity, and reranking.
NeMo AutoModel provides two retrieval fine-tuning recipes:

- **Bi-encoder fine-tuning** trains one encoder to produce query and passage embeddings. Use it when you need fast
  nearest-neighbor search over a document index.
- **Cross-encoder fine-tuning** trains a reranker that scores a query and passage together. Use it after a retriever has
  produced a shortlist and you want stronger ranking quality.

Both recipes use retrieval examples where the first passage is positive and the remaining passages are negatives. A
common workflow is to train a bi-encoder, use it to mine harder negatives, then train either a stronger bi-encoder or a
cross-encoder reranker.

## Workflow Overview

Most retrieval projects move through the same loop:

```text
Prepare retrieval data
  -> Train a bi-encoder
  -> Validate candidate-group ranking quality
  -> Mine hard negatives
  -> Retrain the bi-encoder or train a cross-encoder reranker
  -> Use the consolidated checkpoint for indexing, mining, reranking, or serving
```

Start with a bi-encoder when you need embeddings for approximate nearest-neighbor search. Add hard-negative mining after
the first pass if the model mostly sees easy negatives. Train a cross-encoder when a separate retriever already produces
a small candidate set and you want a stronger reranking stage.

## Quickstart

Before running the examples:

- Use an AutoModel environment with the full GPU training dependencies installed. The NGC container is the safest path
  for multi-GPU runs.
- From a source checkout, use `uv run automodel ...`; from an installed environment, use `automodel ...`.
- Accept access terms for the configured Hugging Face model and set `HF_TOKEN`, or replace the model path with a model
  your environment can download.
- Make sure every rank can read the dataset paths or `hf://` sources.

Run the Llama 3.2 1B bi-encoder example:

```bash
automodel examples/retrieval/bi_encoder/llama3_2_1b.yaml --nproc-per-node 8
```

Run the matching cross-encoder example:

```bash
automodel examples/retrieval/cross_encoder/llama3_2_1b.yaml --nproc-per-node 8
```

Adjust `--nproc-per-node` to the number of GPUs on your machine. The examples use FSDP2 and bfloat16 by default. If the
first launch downloads a model, reads from a slow filesystem, or runs across multiple nodes, increase
`dist_env.timeout_minutes` beyond the short example default.

:::{tip}
For a small smoke test, override the schedule and sample count from the command line:

```bash
automodel examples/retrieval/bi_encoder/llama3_2_1b.yaml --nproc-per-node 1 \
  --step_scheduler.global_batch_size 4 \
  --step_scheduler.local_batch_size 1 \
  --step_scheduler.max_steps 10 \
  --dataloader.dataset.max_train_samples 40
```
:::

## Choose a Recipe

| Need | Use | Why |
|------|-----|-----|
| Search across a large corpus | Bi-encoder | Encodes queries and passages independently, so passage embeddings can be indexed once. |
| RAG candidate generation | Bi-encoder | Produces dense vectors for approximate nearest-neighbor retrieval. |
| Better ranking for a small shortlist | Cross-encoder | Scores each query-passage pair jointly, which is slower but usually more accurate. |
| Better negatives for the next training run | Hard-negative mining | Uses a trained bi-encoder to find confusing passages for each query. |

| Component | Bi-encoder | Cross-encoder |
|-----------|------------|---------------|
| Recipe | `TrainBiEncoderRecipe` | `TrainCrossEncoderRecipe` |
| Model target | `nemo_automodel.NeMoAutoModelBiEncoder.from_pretrained` | `nemo_automodel.NeMoAutoModelCrossEncoder.from_pretrained` |
| Dataset mode | `model_type: bi_encoder` | `model_type: cross_encoder` |
| Collator | `BiEncoderCollator` | `CrossEncoderCollator` |
| Training objective | Cross entropy over one positive plus negatives | Cross entropy over one positive plus negatives |

The bi-encoder computes a query embedding and passage embeddings independently. The cross-encoder formats each
query-passage pair into one sequence and predicts a score for each candidate passage.

## Prepare Data

Use the retrieval dataset format described in [Retrieval Dataset](retrieval-dataset.md). Each record needs:

- a query, stored as `question` or `query`
- at least one positive passage in `pos_doc`
- negatives in `neg_doc` when `n_passages > 1`

For quick custom experiments, inline JSONL is the simplest format. Use the inline dataset factory for these files:

```json
{"query":"What does NVLink do?","pos_doc":"NVLink is a high-bandwidth GPU interconnect.","neg_doc":["CUDA is a programming model.","Tensor Cores accelerate matrix math."]}
{"query":"What is retrieval augmented generation?","pos_doc":"RAG grounds generation by retrieving relevant context.","neg_doc":["Beam search expands candidate tokens.","Dropout regularizes training."]}
```

For larger corpora, use the corpus ID-based JSON format from the dataset guide. Use
`nemo_automodel.components.datasets.llm.make_retrieval_dataset` for corpus ID-based JSON and for `hf://` sources that
already follow the AutoModel retrieval schema, such as:

```yaml
data_dir_list:
  - hf://nvidia/embed-nemotron-dataset-v1/FEVER
  - hf://nvidia/embed-nemotron-dataset-v1/SyntheticClassificationData
```

`n_passages` controls how many passages are sampled for each query. For example, `n_passages: 5` means one positive
and four negatives. If a record has fewer negatives than requested, negatives are repeated to fill the group.

## Minimal Config Anatomy

This minimal bi-encoder config shows the pieces that must be present in a runnable retrieval fine-tuning job. The
sections below explain the model-specific parts in more detail.

```yaml
recipe: TrainBiEncoderRecipe
seed: 42
temperature: 0.02

step_scheduler:
  global_batch_size: 4
  local_batch_size: 1
  max_steps: 10
  ckpt_every_steps: 10
  val_every_steps: 10
  num_epochs: 1

dist_env:
  backend: nccl
  timeout_minutes: 30

model:
  _target_: nemo_automodel.NeMoAutoModelBiEncoder.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
  pooling: avg
  l2_normalize: true
  torch_dtype: bfloat16

tokenizer:
  _target_: nemo_automodel.NeMoAutoTokenizer.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
  add_eos_token: false

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  dataset:
    _target_: nemo_automodel.components.datasets.llm.retrieval_dataset_inline.make_retrieval_dataset
    model_type: bi_encoder
    data_dir_list:
      - /path/to/train.jsonl
    data_type: train
    n_passages: 5
    seed: 42
    do_shuffle: true
    max_train_samples: 40
  collate_fn:
    _target_: nemo_automodel.components.datasets.llm.BiEncoderCollator
    q_max_len: 512
    p_max_len: 512
    query_prefix: "query:"
    passage_prefix: "passage:"
    pad_to_multiple_of: 8
  shuffle: true
  num_workers: 0

optimizer:
  _target_: torch.optim.AdamW
  lr: 5.0e-6
  weight_decay: 0.01

lr_scheduler:
  lr_warmup_steps: 2
  lr_decay_style: linear

checkpoint:
  enabled: true
  checkpoint_dir: ./output/llama3_2_1b_encoder/checkpoints
  model_save_format: safetensors
  save_consolidated: true

distributed:
  strategy: fsdp2
  dp_size: none
  tp_size: 1
  cp_size: 1
```

For a cross-encoder, change `recipe`, `model._target_`, `dataloader.dataset.model_type`, and `dataloader.collate_fn`
to the cross-encoder values shown below.

## Configure a Bi-Encoder

A bi-encoder config has four important parts: the model, tokenizer, retrieval dataset, and `BiEncoderCollator`. This
snippet is an excerpt; keep the scheduler, optimizer, checkpoint, and distributed sections from the full config or one
of the examples.

```yaml
recipe: TrainBiEncoderRecipe

temperature: 0.02

model:
  _target_: nemo_automodel.NeMoAutoModelBiEncoder.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
  pooling: avg
  l2_normalize: true
  torch_dtype: bfloat16

tokenizer:
  _target_: nemo_automodel.NeMoAutoTokenizer.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
  add_eos_token: false

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  dataset:
    _target_: nemo_automodel.components.datasets.llm.retrieval_dataset_inline.make_retrieval_dataset
    model_type: bi_encoder
    data_dir_list:
      - /path/to/train.jsonl
    data_type: train
    n_passages: 5
    seed: 42
    do_shuffle: true
  collate_fn:
    _target_: nemo_automodel.components.datasets.llm.BiEncoderCollator
    q_max_len: 512
    p_max_len: 512
    query_prefix: "query:"
    passage_prefix: "passage:"
    pad_to_multiple_of: 8
  shuffle: true
  num_workers: 0
```

Important knobs:

- `pooling`: controls how token hidden states become one embedding. Common choices are `avg`, `cls`, and `last`.
- `l2_normalize`: normalizes embeddings before scoring. When enabled, the recipe divides scores by `temperature`.
- `q_max_len` and `p_max_len`: set separate truncation lengths for queries and passages.
- `query_prefix` and `passage_prefix`: add task-specific text before tokenization. Keep these aligned between training,
  hard-negative mining, and inference.
- `do_distributed_inbatch_negative`: optional model setting that treats passages from other data-parallel ranks as
  additional negatives. It is useful for larger distributed runs and uses document IDs from corpus-backed datasets to
  avoid treating the same document as a negative for itself. It all-gathers passage embeddings across data-parallel
  ranks, so expect extra communication and score-matrix memory. Use corpus-backed data when you need same-document
  masking, and keep it disabled for ColBERT-style pooling.

The complete example is
[`examples/retrieval/bi_encoder/llama3_2_1b.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/retrieval/bi_encoder/llama3_2_1b.yaml).

## Configure a Cross-Encoder

A cross-encoder config uses the same retrieval dataset factory, but sets `model_type: cross_encoder` and uses
`CrossEncoderCollator`. The dataset transform flattens each query with its positive and negative passages so the model
scores each query-passage pair. This snippet is an excerpt; keep the same scheduler, optimizer, checkpoint, and
distributed structure as the bi-encoder config.

```yaml
recipe: TrainCrossEncoderRecipe

model:
  _target_: nemo_automodel.NeMoAutoModelCrossEncoder.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
  num_labels: 1
  pooling: avg
  temperature: 1.0
  torch_dtype: bfloat16

tokenizer:
  _target_: nemo_automodel.NeMoAutoTokenizer.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
  add_eos_token: false

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  dataset:
    _target_: nemo_automodel.components.datasets.llm.retrieval_dataset_inline.make_retrieval_dataset
    model_type: cross_encoder
    data_dir_list:
      - /path/to/train.jsonl
    data_type: train
    n_passages: 5
    seed: 42
    do_shuffle: true
  collate_fn:
    _target_: nemo_automodel.components.datasets.llm.CrossEncoderCollator
    rerank_max_length: 512
    prompt_template: "question:{query} \n \n passage:{passage}"
    pad_to_multiple_of: 8
  shuffle: true
  num_workers: 0
```

Important knobs:

- `rerank_max_length`: maximum combined query-passage sequence length.
- `prompt_template`: controls how the pair is serialized before tokenization. It must include `{query}` and `{passage}`.
- `n_passages`: number of candidates per query. The positive passage must remain first in each group because labels point
  to index `0`.

The complete example is
[`examples/retrieval/cross_encoder/llama3_2_1b.yaml`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/examples/retrieval/cross_encoder/llama3_2_1b.yaml).

## Distributed Launch and Batch Size

Launch examples with `automodel <config.yaml> --nproc-per-node <gpus>`. The retrieval recipes support data-parallel
training through the configured distributed strategy; pipeline parallelism is not supported for encoder recipes today.

The step scheduler computes gradient accumulation from:

```text
gradient_accumulation_steps = global_batch_size / (local_batch_size * data_parallel_size)
```

`global_batch_size` must be divisible by `local_batch_size * data_parallel_size`, and the result must be at least `1`.
For memory pressure, reduce `step_scheduler.local_batch_size` first, then sequence lengths (`q_max_len`, `p_max_len`, or
`rerank_max_length`), then `n_passages`. Bi-encoders scale memory with query length plus `n_passages` passage sequences;
cross-encoders scale with `n_passages` combined query-passage sequences.

## Add Validation

Both examples include a commented `validation_dataloader` block. Enable it when you have a held-out retrieval file:

```yaml
validation_dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  dataset:
    _target_: nemo_automodel.components.datasets.llm.retrieval_dataset_inline.make_retrieval_dataset
    model_type: bi_encoder
    data_dir_list:
      - /path/to/validation.jsonl
    data_type: eval
    n_passages: 5
    seed: 42
    do_shuffle: false
  collate_fn:
    _target_: nemo_automodel.components.datasets.llm.BiEncoderCollator
    q_max_len: 512
    p_max_len: 512
    query_prefix: "query:"
    passage_prefix: "passage:"
    pad_to_multiple_of: 8
  batch_size: 2
  shuffle: false
  num_workers: 0
```

Validation logs `val_loss`, `val_acc1`, and `val_mrr` to `validation.jsonl` under `checkpoint.checkpoint_dir`. These
metrics measure ranking within each candidate group in the validation file; they are not full-corpus Recall@K or nDCG
metrics. For cross-encoder validation, use `model_type: cross_encoder` and `CrossEncoderCollator` instead.

```bash
tail -n 5 ./output/llama3_2_1b_encoder/checkpoints/validation.jsonl
```

## Monitor Training

Training metrics are written to `training.jsonl` under `checkpoint.checkpoint_dir`. Watch `loss`, `grad_norm`, learning
rate, GPU memory, and step time during smoke tests before scaling to a longer run:

```bash
tail -f ./output/llama3_2_1b_encoder/checkpoints/training.jsonl
```

The examples include a commented `wandb` block. Enable it when you want remote tracking, and tune
`step_scheduler.log_remote_every_steps` to control remote logging cadence.

## Enable LoRA

Retrieval recipes support the same PEFT block used by other AutoModel fine-tuning recipes. Uncomment or add `peft` to
train LoRA adapters instead of updating every weight:

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
  exclude_modules: []
  match_all_linear: true
  dim: 16
  alpha: 32
  use_triton: true
```

Use LoRA when you need lower memory use or want to ship a small adapter. Use full fine-tuning when you can afford the
memory and want maximum adaptation.

## Mine Hard Negatives

After an initial bi-encoder run, mine harder negatives with the consolidated encoder checkpoint:

```bash
torchrun --nproc_per_node=8 examples/retrieval/data_utils/mine_hard_negatives.py \
  --config examples/retrieval/data_utils/mining_config.yaml \
  --mining.model_name_or_path ./output/llama3_2_1b_encoder/checkpoints/LATEST/model/consolidated \
  --mining.train_qa_file_path /path/to/input.json \
  --mining.train_file_output_path /path/to/output.json \
  --mining.cache_embeddings_dir /path/to/cache \
  --mining.query_prefix "query:" \
  --mining.passage_prefix "passage:" \
  --mining.query_max_length 512 \
  --mining.passage_max_length 512
```

Hard-negative mining expects the corpus ID-based retrieval JSON format described in the dataset guide, not the inline
JSONL shortcut. The input must reference one corpus so the miner can build a passage embedding cache, retrieve
candidates, and write mined negatives back to each query.

Key mining settings in `examples/retrieval/data_utils/mining_config.yaml`:

- `hard_negatives_to_mine`: number of negatives to add per query.
- `hard_neg_margin` and `hard_neg_margin_type`: filter near-positive candidates.
- `query_prefix` and `passage_prefix`: keep these consistent with the bi-encoder training config.
- `query_max_length` and `passage_max_length`: keep these consistent with training unless you intentionally change
  truncation.
- `use_negatives_from_file`: include existing negatives from the input file when mining.
- `cache_embeddings_dir`: required for distributed mining so ranks can share cached passage embeddings. Rank `0`
  assembles the final embedding cache and score outputs, so plan memory and local disk accordingly.

Use the mined output as the next `data_dir_list` source for another bi-encoder pass or for cross-encoder training.

## Save, Resume, and Use the Checkpoint

Set checkpointing in the config:

```yaml
checkpoint:
  enabled: true
  checkpoint_dir: ./output/llama3_2_1b_encoder/checkpoints
  model_save_format: safetensors
  save_consolidated: true
```

Each save creates a versioned directory such as:

```text
./output/llama3_2_1b_encoder/checkpoints/epoch_0_step_500/
```

With `save_consolidated: true`, AutoModel also writes a Hugging Face-compatible model under:

```text
./output/llama3_2_1b_encoder/checkpoints/epoch_0_step_500/model/consolidated/
```

The `LATEST` symlink points to the most recent checkpoint. To resume from the latest compatible checkpoint, set:

```yaml
checkpoint:
  enabled: true
  checkpoint_dir: ./output/llama3_2_1b_encoder/checkpoints
  restore_from: LATEST
```

Use a unique `checkpoint_dir` per experiment unless you intentionally want to resume an existing run.

## Use the Model

Use a bi-encoder checkpoint to encode passages, build an approximate nearest-neighbor index, encode queries, and search
the index. Keep the same tokenizer, pooling, normalization, prefixes, and max lengths that you used for training.

Use a cross-encoder checkpoint to rerank a shortlist from a retriever. Cross-encoders score each query-passage pair
jointly, so they are usually too expensive for first-stage full-corpus search.

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Training fails with empty negatives | Ensure every record has `neg_doc` when `n_passages > 1`. |
| Dataset records fail to load | Check the supported schemas in [Retrieval Dataset](retrieval-dataset.md). |
| Loss does not move | Verify the positive passage is first and negatives are not duplicates of the positive. |
| Poor retrieval quality | Mine harder negatives and align train/inference prefixes. |
| OOM at startup or first batch | Lower `local_batch_size`, `q_max_len`, `p_max_len`, or `rerank_max_length`; use LoRA for larger backbones. |
| Distributed launch times out | Increase `dist_env.timeout_minutes`, especially for first model downloads, slow filesystems, or multi-node runs. |
| Batch-size assertion fails | Set `global_batch_size` to a multiple of `local_batch_size * data_parallel_size`. |
| Different mining and training behavior | Match tokenizer settings, prefixes, and max lengths across training and mining. |

## Related Files

- Bi-encoder recipe: [`nemo_automodel/recipes/retrieval/train_bi_encoder.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/retrieval/train_bi_encoder.py)
- Cross-encoder recipe: [`nemo_automodel/recipes/retrieval/train_cross_encoder.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/retrieval/train_cross_encoder.py)
- Retrieval dataset guide: [Retrieval Dataset](retrieval-dataset.md)
- Llama-Embed-Nemotron-8B example:
  [`examples/retrieval/bi_encoder/llama_embed_nemotron_8b/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/retrieval/bi_encoder/llama_embed_nemotron_8b)

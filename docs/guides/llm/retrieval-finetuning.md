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
  your environment can download. Retrieval has custom bidirectional backbones for Llama and Ministral3 embedding
  models, and a custom Llama scoring backbone for cross-encoders. Other Hugging Face model types fall back to
  `AutoModel` for bi-encoders or `AutoModelForSequenceClassification` for cross-encoders; verify that the tokenizer,
  pooling, `num_labels`, and any retrieval-specific model arguments are accepted by the replacement model.
- Make sure every rank can read the dataset paths or `hf://` sources.

The commands below use `automodel`; if you are running from a source checkout, prefix them with `uv run`.

Start with a one-GPU smoke test:

```bash
automodel examples/retrieval/bi_encoder/llama3_2_1b.yaml --nproc-per-node 1 \
  --dist_env.timeout_minutes 30 \
  --step_scheduler.global_batch_size 4 \
  --step_scheduler.local_batch_size 1 \
  --step_scheduler.max_steps 10 \
  --dataloader.dataset.max_train_samples 40
```

The first artifact to check is `training.jsonl` under `checkpoint.checkpoint_dir`. JSONL metrics are buffered, so
stdout/stderr are still the best live signal during a very short run.

Scale the Llama 3.2 1B bi-encoder example to the GPUs on your machine:

```bash
automodel examples/retrieval/bi_encoder/llama3_2_1b.yaml --nproc-per-node 8
```

Run the matching cross-encoder example:

```bash
automodel examples/retrieval/cross_encoder/llama3_2_1b.yaml --nproc-per-node 8
```

Adjust `--nproc-per-node` to the number of GPUs on your machine. The examples use FSDP2 and bfloat16 by default.

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

Use the retrieval dataset format described in [Retrieval Dataset](retrieval-dataset.md). Choose the data path that
matches the workflow you need:

| Data path | Use when | Notes |
|-----------|----------|-------|
| `hf://` AutoModel retrieval schema | You want a tutorial run or shared public dataset | Requires an AutoModel-style HF subset with a companion corpus split. |
| Inline JSONL | You want a small custom run without hard-negative mining | Documents are embedded directly in each record; no document IDs are available for same-document masking. |
| Corpus ID-based JSON | You need hard-negative mining, reusable corpora, or same-document masking | Records reference document IDs in a local corpus that can be loaded by Hugging Face `datasets`. |

The key field requirements differ by source:

| Source | Required query field | Required document fields |
|--------|----------------------|--------------------------|
| Corpus ID JSON | `question` | `question_id`, `corpus_id`, non-empty `pos_doc`, and present `neg_doc` |
| `hf://` AutoModel schema | `question` | non-empty `pos_doc`; `neg_doc` is optional in the source but required before training with negatives |
| Inline JSONL | `query` or `question` | non-empty `pos_doc`, and present `neg_doc` |

`neg_doc` must be present for local JSON and JSONL sources. It may be `[]` only when `n_passages: 1`; when
`n_passages > 1`, provide at least one negative.

For quick custom experiments, inline JSONL is the simplest format. Use the inline dataset factory for these files, and
switch to corpus ID-based JSON before hard-negative mining or full-corpus evaluation:

```json
{"query":"What does NVLink do?","pos_doc":"NVLink is a high-bandwidth GPU interconnect.","neg_doc":["CUDA is a programming model.","Tensor Cores accelerate matrix math."]}
{"query":"What is retrieval augmented generation?","pos_doc":"RAG grounds generation by retrieving relevant context.","neg_doc":["Beam search expands candidate tokens.","Dropout regularizes training."]}
```

To migrate inline data to corpus ID JSON, assign a stable document ID to each unique passage, write those passages into
a corpus split with `id` and `text` columns, then replace inline `pos_doc` and `neg_doc` strings with those IDs. Keep
all known positives for each query in your qrels or source metadata, even if each training row uses only the first
positive. Otherwise, in-batch negatives and mined hard negatives can accidentally treat another relevant passage as a
negative.

For larger corpora, use the corpus ID-based JSON format from the dataset guide. Use
`nemo_automodel.components.datasets.llm.make_retrieval_dataset` for corpus ID-based JSON and for `hf://` sources that
already follow the AutoModel retrieval schema, such as:

```yaml
data_dir_list:
  - hf://nvidia/embed-nemotron-dataset-v1/FEVER
  - hf://nvidia/embed-nemotron-dataset-v1/SyntheticClassificationData
```

`n_passages` controls the size of each query group. For example, `n_passages: 5` means one positive and four negatives.
Training uses the first item in `pos_doc` as the positive, then takes negatives from `neg_doc` in order. If a record has
fewer negatives than requested, negatives are repeated cyclically to fill the group. Treat repetition as a fallback for
shape compatibility; for real training and validation, prefer enough distinct negatives or lower `n_passages`.

The training recipe does not load a separate qrels file. Materialize qrels into retrieval records before fine-tuning.
For training, `pos_doc[0]` is the supervised positive. For mining, keep every known positive for the query in `pos_doc`
so the miner can exclude those IDs; it does not read an external qrels file. If you expand multi-positive queries into
one row per positive, make sure sibling positives are removed from `neg_doc` and audited out of mined negatives before
training with in-batch negatives. Keep the original qrels for offline Recall@K, MRR@K, and nDCG@K evaluation.

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
to the cross-encoder values shown below. Also set `model.num_labels: 1`, keep `model.temperature`, and replace
`q_max_len` / `p_max_len` with `rerank_max_length` in the collator.

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
  additional negatives. Enable it with `model.do_distributed_inbatch_negative: true` or the CLI override
  `--model.do_distributed_inbatch_negative true`. Today it all-gathers over the default process group, so use it only
  for pure DP/FSDP retrieval runs (`tp_size: 1`, `cp_size: 1`). Same-document masking requires `doc_id` fields from
  corpus-backed or custom datasets; inline JSONL does not provide duplicate-document masking. Keep it disabled for
  ColBERT-style pooling.

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

Launch single-node examples with `automodel <config.yaml> --nproc-per-node <gpus>`. The retrieval recipes support
data-parallel training through the configured distributed strategy; pipeline parallelism is not supported for encoder
recipes today.

For multi-node runs, launch with your cluster launcher or an external `torchrun` command so every node has an explicit
rank and rendezvous endpoint:

```bash
torchrun \
  --nnodes 2 \
  --nproc-per-node 8 \
  --node-rank ${NODE_RANK} \
  --rdzv-backend c10d \
  --rdzv-endpoint ${MASTER_ADDR}:${MASTER_PORT} \
  -m nemo_automodel.cli.app examples/retrieval/bi_encoder/llama3_2_1b.yaml
```

Use a shared or pre-populated Hugging Face cache on every node, make dataset paths visible to every rank, and use a
unique `checkpoint_dir` for each experiment. Increase `dist_env.timeout_minutes` for first model downloads, slow shared
filesystems, multi-node collectives, or large checkpoint writes.

The step scheduler computes gradient accumulation from:

```text
gradient_accumulation_steps = global_batch_size / (local_batch_size * data_parallel_size)
```

`global_batch_size` must be divisible by `local_batch_size * data_parallel_size`, and the result must be at least `1`.
In pure data parallelism, `data_parallel_size` is the total GPU count. With tensor or context parallelism enabled, it is
`world_size / (tp_size * cp_size)`. For example, two 8-GPU nodes with `tp_size: 1` and `cp_size: 1` have
`data_parallel_size: 16`; with `tp_size: 2`, they have `data_parallel_size: 8`.

`local_batch_size` is the number of query groups per rank. For memory pressure, reduce
`step_scheduler.local_batch_size` first, then sequence lengths (`q_max_len`, `p_max_len`, or `rerank_max_length`), then
`n_passages`. Bi-encoders scale memory with query length plus `local_batch_size * n_passages` passage sequences;
cross-encoders scale with `local_batch_size * n_passages` combined query-passage sequences.

Current retrieval datasets are map-style datasets loaded in each process, not streaming distributed inputs. Pre-cache
HF data on each node or use a shared cache, and budget CPU RAM and local disk per rank for corpus-backed datasets.

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

## Evaluate Retrieval Quality

Candidate-group validation is a smoke test for the training objective. To decide whether a bi-encoder is useful for RAG
candidate generation, evaluate against a fixed held-out corpus and qrels:

1. Encode corpus passages with the same tokenizer, pooling, normalization, passage prefix, and `p_max_len` used in
   training.
2. Build an ANN or exact top-k index. With `l2_normalize: true`, use inner product or cosine similarity.
3. Encode held-out queries with the matching query prefix and `q_max_len`.
4. Report full-corpus Recall@K, MRR@K, and nDCG@K for the K values your application uses.

AutoModel does not currently provide a one-command full-corpus retrieval evaluator in this guide. Use your existing IR
evaluation stack or a small script around the consolidated checkpoint and report enough run details to make the result
repeatable: query count, corpus size, qrels source, judged/unjudged handling, exact versus ANN search settings, K
values, baseline checkpoint, and whether confidence intervals or significance tests were used.

For cross-encoders, freeze a first-stage retriever, rerank its top-K candidates, and report reranking metrics on that
same candidate set. Do not compare cross-encoder candidate-group validation directly to full-corpus bi-encoder metrics.

## Monitor Training

Training metrics are written to `training.jsonl` under `checkpoint.checkpoint_dir`. The file logger buffers records in
chunks before writing and flushes the remaining records on close, so `tail -f` is useful for completed or longer runs
but may not update during a short smoke test:

```bash
tail -f ./output/llama3_2_1b_encoder/checkpoints/training.jsonl
```

Use stdout/stderr as the live per-step signal today. Watch `loss`, `grad_norm`, learning rate, GPU memory, and step time
before scaling to a longer run. On preempted or timed-out jobs, recent buffered JSONL metrics may be missing even when
stdout/stderr showed them.

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
  --mining.query_prefix "query: " \
  --mining.passage_prefix "passage: " \
  --mining.query_max_length 512 \
  --mining.passage_max_length 512
```

Hard-negative mining expects the corpus ID-based retrieval JSON format described in the dataset guide, not the inline
JSONL shortcut. The input must reference one corpus so the miner can build a passage embedding cache, retrieve
candidates, and write mined negatives back to each query.

Key mining settings in `examples/retrieval/data_utils/mining_config.yaml`:

- `hard_negatives_to_mine`: number of negatives to add per query.
- `hard_neg_margin` and `hard_neg_margin_type`: filter near-positive candidates. With `hard_neg_margin_type: perc`,
  candidates scoring above `min_positive_score * hard_neg_margin` are removed; with `abs`, candidates scoring above
  `min_positive_score - hard_neg_margin` are removed. Inspect mined samples when positive scores are low or negative.
- `query_prefix` and `passage_prefix`: keep these semantically consistent with the bi-encoder training config. The
  miner concatenates prefixes directly, while `BiEncoderCollator` inserts a space after non-empty prefixes; include the
  trailing space in mining prefixes.
- `query_max_length` and `passage_max_length`: keep these consistent with training unless you intentionally change
  truncation.
- `use_negatives_from_file`: include existing negatives from the input file when mining.
- `cache_embeddings_dir`: required for distributed mining so ranks can share cached passage embeddings. Rank `0`
  assembles the final embedding cache and score outputs, so plan memory and local disk accordingly. In multi-node
  mining, this must be a shared writable path mounted at the same location on every node; node-local cache paths leave
  rank `0` unable to read remote-rank shards.

Use the mined output as the next `data_dir_list` source for another bi-encoder pass or for cross-encoder training. Hard
negative mining excludes document IDs listed in each input row's `pos_doc`, but it cannot read an external qrels file or
know every semantically relevant duplicate. Put all known positive IDs for the query in the mining input, deduplicate the
corpus, inspect mined samples, and avoid mining from validation or test corpora.

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
./output/llama3_2_1b_encoder/checkpoints/epoch_0_step_499/
```

Checkpoint directory names use the scheduler step at save time. The saved scheduler state advances to the next step, so
for exact paths prefer the `Saving checkpoint to ...` log line or the `LATEST` pointer over hand-constructing a step
number.

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

If `checkpoint.restore_from` is omitted, AutoModel still auto-detects the latest compatible checkpoint in
`checkpoint_dir` and resumes from it. Use a new or empty `checkpoint_dir` for fresh experiments, and rotate or clear
`training.jsonl` and `validation.jsonl` if you do not want logs from multiple runs appended together.

When `checkpoint.is_async: true`, the `LATEST` symlink can lag the most recent write at job end. For final mining,
export, or evaluation workflows, prefer the explicit `epoch_*_step_*` checkpoint directory or keep async checkpointing
disabled for the final save.

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
| `training.jsonl` does not update during a smoke test | Use stdout/stderr for live monitoring; JSONL metrics are buffered before flush. |
| Run resumes unexpectedly | Use a new or empty `checkpoint_dir`; AutoModel auto-detects compatible checkpoints when `restore_from` is omitted. |
| Different mining and training behavior | Match tokenizer settings, max lengths, and prefix text including trailing spaces across training and mining. |

## Related Files

- Bi-encoder recipe: [`nemo_automodel/recipes/retrieval/train_bi_encoder.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/retrieval/train_bi_encoder.py)
- Cross-encoder recipe: [`nemo_automodel/recipes/retrieval/train_cross_encoder.py`](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/recipes/retrieval/train_cross_encoder.py)
- Retrieval dataset guide: [Retrieval Dataset](retrieval-dataset.md)
- Llama-Embed-Nemotron-8B example:
  [`examples/retrieval/bi_encoder/llama_embed_nemotron_8b/`](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/retrieval/bi_encoder/llama_embed_nemotron_8b)

# Zephon and NeMo AutoModel

[Zephon](https://zephon.io/) is a data loader for training at scale. It reads
and mixes datasets, prepares and packs documents, and keeps data delivery
deterministic across checkpoints and changes in data-parallel world size.
AutoModel owns the model, optimizer, and training loop.

This guide explains how to use Zephon with AutoModel's training recipes. For
Zephon concepts, pipeline operations, and API questions, see the
[Zephon User Guide](https://docs.zephon.io/) and
[API reference](https://docs.zephon.io/api/pipeline.html).

## Get Started

Start with an AutoModel checkout and a CUDA training environment, following the
[AutoModel installation instructions](../../../README.md#getting-started).
Zephon is included in this checkout's Python dependencies. Installing the project
also installs Zephon 0.2.1 or later from PyPI.

The small text example creates a local tokenizer and corpus, then trains a tiny
Llama model from random weights. Run both commands from the repository root:

```bash
uv run python examples/llm_pretrain/zephon/prepare_tiny.py
uv run automodel examples/llm_pretrain/zephon/tiny.yaml --nproc-per-node 1
```

Training requires one CUDA GPU. The corpus is finite: the recipe reads it once
and stops when it runs out. `step_scheduler.max_steps` is an upper bound, so
there is no need to repeat a dataset to reach it. Incomplete final microbatches
are dropped.

## Configure Your Data

The source config chooses tokenization: raw text or instructions are tokenized
online, while pretokenized input keeps its supplied token IDs. Independently,
select a `packing.pack_algorithm` to pack online or set it to `null` to preserve
each supplied sequence. This supports online tokenization and packing, offline
tokenization with online packing, and fully prepacked input.

Choose a Zephon dataset config for the kind of documents you have. AutoModel
selects its matching `ZephonDataloaderConfig` automatically when
`dataloader._target_` is omitted. This thin adapter supplies Zephon's packing
metadata through the existing dataloader interface; the shared build path
delegates complete stream construction to the dataset config. An explicit
loader target takes precedence and must be compatible with the dataset.
For raw text:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.zephon.ZephonTextDatasetConfig
  tokenize_special_tokens: eos
  tokenizer:
    pretrained_model_name_or_path: /path/to/model-or-tokenizer
  zephon:
    sources:
    - name: books
      path: /data/books
      fmt: jsonl
      weight: 0.7
    - name: web
      path: /data/web
      fmt: jsonl
      weight: 0.3
    packing:
      seq_len: 2048
    checkpoint:
      run_id: my-training-run
```

Here each source is a directory of JSONL shards containing documents such as
`{"text": "Some training text."}`. Zephon also accepts cloud sources through
its dataset URI syntax. Text token estimation defaults to `auto`: it is enabled
when there are multiple sources, so weights describe token proportions rather
than document proportions. Set `source.token_estimation: false` to disable it,
`true` or a settings mapping to enable it explicitly, or `null` to use Zephon's
sample-based default. This applies to raw text, instructions, and pretokenized input. See
[mixing datasets](https://docs.zephon.io/worksources/mixing_datasets.html) for
calibration and sampling behavior.

`packing.seq_len` is the model input length. When omitted, it uses the effective
model config's input-token limit. Set it explicitly for shorter training inputs
or when the model has no known limit. A length above a known model limit is
rejected, including when positions reset between subdocuments. Zephon prepares
one extra token for next-token prediction and handles the shift between input and target.
`wrap` can split long text documents; `first_fit` and `best_fit` keep documents whole. With bin packing,
oversized documents are dropped by default; `drop_oversized: false` raises an
error instead. See the [packing guide](https://docs.zephon.io/pipelines/packing.html)
for the algorithms and their options.

The `source`, `packing`, `training`, `workers`, `runtime`, `cache`, and `checkpoint`
groups keep related options together. For example, `workers.tokenize_parallelism`
controls tokenization workers, while `source.stop_after_passes` controls when
reading ends. Options mostly map to the corresponding Zephon pipeline settings;
the [pipeline guide](https://docs.zephon.io/pipelines/index.html) explains how
those settings affect delivery. Concrete config values are integration defaults;
unset values follow the field's documented policy. For example, omitted packing
lengths use the model limit, omitted checkpoint lane counts use the DP world size,
and worker counts set to `null` use the Zephon operator default. With no exhaustion
overrides, Zephon reads a finite source once. Distributed checkpoint aggregation
uses `torch://` automatically unless a store or directory is supplied; choose a
unique `checkpoint.run_id` for each distributed job. Model, optimizer, and
distributed settings stay in their usual AutoModel sections.

Runtime buffering uses Zephon's automatic defaults. Set `runtime.prefetch_batches`,
`runtime.default_stage_prefetch`, or `runtime.op_queue_capacity` to tune pipeline
queues, and `runtime.mtp_buffer` or `runtime.mtp_prefetch` for MTP buffering.
Leaving a value unset delegates its sizing to Zephon; `mtp_prefetch: 0` disables
the background receiver thread.

Source/cache lookahead is a separate optional operator. Set `source.prefetch: {}`
to enable it with Zephon's defaults, or provide `buffer_size` and `parallelism`
inside that mapping to tune it. Omit the group or set it to `null` to leave the
operator out; runtime buffering remains enabled independently.

For remote sources, configure a node-local shard cache. Its disk budget and the
Parquet decoded row-group cache budget are independent; neither bounds pipeline RAM:

```yaml
cache:
  cache_dir: /local_nvme/zephon
  cache_limit_bytes: 80gb
  parquet_rg_cache:
    limit_bytes: 8gb
```

These settings live under `dataset.zephon` and apply to text and images.
`parquet_rg_cache` also accepts `enabled`, `root`, and `min_free_bytes`; omitted
fields use Zephon's defaults. Set `enabled: false` to disable decoded caching.
See [storage and shard caching](https://docs.zephon.io/datasets/storage_backends_and_shard_cache.html)
for cache sizing and the [Parquet guide](https://docs.zephon.io/datasets/parquet.html)
for decoded-cache behavior.

You can also place the data settings in a TOML file and reference it with
`dataset.zephon.data_config`. Inline YAML values override that file; relative
source, cache, and checkpoint paths in TOML are resolved relative to the TOML file.

To reuse native source fields, select a Zephon dataset config and supply one
`dataset.path_or_dataset_id` or `dataset.paths` value instead of `zephon.sources`.
A string or a list containing one path is accepted. A scalar
`path_or_dataset_id: org/dataset` selects a Hugging Face dataset, with `name`
for its subset and a simple named `split` (default `train`). Zephon reads the
uploaded files or its supported Parquet conversion; Hugging Face dataset-card
reader options and feature casting are not applied. Explicit Zephon URIs also
work; encode their subset/split in the URI itself.

Local paths must identify a dataset root containing the intended shards.
Individual files, globs, Megatron binary prefixes and blend files require
migration; the adapter never substitutes a parent directory. Multiple native
paths are rejected because concatenation or natural corpus-size sampling does
not mean equal Zephon weights. Use explicit `zephon.sources` and choose weights
and token estimation for mixtures. Source declarations cannot be combined,
including sources imported from TOML. Unsupported native formatting, truncation
and sample-limit fields raise rather than being ignored.

### Use Pretokenized and Prepacked Text

Tokenization and online packing are independent choices:

| Supplied data | Dataset config | `packing.pack_algorithm` | Online preparation |
| --- | --- | --- | --- |
| Raw text | `ZephonTextDatasetConfig` | `wrap`, `first_fit`, `best_fit`, or `best_fit_wrap` | Tokenize and pack |
| Raw instructions | `ZephonInstructionDatasetConfig` | A packing algorithm | Chat tokenize and pack |
| Pretokenized documents | `ZephonPretokenizedTextDatasetConfig` | A packing algorithm | Pack only |
| Pretokenized, prepacked sequences | `ZephonPretokenizedTextDatasetConfig` | `null` | Preserve supplied sequences |
| Raw text or instructions | The matching raw dataset config | `null` | Tokenize each document separately |

For pretokenized data, `tokens_field` defaults to `input_ids`, and the optional
`loss_mask_field` defaults to `loss_mask`. Supply unshifted, unpadded token lists
from the model's tokenizer. The adapter validates fields, preserves supplied IDs
and binary loss masks, and adds an all-ones mask only when absent. It does not
retokenize or filter short or fully masked documents. Token-aware mixing counts
the supplied tokens, including tokens excluded from loss, using `tokens_field`.

For prepacked text, use the same source config and explicitly disable packing:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.zephon.ZephonPretokenizedTextDatasetConfig
  # Keep the model tokenizer configuration here for vocabulary validation.
  zephon:
    sources:
    - {name: offline-packs, path: /data/prepacked, fmt: jsonl}
    packing:
      pack_algorithm: null
      seq_len: 2048
```

Each supplied token sequence is one document unless it includes Zephon's
`positions` field: a token-aligned integer list starting at zero and increasing
by one, resetting to zero at every subdocument start. For example,
`{"input_ids": [2, 3, 4, 5], "positions": [0, 1, 0, 1]}` retains two subdocuments.
Without `positions`, even an internal EOS token does not imply a boundary.
Supplied positions require `pack_algorithm: null`, since the online packer
creates its own boundaries.

With packing disabled, each sequence must contain at most `seq_len + 1` tokens.
`to_training()` pads shorter sequences and shifts labels exactly once; padding
is excluded from attention and loss even when PAD equals EOS. Oversized inputs
raise instead of being truncated. `training.mask_segment_starts` still controls
boundary-target masking. This is a flat token-sequence format, not an importer
for framework cache files, source padding, cumulative-length tensors, or already
shifted labels. The same `batched` and `thd` output formats remain available.

### Configure Instruction Tuning

For question and answer data, select `ZephonInstructionDatasetConfig`:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.zephon.ZephonInstructionDatasetConfig
  path_or_dataset_id: your-org/instructions
  split: train
  column_mapping: {question: prompt, answer: completion}
  use_hf_chat_template: true
  tokenizer:
    pretrained_model_name_or_path: /path/to/model-or-tokenizer
```

This example reads the `prompt` and `completion` fields of each document.
AutoModel normalizes the columns, and Zephon's `tokenize_chat()` performs chat
tokenization and loss masking. The tokenizer's chat template supplies the
conversation format. Set `use_hf_chat_template: false` for plain prompt and completion
formatting.

Native dataset configs keep their native loader by default. Explicitly setting
`dataloader._target_` to
`nemo_automodel.components.datasets.zephon.ZephonDataloaderConfig` also accepts
`ColumnMappedTextInstructionDatasetConfig` and its dataset-class target. The
adapter translates the source, column mapping and formatting choices into
`ZephonInstructionDatasetConfig` without building the native dataset. Native
padding, truncation and sample limits must remain disabled; they are not
equivalent to Zephon packing or source limits. Unsupported datasets are rejected.
With `use_hf_chat_template: true`, the tokenizer must provide a chat template;
Zephon does not fall back to plain formatting when it is missing.

Use `ZephonInstructionDatasetConfig` directly to customize the Zephon pipeline.
Its loader is selected automatically, and options live under `dataset.zephon`.
Use the source fields above for a single dataset, or `dataset.zephon.sources`
for an explicit mixture.

The local example uses plain formatting and Flash Attention:

```bash
uv run python examples/llm_pretrain/zephon/prepare_tiny.py
uv run automodel examples/llm_pretrain/zephon/sft.yaml --nproc-per-node 1
```

Install the matching Flash Attention dependencies, or select Scaled Dot-Product Attention (SDPA) with
`--model.attn_implementation sdpa`.

## How the Integration Works

The dataset config selects text, chat, or pretokenized input. The Zephon
pipeline reads and mixes sources, prepares documents, packs them, and groups
them into batches; online packing can be disabled. Zephon's `to_training()` handles shifting, padding, and
collation. A small adapter then presents that result in the format expected by
AutoModel's existing recipe and model.

With Zephon selected, AutoModel's `dataset` section holds the complete stream
configuration. `zephon.io.Dataset` descriptors are created from
`dataset.zephon.sources`; Zephon owns reading, mixing, tokenization, packing and
batching. The ordinary AutoModel dataloader builder passes the runtime tokenizer,
seed, batch size and rank information into the selected config's `build()`.
Native sampling, worker, packing and collator settings cannot be combined with
this stream; use the corresponding `dataset.zephon` options instead.

`dataset.output_format` selects the model input layout. The default,
`batched`, uses ordinary `[batch, sequence]` inputs. `thd` supplies the sequence-length
metadata used by compatible native variable-length attention backends.
These are output formats, separate from the packing algorithm. Batched output
can also use Flash Attention; choose the format supported by your model and
attention backend.

By default, subdocuments within a packed document cannot attend to one another,
and positions reset at their boundaries. Set
`dataset.zephon.training.allow_cross_document_attention: true` to let them
attend across boundaries and use continuous positions within the packed
document. Attention remains causal, and different packed documents in a batch
remain independent. The separate `training.mask_segment_starts` option controls whether
next-token targets at subdocument boundaries contribute to the loss.

## Configure Checkpoints for Distributed Training

Zephon's loader state is saved and restored with the AutoModel training
checkpoint. Every training rank participates, including tensor parallelism (TP)
and pipeline parallelism (PP) peers.
Use a unique `dataset.zephon.checkpoint.run_id` for each independent run and
keep it unchanged when resuming that run.

To resume at different data-parallel sizes, set `dataset.zephon.checkpoint.canonical_replicas`
to a common multiple of those sizes before starting. The global number of
microbatches per optimizer step (`global_batch_size / local_batch_size`) must
also be a multiple of that value. Preserve the data configuration, tokenizer,
seed, and global batch size across resumes. See
[distributed training](https://docs.zephon.io/pipelines/distributed_training.html)
and [checkpointing and resuming](https://docs.zephon.io/pipelines/checkpointing_and_resuming.html)
for the underlying Zephon guarantees.

Data parallelism (DP), TP, and PP follow AutoModel's model and backend capabilities. Context
parallelism is not supported by this integration. Zephon currently supplies
training data only; AutoModel's existing validation loaders remain available.

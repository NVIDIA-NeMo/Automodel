# Zephon + NeMo AutoModel

[Zephon](https://zephon.io/) is a data loader for training at scale. It reads
and mixes datasets, prepares and packs documents, and keeps data delivery
deterministic across checkpoints and changes in data-parallel world size.
AutoModel owns the model, optimizer, and training loop.

This guide explains how to use Zephon with AutoModel's training recipes. For
Zephon concepts, pipeline operations, and API questions, see the
[Zephon User Guide](https://docs.zephon.io/) and
[API reference](https://docs.zephon.io/api/pipeline.html).

## Getting started

Start with an AutoModel checkout and a CUDA training environment, following the
[AutoModel installation instructions](../../../README.md#getting-started).
Zephon is included in this checkout's Python dependencies; installing the project
also installs the pinned Zephon revision.

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

## Configuring your data

Choose a dataset config for the kind of documents you have, then use the Zephon
dataloader config in your usual AutoModel recipe. For raw text:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.zephon.ZephonTextSourceConfig
  text_field: text
  tokenize_special_tokens: eos
  tokenizer:
    pretrained_model_name_or_path: /path/to/model-or-tokenizer

dataloader:
  _target_: nemo_automodel.components.datasets.zephon.ZephonDataloaderConfig
  output_format: batched
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
    source:
      token_estimation: true
      stop_after_passes: 1
    packing:
      seq_len: 2048
      pack_algorithm: wrap
    training:
      allow_cross_document_attention: false
    checkpoint:
      run_id: my-training-run
```

Here each source is a directory of JSONL shards containing documents such as
`{"text": "Some training text."}`. Zephon also accepts cloud sources through
its dataset URI syntax. With token estimation enabled, the source weights
describe token proportions rather than document proportions. See
[mixing datasets](https://docs.zephon.io/worksources/mixing_datasets.html) for
calibration and sampling behavior.

`packing.seq_len` is the model input length. Zephon prepares one extra token for
next-token prediction and handles the input/target shift. `wrap` can split long
text documents; `first_fit` and `best_fit` keep documents whole. With bin packing,
oversized documents are dropped by default; `drop_oversized: false` raises an
error instead. See the [packing guide](https://docs.zephon.io/pipelines/packing.html)
for the algorithms and their options.

The `source`, `packing`, `training`, `workers`, `runtime`, `cache`, and `checkpoint`
groups keep related options together. For example, `workers.tokenize_parallelism`
controls tokenization workers, while `source.stop_after_passes` controls when
reading ends. Options mostly map to the corresponding Zephon pipeline settings;
the [pipeline guide](https://docs.zephon.io/pipelines/index.html) explains how
those settings affect delivery. Model, optimizer, and distributed settings stay
in their usual AutoModel sections.

You can also place the data settings in a TOML file and reference it with
`dataloader.zephon.data_config`. Inline YAML values override that file; relative
source, cache, and checkpoint paths in TOML are resolved relative to the TOML file.

## How the integration works

The dataset config selects the text field and tokenizer settings. The Zephon
pipeline reads and mixes sources, tokenizes documents, packs them, and groups
them into batches. Zephon's `to_training()` handles shifting, padding, and
collation. A small adapter then presents that result in the format expected by
AutoModel's existing recipe and model.

With Zephon selected, AutoModel's `dataset` section describes document
preparation rather than constructing a native dataset object. The integration
translates supported settings into Zephon pipeline options. Zephon
reads `dataloader.zephon.sources` and owns mixing, tokenization, packing, and
batching. Configure source traversal and packing under `dataloader.zephon`;
arbitrary native dataset types and their sampling, truncation, or collator
behavior are not automatically converted.

`dataloader.output_format` selects the model input layout. The default,
`batched`, uses ordinary `[batch, sequence]` inputs. `thd` supplies the sequence
length metadata used by compatible native variable-length attention backends.
These are output formats, separate from the packing algorithm. Batched output
can also use Flash Attention; choose the format supported by your model and
attention backend.

By default, subdocuments within a packed document cannot attend to one another,
and positions reset at their boundaries. Set
`dataloader.zephon.training.allow_cross_document_attention: true` to let them
attend across boundaries and use continuous positions within the packed
document. Attention remains causal, and different packed documents in a batch
remain independent. The separate `training.mask_segment_starts` option controls whether
next-token targets at subdocument boundaries contribute to the loss.

## Checkpointing and distributed training

Zephon's loader state is saved and restored with the AutoModel training
checkpoint. Every training rank participates, including TP and PP peers.
Use a unique `dataloader.zephon.checkpoint.run_id` for each independent run and
keep it unchanged when resuming that run.

To resume at different data-parallel sizes, set `checkpoint.canonical_replicas`
to a common multiple of those sizes before starting. The number of local
microbatches per optimizer step (`global_batch_size / local_batch_size`) must
also be a multiple of that value. Preserve the data configuration, tokenizer,
seed, and global batch size across resumes. See
[distributed training](https://docs.zephon.io/pipelines/distributed_training.html)
and [checkpointing and resuming](https://docs.zephon.io/pipelines/checkpointing_and_resuming.html)
for the underlying Zephon guarantees.

For distributed runs over finite sources, set
`dataloader.zephon.runtime.mtp_mode: false` to avoid a known end-of-data
checkpointing hang. See the [Zephon fix](https://github.com/datologyai/zephon/pull/91)
for its status.

DP, TP, and PP follow AutoModel's model and backend capabilities. Context
parallelism is not supported by this integration. Zephon currently supplies
training data only; AutoModel's existing validation loaders remain available.

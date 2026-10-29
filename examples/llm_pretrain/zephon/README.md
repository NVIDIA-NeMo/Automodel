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

The source config chooses tokenization: raw text or instructions are tokenized
online, while pretokenized input keeps its supplied token IDs. Independently,
select a `packing.pack_algorithm` to pack online or set it to `null` to preserve
each supplied sequence. This supports online tokenization and packing, offline
tokenization with online packing, and fully prepacked input.

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

`packing.seq_len` is the model input length. When omitted, it uses the effective
model config's input-token limit. Set it explicitly for shorter training inputs
or when the model has no known limit. A length above a known model limit is
rejected, including when positions reset between subdocuments. Zephon prepares
one extra token for next-token prediction and handles the input/target shift. `wrap` can split long
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

### Pretokenized and prepacked text

Tokenization and online packing are independent choices:

| Supplied data | Dataset config | `packing.pack_algorithm` | Online preparation |
| --- | --- | --- | --- |
| Raw text | `ZephonTextSourceConfig` | `wrap`, `first_fit`, `best_fit`, or `best_fit_wrap` | Tokenize and pack |
| Raw instructions | `ZephonInstructionSourceConfig` | A packing algorithm | Chat tokenize and pack |
| Pretokenized documents | `ZephonPretokenizedTextSourceConfig` | A packing algorithm | Pack only |
| Pretokenized, prepacked sequences | `ZephonPretokenizedTextSourceConfig` | `null` | Preserve supplied sequences |
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
  _target_: nemo_automodel.components.datasets.zephon.ZephonPretokenizedTextSourceConfig
  # Keep the model tokenizer configuration here for vocabulary validation.
dataloader:
  _target_: nemo_automodel.components.datasets.zephon.ZephonDataloaderConfig
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

### Instruction tuning

For question/answer data, select `ZephonInstructionSourceConfig`:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.zephon.ZephonInstructionSourceConfig
  column_mapping: {question: prompt, answer: completion}
  answer_only_loss_mask: true
  use_hf_chat_template: true
  tokenizer:
    pretrained_model_name_or_path: /path/to/model-or-tokenizer
```

This example reads the `prompt` and `completion` fields of each document.
AutoModel normalizes the columns, and Zephon's `tokenize_chat()` performs chat
tokenization and loss masking. The tokenizer's chat template supplies the
conversation format. Set `use_hf_chat_template: false` for plain prompt/completion
formatting.

You can use `ZephonInstructionSourceConfig` directly, as above. AutoModel's
`ColumnMappedTextInstructionDatasetConfig` is an optional compatibility input:
its supported fields are translated into `ZephonInstructionSourceConfig`.
With the same preparation settings, both build the same Zephon instruction
pipeline. In Python, you can pass a `ZephonInstructionSourceConfig` instance as
`ZephonDataloaderConfig.dataset_config`.

The local example uses plain formatting and Flash Attention:

```bash
uv run python examples/llm_pretrain/zephon/prepare_tiny.py
uv run automodel examples/llm_pretrain/zephon/sft.yaml --nproc-per-node 1
```

Install the matching Flash Attention dependencies, or select SDPA with
`--model.attn_implementation sdpa`.

## How the integration works

The dataset config selects text, chat, or pretokenized input. The Zephon
pipeline reads and mixes sources, prepares documents, packs them, and groups
them into batches; online packing can be disabled. Zephon's `to_training()` handles shifting, padding, and
collation. A small adapter then presents that result in the format expected by
AutoModel's existing recipe and model.

With Zephon selected, AutoModel's `dataset` section configures document
preparation. Use a Zephon integration source config directly, or a supported
native config that the integration translates into its Zephon equivalent.
These preparation configs are separate from `zephon.io.Dataset` objects, which
describe data sources and are constructed from `dataloader.zephon.sources`.
Zephon owns reading, mixing, tokenization, packing, and batching. Configure
source traversal and packing under `dataloader.zephon`; arbitrary native dataset
types and their sampling, truncation, or collator behavior are not automatically
converted.

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

DP, TP, and PP follow AutoModel's model and backend capabilities. Context
parallelism is not supported by this integration. Zephon currently supplies
training data only; AutoModel's existing validation loaders remain available.

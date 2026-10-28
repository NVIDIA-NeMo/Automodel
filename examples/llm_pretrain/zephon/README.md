# Zephon raw-text smoke recipe

From the repository root, in an AutoModel CUDA environment:

```sh
uv sync
uv run python examples/llm_pretrain/zephon/prepare_tiny.py
uv run automodel examples/llm_pretrain/zephon/tiny.yaml --nproc-per-node 1
```

The fixture is generated locally. Every JSONL document contains a `text` field.
Source `path` names the **directory of shards**, as required by Zephon; a file
path is not silently broadened to its parent directory. `seq_len` is the model
input length; Zephon packs `seq_len + 1` tokens and `to_training()` performs the
existing next-token shift. There is no native AutoModel dataset packing stage.

This finite corpus ends before `max_steps`. The scheduler trains complete
optimizer steps, coordinates completion across ranks, drops incomplete final
accumulation windows, and saves terminal state. `max_steps` is only a cap.
Use Zephon's `exhausted_policy`, `stop_after_passes`, `max_repeats`, and
`reshuffle_on_repeat` settings to control traversal; repetition is not required.

The pinned Zephon features do not require its separate runtime-fix stack.
Until [Zephon #91](https://github.com/datologyai/zephon/pull/91) lands, use
`zephon.runtime.mtp_mode: false` for distributed finite streams: MTP's automatic terminal
checkpoint can wait for another rank that is still coordinating training.
Ordinary explicit checkpoints retain the all-rank protocol described below.

The recipes use the same typed loader `.build()` entry point for native and
Zephon data. Runtime construction facts select the existing rank ordering and
RNG contexts; no additional YAML switches are needed. Zephon advertises its
synchronized-stream lifecycle to the step scheduler, and the recipes checkpoint
terminal state before calling a loader's optional public `close()` method.
The Zephon adapter closes its owned iterator and uses `Pipeline.close()` when
available; the additional pipeline cleanup API is optional.
The Zephon pipeline ends at `batch()`. The loader calls `to_training()` on the
original `SampleBatch`, then applies AutoModel attention policy, output-format
metadata and field aliases. It never constructs replacement records/batches or
merges replay metadata. All original contributors/counts remain with Zephon.
Post-conversion adaptations update the owned dictionary and reuse tensors where
possible; they do not build dense attention masks on the CPU. The model input
contract is validated once during loader construction, before sources or workers
start. Conversion runs in the consumer process; there is no collation-worker
setting. A TODO records possible future Zephon operations on `to_training()`
results so conversion could later overlap model execution. No throughput
improvement is claimed for this simpler consumer path.

The text adapter supports Zephon's `wrap`, `best_fit_wrap`, `first_fit` and
`best_fit` algorithms and block-causal, HF flash-varlen, or direct document-ID
model interfaces. Cross-document attention is masked by default. Original subdocument positions and a positional padding mask are
retained. PAD may equal EOS without masking real EOS targets. Zephon's default
shift still supervises transitions between subdocuments; attention isolation
does not itself remove these labels. Set `mask_segment_starts: true` to mask
those exact targets independently of attention policy.

At the Zephon API level, `eos_mask_loss` instead excludes predictions made from
every EOS-ID token. The policies coincide at strict distinct-BOS/EOS document
boundaries, but differ for internal turn EOS tokens or a shared BOS/EOS ID.
The integration's `mask_segment_starts` uses boundaries and works without EOS.

Set `allow_cross_document_attention: true` to allow attention to earlier
subdocuments within each batch document. This compatibility adapter runs after
`to_training()`: text position IDs become continuous within each packed document
(`0,1,2,3,4,5` rather than `0,1,2,0,1,2`). With the option off, positions retain
their subdocument resets. Original subdocument IDs, labels and padding stay
unchanged in both modes. Batch documents remain separate and future tokens stay
masked. The same position policy applies to pretraining, SFT and encoded text.
For an ordinary dense-backend causal document, the model-entry hook selects a linear
all-true boolean mask; HF SDPA elides it without constructing a quadratic mask.
This prevents HF training from inferring packed boundaries from reset padding
positions. Flash retains compact joined IDs for explicit cumulative lengths;
processor exclusions and model-specific image policies retain their masks.



## Batch output format

`dataloader.output_format` selects the AutoModel input representation independently
of Zephon's `pack_algorithm`. The default is `batched`, which has the broadest model
coverage and uses `[batch, sequence]` token tensors with document metadata. Native
AutoModel's NEAT producer uses this representation too; Zephon does its own packing.

```yaml
dataloader:
  _target_: nemo_automodel.components.datasets.zephon.ZephonDataloaderConfig
  output_format: thd
  zephon:
    sources:
    - name: text
      path: work/zephon-tiny/corpus
    packing:
      seq_len: 63
      pack_algorithm: wrap
```

THD adds `seq_lens`, `seq_lens_padded` and `qkv_format: thd` in the loader, after `to_training()` and the xdoc policy. AutoModel's existing
input preparation flattens the tensors and builds cumulative sequence lengths.
There is no additional packing or Zephon API change. Select a model/backend
that accepts native THD inputs, for example native Qwen3 with `model.backend.attn: te`;
the recipe validates support before starting the pipeline. THD is not automatically
preferred or selected from the model name.

In both formats, xdoc OFF delimits each subdocument for attention and keeps local
positions. Xdoc ON joins attention and position ranges within each batch document;
batch documents remain independent. Labels do not change; auxiliary MTP boundary
checks follow the selected segmentation. PAD=EOS is safe because lengths come from positional metadata.

Native text loaders can select either format through their existing
`packed_sequence.packing_strategy` (`thd` or `neat`) and collator configuration;
native VLM uses `packing_format`. Their shared `output_format` property reports
that selected representation. `emits_thd` remains a deprecated native text alias.
Zephon uses only `dataloader.output_format` for this choice.

## Document membership and AutoModel input interfaces

`to_training(return_document_ids=True)` emits original per-token subdocument
membership, independent of model positions and SFT supervision. The integration
keeps this as `document_ids`; `attention_document_ids` initially aliases it.
Processor-ineligible tokens receive zero only in attention IDs. Xdoc joins the
positive attention IDs and translates positions; original IDs remain unchanged.
For `[A,A,B,B,PAD]`, original IDs are `[1,1,2,2,0]`; with xdoc enabled the
attention IDs are `[1,1,1,1,0]`. Causality still prevents attention to future tokens.

When yielding to AutoModel, the loader names the selected segmentation
`_packed_seq_ids`; direct-ID models also receive it as `attention_mask`. With
xdoc off this distinguishes subdocuments; with xdoc on all eligible tokens in
each batch document share one ID. Attention, recurrence, THD lengths and MTP
all consume this same policy. Zephon's `mask_segment_starts` remains an
independent choice of next-token labels; enabling xdoc does not rewrite labels.

Original `document_ids` are omitted by default. A model that implements the
shared document-mask composition interface requests them through
`return_document_ids`, for example to keep Gemma's image groups distinct.
This is a model requirement, not another user setting. MTP never uses these
original IDs. Renaming the selected map does not copy tensors or reconstruct
boundaries from positions.
`packed_mask_type` describes the receiving interface, not the packer or attention
kernel: a shared hook can turn IDs into a mask for SDPA, or a model can interpret
IDs itself and then use its own attention backend.

### Loader return contract

Every loader receives the cheap model-derived contract. Querying it installs no
hooks and performs no packed-backend validation. Recipes separately configure
packed batched attention or validate native THD support. These runtime requests
need no extra user flags.

| Contract field | Batched packed output | THD output |
|---|---|---|
| `packed_mask_type` | Selects the mask/Flash input interface | Ignored; native THD consumes lengths |
| `return_packed_sequence_metadata` | Requests the gathered-token bundle below | Ignored; THD constructs kernel metadata from lengths |
| `return_document_ids` | Retains optional original membership | Retains optional original membership |

Model classes can still declare `requires_packed_sequence_metadata`; the query
translates their requirements into loader-facing `return_*` requests. These
requests are runtime input contracts, not extra YAML settings. Unpacked native
loaders ignore packing-only requests.

### Models requesting gathered-token metadata

`return_packed_sequence_metadata=True` requests an additional bundle for batched output.
It requests AutoModel's `packed_token_indices`, `cu_seqlens`, and `max_seqlen` for
the stream after gathering non-padding tokens. For `[A,A,B,B,PAD,PAD]`, gathered
offsets are `[0,2,4]`; physical padded offsets would include `[0,2,4,6]`. Before
PP splitting, AutoModel stores indices and offsets with a batch dimension and
`-1` for unused entries; its consumer normalizes them after splitting.

Qwen3.5 (dense/MoE) and Kimi Linear/K3 text models consume this bundle.
The final batch map calls AutoModel's `build_packed_sequence_metadata` using
**attention** document IDs: xdoc off gives `[0,2,4]`, xdoc on gives `[0,4]`.
Both full attention and recurrent state therefore reset at the selected
boundaries. MTP uses the same selected IDs. The model
normalizes offsets after PP microbatch splitting, never before it. This uses
existing model kernels and does not enable a new attention backend or override
model-specific TP/PP restrictions. Kimi Linear, for example, currently declares
both TP and PP unsupported. Packed Qwen3.5-MoE with TE is rejected before
loader startup because that model path lacks the required token gather/scatter;
its SDPA and native FA4 interfaces remain available.

CPU tests exercise actual loader output, microbatch offsets and the models'
attention/metadata consumers; the CUDA recurrence tests exercise real GDN/KDA
kernels when their dependencies are installed. CPU passes do not certify those
CUDA kernels. Qwen3.5/Kimi image preprocessing is not added by this text handoff;
the image adapter's supported processor families are documented separately.


## Grouped configuration

`dataloader.zephon` keeps `sources` and `seed` at the top level. Both LLM and
VLM loaders use the same typed groups:

| Group | Settings |
| --- | --- |
| `source` | Source shuffling, chunk size, token estimation, exhaustion/repeat limits |
| `packing` | Sequence length, existing packing algorithm, bins, document limits, padding, post-pack shuffle |
| `training` | Cross-document attention/positions and `mask_segment_starts` |
| `workers` | Per-operator parallelism, including chat tokenization and normalization |
| `runtime` | Runner, MTP transport and prefetch buffer size |
| `cache` | Download cache directory and byte limit |
| `checkpoint` | Canonical replicas, aggregation backend/directory and stream run ID |

For example:

```yaml
dataloader:
  _target_: nemo_automodel.components.datasets.zephon.ZephonDataloaderConfig
  zephon:
    sources:
      - name: corpus
        path: data/corpus
    packing:
      seq_len: 2048
      pack_algorithm: best_fit
    source:
      token_estimation: true
      exhausted_policy: redistribute
    training:
      allow_cross_document_attention: false
    workers:
      tokenize_parallelism: 4
      pack_parallelism: 2
    checkpoint:
      run_id: training-1
```

The reference integrations' flat TOML names are accepted at the recipe boundary;
existing flat YAML settings are accepted there as well. Runtime code uses only
the typed groups. Inline YAML overrides individual TOML settings, retaining other
settings in the same group; an inline source list replaces the entire file list.
Duplicate flat/grouped spellings in one input and unknown group fields raise.
Encoding and special-token policy remain on `dataset`, and `output_format`
remains on `dataloader`.

## Packing and source configuration

`first_fit` and `best_fit` preserve whole documents and default to dropping
oversized documents. `drop_oversized: false` requests an error instead. These
paths do not truncate text before packing. `pack_num_bins` defaults to 64 for
these algorithms. Wrapping algorithms split long text and reject
`drop_oversized: true`. `max_num_documents` optionally caps contributors; its
`max_num_documents_scope` is `batch` by default or `bin` explicitly.

Supply multiple named `sources` with positive `weight` values for mixtures.
`token_estimation` defaults to enabled for multiple sources; `false` selects
document-count weighting. `true` measures tokenizer costs when the source is primed at iteration or
checkpoint time; tokenizer preflight validates loading and templates first. A
mapping passes existing Zephon `TokenEstimation` fields, for example:

```yaml
dataloader:
  zephon:
    source:
      token_estimation:
        calibration_samples: 256
        calibration_shards_min: 1
        calibration_shards_max: 4
        fallback_tokens_per_byte: 0.25
```

`primer` may be `measure`, a positive global tokens/byte ratio, or per-source
ratios (unlisted sources are measured). Calibration is checkpointed by Zephon.
Mixing uses source-level token allocation; filtering and oversized-document
drops can change the delivered proportions. Loader logs show emitted component
token shares. `exhausted_policy: stop` ends at the first
exhausted source, `redistribute` drains remaining sources, and `repeat` opts into
repetition; source-specific mappings and repeat/pass caps are forwarded to
Zephon unchanged.

Optional `data_config: /path/to/recipe.toml` under `dataloader.zephon` reads the
reference-style source/execution recipe. Explicit YAML settings override file
values; a YAML source list replaces the file's entire list. Relative file paths
resolve from the TOML directory. Unknown settings are rejected. `repeat: false`
uses Zephon's finite-pass default; it cannot be combined with explicit exhaustion
settings. This adapter accepts the reference-style field names, with the defaults
below; a recipe from another integration should set these explicitly when
reproducing a run.

The reference TOML's top-level `tokenize_special_tokens` key maps silently to
`dataset.tokenize_special_tokens`. The same key under `dataloader.zephon` in YAML
is also supported. An explicit `dataset.tokenize_special_tokens` takes precedence
over either source; otherwise inline YAML overrides the TOML value.

| Setting | AutoModel default |
| --- | --- |
| Source traversal | One finite pass |
| `shuffle_after_pack` | `true` for first-fit/best-fit, `false` for wrapping |
| `dataset.tokenize_special_tokens` | `bos_eos` (raw text only) |
| `mtp_mode` | `true` |
| Distributed aggregation | `torch://` by default; explicit `run_id` required |

Raw text brackets each document with BOS and EOS by default, matching Zephon
text tokenization. Set `dataset.tokenize_special_tokens: eos` when the tokenizer
has no BOS, or choose another supported policy explicitly. `tokenizer_default`
can produce BOS-only documents when that is the tokenizer's own policy.
The adapter passes the effective tokenizer object to
Zephon, including unsaved vocabulary and special-token changes. Worker counts
are explicit controls rather than hardware-dependent defaults.

For explicit insertion, `dataset.bos_token_id` and `dataset.eos_token_id`
override the effective tokenizer's IDs. These must be existing vocabulary IDs
and require the corresponding `bos`, `eos`, or `bos_eos` insertion policy;
unused overrides are rejected. They change only the IDs Zephon inserts, leaving
the tokenizer's special-token settings and postprocessor unchanged. For example:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.zephon.ZephonTextSourceConfig
  tokenize_special_tokens: eos
  eos_token_id: 1  # An existing vocabulary ID chosen for this corpus.
```

Operator worker counts live under `dataloader.zephon.workers`:

| Setting | Stage | Default |
| --- | --- | --- |
| `fetch_parallelism` | Read records from source shards | `2` |
| `prefetch_parallelism` | Download ahead when `prefetch_buffer_size > 0` | `null` |
| `map_parallelism` | Normalize SFT columns or validate encoded documents | `null` |
| `tokenize_parallelism` | Raw-text `tokenize`, SFT `tokenize_chat`, or VLM image/text encoding | `3` |
| `pack_parallelism` | Materialize `pack_flat` or VLM `pack_sequences` bins | `1` |
| `shuffle_parallelism` | Shuffle packed documents when `shuffle_after_pack: true` | `null` |

Each accepts a positive integer or `null` to use the operator's default.
`null` is not an adaptive CPU-count policy. The VLM encoder resolves
`tokenize_parallelism: null` to one worker; built-in text/chat tokenizers use
their own operator defaults. Unused stages are not added: raw
text needs no normalization map, and encoded input needs no tokenizer.
Zephon's `batch()` operator exposes no parallelism argument. Text and image
collation both run in the loader consumer, with no post-batch operator.
For example, an SFT recipe can set:

```yaml
dataloader:
  zephon:
    workers:
      map_parallelism: 2
      tokenize_parallelism: 4
      pack_parallelism: 2
```

Checkpoints use the reference integrations' **aggregated opaque pickle bytes**
through AutoModel's all-rank DCP state route. Load only trusted training
checkpoints. Multi-rank runs require an explicit stream `run_id`; when no
`aggregate_store` or shared `aggregate_dir` is configured, the integration uses
`torch://`, connecting to the launcher's `MASTER_ADDR`/`MASTER_PORT` TCPStore.
Launchers without that store must select an explicit aggregation backend.
Restore through AutoModel's usual
`restore_from.path`. Validation continues using native `validation_dataset` /
`validation_dataloader` configs and does not advance the training stream.
When exhaustion follows a periodic step checkpoint, AutoModel publishes a
distinct terminal epoch checkpoint with the same weights and optimizer state
but newer scheduler and loader state. Retention may keep both checkpoints, so
this terminal publication incurs another complete checkpoint write.

For elastic resume, pin `canonical_replicas` to a common multiple of all intended
DP sizes. `global_batch_size / local_batch_size` must also be a multiple of that
value, so optimizer steps and step-boundary checkpoints consume complete lane
windows. Invalid pinned configurations fail before the model is built.

TP and PP use the existing `distributed` configuration. Every global rank owns a
loader, and TP/PP peers sharing a DP coordinate read identical documents. Only
DP shards the source; every global rank participates in aggregated checkpoints.
The shared SDPA/eager/Flash handoff receives a compact CPU attention-document
map. Models declaring `packed_mask_type="document_ids"` receive that map directly
as their `attention_mask`, following their own device-transfer and masking path. After PP microbatch splitting, model-entry hooks build boolean SDPA or
model-dtype additive eager masks on the input device. Flash cumulative offsets
are computed on the host and only their small int32 vectors move to the model;
the IDs stay on the CPU during ordinary recipe batch transfer. Original
subdocument boundaries remain available independently of the selected text RoPE
positions. No user packing switch is needed. A four-GPU
TP2 × PP2 example is available:

```sh
uv run automodel examples/llm_pretrain/zephon/model_parallel.yaml --nproc-per-node 4
```

CP, KD, pre-packed THD input and prepacked caches remain unsupported.
VLM uses its dedicated adapter. CPU coverage exercises actual Zephon pipelines, spawned
workers, finite completion, tiny Llama TP and PP forward/backward against an unsplit reference, and fresh-loader
DCP replay. A CPU attention oracle also verifies the Flash metadata handoff
through the public PP schedule, including cross-document attention and gradients.
The full CUDA recipe and FlashAttention forward/backward still require a GPU
validation run; the CPU oracle does not exercise FlashAttention kernels.
The GPU comparison should measure native versus Zephon throughput with the same
model and batches, the per-step availability collective overhead, and a
fetch/tokenize worker sweep of 1/1 versus 2/3 at eight ranks per node.


## Instruction and encoded sources

`prepare_tiny.py` also writes an instruction corpus. Run the SFT variant with:

```sh
uv run automodel examples/llm_pretrain/zephon/sft.yaml --nproc-per-node 1
```

The existing `ColumnMappedTextInstructionDatasetConfig` target selects SFT from
its `column_mapping`, `answer_only_loss_mask` and `use_hf_chat_template` fields.
Transport comes from `dataloader.zephon.sources`. Alternatively,
`dataset.path_or_dataset_id` can supply Zephon-compatible shard directories/URIs;
setting both is rejected. HF dataset IDs and individual JSONL files must first
be supplied as Zephon-supported shard sources. Native dataset padding,
truncation, subset selection and sample limits are rejected rather than ignored;
Zephon owns packing and source traversal.

All instruction sources use Zephon `tokenize_chat`. AutoModel only normalizes
columns into messages, preserving its existing prompt whitespace or
system/user/assistant order. Plain prompt/completion uses
`apply_chat_template=False` with `special_tokens="tokenizer_default"`; chat uses
the effective tokenizer template with `special_tokens="none"`. These are the
same special-token policies exposed by Zephon text tokenization. Zephon selects
generation-tag or prefix-based masking internally. Untagged templates supervise
the assistant header as well as its answer, preserving AutoModel's default header
policy. Full-token supervision
uses `loss_scope="all_tokens"`. The adapter adds no EOS beyond tokens supplied
by the tokenizer or template. A tokenizer/template that does not supply EOS
therefore produces SFT documents without EOS, matching the native formatting
policy. To choose a different policy explicitly, use the Zephon instruction
source target and set `dataset.special_tokens: eos` or `bos_eos`; these bracket
the complete conversation, while `assistant_eos` is the chat-only per-assistant
policy with `use_hf_chat_template: false`. No deprecated EOS boolean is needed.

SFT supports the same numeric insertion overrides as raw text:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.zephon.ZephonInstructionSourceConfig
  column_mapping: {question: question, answer: answer}
  use_hf_chat_template: true
  special_tokens: eos
  eos_token_id: 1  # An existing vocabulary ID, appended after the rendered conversation.
```

`special_tokens: bos_eos` additionally permits `bos_token_id`; omitted IDs use
the effective tokenizer's BOS/EOS. These policies bracket the complete encoded
conversation. Template turn markers remain where the template puts them, so a
template-provided end-of-turn token may precede the explicitly inserted EOS.
The default policies (`none` with a template, `tokenizer_default` without one)
and `assistant_eos` reject numeric overrides because those modes do not perform
the corresponding explicit conversation-level insertion. An EOS string loader
override is a separate Zephon API feature and is not configured here.

Zephon masks tokens by overlap with assistant text. This deliberately differs
from native prompt-length arithmetic when tokenization merges across a boundary:
for `"hello " + "world"`, a token containing the space and `world` remains
supervised. The first token of each source document remains masked, including
with full-token supervision; cross-document attention does not change loss
masks. Zephon logs empty or unsupervised document drops. Oversized documents
follow the configured packer policy without prior truncation.

Select `nemo_automodel.components.datasets.zephon.ZephonEncodedTextSourceConfig`
for encoded text. Set `dataset.tokens_field` (default `input_ids`); the optional
`loss_mask_field` defaults to `loss_mask`. Each source document contains a
one-dimensional integer token list and, when present, an equally sized binary
mask selecting target tokens **before** shifting. A supplied mask must exclude
the first token, so the previous document cannot predict a supervised SFT start.
Without a mask, ordinary pretraining supervision is retained. Tokens must belong to the
resolved model tokenizer. The adapter skips encoding and uses normal Zephon
packing and `to_training()` shifting. Already shifted `labels`, existing packing
boundaries, and padded documents are rejected; this is an encoded-document
adapter, not a universal prepacked-cache importer. No legacy window inference
is performed. Raw-text special-token policy lives on its own dataset config.

Token-aware allocation also works for instruction and encoded sources. Zephon's
built-in SFT counter replays deterministic column normalization and the same
tokenizer settings used for delivery. Encoded inputs provide a token-count
function because they skip tokenization. No mixture correction stage is
installed. CPU tests cover chat/header masks, tokenizer special tokens,
cross-boundary token merges, raw/encoded delivery, source-token calibration and
fresh-loader replay through process workers.

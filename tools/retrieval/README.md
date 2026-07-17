# Retrieval Data Tools

This directory contains CPU-side utilities for preparing retrieval training data before launching GPU training jobs.

## Which Option Should I Use?

Use **normalized Arrow** for most large retrieval runs.

Despite the `vl` in some script names, these tools also support text-only retrieval. Image fields are optional; text-only
corpus documents are stored with an empty image field.

- **Normalized Arrow:** recommended. Prepare a portable dataset bundle once, then train with
  `nemo_automodel.components.datasets.llm.retrieval_dataset_normalized.NormalizedRetrievalDatasetConfig`.
- **Warm HF cache:** use only when you want to keep the original
  `nemo_automodel.components.datasets.llm.make_retrieval_dataset` path unchanged on the same cluster.

## Normalized Arrow (Recommended)

Normalized Arrow keeps the original retrieval structure, but moves the expensive corpus materialization into a reusable
prepared artifact:

- Train rows store query text plus positive/negative document IDs.
- Local corpus shards store each referenced document and optional image once.
- Training still resolves `doc_id -> document`, but it reads from local Arrow instead of rebuilding Hugging Face cache
  state in the GPU job.

This is the recommended full-dataset format because it is portable and avoids duplicating image payload in every
training row.

### Prepare

```bash
uv run python tools/retrieval/prepare_normalized_vl_retrieval_data.py \
  --config /path/to/original_retrieval_config.yaml \
  --output-dir /path/to/normalized_vl_retrieval \
  --resume
```

On Slurm CPU nodes:

```bash
CONFIG=/path/to/original_retrieval_config.yaml \
OUT_DIR=/path/to/normalized_vl_retrieval \
PARTITION=cpu_short \
TIME=08:00:00 \
CPUS_PER_TASK=32 \
EXTRA_CONTAINER_MOUNTS=/path/to/source_data:/path/to/source_data \
tools/retrieval/submit_prepare_normalized_vl_retrieval_data_cpu.sh
```

### Train

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.retrieval_dataset_normalized.NormalizedRetrievalDatasetConfig
  data_dir_list: /path/to/normalized_vl_retrieval
  model_type: bi_encoder
  data_type: train
  n_passages: 5
  do_shuffle: true
```

### Choose Sources Or Sample Caps

A normalized bundle stores each original `data_dir_list` entry as a separate numeric source directory:
`sources/source-00000`, `sources/source-00001`, and so on. The top-level metadata records the readable `source_name`,
stable `source_key`, original `source_entry`, and source path.

For normal training, pass the top-level bundle path. If you want to choose specific prepared sources or cap samples per
source, pass a list instead:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.retrieval_dataset_normalized.NormalizedRetrievalDatasetConfig
  data_dir_list:
    - path: /path/to/normalized_vl_retrieval/sources/source-00000
      num_samples: 10000
    - path: /path/to/normalized_vl_retrieval/sources/source-00001
      num_samples: null
  model_type: bi_encoder
  data_type: train
  n_passages: 5
```

The same list form can also combine different normalized bundle roots:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.retrieval_dataset_normalized.NormalizedRetrievalDatasetConfig
  data_dir_list:
    - path: /path/to/normalized_vl_retrieval_a
      num_samples: 20000
    - path: /path/to/normalized_vl_retrieval_b
      num_samples: null
  model_type: bi_encoder
  data_type: train
  n_passages: 5
```

You can mix top-level bundle roots and individual `sources/source-*` paths in the same list. Avoid overlapping entries,
for example listing both `/path/to/normalized_vl_retrieval` and
`/path/to/normalized_vl_retrieval/sources/source-00000`, unless you intentionally want duplicated samples.

### Add More Normalized Data

There are two supported workflows.

The simplest workflow is to prepare the new source into a separate normalized bundle, then list both bundle roots in
training `data_dir_list`. This avoids modifying the existing bundle.

If you want one shared bundle path, use append mode. Run prep with a config that contains only the new source(s), the
same `OUT_DIR`, and `APPEND=1`:

```bash
CONFIG=/path/to/new_source_only_config.yaml \
OUT_DIR=/path/to/normalized_vl_retrieval \
APPEND=1 \
tools/retrieval/submit_prepare_normalized_vl_retrieval_data_cpu.sh
```

Append mode skips exact duplicate source entries already present in metadata. It stages each new source in a temporary
directory and updates top-level metadata only after all new sources finish. It does not deduplicate documents across
different sources, so do not re-list old sources under a modified path or config entry.

### Resume Interrupted Normalized Prep

Use `--resume` when a normalized CPU prep job was interrupted and you want to reuse readable train shards and complete
corpus directories from the same output directory. Resume is for continuing an interrupted prep run; append mode is for
adding new source entries to an already prepared bundle.

## Warm Hugging Face Dataset Cache

This option keeps the original training dataset unchanged:
`nemo_automodel.components.datasets.llm.make_retrieval_dataset`.

It moves expensive `datasets.load_dataset(...)` cache construction to CPU nodes, but it does not create a portable
dataset artifact. Use it only when training will run on the same cluster with the same cache directory and effective
dataset paths.

```bash
CONFIG=/path/to/original_retrieval_config.yaml \
CACHE_DIR=/path/to/shared/hf_cache \
PARTITION=cpu_short \
TIME=02:00:00 \
CPUS_PER_TASK=32 \
TOUCH_SAMPLES=128 \
EXTRA_CONTAINER_MOUNTS=/path/to/source_data:/path/to/source_data \
tools/retrieval/submit_warm_retrieval_hf_cache_cpu.sh
```

The same `CACHE_DIR` should be mounted into GPU training and exported as `HF_HOME`, `HF_DATASETS_CACHE`,
`HUGGINGFACE_HUB_CACHE`, and `TRANSFORMERS_CACHE`.

HF cache reuse is exact-key based. A warm cache is reused only when the later training run uses the same cache directory
and the same effective dataset fingerprint. The fingerprint can change if the same source data is referenced through a
different mount alias, symlink, or config path, for example `/lustre/fsw/...` versus `/lustre/fs11/...`.

For local/non-Slurm use:

```bash
uv run python tools/retrieval/warm_retrieval_hf_cache.py \
  --config /path/to/original_retrieval_config.yaml \
  --cache-dir /path/to/shared/hf_cache \
  --touch-samples 128
```

`--touch-samples` reads transformed examples to validate corpus lookup and image decoding. Decoded images are not
persisted.

## Storage Comparison

Measured on the Nemotron VL 1B image-retrieval training set used in debugging, with 262,197 train rows:

| Path | What it stores | Size observed | Recommendation |
| --- | --- | ---: | --- |
| Original HF cache | Hugging Face cache fingerprints and materialized source corpora under `HF_DATASETS_CACHE` | `3.7T` in one observed active cache dir; varies by cache history | Fast when warm, but not portable and can grow with each fingerprint/config/path variant |
| Normalized Arrow | Train refs plus deduplicated local corpus Arrow shards | `176G` | Recommended full-dataset portable format |

The exact HF cache size can vary because Hugging Face Datasets may keep multiple fingerprints, source downloads,
intermediate Arrow files, and old variants. Normalized Arrow is an explicit dataset artifact, so the size is much more
predictable.

## Adding New Corpus Schemas

For new corpus schemas, use the original AutoModel extension point:

1. Add an `AbstractDataset` implementation with `get_document_by_id()` and `get_all_ids()`.
2. Register it in `DATASETS`.
3. Add the source JSON to the original retrieval config.
4. Prepare a normalized bundle or warm the original Hugging Face cache.

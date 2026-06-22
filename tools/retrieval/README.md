# Retrieval Data Tools

This directory contains offline data utilities for retrieval training.

## Overview

These tools provide three ways to move expensive source dataset and corpus setup out of GPU training jobs:

- **Normalized Arrow:** recommended for large VL retrieval training. It creates a portable dataset bundle that stores
  train rows separately from deduplicated corpus documents/images. Train with
  `nemo_automodel.components.datasets.llm.make_normalized_retrieval_dataset`.
- **Warm HF cache:** keeps the original `make_retrieval_dataset` training path unchanged, but prebuilds Hugging Face
  cache files on CPU nodes. Train with the original
  `nemo_automodel.components.datasets.llm.make_retrieval_dataset`. This is useful on one cluster, but the result is not
  a portable dataset artifact.
- **Resolved Arrow:** useful for small repro/debug datasets. It writes fully materialized training rows where each row
  already contains the selected document text and image bytes. Train with
  `nemo_automodel.components.datasets.llm.make_resolved_retrieval_dataset`.

Use normalized Arrow unless you specifically need to keep the original data path unchanged or need a small fully
self-contained debug dataset.

## Normalized Arrow (Recommended)

Normalized Arrow keeps the original corpus-id retrieval model but stores the referenced corpus locally:

- `sources/source-*/train/*.arrow` stores query rows and positive/negative document IDs.
- `sources/source-*/corpus/*/*.arrow` stores each referenced document/image once.
- Training still resolves `doc_id -> document` through the normal retrieval transform.

This is the recommended portable format because it avoids hidden Hugging Face cache rebuilds and avoids duplicating
image payload in every training row.

```bash
python tools/retrieval/prepare_normalized_vl_retrieval_data.py \
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

Then train with:

```yaml
dataloader:
  dataset:
    _target_: nemo_automodel.components.datasets.llm.make_normalized_retrieval_dataset
    data_dir_list: /path/to/normalized_vl_retrieval
    model_type: bi_encoder
    data_type: train
    n_passages: 5
    do_shuffle: true
```

Normalized prep keeps each `data_dir_list` entry as a separate numeric source bundle under `sources/`. Numeric
directories keep the artifact portable and avoid path-name collisions; duplicate detection uses the top-level metadata
instead. The metadata records a readable `source_name`, stable `source_key`, original `source_entry`, and source path.
Training can use the top-level bundle, or choose prepared sources explicitly and apply per-source sample caps:

```yaml
dataloader:
  dataset:
    _target_: nemo_automodel.components.datasets.llm.make_normalized_retrieval_dataset
    data_dir_list:
      - path: /path/to/normalized_vl_retrieval/sources/source-00000
        num_samples: 10000
      - path: /path/to/normalized_vl_retrieval/sources/source-00001
        num_samples: null
    model_type: bi_encoder
    data_type: train
    n_passages: 5
```

The same list form can also combine different normalized bundle roots. Use this when sources were prepared into
separate bundle directories, or when you want independent `num_samples` caps without appending into one bundle.
Append mode is only a convenience feature for maintaining one shared top-level bundle.

To add a new source later, run prep again with a config that contains only the new source(s), the same `OUT_DIR`, and
append enabled:

```bash
CONFIG=/path/to/new_source_only_config.yaml \
OUT_DIR=/path/to/normalized_vl_retrieval \
APPEND=1 \
tools/retrieval/submit_prepare_normalized_vl_retrieval_data_cpu.sh
```

Append mode stages each new source in a temporary directory and updates the top-level metadata only after all new
sources finish. It skips exact duplicate source entries already present in metadata. It does not deduplicate documents
across different sources, so do not re-list old sources under a modified path or config entry.

## Warm Hugging Face Dataset Cache

Use this only when you want to keep the original `make_retrieval_dataset` training path unchanged on the same cluster.
It moves expensive `datasets.load_dataset(...)` cache construction to CPU nodes, but it does not create a portable
dataset artifact.

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

HF cache reuse is exact-key based. A warm cache is reused only when the later training run uses the same cache
directory and the same effective dataset fingerprint. The fingerprint can change if the same source data is referenced
through a different mount alias, symlink, or config path, for example `/lustre/fsw/...` versus `/lustre/fs11/...`.

For local/non-Slurm use:

```bash
python tools/retrieval/warm_retrieval_hf_cache.py \
  --config /path/to/original_retrieval_config.yaml \
  --cache-dir /path/to/shared/hf_cache \
  --touch-samples 128
```

`--touch-samples` reads transformed examples to validate corpus lookup and image decoding. Decoded images are not
persisted.

## Resolved Arrow Debug Data

`prepare_resolved_vl_retrieval_data.py` writes packed Arrow shards where every training row already contains the
selected document text and image bytes. This avoids corpus lookup during training, but it duplicates payload whenever
the same document appears in multiple rows.

Use resolved Arrow for:

- small self-contained repro datasets;
- debugging exact post-transform samples;
- copying a tiny subset to another cluster.

Do not use it as the default full-dataset format when normalized Arrow is available.

Example:

```bash
CONFIG=/path/to/original_retrieval_config.yaml \
OUT_DIR=/path/to/resolved_vl_retrieval \
NUM_BUILD_SHARDS=32 \
ARRAY_SPEC=0-31 \
PARTITION=cpu_short \
TIME=04:00:00 \
CPUS_PER_TASK=32 \
EXTRA_CONTAINER_MOUNTS=/path/to/source_data:/path/to/source_data \
tools/retrieval/submit_prepare_resolved_vl_retrieval_data_cpu_array.sh
```

Training from resolved Arrow:

```yaml
dataloader:
  dataset:
    _target_: nemo_automodel.components.datasets.llm.make_resolved_retrieval_dataset
    data_dir_list: /path/to/resolved_vl_retrieval
    model_type: bi_encoder
    data_type: train
    n_passages: 5
```

Resolved Arrow is map-style, so normal DataLoader sampler/shuffle behavior applies.

Resolved prep can pack multiple original `data_dir_list` entries into one output directory. If you want to choose
prepared sources independently later, or apply different per-source sample caps, prepare them into separate output
directories and list those directories:

```yaml
dataloader:
  dataset:
    _target_: nemo_automodel.components.datasets.llm.make_resolved_retrieval_dataset
    data_dir_list:
      - path: /path/to/resolved_vl_retrieval_a
        num_samples: 10000
      - path: /path/to/resolved_vl_retrieval_b
        num_samples: null
    model_type: bi_encoder
    data_type: train
    n_passages: 5
```

## Storage Comparison

Measured on the Nemotron VL 1B image-retrieval training set used in debugging, with 262,197 train rows:

| Path | What it stores | Size observed | Recommendation |
| --- | --- | ---: | --- |
| Original HF cache | Hugging Face cache fingerprints and materialized source corpora under `HF_DATASETS_CACHE` | `3.7T` in one observed active cache dir; varies by cache history | Fast when warm, but not portable and can grow with each fingerprint/config/path variant |
| Resolved Arrow | Fully materialized train rows with document/image payload repeated per row | `508G` | Keep only for small self-contained repro/debug datasets |
| Normalized Arrow | Train refs plus deduplicated local corpus Arrow shards | `176G` | Recommended full-dataset portable format |

The exact HF cache size can vary because Hugging Face Datasets may keep multiple fingerprints, source downloads,
intermediate Arrow files, and old variants. Normalized Arrow is an explicit dataset artifact, so the size is much more
predictable.

## Adding More Data

For existing corpus schemas:

- **Normalized Arrow:** either run prep into a separate bundle and list both bundle roots in training `data_dir_list`,
  or use `APPEND=1` with the same `OUT_DIR` when you want one shared top-level bundle.
- **Warm HF cache:** rerun cache warmup with the updated original config and the same shared cache directory.
- **Resolved Arrow:** either rebuild one resolved output from the full updated config, or prepare the new source into a
  separate output directory and list both old and new resolved directories in training `data_dir_list` when you want
  independent source selection or sample caps.

For new corpus schemas, use the original AutoModel extension point:

1. Add an `AbstractDataset` implementation with `get_document_by_id()` and `get_all_ids()`.
2. Register it in `DATASETS`.
3. Add the source JSON to the original retrieval config.
4. Run one of the prep paths above.

`--resume` on normalized prep reuses readable train shards and complete corpus directories from an interrupted run.
Use `APPEND=1` to add new sources to an existing normalized bundle. Resolved prep writes flat materialized Arrow
shards and does not append into an existing output directory.

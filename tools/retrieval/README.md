# Retrieval Data Tools

This directory contains offline data utilities for retrieval training.

There are two separate preprocessing paths:

| Path | What it writes | Training dataset target | Best for | Portability |
| --- | --- | --- | --- | --- |
| Hugging Face cache warmup | HF `datasets` Arrow cache under `HF_DATASETS_CACHE` | Original `make_retrieval_dataset` | Keeping training behavior exactly unchanged while moving first-load cache construction to CPU nodes | Fragile across users/configs unless the cache dir, source path strings, and dataset code match exactly |
| Resolved VL retrieval data | Explicit JSONL, SQLite, or Parquet shards with resolved document text and image bytes | `make_resolved_retrieval_dataset` | Sharing/copying a self-contained dataset, reproducing on another cluster, avoiding corpus lookup at training time | Portable as normal files; copy the output directory or a subset of shards |

Use HF cache warmup when the goal is to run the original config faster on the same cluster. Use resolved Parquet when the goal is to create a reusable dataset artifact that is independent of the original corpus layout. Both CPU tools read the original retrieval config, but only the resolved-data path changes the GPU training dataset target.

## Warm Hugging Face Dataset Cache

For production training that already uses `make_retrieval_dataset`, prefer warming the Hugging Face `datasets` cache on CPU nodes before launching GPU jobs. This keeps the training data path identical to the normal AutoModel config while moving expensive `datasets.load_dataset(...)` materialization out of the GPU allocation.

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

The same `CACHE_DIR` should be mounted into GPU training and exported as `HF_HOME`, `HF_DATASETS_CACHE`, `HUGGINGFACE_HUB_CACHE`, and `TRANSFORMERS_CACHE`.

HF cache reuse is exact-key based. A warm cache is reused only when the later training run uses the same cache directory and the same effective dataset fingerprint. In practice, the fingerprint can change if the same source data is referenced through a different mount alias, symlink, or config path, for example `/lustre/fsw/...` versus `/lustre/fs11/...`. If the fingerprint changes, HF `datasets` will build a second Arrow cache instead of reusing the previous one. If the fingerprint matches, the warmup is fast and skips the expensive Arrow materialization.

This path does not create a portable dataset artifact. It only pre-populates the cache expected by the existing training dataset.

For local/non-Slurm use:

```bash
python tools/retrieval/warm_retrieval_hf_cache.py \
  --config /path/to/original_retrieval_config.yaml \
  --cache-dir /path/to/shared/hf_cache \
  --touch-samples 128
```

`--touch-samples` reads transformed examples to validate corpus lookup and image decoding. It is not required to build the Arrow cache, and decoded images are not persisted.

## Resolved VL Retrieval Data

`prepare_resolved_vl_retrieval_data.py` creates portable JSONL, SQLite, or Parquet shards by resolving corpus IDs into document text and image bytes ahead of training. Unlike a Hugging Face cache directory, the resolved output is an explicit dataset artifact: it can be inspected with common tools, copied to another cluster, and subsetted by copying a few shard files.

Resolved data is not an HF cache. It is a new training input format. After generating it once, GPU training should point `dataloader.dataset._target_` to `make_resolved_retrieval_dataset` and `dataloader.dataset.data_dir_list` to the resolved output directory. This path avoids the original corpus lookup and HF Arrow cache construction during training, but it does intentionally change the dataset loader used by training.

This path does not rely on the original HF cache during training. It reads the resolved shard files directly.

This format is useful when:

- the dataset needs to be copied to another cluster;
- corpus lookup paths are not available in the target environment;
- a small, self-contained reproduction dataset is needed;
- CPU-only preprocessing is desired for debugging data internals.

For steady-state training, prefer Parquet over JSONL+loose JPEGs or SQLite when a portable resolved dataset is needed. Parquet is a common interchange format for large datasets, avoids many small image files, and can be consumed by pyarrow, pandas, Spark, and Hugging Face Datasets. The native AutoModel loader reads Parquet sequentially and supports bounded-memory row shuffle through `shuffle_buffer_size`, so it does not require building a second Arrow cache before training.

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

Training from resolved Parquet:

```yaml
dataloader:
  dataset:
    _target_: nemo_automodel.components.datasets.llm.make_resolved_retrieval_dataset
    data_dir_list: /path/to/resolved_vl_retrieval
    model_type: bi_encoder
    data_type: train
    n_passages: 5
    do_shuffle: true
    shuffle_buffer_size: 1024
  shuffle: false
```

`dataloader.shuffle` must stay `false` because resolved data is an `IterableDataset`; row-level approximate shuffle happens inside the dataset via `shuffle_buffer_size`. The default Parquet sharding mode assigns row groups to ranks/workers for efficient sequential reads. For small diagnostics that need exact row-level rank sharding inside a row group, set `parquet_sharding: row`, but this can duplicate Parquet reads and is not the recommended production default.

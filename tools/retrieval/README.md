# Retrieval Data Tools

This directory contains offline data utilities for retrieval training.

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

For local/non-Slurm use:

```bash
python tools/retrieval/warm_retrieval_hf_cache.py \
  --config /path/to/original_retrieval_config.yaml \
  --cache-dir /path/to/shared/hf_cache \
  --touch-samples 128
```

`--touch-samples` reads transformed examples to validate corpus lookup and image decoding. It is not required to build the Arrow cache, and decoded images are not persisted.

## Resolved VL Retrieval Data

`prepare_resolved_vl_retrieval_data.py` creates portable JSONL, SQLite, or Parquet shards by resolving corpus IDs into document text and image bytes ahead of training.

This format is useful when:

- the dataset needs to be copied to another cluster;
- corpus lookup paths are not available in the target environment;
- a small, self-contained reproduction dataset is needed;
- CPU-only preprocessing is desired for debugging data internals.

It is not currently recommended as a steady-state speed replacement for the original `make_retrieval_dataset` path. In Nemotron VL retrieval profiling, the original HF/map-style dataset path had similar median step time but fewer long-tail 8-node steps than resolved JSONL/SQLite/Parquet. The likely reason is different shuffle/sharding behavior: original training uses a map-style dataset and sampler, while resolved data is streamed as an `IterableDataset`.

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

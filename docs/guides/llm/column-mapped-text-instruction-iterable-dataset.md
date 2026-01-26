# Use the ColumnMappedTextInstructionIterableDataset (Streaming)

This guide explains how to use `ColumnMappedTextInstructionIterableDataset` to **stream** instruction datasets for LLM fine-tuning, including **Delta Lake/Databricks** sources.

Unlike `ColumnMappedTextInstructionDataset` (map-style, non-streaming), this class is a `torch.utils.data.IterableDataset` and **always** loads data in streaming mode. This is intentional: it helps ensure data is consumed as a stream and avoids accidentally materializing full datasets/tables to disk or memory (which is especially important for large or sensitive corpora).

---
## When to Use This Dataset

Use `ColumnMappedTextInstructionIterableDataset` when you need:

- **Streaming-only behavior** (e.g., to reduce accidental data leakages from full dataset materialization)
- **Delta Lake/Databricks** (Unity Catalog, cloud lakehouse storage, DBFS, etc.)
- **Very large datasets** where map-style loading/caching is undesirable

If you do *not* need streaming (and you want `len(ds)` / `ds[i]`), use [`ColumnMappedTextInstructionDataset`](column-mapped-text-instruction-dataset.md).

---
## Key Differences vs ColumnMappedTextInstructionDataset

- **Iterable**: you iterate (`for sample in ds:`); you cannot rely on `len(ds)` or `ds[i]`.
- **Always streaming**: there is no `streaming=` flag; it is always enabled.
- **Repeat behavior**: by default, `repeat_on_exhaustion=True` (infinite stream). Set `repeat_on_exhaustion=False` to do a single pass.
- **(Optional) sharding/shuffle helpers**: use `.shard(num_shards, index)` / `.shuffle(buffer_size, seed)` when supported by the underlying backend.

The column mapping and tokenization logic are shared with `ColumnMappedTextInstructionDataset`. See [Tokenization Paths](column-mapped-text-instruction-dataset.md#tokenization-paths) for details on output fields (`input_ids`, `labels`, `attention_mask`) and masking behavior.

---
## Quickstart (Hugging Face Streaming)

```python
from transformers import AutoTokenizer

from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_iterable_dataset import (
    ColumnMappedTextInstructionIterableDataset,
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

ds = ColumnMappedTextInstructionIterableDataset(
    path_or_dataset_id="Muennighoff/natural-instructions",
    split="train",
    column_mapping={
        "context": "definition",
        "question": "inputs",
        "answer": "targets",
    },
    tokenizer=tokenizer,
    # Optional:
    # limit_dataset_samples=10_000,
    # repeat_on_exhaustion=False,   # do one pass instead of infinite stream
)

sample = next(iter(ds))
print(sample.keys())  # input_ids / labels / attention_mask (and ___PAD_TOKEN_IDS___)
```

---
## Delta Lake/Databricks

`ColumnMappedTextInstructionIterableDataset` supports Delta Lake tables from:

- Local Delta tables (directories containing `_delta_log`)
- Cloud storage (S3, Azure Blob/ADLS via `abfss://`, GCS via `gs://`)
- Databricks (DBFS paths and Unity Catalog tables)

### Installation

Install the basic Delta Lake reader:

```bash
pip install deltalake
```

For **Unity Catalog access outside of Spark** (optional), install:

```bash
pip install databricks-sql-connector
```

### Local Delta Table

```python
ds = ColumnMappedTextInstructionIterableDataset(
    path_or_dataset_id="/path/to/delta_table",  # directory containing _delta_log
    column_mapping={"question": "prompt", "answer": "completion"},
    tokenizer=tokenizer,
)
```

### Databricks Unity Catalog

Use the `delta://` prefix so the loader selects the Delta backend:

```python
ds = ColumnMappedTextInstructionIterableDataset(
    path_or_dataset_id="delta://catalog.schema.instruction_data",
    column_mapping={
        "context": "system_prompt",
        "question": "user_input",
        "answer": "assistant_response",
    },
    tokenizer=tokenizer,
    delta_storage_options={
        "DATABRICKS_TOKEN": "dapi...",  # or set DATABRICKS_TOKEN env var
        "DATABRICKS_HOST": "https://your-workspace.databricks.com",
        # Optional (depending on how you connect):
        # "DATABRICKS_HTTP_PATH": "/sql/1.0/warehouses/...",
    },
)
```

### Cloud Storage (S3/Azure/GCS)

```python
# S3 Delta table
ds = ColumnMappedTextInstructionIterableDataset(
    path_or_dataset_id="s3://my-bucket/path/to/delta_table",
    column_mapping={"question": "input", "answer": "output"},
    tokenizer=tokenizer,
    delta_storage_options={
        "AWS_ACCESS_KEY_ID": "...",
        "AWS_SECRET_ACCESS_KEY": "...",
        "AWS_REGION": "us-east-1",
    },
)

# Azure (ADLS Gen2/ABFS)
ds = ColumnMappedTextInstructionIterableDataset(
    path_or_dataset_id="abfss://container@account.dfs.core.windows.net/delta_table",
    column_mapping={"question": "input", "answer": "output"},
    tokenizer=tokenizer,
    delta_storage_options={
        "AZURE_STORAGE_ACCOUNT_NAME": "...",
        "AZURE_STORAGE_ACCOUNT_KEY": "...",
    },
)
```

### YAML Configuration (Delta Lake/Databricks)

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_iterable_dataset.ColumnMappedTextInstructionIterableDataset
  path_or_dataset_id: delta://catalog.schema.training_data
  column_mapping:
    context: system_prompt
    question: user_message
    answer: assistant_message
  answer_only_loss_mask: true
  delta_storage_options:
    DATABRICKS_TOKEN: ${oc.env:DATABRICKS_TOKEN}
    DATABRICKS_HOST: ${oc.env:DATABRICKS_HOST}
    # Optional:
    # DATABRICKS_HTTP_PATH: ${oc.env:DATABRICKS_HTTP_PATH}
```

---
## Streaming from a Delta SQL Query (Computed/Aliased Columns)

If you want to generate columns dynamically (joins, filters, computed prompt strings, etc.), pass a SQL query that returns the fields referenced by your `column_mapping`.

```python
ds = ColumnMappedTextInstructionIterableDataset(
    path_or_dataset_id="delta://catalog.schema.training_data",
    column_mapping={"question": "question", "answer": "answer"},
    tokenizer=tokenizer,
    delta_storage_options={
        "DATABRICKS_HOST": "https://your-workspace.databricks.com",
        "DATABRICKS_TOKEN": "dapi...",
        "DATABRICKS_HTTP_PATH": "/sql/1.0/warehouses/...",
    },
    delta_sql_query="""
      SELECT
        concat(system_prompt, '\n', user_message) AS question,
        assistant_message AS answer
      FROM catalog.schema.training_data
      WHERE split = 'train'
    """,
)
```

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_iterable_dataset.ColumnMappedTextInstructionIterableDataset
  path_or_dataset_id: delta://catalog.schema.training_data
  column_mapping:
    question: question
    answer: answer
  delta_sql_query: |
    SELECT
      concat(system_prompt, '\n', user_message) AS question,
      assistant_message AS answer
    FROM catalog.schema.training_data
    WHERE split = 'train'
  delta_storage_options:
    DATABRICKS_HOST: ${oc.env:DATABRICKS_HOST}
    DATABRICKS_TOKEN: ${oc.env:DATABRICKS_TOKEN}
    DATABRICKS_HTTP_PATH: ${oc.env:DATABRICKS_HTTP_PATH}
```

:::{note}
**SQL engine requirement:** `delta_sql_query` is executed via Spark (Databricks runtime/pyspark) when available, otherwise via `databricks-sql-connector`. It is not supported in a deltalake-only environment.
:::

:::{note}
**Authentication:** The Delta Lake loader automatically picks up credentials from environment variables (`DATABRICKS_TOKEN`, `AWS_ACCESS_KEY_ID`, `AZURE_STORAGE_ACCOUNT_KEY`, etc.) if not explicitly provided in `delta_storage_options`.
:::

---
## Common Arguments

| Arg                     | Default | Description |
|-------------------------|---------|-------------|
| `split`                 | `None`  | Which split to stream from a HF repo (`train`, `validation`, etc.). Ignored for local files and Delta tables. |
| `name`                  | `None`  | Name of the dataset configuration/subset to load. |
| `answer_only_loss_mask` | `True`  | Mask prompt tokens in `labels` with `-100` (CrossEntropy `ignore_index`). |
| `use_hf_chat_template`  | `False` | If `True` and the tokenizer supports chat templates, format via `tokenizer.apply_chat_template(...)`. |
| `seq_length`            | `None`  | Optional max sequence length; used for padding/truncation when enabled. |
| `padding`               | `"do_not_pad"` | Padding strategy passed to the tokenizer. |
| `truncation`            | `"do_not_truncate"` | Truncation strategy passed to the tokenizer. |
| `limit_dataset_samples` | `None`  | Optionally limit the stream to the first \(N\) samples (best-effort; depends on backend). |
| `repeat_on_exhaustion`  | `True`  | If `True`, restart the stream on exhaustion (useful for step-based training). |
| `delta_storage_options` | `None`  | Storage/auth options for Delta Lake backends (Databricks, S3, Azure, GCS). |
| `delta_version`         | `None`  | Specific Delta table version to read. |
| `delta_sql_query`       | `None`  | SQL query to generate rows for Delta sources (Spark / Databricks SQL only). |


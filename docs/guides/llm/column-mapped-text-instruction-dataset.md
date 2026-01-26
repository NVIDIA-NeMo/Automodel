# Use the ColumnMappedTextInstructionDataset

This guide explains how to use `ColumnMappedTextInstructionDataset` to quickly and flexibly load instruction-answer datasets for LLM fine-tuning, with minimal code changes and support for common tokenization strategies.

The `ColumnMappedTextInstructionDataset` is a lightweight, plug-and-play helper that lets you train on instruction-answer style corpora without writing custom Python for every new schema. You simply specify which columns map to logical fields like `context`, `question`, and `answer`, and the loader handles the rest automatically. This enables:

- Quick prototyping across diverse instruction datasets
- Schema flexibility without requiring code changes
- Consistent field names for training loops, regardless of dataset source

`ColumnMappedTextInstructionDataset` is a **map-style** dataset (`torch.utils.data.Dataset`): it supports `len(ds)` and `ds[i]`, and it loads data **non-streaming**.

It supports two data sources out-of-the-box:

1. **Local JSON/JSONL files** - pass a single file path or a list of paths on disk. Newline-delimited JSON works great.
2. **Hugging Face Hub** - point to any dataset repo (`org/dataset`) that contains the required columns.

For **streaming** (including **Delta Lake / Databricks**), use [`ColumnMappedTextInstructionIterableDataset`](column-mapped-text-instruction-iterable-dataset.md). The iterable variant always streams by design to avoid accidentally materializing entire datasets to disk/memory.

---
## Quickstart
The fastest way to sanity-check the loader is to point it at an existing Hugging Face dataset and print the first sample. This section provides a minimal, runnable example to help you quickly try out the dataset.

```python
from transformers import AutoTokenizer
from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset import ColumnMappedTextInstructionDataset

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

ds = ColumnMappedTextInstructionDataset(
    path_or_dataset_id="Muennighoff/natural-instructions",
    column_mapping={
      "context": "definition",
      "question": "inputs",
      "answer": "targets"
    },
    tokenizer=tokenizer,
    answer_only_loss_mask=True,
)

sample = ds[0]
print(sample.keys())

# Typical keys include: input_ids, labels, attention_mask (and an internal ___PAD_TOKEN_IDS___ helper).
# Note: when answer_only_loss_mask=True, prompt tokens are masked in labels with -100
# (the standard CrossEntropy "ignore_index").
```

The code above is intended only for a quick sanity check of the dataset and its tokenization output. For training or production use, configure the dataset using YAML as shown below. YAML offers a reproducible, maintainable, and scalable way to specify dataset and tokenization settings.

---
## Usage Examples

This section provides practical usage examples, including how to load remote datasets, work with local files, and configure pipelines using YAML recipes.

### Local JSONL Example

Assume you have a local newline-delimited JSON file at `/data/my_corpus.jsonl`
with the simple schema `{instruction, output}`. A few sample rows:

```jsonl
{"instruction": "Translate 'Hello' to French", "output": "Bonjour"}
{"instruction": "Summarize the planet Neptune.", "output": "Neptune is the eighth planet from the Sun."}
```

You can load it using Python code like:

```python
local_ds = ColumnMappedTextInstructionDataset(
    path_or_dataset_id=["/data/my_corpus_1.jsonl", "/data/my_corpus_2.jsonl"], # can also be a single path (string)
    column_mapping={
        "question": "instruction",
        "answer": "output",
    },
    tokenizer=tokenizer,
    answer_only_loss_mask=False,  # compute loss over full sequence
)

print(local_ds[0].keys())   # dict_keys(['input_ids', 'labels', 'attention_mask', '___PAD_TOKEN_IDS___'])
```

You can configure the dataset entirely from your recipe YAML. For example:
```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id:
    - /data/my_corpus_1.jsonl
    - /data/my_corpus_2.jsonl
  column_mapping:
    question: instruction
    answer: output
  answer_only_loss_mask: false
```


### Remote Dataset Example

In the following section, we demonstrate how to load the instruction-tuning corpus
[`Muennighoff/natural-instructions`](https://huggingface.co/datasets/Muennighoff/natural-instructions).
The dataset schema is `{task_name, id, definition, inputs, targets}`.

The following are examples from the training split:

```jsonl
{
  "task_name": "task001_quoref_question_generation",
  "id": "task001-abc123",
  "definition": "In this task, you're given passages that...",
  "inputs": "Passage: A man is sitting at a piano...",
  "targets": "What is the first name of the person who doubted it would be explosive?"
}
{
  "task_name": "task002_math_word_problems",
  "id": "task002-def456",
  "definition": "Solve the following word problem.",
  "inputs": "If there are 3 apples and you take 2...",
  "targets": "1"
}
```

For basic QA fine-tuning, we usually map `definition → context`, `inputs → question`, and `targets → answer` as follows:

```python
from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset import (
    ColumnMappedTextInstructionDataset,
)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

remote_ds = ColumnMappedTextInstructionDataset(
    path_or_dataset_id="Muennighoff/natural-instructions",  # Hugging Face repo ID
    column_mapping={
        "context": "definition",  # high-level context
        "question": "inputs",      # the actual prompt / input
        "answer": "targets",       # expected answer string
    },
    tokenizer=tokenizer,
    split="train[:5%]",        # demo slice; omit (i.e., `split="train",`) for full data
    answer_only_loss_mask=True,
)
```

You can configure the entire dataset directly from your recipe YAML. For example:
```yaml
# dataset section of your recipe's config.yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: Muennighoff/natural-instructions
  split: train
  column_mapping:
    context: definition
    question: inputs
    answer: targets
  answer_only_loss_mask: true
```

### Streaming / Delta Lake / Databricks

:::{note}
`ColumnMappedTextInstructionDataset` does not support streaming or Delta Lake / Databricks sources. For those, use [`ColumnMappedTextInstructionIterableDataset`](column-mapped-text-instruction-iterable-dataset.md).
:::

:::{note}
Delta Lake / Databricks (including `delta_sql_query` and authentication) is supported only by `ColumnMappedTextInstructionIterableDataset`. See [`column-mapped-text-instruction-iterable-dataset.md`](column-mapped-text-instruction-iterable-dataset.md) for details.
:::

### Advanced Options
| Arg                     | Default | Description |
|-------------------------|---------|-------------|
| `split`                 | `"train"` | Which split to pull from a HF repo (`train`, `validation`, etc.). Ignored for local JSON/JSONL. |
| `name`                  | `None`    | Name of the Hugging Face dataset configuration/subset to load. |
| `answer_only_loss_mask` | `True`    | Mask prompt tokens in `labels` with `-100` (the standard CrossEntropy `ignore_index`). |
| `use_hf_chat_template`  | `False`   | If `True` and the tokenizer supports chat templates, format as a system/user/assistant conversation via `tokenizer.apply_chat_template(...)`. |
| `seq_length`            | `None`    | Optional max sequence length; used for padding/truncation when enabled. |
| `padding`               | `"do_not_pad"` | Padding strategy passed to the tokenizer (`"do_not_pad"`, `"max_length"`, `True`, etc.). |
| `truncation`            | `"do_not_truncate"` | Truncation strategy passed to the tokenizer (`"do_not_truncate"`, `True`, etc.). |
| `limit_dataset_samples` | `None`    | Optionally load only the first \(N\) samples (useful for debugging). |

---
## Tokenization Paths
This section explains how the dataset formats and tokenizes samples.

`ColumnMappedTextInstructionDataset` produces standard next-token training tensors:

- `input_ids`
- `labels`
- `attention_mask`

When `answer_only_loss_mask=True`, prompt tokens are masked in `labels` with `-100` (the standard CrossEntropy `ignore_index`).

The dataset supports two formatting paths:

1. **Chat-template path (opt-in)**: if `use_hf_chat_template=True` and the tokenizer exposes a `chat_template` and `apply_chat_template`, the dataset builds messages like:

   `[{"role": "system", "content": <context or "">}, {"role": "user", "content": <question or "">}, {"role": "assistant", "content": <answer>}]`

   and tokenizes them via `tokenizer.apply_chat_template(..., tokenize=True, return_dict=True)`.

2. **Plain prompt/completion path (default)**: otherwise the dataset concatenates prompt and answer and tokenizes the result.

In both cases, `labels` are the next-token targets (shifted by one relative to `input_ids`). The dataset also includes an internal `___PAD_TOKEN_IDS___` field used downstream for padding.

---
## Parameter Requirements

The following section lists important requirements and caveats for correct usage.
- `column_mapping` must include `answer`, and must include at least one of `context` or `question` (2- or 3-column mapping only).
- If `use_hf_chat_template=True`, the tokenizer must support chat templates (`chat_template` + `apply_chat_template`).

---
## Slurm Configuration for Distributed Training

For distributed training on Slurm clusters, add a `slurm` section to your YAML configuration. This section configures the Slurm batch job parameters and automatically generates the appropriate `#SBATCH` directives.

### Basic Slurm Configuration

Add the following section to your YAML configuration:

```yaml
# Your existing model, dataset, training config...
step_scheduler:
  grad_acc_steps: 4
  num_epochs: 1

model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B

dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: Muennighoff/natural-instructions
  column_mapping:
    context: definition
    question: inputs
    answer: targets

# Add Slurm configuration
slurm:
  job_name: llm-finetune
  nodes: 1
  ntasks_per_node: 8
  time: 00:30:00
  account: your_account
  partition: gpu
  container_image: nvcr.io/nvidia/nemo-automodel:25.11.00
  gpus_per_node: 8
```

### Multi-Node Slurm Configuration

:::{note}
**Multi-Node Training**: When using Hugging Face datasets in multi-node setups, you need shared storage accessible by all nodes. Set the `HF_DATASETS_CACHE` environment variable to point to a shared directory (e.g., `HF_DATASETS_CACHE=/shared/hf_cache`) in the YAML file as shown, to ensure all nodes can access the cached datasets.
:::

When using multiple nodes with Hugging Face datasets:

1. **Shared Storage**: Ensure all nodes can access the same storage paths
2. **HF Cache**: Set `hf_home` to a shared directory accessible by all nodes
3. **Environment Variables**: Use `env_vars` to set `HF_DATASETS_CACHE` to the shared location

```yaml
slurm:
  job_name: llm-finetune-multi-node # name of the slurm job
  nodes: 4 # number of nodes to use
  ntasks_per_node: 8 # Number of tasks per node (typically equals number of GPUs)
  time: 02:00:00 # Maximum job runtime (format: `HH:MM:SS`)
  account: your_account # Slurm account to charge resources to
  partition: gpu # Slurm partition to submit to
  container_image: nvcr.io/nvidia/nemo-automodel:25.11.00 # Container image to use for the job
  gpus_per_node: 8 # Number of GPUs per node (adds `#SBATCH --gpus-per-node=N`)
  # Optional: Add extra mount points if needed
  extra_mounts: # Additional mount points for the container
    - /lustre:/lustre
    - /shared:/shared
  # Optional: Specify custom HF_HOME location
  hf_home: /shared/hf_cache # Custom Hugging Face cache directory on shared disk space.
  # Optional: Specify custom env vars
  env_vars: # Additional environment variables
     HF_DATASETS_CACHE: /shared/hf_cache  # Similar to hf_home; useful when you use a different directory for datasets.
  # Optional: Specify custom job directory
  job_dir: /path/to/slurm/jobs
```


---
### That's It!
With the mapping specified, the rest of the NeMo Automodel pipeline (pre-tokenization, packing, collate-fn, *etc.*) works as usual.

# Use the ColumnMappedTextInstructionDataset

This guide explains how to use `ColumnMappedTextInstructionDataset` to quickly and flexibly load instruction-answer datasets for LLM fine-tuning, with minimal code changes and support for various data formats and tokenization strategies.

The `ColumnMappedTextInstructionDataset` is a lightweight, plug-and-play helper that lets you train on instruction–answer style corpora without writing custom Python for every new schema. You simply specify which columns map to logical fields like `context`, `question`, and `answer`, and the loader handles the rest automatically. This enables:

* Quick prototyping across diverse instruction datasets
* Schema flexibility without needing codebase changes
* Consistent field names for training loops, regardless of dataset source

It supports two data sources out-of-the-box and optionally streams them so they never fully reside in memory:

1. **Local JSON/JSONL files** - pass a single file path or a list of paths on disk. The newline-delimited JSON works great.
2. **Hugging Face Hub** - point to any dataset repo (`org/dataset`) that contains the required columns.

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
    answer_only_loss_mask=False,
)

print(next(iter(ds)))

# The above command will print:
# {
#   'input_ids': [128000, 12465, 425, 25, 578, 38413, 61941, ... 78863, 36373, 7217],
#   'labels':    [12465, 425, 25, 578, 38413, 61941, ..., 78863, 36373, 7217, 30],
#   'loss_mask': [1, 1, 1, 1, ..., 1, 1, 1, 1]
# }

# if you disable streaming (i.e., pass `streaming=False`), then you can inspect samples with
# print(ds[0])
# or inspect the length of the dataset
# print(len(ds))
```

The code above is intended only for a quick sanity check of the dataset and its tokenization output. For training or production use, configure the dataset using YAML as shown below. YAML offers a reproducible, maintainable, and scalable way to specify dataset and tokenization settings.

---
## Usage Examples

This section provides practical usage examples, including how to load remote datasets, work with local files, and configure pipelines using YAML recipes.

### Local JSONL Example

Assume you have a local newline-delimited JSON file at `/data/my_corpus.jsonl`
with the simple schema `{instruction, output}`.  A few sample rows:

```json
{"instruction": "Translate 'Hello' to French", "output": "Bonjour"}
{"instruction": "Summarize the planet Neptune.", "output": "Neptune is the eighth planet from the Sun."}
```

You can load it using python code like:

```python
local_ds = ColumnMappedTextInstructionDataset(
    path_or_dataset_id=["/data/my_corpus_1.jsonl", "/data/my_corpus_2.jsonl"]  # can also be a single path (string)
    column_mapping={
        "question": "instruction",
        "answer": "output",
    },
    tokenizer=tokenizer,
    answer_only_loss_mask=False,  # compute loss over full sequence
)

print(remote_ds[0].keys())  # {'context', 'question', 'answer'}
print(local_ds[0].keys())   # {'question', 'answer'}
```

You can configure the dataset entirely from your recipe YAML.  For example:
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

```json
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
        "question": "inputs",         # the actual prompt / input
        "answer": "targets",          # expected answer string
    },
    tokenizer=tokenizer,
    split="train[:5%]",        # demo slice; omit (i.e. `split="train",`) for full data
    answer_only_loss_mask=True,
    start_of_turn_token="<|assistant|>",
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
  start_of_turn_token: "<|assistant|>"
```

### Advanced Options
| Arg                     | Default | Description |
|-------------------------|---------|-------------|
| `split`                 | `None`  | Which split to pull from a HF repo (`train`, `validation`, *etc.*). Ignored for local files. |
| `name`                  | `None`  | Name of the dataset configuration/subset to load |
| `answer_only_loss_mask` | `True`  | Create a `loss_mask` where only the answer tokens contribute to the loss. Requires `start_of_turn_token`. |
| `start_of_turn_token`   | `None`  | String token marking the assistant's response. Required when `answer_only_loss_mask=True` for tokenizers with chat template. |

---
## Tokenisation Paths
This section explains how the dataset tokenizes both inputs and outputs, and how it adapts to different tokenizers.
`ColumnMappedTextInstructionDataset` automatically picks one of two tokenization
strategies depending on the capabilities of the provided tokenizer:

1. **Chat-template path**: if the tokenizer exposes a
   `chat_template` attribute and an `apply_chat_template` method, the
   dataset will:

   * build a list of messages in the form
      `[{"role": "user", "content": <prompt>}, {"role": "assistant", "content": <answer>}]`,
   * call `tokenizer.apply_chat_template(messages)` to convert them to
      `input_ids`,
   * derive `labels` by shifting `input_ids` one position to the right, and
   * compute `loss_mask` by locating the second occurrence of
      `start_of_turn_token` (this marks the assistant response boundary).  All
      tokens that belong to the user prompt are set to `0`, while the answer
      tokens are `1`.

2. **Plain prompt/completion path**: if the tokenizer has no chat template, the
   dataset falls back to a classic prompt and answer concatenation:

   ```text
   "<context> <question> " + "<answer>"
   ```

   The helper strips any trailing *eos* from the prompt and leading *bos* from
   the answer so that the two halves join cleanly.

Regardless of the path, the output dict is always:

```python
{
    "input_ids": [...],  # one token shorter than the full sequence
    "labels":     [...], # next-token targets
    "loss_mask":  [...], # 1 for tokens contributing to the loss
}
```

---
## Parameter Requirements

The following section lists important requirements and caveats for correct usage.
* `answer_only_loss_mask=True` requires a start_of_turn_token string that exists in the tokenizer's vocabulary and can be successfully encoded when the helper performs a lookup. Otherwise, a `ValueError` is raised at instantiation time.
* Each sample must include at least one of `context` or `question`; omitting both will result in a `ValueError`.

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
  container_image: nvcr.io/nvidia/nemo:25.07
  gpus_per_node: 8
```

### Multi-Node Slurm Configuration

:::note
**Note for Multi-Node Training**: When using Hugging Face datasets in multi-node setups, you need shared storage accessible by all nodes. Set the `HF_DATASETS_CACHE` environment variable to point to a shared directory (e.g., `HF_DATASETS_CACHE=/shared/hf_cache`) in the yaml file as shown, to ensure all nodes can access the cached datasets.
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
  container_image: nvcr.io/nvidia/nemo:25.07 # Container image to use for the job
  gpus_per_node: 8 # Number of GPUs per node (adds `#SBATCH --gpus-per-node=N`)
  # Optional: Add extra mount points if needed
  extra_mounts: # Additional mount points for the container
    - /lustre:/lustre
    - /shared:/shared
  # Optional: Specify custom HF_HOME location
  hf_home: /shared/hf_cache # Custom Hugging Face cache directory on shared dist space.
  # Optional: Specify custom env vars
  env_vars: # Additional environment variables
    HF_DATASETS_CACHE: /shared/hf_cache  # similar to hf_home, useful when use a different directory for datasets.
  # Optional: Specify custom job directory
  job_dir: /path/to/slurm/jobs
```


---
### That's It!
With the mapping specified, the rest of the NeMo Automodel pipeline (pre-tokenisation, packing, collate-fn, *etc.*) works as usual.

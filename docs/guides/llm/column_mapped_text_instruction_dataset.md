# ColumnMappedTextInstructionDataset

The `ColumnMappedTextInstructionDataset` is a **light-weight, plug-and-play** helper that lets you train on *instruction-answer* style corpora **without writing custom Python for every new schema**.  
You simply specify **which column in your source dataset maps to which logical field** (`context`, `question`, `answer`, *etc.*) and the loader does the rest.

It supports two data sources out-of-the-box **and optionally streams them so they never fully reside in memory**:

1. **Hugging Face Hub** - point to any dataset repo (`org/dataset`) that contains your desired columns.
2. **Local JSON/JSONL files** - pass one file path *or* a list of paths on disk (newline-delimited JSON works great).

> **When to use it?**
> - Quick prototyping across many instruction datasets.  
> - No need to edit the codebase for each new schema.  
> - Unified field names downstream ‑- your training loop can rely on the same keys regardless of origin.

---
## Basic Python usage
### Remote dataset example

Below we demonstrate how to load the instruction-tuning corpus
[`Muennighoff/natural-instructions`](https://huggingface.co/datasets/Muennighoff/natural-instructions).
The dataset schema is `{task_name, id, definition, inputs, targets}`.

Example lines (train split):

```json
{"task_name":"task001_quoref_question_generation","id":"task001-abc123","definition":"In this task, you're given passages that...","inputs":"Passage: A man is sitting at a piano...","targets":"What is the first name of the person who doubted it would be explosive?"}
{"task_name":"task002_math_word_problems","id":"task002-def456","definition":"Solve the following word problem.","inputs":"If there are 3 apples and you take 2...","targets":"1"}
```

For basic QA fine-tuning we usually map `definition → instruction`, `inputs → question`, and `targets → answer`:

```python
from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset import (
    ColumnMappedTextInstructionDataset,
)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

remote_ds = ColumnMappedTextInstructionDataset(
    path_or_dataset_id="Muennighoff/natural-instructions",  # Hugging Face repo ID
    column_mapping={
        "instruction": "definition",  # high-level instruction
        "question": "inputs",         # the actual prompt / input
        "answer": "targets",          # expected answer string
    },
    tokenizer=tokenizer,
    split="train[:5%]",        # demo slice; omit (i.e. `split="train",`) for full data
    answer_only_loss_mask=True,
    start_of_turn_token="<|assistant|>",
    streaming=True,              # <── stream instead of download whole dataset
)
```

### Local JSONL example

Assume you have a local newline-delimited JSON file at `/data/my_corpus.jsonl`
with the simple schema `{instruction, output}`.  A few sample rows:

```json
{"instruction": "Translate 'Hello' to French", "output": "Bonjour"}
{"instruction": "Summarize the planet Neptune.", "output": "Neptune is the eighth planet from the Sun."}
```

You can load it like so:

```python
local_ds = ColumnMappedTextInstructionDataset(
    path_or_dataset_id="/data/my_corpus.jsonl",  # can also be [list_of_paths]
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

---
## YAML integration (NeMo Automodel recipe)
You can configure the dataset **entirely from your recipe YAML**.  Example:
```yaml
# dataset section of your recipe's config.yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: Muennighoff/natural-instructions
  split: train
  column_mapping:
    context: context
    question: question
    answer: answer
  answer_only_loss_mask: true
  start_of_turn_token: "<|assistant|>"
```
For a local file you would write:
```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: 
    - /data/alpaca_part1.jsonl
    - /data/alpaca_part2.jsonl
  column_mapping:
    question: instruction
    answer: output
  answer_only_loss_mask: false
```

---
## Column mapping cheat-sheet
| **Logical field** | **Enum** (optional)          | **Typical meaning**                 |
|-------------------|------------------------------|-------------------------------------|
| `context`         | `ColumnTypes.Context`        | Additional supporting text          |
| `question`        | `ColumnTypes.Question`       | The user’s instruction / query      |
| `answer`          | `ColumnTypes.Answer`         | The model’s target answer / output  |

You *don’t* need to use the enum in the mapping - plain strings work fine.  The enum is provided merely to avoid typos when building mappings in code.

---
## Advanced options
| Arg                     | Default | Description |
|-------------------------|---------|-------------|
| `split`                 | `None`  | Which split to pull from a HF repo (`train`, `validation`, *etc.*). Ignored for local files. |
| `streaming`             | `False` | If `True`, loads the dataset in *streaming* mode (an HF `IterableDataset`). Useful for very large corpora or when you want to start training before the full download completes.  When enabled, `len(...)` and random access (`dataset[idx]`) are **not** available — iterate instead. |
| `answer_only_loss_mask` | `True`  | Create a `loss_mask` where only the answer tokens contribute to the loss. Requires `start_of_turn_token`. |
| `start_of_turn_token`   | `None`  | String token marking the assistant’s response. Required when `answer_only_loss_mask=True`. |

---
## Tokenisation paths

`ColumnMappedTextInstructionDataset` automatically picks *one of two* tokenization
strategies depending on the capabilities of the provided tokenizer:

1. **Chat-template path**: if the tokenizer exposes a
   `chat_template` attribute **and** an `apply_chat_template` method, the
   dataset will:

   a. build a list of messages in the form
      `[{"role": "user", "content": <prompt>}, {"role": "assistant", "content": <answer>}]`,
   b. call `tokenizer.apply_chat_template(messages)` to convert them to
      `input_ids`,
   c. derive `labels` by shifting `input_ids` one position to the right, and
   d. compute `loss_mask` by locating the *second* occurrence of
      `start_of_turn_token` (this marks the assistant response boundary).  All
      tokens that belong to the user prompt are set to **0**, while the answer
      tokens are **1**.

2. **Plain prompt/completion path**: if the tokenizer has no chat template the
   dataset falls back to a classic *prompt + answer* concatenation:

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
### Parameter gotchas

* `answer_only_loss_mask=True` (**default**) requires *both*:
  - a **valid** `start_of_turn_token` string that exists in the tokenizer
    vocabulary, and
  - the tokenizer to be able to encode that token when the helper looks it up.

  Otherwise a `ValueError` is raised at instantiation time.
* At least **one** of `context` *or* `question` must be present in the mapping;
  passing a sample with both missing will raise a `ValueError`.

---
## Dataset schema examples
Below are two minimal JSONL rows and the corresponding `column_mapping` you would use.

### Simple QA pair (local JSONL)
```json
{"question": "Who wrote *Pride and Prejudice*?", "answer": "Jane Austen."}
```
mapping:
```python
{"question": "question", "answer": "answer"}
```

### Chat-style with context (HF)
```json
{
  "context": "You are an AI writing assistant.",
  "question": "Rewrite the following sentence in active voice...",
  "answer": "The cat chased the mouse."
}
```
mapping:
```yaml
context: context
question: question
answer: answer
```

---
### That’s it!
With the mapping specified, the rest of the NeMo Automodel pipeline (pre-tokenisation, packing, collate-fn, *etc.*) works as usual.  Happy finetuning! 🚀 
# Configure Chat Templates for LLM Fine-Tuning

When [fine-tuning](finetune.md) an instruct model on a custom dataset, the training data must be formatted with the same special tokens the model expects at inference time. A **chat template** is a template string (stored in `tokenizer.chat_template`) that defines this formatting — it controls how system prompts, user messages, and assistant responses are wrapped with model-specific control tokens such as `<|im_start|>` / `<|im_end|>` (Qwen) or `<|start_header_id|>` / `<|end_header_id|>` (Llama-3). Chat templates are most commonly written in [Jinja](https://jinja.palletsprojects.com/), a widely used templating language, though other formats exist.

This guide shows how to enable chat template formatting in NeMo AutoModel for LLM fine-tuning. It covers two dataset classes and one override mechanism.

:::{note}
**When to use a chat template:** You are fine-tuning an instruct model (model name typically contains `Instruct`, `Chat`, or an `-it` suffix) and want inference to work with `tokenizer.apply_chat_template(...)` or an OpenAI-compatible API.

**When you do NOT need one:** You are fine-tuning a base model for completion-style tasks, or doing language modeling / pretraining. In that case, see [Integrate Your Own Text Dataset](dataset.md).
:::

---

## Choose Your Path

Your data format determines which dataset class to use:

| Your data format | Dataset class | Chat template behavior | YAML key to set |
|------------------|--------------|----------------------|-----------------|
| Flat columns (e.g. `instruction`, `input`, `output`) | [`ColumnMappedTextInstructionDataset`](../../../nemo_automodel/components/datasets/llm/column_mapped_text_instruction_dataset.py) | Off by default; opt in | `use_hf_chat_template: true` |
| OpenAI messages list (`{"messages": [...]}`) | [`ChatDataset`](../../../nemo_automodel/components/datasets/llm/chat_dataset.py) | Always on (required) | None needed |

If you need to override the tokenizer's built-in template or provide one where none exists, see [Custom or Missing Template](#custom-or-missing-template). If your data format does not fit either class, see [Integrate Your Own Text Dataset](dataset.md) for writing a custom dataset class.

:::{note}
The examples below use `Qwen/Qwen3-4B`, which ships with a built-in chat template. Any instruct model with a `chat_template` in its tokenizer config works the same way.
:::

---

## Flat Instruction Columns

Use `ColumnMappedTextInstructionDataset` when your data has simple columns like `instruction`, `input`, `output` (or any other names).

### Column-to-Role Mapping

The `column_mapping` keys map your dataset columns to chat roles:

| `column_mapping` key | Chat role | Required |
|---------------------|-----------|----------|
| `context` | `system` | No (defaults to `""`) |
| `question` | `user` | Yes (or `context`) |
| `answer` | `assistant` | Yes |

### Example Data

```json
{"instruction": "Translate 'Hello' to French", "input": "", "output": "Bonjour"}
{"instruction": "What is the capital of Japan?", "input": "Geography quiz", "output": "Tokyo"}
```

### YAML Configuration

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: Qwen/Qwen3-4B

dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: /path/to/train.jsonl
  column_mapping:
    context: input
    question: instruction
    answer: output
  answer_only_loss_mask: true
  use_hf_chat_template: true  # <-- enables chat template formatting
```

- **`answer_only_loss_mask`**: When `true`, prompt tokens are masked with `-100` so the training loss is computed only on the assistant response. This is the recommended setting for instruction tuning.
- The tokenizer is injected into the dataset by the recipe at runtime — you do not need to specify it in the YAML `dataset` block.

If your dataset has no system/context field, use a two-column mapping:

```yaml
  column_mapping:
    question: instruction
    answer: output
```

### What the Model Sees

For Qwen3-4B with the example data above, the two formatting paths produce:

**Without chat template** (default — plain concatenation):
```
What is the capital of Japan? Tokyo
```

**With chat template** (`use_hf_chat_template: true`):
```
<|im_start|>system
Geography quiz<|im_end|>
<|im_start|>user
What is the capital of Japan?<|im_end|>
<|im_start|>assistant
Tokyo<|im_end|>
```

With `answer_only_loss_mask: true`, only the assistant response tokens contribute to the training loss in both cases.

---

## OpenAI Messages Format

Use `ChatDataset` when your data is already structured as a `messages` list in OpenAI chat format. This is the preferred class for multi-turn conversations and tool-calling datasets (see also the [Tool Calling guide](toolcalling.md)).

`ChatDataset` **always** applies the tokenizer's chat template — there is no opt-in flag.

### Example Data

```json
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
```

Multi-turn:

```json
{
  "messages": [
    {"role": "system", "content": "You are a travel assistant."},
    {"role": "user", "content": "I want to visit Paris."},
    {"role": "assistant", "content": "Paris is a wonderful choice! When are you planning to go?"},
    {"role": "user", "content": "Next summer."},
    {"role": "assistant", "content": "Summer is ideal. I recommend visiting the Eiffel Tower and the Louvre."}
  ]
}
```

### YAML Configuration

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: Qwen/Qwen3-4B

dataset:
  _target_: nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset
  path_or_dataset_id: /path/to/train.jsonl
```

---

## Custom or Missing Template

There are two distinct cases where you need the `chat_template` parameter on `ChatDataset`:

1. **Override**: The tokenizer has a built-in template, but you want a different format (e.g., to match your deployment API).
2. **Provision**: The tokenizer has no template at all (common with base model tokenizers), and you need to supply one.

Here is a simplified ChatML-style template for reference:

```jinja
{% for message in messages %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
```

Every model family defines its own template with its own control tokens — the template your model ships with will likely differ from this example. Always inspect the tokenizer's built-in template (see [Verifying the Template Before Training](#verifying-the-template-before-training)) before writing a custom one.

In both cases, the `chat_template` parameter accepts the following input forms:

| Input | Behavior |
|-------|----------|
| Path to a `.jinja` file | File content is used as the template |
| Path to a JSON file with a `chat_template` key | That key's value is extracted and used |
| A literal Jinja string | Used directly |
| `null` or omitted | Tokenizer's built-in template is used (default) |

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset
  path_or_dataset_id: /path/to/train.jsonl
  chat_template: /path/to/my_template.jinja   # or inline Jinja string
```

:::{note}
The `chat_template` override is available on `ChatDataset` only. `ColumnMappedTextInstructionDataset` always uses the tokenizer's built-in template when `use_hf_chat_template: true`.
:::

---

## Verifying the Template Before Training

Run this snippet to confirm the template produces the expected output:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris"},
]

print(tokenizer.apply_chat_template(messages, tokenize=False))
```

---

## Troubleshooting

### "Tokenizer lacks a usable chat template"
The tokenizer does not have a `chat_template` attribute. This is common with base model tokenizers. There are three ways to resolve it:

**Option 1 — Switch to the instruct variant of the model.** Instruct models (names containing `Instruct`, `Chat`, or `-it`) ship with a built-in template. For example, `Qwen/Qwen3-4B` has one but the corresponding base model may not.

**Option 2 — Supply a custom template via `ChatDataset`.** Point the `chat_template` parameter to a `.jinja` file or pass a literal Jinja string:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.chat_dataset.ChatDataset
  path_or_dataset_id: /path/to/train.jsonl
  chat_template: /path/to/my_template.jinja
```

You can find ready-made templates in the model's Hugging Face repository (look for `tokenizer_config.json` → `chat_template` key) or write your own following the example in [Custom or Missing Template](#custom-or-missing-template).

**Option 3 — Disable chat template formatting.** If you do not need special tokens (e.g., you are fine-tuning for plain completion), set `use_hf_chat_template: false` on `ColumnMappedTextInstructionDataset` to fall back to plain concatenation:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: /path/to/train.jsonl
  column_mapping:
    question: instruction
    answer: output
  use_hf_chat_template: false  # plain concatenation, no special tokens
```

### Loss not decreasing
- Ensure `answer_only_loss_mask: true` so loss is computed only on the assistant tokens.
- Check that your `column_mapping` keys match columns that actually exist in your dataset.
- Verify the template output with the snippet above.

### Tokenizer produces unexpected output
- Some tokenizers require `trust_remote_code=True`. NeMo AutoModel handles this automatically for supported models.
- If using a custom `chat_template`, test the Jinja syntax with `tokenizer.apply_chat_template()` before launching training.

---

## Appendix: Internals

When `use_hf_chat_template: true`, `ColumnMappedTextInstructionDataset` converts each sample to a message list and delegates to `tokenizer.apply_chat_template`:

```python
messages = [
    {"role": "system", "content": "<value of context column>"},
    {"role": "user", "content": "<value of question column>"},
    {"role": "assistant", "content": "<value of answer column>"},
]
tokenized = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True)
```

`ChatDataset` follows the same path but reads the `messages` list directly from the data row. Some chat templates include a `{%- generation -%}` block — a Jinja directive that marks where the assistant's response begins. When this block is present, `apply_chat_template` can accept a `return_assistant_tokens_mask` flag to return a per-token boolean mask that identifies assistant tokens directly. When the block is absent, the mask is derived by tokenizing the prompt portion separately and computing the length difference.

The underlying formatting functions live in [`nemo_automodel/components/datasets/llm/formatting_utils.py`](../../../nemo_automodel/components/datasets/llm/formatting_utils.py).

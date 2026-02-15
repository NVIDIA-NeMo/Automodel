# Chat Templates

:::{tip}
**TL;DR** -- Chat templates control how conversations are tokenized for training and inference. Mismatched templates between training and inference cause bad outputs.
:::

## What Are Chat Templates?

- **Chat templates** define how user/assistant turns are formatted before tokenization (e.g., `<|user|>`, `<|assistant|>`, special tokens)
- **Every instruct model** ships with a chat template baked into its tokenizer
- **Training and inference must use the same template** -- otherwise the model sees a different format than it learned

## Check Your Model's Template

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
print(tokenizer.chat_template)
```

## Key YAML Fields

| Field | What It Does | Default |
|-------|-------------|---------|
| `use_hf_chat_template` | Use the tokenizer's built-in template | `true` |
| `chat_template` | Override with a custom Jinja2 template string | (none) |
| `start_of_turn_token` | Token marking start of a user/assistant turn | Model-specific |
| `start_of_response_token` | Token marking where the model should start generating | Model-specific |

## Common Patterns

| Model Family | Turn Format | Notes |
|-------------|------------|-------|
| **Qwen** | `<|im_start|>system\n...<|im_end|>` | ChatML format |
| **Llama 3** | `<|start_header_id|>user<|end_header_id|>\n\n...` | Custom format |
| **Gemma** | `<start_of_turn>user\n...<end_of_turn>` | Google format |
| **Mistral** | `[INST] ... [/INST]` | Bracket format |

## Common Mistakes

:::{warning}
**Training with one template, inferring with another** -- The most common source of garbled or repetitive output. Always verify that your inference code uses the same tokenizer and template as training.
:::

:::{warning}
**Forgetting `answer_only_loss_mask`** -- Without this, the model trains on the user prompt too (not just the response). Set `answer_only_loss_mask: true` in your dataset config for instruction-tuning tasks.
:::

:::{warning}
**Custom template syntax errors** -- Chat templates use Jinja2. A missing `{% endif %}` or wrong variable name silently corrupts your data. Test with `tokenizer.apply_chat_template()` before training.
:::

## Testing Your Template

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

messages = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]

# Verify the formatted output looks correct
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
print(formatted)
```

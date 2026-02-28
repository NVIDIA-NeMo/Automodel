# Use the ChatDataset for Conversations and Tool Calling

`ChatDataset` is the recommended way to fine-tune LLMs on conversational data in NeMo Automodel. If your training data uses the [OpenAI messages format](https://platform.openai.com/docs/guides/text?api=chat) — or you can convert it — this is the dataset class to use.

It handles single-turn Q&A, multi-turn dialogue, system prompts, tool calling, and tool responses out of the box. You supply a JSON/JSONL file (or a Hugging Face dataset ID), point your YAML config at it, and start training.

- **Class**: `nemo_automodel.components.datasets.llm.ChatDataset`
- **Data format**: OpenAI messages (`{"messages": [...], "tools": [...]}`)
- **Sources**: local JSON / JSONL files, or any Hugging Face Hub dataset with a `messages` column

:::{tip}
If your data is **not** in OpenAI messages format but has simple `instruction` / `output` columns, see [ColumnMappedTextInstructionDataset](column-mapped-text-instruction-dataset.md) instead.
:::

---

## Data Format

Each sample is a JSON object with a `messages` list. An optional `tools` list provides tool/function schemas for function-calling scenarios.

### Simple Conversation (no tools)

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

### Multi-Turn Conversation

```json
{
  "messages": [
    {"role": "user", "content": "Summarize photosynthesis."},
    {"role": "assistant", "content": "Photosynthesis is the process by which plants convert sunlight into energy."},
    {"role": "user", "content": "What role does chlorophyll play?"},
    {"role": "assistant", "content": "Chlorophyll absorbs light energy, primarily from the blue and red wavelengths."}
  ]
}
```

### Tool Calling

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather in Seattle?"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "get_weather",
            "arguments": "{\"city\": \"Seattle\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "tool_call_id": "call_1",
      "content": "{\"temperature\": 65, \"condition\": \"cloudy\"}"
    },
    {
      "role": "assistant",
      "content": "It's 65°F and cloudy in Seattle."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string"}
          },
          "required": ["city"]
        }
      }
    }
  ]
}
```

Store your samples as **one JSON object per line** in a `.jsonl` file.

---

## Quickstart

The fastest way to try it out — point at a local JSONL file and print the first tokenized sample:

```python
from transformers import AutoTokenizer
from nemo_automodel.components.datasets.llm import ChatDataset

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

ds = ChatDataset(
    path_or_dataset_id="data/my_chat_data/train.jsonl",
    tokenizer=tokenizer,
    seq_length=2048,
    padding="max_length",
    truncation="longest_first",
)

sample = ds[0]
print(sample.keys())  # dict_keys(['input_ids', 'labels', 'attention_mask'])
```

For training, configure the dataset entirely from YAML as shown below.

---

## YAML Configuration

### Minimal Config (local JSONL)

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.ChatDataset
  path_or_dataset_id: data/my_chat_data/train.jsonl
  seq_length: 2048
  padding: max_length
  truncation: longest_first

validation_dataset:
  _target_: nemo_automodel.components.datasets.llm.ChatDataset
  path_or_dataset_id: data/my_chat_data/validation.jsonl
  seq_length: 2048
  padding: max_length
  truncation: longest_first
```

### With a Hugging Face Dataset

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.ChatDataset
  path_or_dataset_id: Salesforce/xlam-function-calling-60k
  split: train
  seq_length: 2048
  start_of_turn_token: "<start_of_turn>"
```

### Answer-Only Loss

To train the model only on assistant responses (masking user/system tokens in the loss), provide `start_of_turn_token` — the token that marks where the assistant response begins in the rendered chat template. This is model-specific:

| Model family | `start_of_turn_token` |
|---|---|
| Llama 3 Instruct | `<|start_header_id|>assistant` |
| Gemma / FunctionGemma | `<start_of_turn>` |
| Qwen 2.5 Instruct | `<|im_start|>assistant` |

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.ChatDataset
  path_or_dataset_id: data/my_chat_data/train.jsonl
  seq_length: 2048
  padding: max_length
  truncation: longest_first
  start_of_turn_token: "<|start_header_id|>assistant"
```

---

## End-to-End Example: MASSIVE NLU

The repository includes a complete worked example that fine-tunes Llama-3.2-3B-Instruct on the [AmazonScience/MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) NLU dataset for intent detection and entity extraction.

**Step 1** — Prepare the data (converts MASSIVE to OpenAI messages JSONL):

```bash
python examples/llm_finetune/prepare_massive_nlu.py --output_dir data/massive_nlu
```

**Step 2** — Launch training:

```bash
automodel examples/llm_finetune/llama3_2/llama_3_2_3b_instruct_massive_nlu_peft.yaml
```

See also the Qwen chat example config at `examples/llm_finetune/qwen/qwen2_5_7b_instruct_chat.yaml`.

---

## Parameter Reference

| Arg | Default | Description |
|---|---|---|
| `path_or_dataset_id` | *(required)* | Path to local JSON/JSONL file(s), or a Hugging Face dataset ID. |
| `tokenizer` | *(required)* | Tokenizer instance. Must have chat template support. Automatically injected when using YAML. |
| `split` | `None` | Dataset split (e.g., `"train"`, `"validation"`). Required for HF datasets. |
| `name` | `None` | HF dataset configuration/subset name. |
| `seq_length` | `None` | Maximum sequence length for padding/truncation. |
| `padding` | `"do_not_pad"` | Padding strategy: `"do_not_pad"`, `"max_length"`, `True`, etc. |
| `truncation` | `"do_not_truncate"` | Truncation strategy: `"do_not_truncate"`, `"longest_first"`, `True`, etc. |
| `start_of_turn_token` | `None` | Token marking assistant response start, for answer-only loss masking. |
| `chat_template` | `None` | Optional Jinja2 template to override the tokenizer's built-in chat template. |

---

## When to Use ChatDataset vs. Other Dataset Classes

| Scenario | Recommended class |
|---|---|
| Conversational data in OpenAI messages format | **ChatDataset** |
| Tool-calling / function-calling fine-tuning | **ChatDataset** |
| Simple instruction → response with column mapping | [ColumnMappedTextInstructionDataset](column-mapped-text-instruction-dataset.md) |
| Completion-style LM (context → continuation) | [HellaSwag](../dataset-overview.md#hellaswag-completion-sft) or custom class with `SFTSingleTurnPreprocessor` |
| Streaming over very large instruction datasets | [ColumnMappedTextInstructionIterableDataset](column-mapped-text-instruction-iterable-dataset.md) |

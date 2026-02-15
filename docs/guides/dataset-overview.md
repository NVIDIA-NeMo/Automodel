# Dataset Overview: LLM, VLM, and Retrieval Datasets in NeMo Automodel

:::{tip}
**TL;DR** -- Use the table below to pick the right dataset class for your data format. Then see the linked guide for setup details.
:::

## Which Dataset Class Should I Use?

| Your Data Format | Dataset Class | Guide |
|-----------------|--------------|-------|
| **JSONL instruction pairs** (`{"instruction": ..., "output": ...}`) | `ColumnMappedTextInstructionDataset` | [Column-Mapped Dataset](llm/column-mapped-text-instruction-dataset.md) |
| **Multi-turn conversations** (chat format) | `ColumnMappedTextInstructionDataset` with chat template | [Chat Templates](chat-templates.md) |
| **Completion / continuation** (prompt + gold ending) | `HellaSwag` pattern | See below |
| **Streaming / very large datasets** | `ColumnMappedTextInstructionIterableDataset` | [Iterable Dataset](llm/column-mapped-text-instruction-iterable-dataset.md) |
| **Image + text (VLM)** | `make_rdr_dataset` / `make_cord_v2_dataset` | [VLM Datasets](vlm/dataset.md) |
| **Pretraining tokens** | `MegatronPretraining` or `NanogptDataset` | [Pretraining Guide](llm/pretraining.md) |
| **Retrieval / bi-encoder** | `RetrievalDataset` | [Retrieval Dataset](llm/retrieval-dataset.md) |
| **Question-answering** (SQuAD-style) | `SQuAD` | See below |

---

- See also: [LLM datasets](llm/dataset.md), [VLM datasets](vlm/dataset.md), and [Biencoder retrieval dataset](llm/retrieval-dataset.md) for deeper, task-specific guides.

- If a dataset you need is missing, please open a [GitHub issue](https://github.com/NVIDIA-NeMo/Automodel/issues) with a short description and example schema so we can prioritize support.
---

## LLM Datasets

NeMo Automodel supports several common patterns for language modeling and instruction tuning.

- **HellaSwag (completion SFT)**
  - Wrapper: `nemo_automodel.components.datasets.llm.hellaswag.HellaSwag`
  - Use case: single-turn completion style SFT where a prompt (ctx) is followed by a gold continuation (ending)
  - Key args: `path_or_dataset`, `split`, `num_samples_limit`
### HellaSwag (Completion SFT)
- Wrapper: `nemo_automodel.components.datasets.llm.hellaswag.HellaSwag`
- Use case: single-turn completion-style SFT where a prompt (ctx) is followed by a gold continuation (ending)
- Key args: `path_or_dataset`, `split`, `num_samples_limit`
- Example YAML:
```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.hellaswag.HellaSwag
  path_or_dataset: rowan/hellaswag
  split: train
```

- **SQuAD-style Question Answering (QA) (instruction SFT)**
  - Factory: `nemo_automodel.components.datasets.llm.squad.make_squad_dataset`
  - Use case: instruction/QA tuning with either prompt+answer formatting or chat-template formatting
  - Notes:
    - If the tokenizer has a chat template and you want answer-only loss, you must provide `start_of_turn_token`.
    - Optional `seq_length` can be used for padding/truncation.
### SQuAD-Style Question Answering (QA) (Instruction SFT)
- Factory: `nemo_automodel.components.datasets.llm.squad.make_squad_dataset`
- Use case: instruction/QA tuning with either prompt-and-answer formatting or chat-template formatting
:::{note}
- If the tokenizer has a chat template and you want answer-only loss, you must provide `start_of_turn_token`.
- Optional `seq_length` can be used for padding/truncation.
:::
- Example YAML:
```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  split: train
  dataset_name: rajpurkar/squad
  start_of_turn_token: "<|assistant|>"
```

- **ColumnMappedTextInstructionDataset (generic instruction SFT)**
  - Class: `nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset`
  - Use case: quickly adapt instruction datasets by mapping your schema's columns to `context`, `question`, `answer`
  - Sources: local JSON/JSONL or Hugging Face Hub dataset ID
  - Notes:
    - For tokenizers with chat templates and answer-only loss, you may set `answer_only_loss_mask: true` and provide `start_of_turn_token`.
    - Supports streaming mode for large datasets (see [Streaming Datasets](#streaming-datasets) section below).
    - Map-style, non-streaming dataset (supports `len(ds)` and `ds[i]`)
    - For streaming (including Delta Lake / Databricks), use `ColumnMappedTextInstructionIterableDataset`
  - Example YAML:
```yaml
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
See the detailed guide, [Column-Mapped Text Instruction Dataset](llm/column-mapped-text-instruction-dataset.md), for more information.

- **ChatDataset (multi-turn conversations and tool calling)**
  - Class: `nemo_automodel.components.datasets.llm.ChatDataset`
  - Use case: multi-turn conversations and tool calling in OpenAI chat format
  - Sources: local JSON/JSONL or Hugging Face Hub dataset ID
  - Key args:
    - `path_or_dataset_id`: path to local file(s) or HuggingFace dataset ID
    - `tokenizer`: tokenizer instance (required. Must have chat template support)
    - `split`: dataset split (e.g., "train", "validation")
    - `name`: dataset configuration/subset name
    - `seq_length`: maximum sequence length for padding/truncation
    - `padding`: padding strategy ("do_not_pad", "max_length", etc.)
    - `truncation`: truncation strategy ("do_not_truncate", "longest_first", etc.)
    - `start_of_turn_token`: token marking assistant response start (for answer-only loss)
    - `chat_template`: optional override for tokenizer's chat template
  - Notes:
    - Requires a tokenizer with chat template support
    - Supports both single-turn and multi-turn tool calling
    - Tool definitions are provided in a `tools` field at the conversation level
    - Tool calls appear in assistant messages via `tool_calls` field
    - Tool responses use the `tool` role
### ChatDataset (Multi-Turn Conversations and Tool Calling)
- Class: `nemo_automodel.components.datasets.llm.ChatDataset`
- Use case: multi-turn conversations and tool calling in OpenAI chat format
- Sources: local JSON/JSONL or Hugging Face Hub dataset ID
- Key args:
  - `path_or_dataset_id`: path to local file(s) or Hugging Face dataset ID
  - `tokenizer`: tokenizer instance (required; must have chat template support)
  - `split`: dataset split (e.g., "train", "validation")
  - `name`: dataset configuration/subset name
  - `seq_length`: maximum sequence length for padding/truncation
  - `padding`: padding strategy ("do_not_pad", "max_length", etc.)
  - `truncation`: truncation strategy ("do_not_truncate", "longest_first", etc.)
  - `start_of_turn_token`: token marking assistant response start (for answer-only loss)
  - `chat_template`: optional override for tokenizer's chat template
:::{note}
- Requires a tokenizer with chat template support
- Supports both single-turn and multi-turn tool calling
- Tool definitions are provided in a `tools` field at the conversation level
- Tool calls appear in assistant messages through the `tool_calls` field
- Tool responses use the `tool` role
:::
- Example YAML:
```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.ChatDataset
  path_or_dataset_id: Salesforce/xlam-function-calling-60k
  split: train
  tokenizer:
    _target_: transformers.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: google/functiongemma-270m-it
  seq_length: 2048
  start_of_turn_token: "<start_of_turn>"
```
  - Expected data format (OpenAI messages format):
```json
{
  "messages": [
    {
      "role": "user",
      "content": "What's the weather in Seattle?"
    },
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
  - For single-turn tool calling (one tool call per conversation), omit the tool response and final assistant message:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Book a table for two at 7pm in Seattle."
    },
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "book_table",
            "arguments": "{\"party_size\": 2, \"time\": \"19:00\", \"city\": \"Seattle\"}"
          }
        }
      ]
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "book_table",
        "description": "Book a restaurant table",
        "parameters": {
          "type": "object",
          "properties": {
            "party_size": {"type": "integer"},
            "time": {"type": "string"},
            "city": {"type": "string"}
          }
        }
      }
    }
  ]
}
```
See the [Function Calling guide](llm/toolcalling.md) for an end-to-end example with FunctionGemma.

### Retrieval/Biencoder (Embedding Fine-Tuning)
- Factory: `nemo_automodel.components.datasets.llm.make_retrieval_dataset`
- Collator: `nemo_automodel.components.datasets.llm.RetrievalBiencoderCollator`
- Use case: embedding model fine-tuning with (query, positive doc, negative docs) contrastive learning
- Supported schemas:
  - Corpus-ID JSON (Merlin/NeMo-retriever style)
  - Inline-text JSONL (e.g., `{"query": "...", "pos_doc": "...", "neg_doc": ["...", "..."]}`)
- Example YAML:
```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.make_retrieval_dataset
  data_dir_list: /abs/path/to/train.jsonl
  data_type: train
  train_n_passages: 5
collate_fn:
  _target_: nemo_automodel.components.datasets.llm.RetrievalBiencoderCollator
  q_max_len: 512
  p_max_len: 512
```
See the detailed guide, [Biencoder retrieval dataset](llm/retrieval-dataset.md), for more information.

- **NanoGPT Binary Shards (pretraining)**
  - Class: `nemo_automodel.components.datasets.llm.nanogpt_dataset.NanogptDataset`
  - Use case: token-level LM pretraining over `.bin` shards produced by NanoGPT-style preprocessors (supports legacy and current formats)
  - Notes:
    - Streams contiguous `seq_len` slices, supports optional BOS alignment and `.bos.idx` sidecar files
### NanoGPT Binary Shards (Pretraining)
- Class: `nemo_automodel.components.datasets.llm.nanogpt_dataset.NanogptDataset`
- Use case: token-level LM pretraining over `.bin` shards produced by NanoGPT-style preprocessors (supports legacy and current formats)
:::{note}
- Streams contiguous `seq_len` slices, supports optional BOS alignment and `.bos.idx` sidecar files
- Related tool: `tools/nanogpt_data_processor.py`
:::

- **Megatron (pretraining; interoperable with pre-tokenized Megatron data)**
  - Class: `nemo_automodel.components.datasets.llm.megatron_dataset.MegatronPretraining`
  - Use case: large-scale LM pretraining over Megatron-LM formatted tokenized corpora
  - Interoperability: If your corpus has already been tokenized/indexed for Megatron (i.e., `.bin`/`.idx` pairs), you can point Automodel to those assets directly. No re-tokenization required.
  - Key args: `paths` (single path, glob, weighted list, or per-split dict), `seq_length`, `tokenizer`, `split`, `index_mapping_dir`, `splits_to_build`
### Megatron (Pretraining; Interoperable With Pre-Tokenized Megatron Data)
- Class: `nemo_automodel.components.datasets.llm.megatron_dataset.MegatronPretraining`
- Use case: large-scale LM pretraining over Megatron-LM formatted tokenized corpora
- Interoperability: If your corpus has already been tokenized/indexed for Megatron (i.e., `.bin`/`.idx` pairs), you can point Automodel to those assets directly. No re-tokenization required.
- Key args: `paths` (single path, glob, weighted list, or per-split dict), `seq_length`, `tokenizer`, `split`, `index_mapping_dir`, `splits_to_build`
- Example YAML:
```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.megatron_dataset.MegatronPretraining
  paths: /abs/path/to/processed_data_*_text_document*  # glob or explicit list
  index_mapping_dir: /abs/path/to/mapping_dir
  tokenizer:
    _target_: transformers.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: openai-community/gpt2
  seq_length: 1024
  split: "0.99, 0.01, 0.00"  # train, validation, test
  splits_to_build: "train"
```
See the detailed [pretraining guide](llm/pretraining.md), which uses MegatronPretraining data.

## Streaming Datasets

Streaming datasets enable processing very large datasets without loading them entirely into memory. This is particularly useful when working with datasets that exceed available RAM or when you want to start training immediately without waiting for the full dataset to download.

### What Are Streaming Datasets?

Streaming datasets load and process data incrementally, one batch at a time, rather than loading the entire dataset into memory upfront. This approach:

- **Reduces memory footprint**: Only the current batch resides in memory
- **Enables training on massive datasets**: Process terabyte-scale datasets on machines with limited RAM
- **Faster startup**: Begin training immediately without waiting for full dataset download
- **Efficient for remote datasets**: Stream directly from Hugging Face Hub without local storage

### When to Use Streaming

Use streaming mode when:

- Your dataset is very large (hundreds of GB or TB)
- Available memory is limited compared to dataset size
- You want to start training quickly without downloading the full dataset
- You're experimenting with a subset of a large dataset

Avoid streaming when:

- Your dataset is small enough to fit comfortably in memory
- You need random access to samples (e.g., for certain sampling strategies)
- You need to know the exact dataset length upfront
- Training requires multiple passes with different orderings

### How to Enable Streaming

For `ColumnMappedTextInstructionDataset`, use the streaming variant by changing the class to `ColumnMappedTextInstructionIterableDataset`:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_iterable_dataset.ColumnMappedTextInstructionIterableDataset
  path_or_dataset_id: Muennighoff/natural-instructions
  split: train
  column_mapping:
    context: definition
    question: inputs
    answer: targets
  answer_only_loss_mask: true
  start_of_turn_token: "<|assistant|>"
```

For Hugging Face datasets loaded directly, set `streaming=True`:

```python
from datasets import load_dataset

# Non-streaming (loads entire dataset into memory)
dataset = load_dataset("large-dataset/corpus", split="train", streaming=False)

# Streaming (loads data incrementally)
dataset = load_dataset("large-dataset/corpus", split="train", streaming=True)
```

### Streaming Limitations

When using streaming datasets, be aware of these limitations:

1. **No random access**: You cannot use `dataset[index]` to access specific samples. Streaming datasets only support iteration.

2. **No length information**: The `len(dataset)` operation is not available. You cannot determine the total number of samples upfront.

3. **Single-pass iteration**: Each iteration consumes the stream. To iterate multiple times, you need to recreate the dataset or use the `repeat_on_exhaustion` parameter.

4. **Limited shuffling**: Shuffling is done with a buffer (not the entire dataset), which may not provide perfect randomization.

### Distributed Training with Streaming

Streaming datasets support distributed training through sharding:

```python
from nemo_automodel.components.datasets.llm.column_mapped_text_instruction_iterable_dataset import (
    ColumnMappedTextInstructionIterableDataset
)

dataset = ColumnMappedTextInstructionIterableDataset(
    path_or_dataset_id="large-dataset/corpus",
    column_mapping={"question": "input", "answer": "output"},
    tokenizer=tokenizer,
)

# Shard the dataset across workers
dataset = dataset.shard(num_shards=8, index=worker_id)

# Enable shuffling with a buffer
dataset = dataset.shuffle(buffer_size=10000, seed=42)

# Set epoch for deterministic shuffling across epochs
dataset.set_epoch(epoch_num)
```

### Performance Considerations

**Memory vs. Speed Trade-offs**:
- Streaming reduces memory usage but may be slower than in-memory datasets
- Network latency can impact streaming performance for remote datasets
- Use local caching when repeatedly accessing the same remote dataset

**Buffer Size for Shuffling**:
- Larger buffers provide better randomization but use more memory
- A buffer size of 10,000-100,000 samples is typically a good balance
- For perfect shuffling, you need a buffer size equal to the dataset size (defeating the purpose of streaming)

**Prefetching**:
- Most streaming implementations prefetch data in the background
- This helps hide network latency and keeps GPUs busy
- Adjust prefetch settings based on your network speed and batch size

### Example: Streaming a Large Dataset

Here's a complete example of using streaming for a large instruction-tuning dataset:

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_iterable_dataset.ColumnMappedTextInstructionIterableDataset
  path_or_dataset_id: HuggingFaceH4/ultrachat_200k
  split: train_sft
  column_mapping:
    question: prompt
    answer: completion
  answer_only_loss_mask: true
  start_of_turn_token: "<|assistant|>"
  repeat_on_exhaustion: true  # Automatically restart when stream ends

dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 4
  num_workers: 4
```

This configuration:
- Streams the dataset without loading it fully into memory
- Automatically repeats when the stream is exhausted
- Uses multiple workers for efficient data loading
- Applies answer-only loss masking during tokenization

## Packed Sequence Support
To reduce padding and improve throughput with variable-length sequences:
```yaml
packed_sequence:
  packed_sequence_size: 8192   # > 0 enables packing
  split_across_pack: false
```
Use a collator that pads to an FP8-friendly multiple when training with FP8:
```yaml
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  collate_fn:
    _target_: nemo_automodel.components.datasets.utils.default_collater
    pad_seq_len_divisible: 16
```

---

## VLM Datasets (Vision/Audio + Language)
VLM datasets are represented as conversations (message lists) that combine text with images or audio and are processed with the model's `AutoProcessor.apply_chat_template` and a suitable collate function.

Built-in dataset makers (return lists of `conversation` dicts):
- **RDR items**: `nemo_automodel.components.datasets.vlm.datasets.make_rdr_dataset` (HF: `quintend/rdr-items`)
- **CORD-V2 receipts (Consolidated Receipt Dataset for Post-OCR Parsing)**: `nemo_automodel.components.datasets.vlm.datasets.make_cord_v2_dataset` (HF: `naver-clova-ix/cord-v2`)
- **MedPix-VQA (Medical Pixel Question Answering)**: `nemo_automodel.components.datasets.vlm.datasets.make_medpix_dataset`
- **CommonVoice 17 (CV17) (audio)**: `nemo_automodel.components.datasets.vlm.datasets.make_cv17_dataset`


Each example follows the conversation schema expected by `apply_chat_template`, e.g.:
```python
{
  "conversation": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": example_image},
        {"type": "text",  "text":  "Describe this image."}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": ground_truth_text}]
    }
  ]
}
```

### Collate Functions
- `nemo_automodel.components.datasets.vlm.collate_fns.default_collate_fn`
- `nemo_automodel.components.datasets.vlm.collate_fns.qwen2_5_collate_fn` (Qwen2.5 VL)
- `nemo_automodel.components.datasets.vlm.collate_fns.phi4_mm_collate_fn` (audio)

Select in your YAML:
```yaml
dataloader:
  _target_: torchdata.stateful_dataloader.StatefulDataLoader
  batch_size: 1
  collate_fn:
    _target_: nemo_automodel.components.datasets.vlm.collate_fns.qwen2_5_collate_fn
```
If you want answer-only loss masking, provide a model-appropriate `start_of_response_token` to the collate function.

See [Gemma-3n](omni/gemma3-3n.md) and [VLM dataset](vlm/dataset.md) for end-to-end examples.

---

## Bring Your Own Dataset
You can integrate custom datasets with zero code changes to NeMo Automodel by using `_target_` in YAML. There are three approaches:

### Point to an Existing Class or Function (Dotted Path)
- LLM example (class):
```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.hellaswag.HellaSwag
  path_or_dataset: rowan/hellaswag
  split: train
```
- LLM example (factory function):
```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.squad.make_squad_dataset
  split: train
  dataset_name: rajpurkar/squad
```
- VLM example (factory function):
```yaml
dataset:
  _target_: nemo_automodel.components.datasets.vlm.datasets.make_cord_v2_dataset
  split: train
```

### Point to a Local Python File and Function
```yaml
dataset:
  _target_: /abs/path/to/my_custom_dataset.py:build_my_dataset
  some_arg: 123
  split: train
```
Where `build_my_dataset` returns either a `datasets.Dataset` or a list/iterator of conversation dicts (for VLM).

### Use ColumnMappedTextInstructionDataset for Most Instruction Datasets (LLM)
- Ideal when your data has columns like `instruction`, `input`, or `output` but with arbitrary names
- Supports local JSON/JSONL and HF Hub
```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_dataset.ColumnMappedTextInstructionDataset
  path_or_dataset_id: /abs/path/to/*.jsonl  # or org/repo on HF
  column_mapping:
    context: definition
    question: inputs
    answer: targets
  answer_only_loss_mask: true
  start_of_turn_token: "<|assistant|>"
```

### Implement a Minimal Custom Class Pattern (LLM Completion)
If you prefer Python, implement `get_context` and `get_target` and reuse the built-in preprocessor:
```python
from datasets import load_dataset
from nemo_automodel.components.datasets.utils import SFTSingleTurnPreprocessor

class MyCompletionDataset:
    def __init__(self, path_or_dataset, tokenizer, split="train"):
        raw_ds = load_dataset(path_or_dataset, split=split)
        self.dataset = SFTSingleTurnPreprocessor(tokenizer).process(raw_ds, self)

    def get_context(self, examples):
        return examples["my_context_field"]

    def get_target(self, examples):
        return examples["my_target_field"]
```
Then reference your class via `_target_` in YAML.

### Important Considerations
- **Chat templates**: If your tokenizer has a chat template and you want answer-only loss, provide the correct `start_of_turn_token` (LLM) or `start_of_response_token` (VLM collate functions).
- **Padding for FP8**: If training with FP8, set `pad_seq_len_divisible: 16` in your collate function to align sequence lengths.
- **Packed sequences**: Prefer packed sequences for throughput when fine-tuning LLMs on variable-length corpora.
- **Validation**: You can define a separate `validation_dataset` and `validation_dataloader` block mirroring your training config.

For detailed, end-to-end recipes, browse the example configs under [examples/llm_finetune/](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_finetune), [examples/llm_pretrain/](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_pretrain), and [examples/vlm_finetune/](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/vlm_finetune).

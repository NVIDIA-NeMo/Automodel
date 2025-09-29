# Dataset Overview: LLM and VLM Datasets in NeMo Automodel

This page summarizes the datasets already supported in NeMo Automodel for LLM and VLM, and shows how to plug in your own datasets via simple Python functions or purely through YAML using the `_target_` mechanism.

- See also: [LLM datasets](llm/dataset.md) and [VLM datasets](vlm/dataset.md) for deeper, task-specific guides.

- If a dataset you need is missing, please open a [GitHub issue](https://github.com/NVIDIA-NeMo/Automodel/issues) with a short description and example schema so we can prioritize support.
---

## LLM Datasets

NeMo Automodel supports several common patterns for language modeling and instruction tuning.

- **HellaSwag (completion SFT)**
  - Wrapper: `nemo_automodel.components.datasets.llm.hellaswag.HellaSwag`
  - Use case: single-turn completion style SFT where a prompt (ctx) is followed by a gold continuation (ending)
  - Key args: `path_or_dataset`, `split`, `num_samples_limit`
  - Example YAML:
```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.hellaswag.HellaSwag
  path_or_dataset: rowan/hellaswag
  split: train
```

- **SQuAD-style QA (instruction SFT)**
  - Factory: `nemo_automodel.components.datasets.llm.squad.make_squad_dataset`
  - Use case: instruction/QA tuning with either prompt+answer formatting or chat-template formatting
  - Notes:
    - If the tokenizer has a chat template and you want answer-only loss, you must provide `start_of_turn_token`.
    - Optional `seq_length` can be used for padding/truncation.
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
  - See detailed guide: (here)[llm/column-mapped-text-instruction-dataset.md].

- **NanoGPT Binary Shards (pretraining)**
  - Class: `nemo_automodel.components.datasets.llm.nanogpt_dataset.NanogptDataset`
  - Use case: token-level LM pretraining over `.bin` shards produced by NanoGPT-style preprocessors (supports legacy and current formats)
  - Notes:
    - Streams contiguous `seq_len` slices, supports optional BOS alignment and `.bos.idx` sidecar files
    - Related tool: `tools/nanogpt_data_processor.py`

- **Megatron (pretraining; interoperable with pre-tokenized Megatron data)**
  - Class: `nemo_automodel.components.datasets.llm.megatron_dataset.MegatronPretraining`
  - Use case: large-scale LM pretraining over Megatron-LM formatted tokenized corpora
  - Interoperability: if your corpus has already been tokenized/indexed for Megatron (i.e., `.bin`/`.idx` pairs), you can point Automodel to those assets directly; no re-tokenization required
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
 - See detailed pretrained guide (here)[llm/mcore_pretraining.md] using MegatronPretraining data.

> ⚠️ Note: Multi-turn conversational and tool-calling/function-calling dataset support is coming soon.

## Packed Sequence Support
To reduce padding and improve throughput with variable-length sequences:
```yaml
packed_sequence:
  packed_sequence_size: 8192   # > 0 enables packing
  split_across_pack: false
```
Use a collater that pads to an FP8-friendly multiple when training with FP8:
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
- **CORD-V2 receipts**: `nemo_automodel.components.datasets.vlm.datasets.make_cord_v2_dataset` (HF: `naver-clova-ix/cord-v2`)
- **MedPix-VQA (medical)**: `nemo_automodel.components.datasets.vlm.datasets.make_medpix_dataset`
- **CommonVoice 17 (audio)**: `nemo_automodel.components.datasets.vlm.datasets.make_cv17_dataset`

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

### Collate functions
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
You can integrate custom datasets with zero code changes to NeMo Automodel by using `_target_` in YAML. Two main approaches:

### 1) Point to an existing class or function (dotted path)
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

### 2) Point to a local Python file and function
```yaml
dataset:
  _target_: /abs/path/to/my_custom_dataset.py:build_my_dataset
  some_arg: 123
  split: train
```
Where `build_my_dataset` returns either a `datasets.Dataset` or a list/iterator of conversation dicts (for VLM).

### 3) Use ColumnMappedTextInstructionDataset for most instruction datasets (LLM)
- Ideal when your data has columns like `instruction`, `input`, `output` but with arbitrary names
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

### Minimal custom class pattern (LLM completion)
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

### Tips and gotchas
- **Chat templates**: If your tokenizer has a chat template and you want answer-only loss, provide the correct start-of-turn token (LLM) or start-of-response token (VLM collate functions).
- **Padding for FP8**: If training with FP8, set `pad_seq_len_divisible: 16` in your collate function to align sequence lengths.
- **Packed sequences**: Prefer packed sequences for throughput when fine-tuning LLMs on variable-length corpora.
- **Validation**: You can define a separate `validation_dataset` and `validation_dataloader` block mirroring your training config.

For detailed, end-to-end recipes, browse the example configs under [examples/llm_finetune/](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_finetune), [examples/llm_pretrain/](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/llm_pretrain), and [examples/vlm_finetune/](https://github.com/NVIDIA-NeMo/Automodel/tree/main/examples/vlm_finetune).
